"""Playable browser demo: tree B1 Stage2 world model, real time on ONE GPU, controlled with W/A/S/D.

The generation loop is the single-GPU real-time path of shownow/latency/fast_long_infer.py (FastTree + CUDA graph +
per-layer compile + chosen-only commit + TAEHV decode), driven by the browser instead of a key script:

  1. the 4 candidates (W/A/S/D) of the next latent (0.25 s = 4 frames) are generated speculatively;
  2. the server waits until the frames already on screen are nearly used up (--lead frames left), then reads the key
     held right now, commits only that candidate into the K/V ring and decodes it;
  3. no key held -> the world pauses (the model only knows W/A/S/D); the precomputed candidates stay valid, so the
     next key press is served immediately.

Browser keys: W/A/S/D hold to move, Q/E turn, F fire, Space jump (8-action models), R restart the current scene from its real 2 s history, N next scene.
Video is MJPEG over HTTP (16 fps), keys are small POSTs; no extra Python packages needed. The server binds to
127.0.0.1 by default: reach it through an ssh tunnel (ssh -L 8765:localhost:8765 <this node>).

  SHOWNOW_N_ACTIONS=9 python -m shownow.demo.play_server --student checkpoints/shownow_8a_stage2_step001000.pt [--port 8765]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("SOLARWM_COMPILE_FLEX", "1")
os.environ["SHOWNOW_COMPILE_LAYER"] = "1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
sys.path.insert(0, str(REPO / "third_party/taehv"))
import cv2  # noqa: E402
import torch  # noqa: E402

from shownow.eval.camera_prior_eval import LAT  # noqa: E402
from shownow.eval.long_drift_eval import KEYS, camera_tokens_long  # noqa: E402
from shownow.latency.tree_fast import FRAME_TOKENS, FastTree, commit_view  # noqa: E402
from shownow.train.student_common import HIST_LATENTS, allocate_caches, tokens  # noqa: E402

FPS = 16
KEY_OF = {k: i for k, i in {"w": 1, "a": 2, "s": 3, "d": 4, "q": 5, "e": 6, "f": 7, "space": 8}.items() if i in KEYS}
# browser keys -> action ids (8-action models: Q / E turn left / right, F attack, Space jump)


class Controls:
    """Keys held in the browser (most recently pressed wins) + reset / scene requests."""

    def __init__(self):
        self.lock = threading.Lock()
        self.held: list[int] = []
        self.reset = False
        self.scene_step = 0

    def key(self, k: int, down: bool):
        with self.lock:
            if k in self.held:
                self.held.remove(k)
            if down:
                self.held.append(k)

    def current(self):
        with self.lock:
            return self.held[-1] if self.held else None

    def request_reset(self, next_scene: bool):
        with self.lock:
            self.reset = True
            self.scene_step += int(next_scene)
            self.held.clear()

    def take_reset(self):
        with self.lock:
            r, self.reset = self.reset, False
            s, self.scene_step = self.scene_step, 0
            return r, s

    def reset_pending(self):
        with self.lock:
            return self.reset


class FramePipe:
    """Queue of JPEG frames played out at 16 fps; HTTP clients wait for the next displayed frame."""

    def __init__(self):
        self.q = collections.deque()
        self.cond = threading.Condition()
        self.cur, self.seq = None, 0
        threading.Thread(target=self._pace, daemon=True).start()

    def push(self, jpgs):
        with self.cond:
            self.q.extend(jpgs)

    def pending(self):
        return len(self.q)

    def clear(self):
        with self.cond:
            self.q.clear()

    def _pace(self):
        t = time.time()
        while True:
            with self.cond:
                if self.q:
                    self.cur = self.q.popleft()
                    self.seq += 1
                    self.cond.notify_all()
            t += 1.0 / FPS
            time.sleep(max(0.0, t - time.time()))
            if time.time() - t > 0.5:  # fell far behind (e.g. machine hiccup): resync the clock
                t = time.time()

    def wait_next(self, seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq and self.cur is not None, timeout)
            return self.seq, self.cur


class Engine:
    def __init__(self, args, ctl: Controls, pipe: FramePipe):
        self.args, self.ctl, self.pipe = args, ctl, pipe
        self.status = {"state": "loading model", "scene": None, "step": 0, "gen_ms": None, "commit_ms": None,
                       "key": None, "video_s": 0.0}
        torch._dynamo.config.cache_size_limit = 64
        torch.backends.cuda.matmul.allow_tf32 = True
        self.dev = dev = torch.device(getattr(args, "device", "cuda"))
        from shownow.train.student_stage2 import build_stage2_student
        from solarwm.backends.wan22.sgf import warp_denoising_steps
        from taehv import TAEHV, StreamingTAEHV
        self.BL = int(getattr(args, "block", 1))  # latents per AR block: 1 = ShowNow B1, 3 = SolarWM B3
        d, _, _ = build_stage2_student(args.student, field="generator_ema", block=self.BL, history=args.history)
        self.m = d.module.to(dev, torch.bfloat16).eval()
        self.m.freqs = self.m.freqs.to(dev)
        self.fewstep = [float(s) for s in warp_denoising_steps(
            [int(v) for v in getattr(args, "denoising_steps", "1000,750,500,250").split(",")], d.scheduler.timesteps.to(dev),
                                                               num_train_timesteps=1000)]
        tab_t, tab_s = d.scheduler.timesteps.float(), d.scheduler.sigmas.float()
        self.sig = [float(tab_s[(tab_t - t).abs().argmin()]) for t in self.fewstep]
        tae = TAEHV(str(REPO / "third_party/taehv/taew2_2.pth")).to(dev, torch.float16).eval()
        self.tae = StreamingTAEHV(tae)
        if args.data == "sw8a":  # 8-action key-switch forks on the held-out maps (shownow/eval/sw8a_split.py)
            from shownow.eval.sw8a_split import split
            from shownow.tree.cond_v3 import split_name
            self.scenes = split()["test_e3" if split_name() == "test" else "scenes"] \
                if not os.environ.get("SHOWNOW_SW8A_SCENES") else os.environ["SHOWNOW_SW8A_SCENES"].split(",")  # dev / timing
        elif args.data in ("v3", "mix"):  # roaming / key-switch validation forks (map-disjoint), key-stream cameras
            from shownow.tree.cond_v3 import v3_split
            _, _, vd, vs = v3_split()
            val = sorted(vd, key=lambda f: f.name)
            self.scenes = [str(f) for f in val[::7]]
            from shownow.tree.cond_v3 import split_name, test_forks
            if split_name() == "test":  # held-out test scenes (E3 on the test split)
                self.scenes = [str(f) for f in test_forks(val, 20, offset=96)]
            if args.data == "mix":  # interleave held-out SolarWM-Data clips (real walking / walkthroughs / games)
                by_src = {}
                for f in sorted(vs, key=lambda f: f.name):
                    by_src.setdefault(f.name.split("__")[0], []).append(str(f))
                order = ["sekai_walking-clean", "dl3dv-10s", "realcam_vid", "sekai_game", "abot"]
                real = [x for i in range(20) for src in order if i < len(by_src.get(src, [])) for x in [by_src[src][i]]]
                doom = self.scenes
                self.scenes = [x for pair in zip(real, doom) for x in pair] + real[len(doom):]
        else:
            split = json.load(open(REPO / "runs/teacher_ft/wasd_v1/split.json"))
            val = sorted(split["val_ood"])
            self.scenes = [val[i * 7] for i in range(len(val) // 7)]  # first 5 = the forks of the published videos
        self.scene_idx = 0

    # ------------------------------------------------------------------ helpers
    def dec_push(self, lat):
        outf = []
        f = self.tae.decode(lat.half())
        while f is not None:
            outf.append(f)
            f = self.tae.decode()
        x = torch.cat(outf, 1)[0].clamp(0, 1)  # [n, 3, H, W]
        x = (x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [cv2.imencode(".jpg", fr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])[1].tobytes()
                for fr in x]

    def decide(self, warm):
        """Wait until the screen buffer is nearly empty and a key is held; None = reset requested."""
        if warm:
            return 1 + self.status["step"] % len(KEYS)
        while True:
            if self.ctl.reset_pending():
                return None
            k = self.ctl.current()
            if k is not None and self.pipe.pending() <= self.args.lead:
                return k
            if k is None and self.pipe.pending() == 0:
                self.status["state"] = "paused (hold W/A/S/D)"
            time.sleep(0.002)

    # ------------------------------------------------------------------ scenes
    @torch.no_grad()
    def load_scene(self, sid):
        """Commit the first P real latents of the fork (2 s history + branch-0 future) with fresh text caches."""
        dev, m, P = self.dev, self.m, self.args.history
        n_tok = FRAME_TOKENS
        static, scale = False, self.args.cam_scale
        if "swm_sub" in sid:  # SolarWM-Data clip: real camera for the intro, then action-only (static camera)
            from shownow.tree.cond_v3 import cam_dir_for
            fpath = Path(sid)
            fork = torch.load(fpath, map_location="cpu", weights_only=False)
            nominal = torch.load(cam_dir_for(fpath.parent) / fpath.name, weights_only=True)["true"].float()
            stream, static, scale = None, True, float(fork.get("cam_scale", 1.0))
            sid = fork["meta"]["source"] + ":" + fpath.stem.split("__")[1][:10]
        elif sid.endswith(".pt"):  # v3 scene: file path
            from shownow.data.nominal_8a import KeyStream8
            from shownow.data.nominal_switch import KeyStream
            from shownow.tree.cond_v3 import cam_dir_for
            fpath = Path(sid)
            fork = torch.load(fpath, map_location="cpu", weights_only=False)
            camf = torch.load(cam_dir_for(fpath.parent) / fpath.name, weights_only=True)
            nominal = camf["nominal"]
            stream = (KeyStream8 if len(KEYS) == 8 else KeyStream)(camf["fork"].double().numpy())
            for s_ in fork["meta"]["switch_schedules"]["W"]:  # replay branch 0's keys over the intro latents 9-14
                for _ in range(s_["n_latents"]):
                    if stream.j < P - HIST_LATENTS:
                        stream.commit({v: k for k, v in KEYS.items()}[s_["key"]])
            sid = fpath.stem
        else:
            stream = None
            fork = torch.load(LAT / f"{sid}.pt", map_location="cpu", weights_only=False)
            nominal = torch.load(REPO / "data/cf_vizdoom_10k_cam" / f"{sid}.pt", weights_only=True)["nominal"]
        self.cur_scale = scale
        sc = {"sid": sid, "static": static, "scale": scale,
              "vel": {k: (nominal[k - 1, 20, :3, 3] - nominal[k - 1, HIST_LATENTS - 1, :3, 3]) / (21 - HIST_LATENTS)
                      for k in KEYS} if nominal.shape[0] == 4 else {k: torch.zeros(3) for k in KEYS},
              "rot": nominal[0, 20, :3, :3], "lat": fork["latents"][:1, :P].to(dev),
              "poses": [nominal[0, i] for i in range(P)], "committed": [], "stream": stream}
        acts = [int(a) for a in fork["actions"][0, :P]]
        xa = allocate_caches(m, 1, 1, torch.bfloat16, dev)[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = m.text_embedding(fork["prompt"].to(dev)[None])
            BL = self.BL
            for c0 in range(0, P, BL):  # one AR block at a time
                ft = self.make_ft(sc["committed"], c0, 1, share=False)
                cam = self.cams_for([sc["poses"][c0:c0 + BL]])
                act = tokens(torch.tensor([acts[c0:c0 + BL]], device=dev).float()).long()
                fr = ft.freqs_for_positions(torch.arange(c0, c0 + BL, device=dev))
                if c0:
                    ft.prepare_prefix()
                ft.forward(sc["lat"][:, c0:c0 + BL], torch.zeros(1, BL, device=dev), act, ctx, xa, cam, fr, commit=True)
                sc["committed"].append((ft.ck[:, 0:1].clone(), ft.cv[:, 0:1].clone(),
                                        torch.cat([cam["viewmats"].flatten(2), cam["K"].flatten(2)], 2)))
                del ft
        sc["ctx"], sc["xa"] = ctx, xa
        assert all(c["is_init"] for c in xa)
        return sc

    def make_ft(self, committed, n_prefix, K, share):
        n_tok = FRAME_TOKENS
        ft = FastTree(self.m, K, prefix_latents=n_prefix, share=share, h=15, w=27, block=self.BL)
        self.fill_ring(ft, committed[-(n_prefix // self.BL):] if n_prefix else [])
        return ft

    def fill_ring(self, ft, committed):
        n_tok = FRAME_TOKENS * self.BL  # one ring entry = one AR block
        for i, (k_, v_, c_) in enumerate(committed):
            s = i * n_tok
            ft.rk[:, :, s:s + n_tok].copy_(k_)
            ft.rv[:, :, s:s + n_tok].copy_(v_)
            ft.pcam_v[:, s:s + n_tok].copy_(c_[..., :16].unflatten(2, (4, 4)))
            ft.pcam_K[:, s:s + n_tok].copy_(c_[..., 16:].unflatten(2, (3, 3)))
        ft.ppos.copy_(torch.arange(len(committed) * self.BL, device=self.dev))

    def cams_for(self, pose_lists):
        return camera_tokens_long(torch.stack([q for ps in pose_lists for q in ps])[None], self.dev,
                                  getattr(self, "cur_scale", self.args.cam_scale))

    @torch.no_grad()
    def build_graphs(self, sc):
        """Static FastTree (prefix P, 4 candidates) + CUDA graphs, captured ONCE; scenes are swapped in place."""
        dev, P, KL, BL = self.dev, self.args.history, len(KEYS), self.BL
        n_tok = FRAME_TOKENS * BL  # tokens of one candidate block
        C, Hh, Ww = 48, 30, 54
        fewstep, sig = self.fewstep, self.sig
        self.xa, self.ctx = sc["xa"], sc["ctx"]  # persistent cross-attn caches: later scenes copy into these
        self.steady = steady = self.make_ft(sc["committed"], P, KL, share=True)
        self.act = act = tokens(torch.tensor([[k for k in KEYS for _ in range(BL)]], device=dev).float()).long()
        cam = self.cams_for([[sc["poses"][-1]] * BL] * KL)
        self.st_cam = {k_: v_.clone() for k_, v_ in cam.items()}
        st_fr = steady.freqs_for_positions(torch.arange(P, P + BL, device=dev).repeat(KL))
        st_ts = [torch.full((1, KL * BL), t, device=dev) for t in fewstep]
        self.st_x = torch.empty(1, KL * BL, C, Hh, Ww, device=dev, dtype=torch.bfloat16)
        self.cv1 = cv1 = commit_view(steady, 1)
        self.c1_x = torch.zeros(1, BL, C, Hh, Ww, device=dev, dtype=torch.bfloat16)
        self.c1_cam = {k_: v_[:, :n_tok].clone() for k_, v_ in cam.items()}
        self.c1_act = act[:, :n_tok].clone()
        c1_fr = steady.freqs_for_positions(torch.arange(P, P + BL, device=dev))
        c1_zero = torch.zeros(1, BL, device=dev)
        xa, ctx, st_cam, st_x = self.xa, self.ctx, self.st_cam, self.st_x

        def gen_block():
            steady.prepare_prefix()
            x = torch.randn_like(st_x)
            for i in range(len(fewstep)):
                flow = steady.forward_fused(x, st_ts[i], act, ctx, xa, st_cam, st_fr, commit=False)
                x0 = x.float() - sig[i] * flow.float()
                x = x0.to(torch.bfloat16) if i == len(fewstep) - 1 else \
                    ((1 - sig[i + 1]) * x0 + sig[i + 1] * torch.randn_like(x0)).to(torch.bfloat16)
            st_x.copy_(x)

        def commit_block():
            cv1.forward_fused(self.c1_x, c1_zero, self.c1_act, ctx, xa, self.c1_cam, c1_fr, commit=True)

        graphs = []
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for fn in (gen_block, commit_block):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                    fn()
                graphs.append(g)
        self.graph, self.graph_c1 = graphs
        # the graphs read these by address: keep them (and the closures holding them) alive for the server's lifetime
        self._graph_inputs = (gen_block, commit_block, st_fr, st_ts, c1_fr, c1_zero)

    def install(self, sc):
        """Swap a freshly loaded scene into the captured graphs' buffers (text caches + K/V ring)."""
        if sc["xa"] is not self.xa:
            for dst, src in zip(self.xa, sc["xa"]):
                dst["k"].copy_(src["k"])
                dst["v"].copy_(src["v"])
            self.ctx.copy_(sc["ctx"])
        self.fill_ring(self.steady, sc["committed"][-self.args.history:])
        self.ring_ptr = 0

    # ------------------------------------------------------------------ play one scene
    @torch.no_grad()
    def episode(self, sc, warm_steps=0):
        assert self.BL == 1, "the browser demo runs the B1 ShowNow path"
        dev, P, n_tok, KL = self.dev, self.args.history, FRAME_TOKENS, len(KEYS)
        steady, cam_buf, act = self.steady, self.st_cam, self.act
        poses, vel, rot = list(sc["poses"]), sc["vel"], sc["rot"]
        self.cur_scale = sc.get("scale", self.args.cam_scale)
        self.install(sc)
        self.tae.reset()
        jpgs = self.dec_push(sc["lat"])  # the real 3.75 s intro plays first
        if not warm_steps:
            self.pipe.clear()
            self.pipe.push(jpgs)
        self.status.update(scene=sc["sid"], step=0, video_s=0.0, key=None, state="playing")
        step = 0
        while True:
            if warm_steps and step >= warm_steps:
                return
            if not warm_steps and self.ctl.reset_pending():
                return
            t0 = time.time()
            cand = {}
            for k in KEYS:
                if sc.get("static"):  # real-scene clips: action-only input, camera frozen at the last pose
                    cand[k] = poses[-1].clone()
                    continue
                if sc["stream"] is not None:  # v3: same key-stream camera as the training data
                    cand[k] = torch.from_numpy(sc["stream"].peek(k)).float()
                    continue
                p = poses[-1].clone()
                p[:3, :3] = rot
                p[:3, 3] = p[:3, 3] + vel[k]
                cand[k] = p
            cam = self.cams_for([[cand[k]] for k in KEYS])
            for k_ in cam_buf:
                cam_buf[k_].copy_(cam[k_])
            self.graph.replay()  # 4 candidates of the next latent
            torch.cuda.synchronize()
            gen_ms = (time.time() - t0) * 1000
            key = self.decide(bool(warm_steps))  # <- the key is read only now, after all 4 candidates exist
            if key is None:
                return
            t1 = time.time()
            j = key - 1
            lat = self.st_x[:, j:j + 1]
            self.c1_x.copy_(lat)
            for k_ in self.c1_cam:
                self.c1_cam[k_].copy_(cam[k_][:, j * n_tok:(j + 1) * n_tok])
            self.c1_act.copy_(act[:, j * n_tok:(j + 1) * n_tok])
            self.graph_c1.replay()  # commit pass for the pressed key only
            s = self.ring_ptr * n_tok  # chosen latent -> oldest ring slot
            steady.rk[:, :, s:s + n_tok].copy_(self.cv1.ck[:, 0:1])
            steady.rv[:, :, s:s + n_tok].copy_(self.cv1.cv[:, 0:1])
            steady.pcam_v[:, s:s + n_tok].copy_(cam["viewmats"][:, j * n_tok:(j + 1) * n_tok])
            steady.pcam_K[:, s:s + n_tok].copy_(cam["K"][:, j * n_tok:(j + 1) * n_tok])
            steady.ppos.sub_(1)
            steady.ppos[self.ring_ptr] = P - 1
            self.ring_ptr = (self.ring_ptr + 1) % P
            jpgs = self.dec_push(lat)
            if not warm_steps:
                self.pipe.push(jpgs)
            commit_ms = (time.time() - t1) * 1000
            poses.append(cand[key])
            if sc["stream"] is not None:
                sc["stream"].commit(key)
            step += 1
            self.status.update(step=step, gen_ms=round(gen_ms, 1), commit_ms=round(commit_ms, 1),
                               key=KEYS[key], video_s=round(step * 0.25, 2), state="playing")

    def run(self):
        try:
            self.status["state"] = "warming up (compile + CUDA graphs)"
            t = time.time()
            sc = self.load_scene(self.scenes[0])
            self.build_graphs(sc)
            self.episode(sc, warm_steps=4)
            print(f"warm-up done in {time.time() - t:.0f} s", flush=True)
            while True:
                _, ds = self.ctl.take_reset()
                self.scene_idx = (self.scene_idx + ds) % len(self.scenes)
                self.status["state"] = "starting scene"
                t = time.time()
                sc = self.load_scene(self.scenes[self.scene_idx])
                print(f"scene {sc['sid']} loaded in {time.time() - t:.2f} s", flush=True)
                self.episode(sc)
        except Exception:
            self.status["state"] = "ERROR: " + traceback.format_exc().splitlines()[-1]
            traceback.print_exc()


def make_handler(ctl: Controls, pipe: FramePipe, engine_ref):
    page = (Path(__file__).with_name("play.html")).read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/status":
                e = engine_ref[0]
                st = dict(e.status) if e else {"state": "loading model"}
                st["buffer_frames"] = pipe.pending()
                return self._send(200, json.dumps(st).encode(), "application/json")
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        new, jpg = pipe.wait_next(seq)
                        if new == seq or jpg is None:
                            continue
                        seq = new
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/key":
                k = KEY_OF.get(str(body.get("key", "")).lower())
                if k:
                    ctl.key(k, bool(body.get("down")))
            elif self.path == "/release_all":
                for k in KEY_OF.values():
                    ctl.key(k, False)
            elif self.path == "/reset":
                ctl.request_reset(bool(body.get("next")))
            return self._send(200, b"{}", "application/json")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(REPO / "checkpoints/shownow_8a_stage2_step001000.pt"))
    ap.add_argument("--denoising-steps", default="1000,667,333", help="SGF schedule (the 8-action student: 3 steps)")
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lead", type=int, default=1, help="frames still queued when the key is read (1 = 62.5 ms)")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--data", default="sw8a", choices=["sw8a", "v3", "mix", "v1"],
                    help="sw8a = 8-action key-switch evaluation scenes (shownow/eval/sw8a_split.py; default); v3 / mix / v1 "
                         "= 4-action data sets of the earlier models")
    args = ap.parse_args()
    ctl, pipe, engine_ref = Controls(), FramePipe(), [None]
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(ctl, pipe, engine_ref))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    engine_ref[0] = Engine(args, ctl, pipe)
    engine_ref[0].run()


if __name__ == "__main__":
    main()
