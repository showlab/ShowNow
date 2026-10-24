"""E3 / checkpoint selection: 60 s rollouts of the final student on fixed v3 Doom validation scenes, a new key every
2 s (same key script and camera trajectory for every method), and their long-horizon metrics.

  gen --mode realtime   the deployed ShowNow path (play_server / latency_harness engine: 4 candidates per latent in
                        one shared-history tree forward, CUDA graphs, the scripted key's candidate committed into the
                        K/V ring). Saves latents, keys and the camera poses actually used.
  gen --mode reference  D1, normal non-speculative inference: the training-faithful cache-free sliding window
                        (15 prior latents + 1 noisy, window-relative RoPE, 4 SGF steps), one future, replaying the
                        keys and poses saved by a realtime run (--from).
  score                 TAEHV decode (same decoder for every method) and metrics per run directory:
    key_idm_acc   key IDM (shownow/eval/idm.py) reads the scripted key, settled latents (>= 2 latents after a switch);
                  no real 60 s future exists, so the key is the reference (walls may legitimately stop the player)
    along_key     motion IDM: generated motion points along the key (cos > 0.5), settled latents
    fd            per 5 s window: Frechet distance between the latent channel mean/std statistics (96-d per latent) of
                  all videos' latents in that window and all real latents of the same scenes (4 branches x 21);
                  floor = FD between two disjoint halves of the real latents. Main drift metric.
    drift         per video and 5 s window: L2 distance of the window-mean statistics to the real scenes' mean
                  (floor = the real scenes' own spread around that mean)
    sharp/contrast per 5 s window: Laplacian variance / gray std of the frames relative to the real frames of the same
                  scenes (pooled; the player walks into other rooms, so the own 2 s history is not a reference);
                  washed-out window = sharpness or contrast ratio < 0.5
    boundary      mean |frame diff| across latent boundaries / within latents (1 = seamless); real reference from the
                  decoded real latents of the same scenes
    red_jumps     frame-to-frame mean redness change > 15, per minute

  python -m shownow.eval.long_eval gen --mode realtime --student <ckpt> --out runs/long_eval/<name>/rt [--rank r --world n]
  python -m shownow.eval.long_eval gen --mode reference --student <ckpt> --from runs/long_eval/<name>/rt --out .../ref
  python -m shownow.eval.long_eval score runs/long_eval/<name>/rt [more run dirs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
sys.path.insert(0, str(REPO / "third_party/taehv"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

KEYS = {1: "W", 2: "A", 3: "S", 4: "D"}
P = 15  # real latents committed before the scripted minute (play_server.load_scene)
WIN = 20  # latents per 5 s window
SETTLE = 2  # latents after a switch that are not scored (Doom momentum, evaluation_plan §3.4)


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def gen_realtime(args):
    from shownow.eval.latency_harness import LatencyEngine
    from shownow.latency.long_infer import key_schedule
    from shownow.latency.tree_fast import FRAME_TOKENS
    args.jpeg_quality, args.lead, args.data = 85, 2, "v3"
    out = Path(args.out)
    (out / "lat").mkdir(parents=True, exist_ok=True)
    assert args.block == 1, "the ShowNow real-time path is B1 (use --mode solarwm for SolarWM-FT blocks)"
    eng = LatencyEngine(args, None, None)
    n_tok, n_lat, per = FRAME_TOKENS, args.seconds * 4, int(round(args.switch_sec * 4))
    mine = list(range(args.videos))[args.rank::args.world]
    if not mine:
        return
    sc = eng.load_scene(eng.scenes[mine[0] % len(eng.scenes)])
    eng.build_graphs(sc)
    for vi in mine:
        scene = eng.scenes[vi % len(eng.scenes)]
        sc = eng.load_scene(scene)
        keys = key_schedule(n_lat, per, 1000 + vi)
        eng.install(sc)
        poses = list(sc["poses"])
        lats = [sc["lat"][0, i] for i in range(P)]
        steady, act = eng.steady, eng.act
        for j in range(n_lat):
            key = int(keys[j])
            cand = {k: eng.cand_cam(sc, poses, k) for k in (1, 2, 3, 4)}
            cam = eng.cams_for([[cand[k]] for k in (1, 2, 3, 4)])
            for c_ in eng.st_cam:
                eng.st_cam[c_].copy_(cam[c_])
            eng.graph.replay()
            kk = key - 1
            lat = eng.st_x[:, kk:kk + 1]
            eng.c1_x.copy_(lat)
            for c_ in eng.c1_cam:
                eng.c1_cam[c_].copy_(cam[c_][:, kk * n_tok:(kk + 1) * n_tok])
            eng.c1_act.copy_(act[:, kk * n_tok:(kk + 1) * n_tok])
            eng.graph_c1.replay()
            s0 = eng.ring_ptr * n_tok
            steady.rk[:, :, s0:s0 + n_tok].copy_(eng.cv1.ck[:, 0:1])
            steady.rv[:, :, s0:s0 + n_tok].copy_(eng.cv1.cv[:, 0:1])
            steady.pcam_v[:, s0:s0 + n_tok].copy_(cam["viewmats"][:, kk * n_tok:(kk + 1) * n_tok])
            steady.pcam_K[:, s0:s0 + n_tok].copy_(cam["K"][:, kk * n_tok:(kk + 1) * n_tok])
            steady.ppos.sub_(1)
            steady.ppos[eng.ring_ptr] = args.history - 1
            eng.ring_ptr = (eng.ring_ptr + 1) % args.history
            lats.append(lat[0, 0].clone())
            sc["stream"].commit(key)
            poses.append(cand[key])
        torch.save({"lat": torch.stack(lats).cpu(), "keys": torch.from_numpy(keys), "poses": torch.stack(poses),
                    "scene": scene, "sid": sc["sid"], "mode": "realtime", "student": args.student},
                   out / "lat" / f"video{vi}.pt")
        print(json.dumps({"video": vi, "sid": sc["sid"], "keys": "".join(KEYS[int(k)] for k in keys[::per])}),
              flush=True)


@torch.no_grad()
def gen_realtime_opt(args):
    """gen_realtime on the optimised path (tree_opt.RTStepper with --opts, CUDA graphs, same scenes / key scripts /
    cameras). Fused commit: the scripted key's commit runs inside the next step's first denoising pass."""
    from shownow.eval.latency_harness import LatencyEngine
    from shownow.latency.long_infer import key_schedule
    from shownow.latency.tree_opt import RTStepper, opts_name, parse_opts
    args.jpeg_quality, args.lead, args.data = 85, 2, "v3"
    out = Path(args.out)
    (out / "lat").mkdir(parents=True, exist_ok=True)
    eng = LatencyEngine(args, None, None)
    P, n_lat, per = args.history, args.seconds * 4, int(round(args.switch_sec * 4))
    mine = list(range(args.videos))[args.rank::args.world]
    if not mine:
        return
    st = RTStepper(eng.m, 4, eng.fewstep, eng.sig, P=P, opts=parse_opts(args.opts), device=eng.dev)
    captured = False
    for vi in mine:
        scene = eng.scenes[vi % len(eng.scenes)]
        sc = eng.load_scene(scene)
        keys = key_schedule(n_lat, per, 1000 + vi)
        st.set_text(sc["ctx"])
        st.load_ring(sc["committed"])
        poses = list(sc["poses"])
        lats = [sc["lat"][0, i] for i in range(P)]
        for j in range(n_lat):
            key = int(keys[j])
            cand = {k: eng.cand_cam(sc, poses, k) for k in (1, 2, 3, 4)}
            st.set_inputs(eng.cams_for([[cand[k]] for k in (1, 2, 3, 4)]), [1, 2, 3, 4])
            if not captured:
                st.capture()
                captured = True
            x = st.replay_gen()
            lats.append(x[0, key - 1].clone())
            st.replay_choose(key - 1)
            sc["stream"].commit(key)
            poses.append(cand[key])
        torch.save({"lat": torch.stack(lats).cpu(), "keys": torch.from_numpy(keys), "poses": torch.stack(poses),
                    "scene": scene, "sid": sc["sid"], "mode": f"realtime_opt[{opts_name(st.o)}]",
                    "student": args.student}, out / "lat" / f"video{vi}.pt")
        print(json.dumps({"video": vi, "sid": sc["sid"], "keys": "".join(KEYS[int(k)] for k in keys[::per])}),
              flush=True)


def gen_solarwm(args):
    """SolarWM-FT deployment content: one AR block (--block latents) per step, the key read once per block, every
    candidate an independent K = 1 rollout from the K/V cache; the held key's candidate is committed. Only that
    candidate is generated here (the other three would be discarded; content and cache are identical)."""
    from shownow.eval.latency_harness import LatencyEngine
    from shownow.latency.long_infer import key_schedule
    from shownow.latency.tree_fast import FRAME_TOKENS
    args.jpeg_quality, args.lead, args.data = 85, 2, "v3"
    out = Path(args.out)
    (out / "lat").mkdir(parents=True, exist_ok=True)
    eng = LatencyEngine(args, None, None)
    BL = eng.BL
    n_tok, n_lat, per = FRAME_TOKENS * BL, args.seconds * 4, int(round(args.switch_sec * 4))
    assert n_lat % BL == 0
    mine = list(range(args.videos))[args.rank::args.world]
    if not mine:
        return
    sc = eng.load_scene(eng.scenes[mine[0] % len(eng.scenes)])
    eng.build_graphs(sc)
    eng.build_k1_graph()
    new_pos = torch.arange(args.history - BL, args.history, device=eng.dev)
    for vi in mine:
        scene = eng.scenes[vi % len(eng.scenes)]
        sc = eng.load_scene(scene)
        keys = key_schedule(n_lat, per, 1000 + vi)
        eng.install(sc)
        poses = list(sc["poses"])
        lats = [sc["lat"][0, i] for i in range(P)]
        applied = []
        steady, act = eng.steady, eng.act
        for b in range(n_lat // BL):
            key = int(keys[b * BL])  # read once per block
            cam = eng.cams_for([eng.cand_cams(sc, poses, key)])
            sl = slice((key - 1) * n_tok, key * n_tok)
            for c_ in eng.g1_cam:
                eng.g1_cam[c_].copy_(cam[c_])
            eng.g1_act.copy_(act[:, sl])
            eng.graph_g1.replay()
            eng.c1_x.copy_(eng.g1_x)
            for c_ in eng.c1_cam:
                eng.c1_cam[c_].copy_(cam[c_])
            eng.c1_act.copy_(act[:, sl])
            eng.graph_c1.replay()
            s0 = eng.ring_ptr * n_tok
            steady.rk[:, :, s0:s0 + n_tok].copy_(eng.cv1.ck[:, 0:1])
            steady.rv[:, :, s0:s0 + n_tok].copy_(eng.cv1.cv[:, 0:1])
            steady.pcam_v[:, s0:s0 + n_tok].copy_(cam["viewmats"])
            steady.pcam_K[:, s0:s0 + n_tok].copy_(cam["K"])
            steady.ppos.sub_(BL)
            steady.ppos[eng.ring_ptr * BL:(eng.ring_ptr + 1) * BL].copy_(new_pos)
            eng.ring_ptr = (eng.ring_ptr + 1) % (args.history // BL)
            lats += [eng.g1_x[0, i].clone() for i in range(BL)]
            applied += [key] * BL
            eng.advance(sc, poses, key)
        torch.save({"lat": torch.stack(lats).cpu(), "keys": torch.from_numpy(keys),
                    "keys_applied": torch.tensor(applied), "poses": torch.stack(poses), "scene": scene,
                    "sid": sc["sid"], "mode": f"solarwm_b{BL}", "student": args.student}, out / "lat" / f"video{vi}.pt")
        print(json.dumps({"video": vi, "sid": sc["sid"]}), flush=True)


@torch.no_grad()
def gen_reference(args):
    from shownow.eval.action_follow_eval import load
    from shownow.eval.long_drift_eval import camera_tokens_long
    from shownow.train.student_common import tokens
    from shownow.tree.tree import tree_call, window_layout
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    (out / "lat").mkdir(parents=True, exist_ok=True)
    BL = args.block
    d = load("s2", args.student, dev, BL, args.history)
    root = d.module
    fewstep = [float(s) for s in warp_denoising_steps([int(v) for v in args.denoising_steps.split(",")],
                                                      d.scheduler.timesteps.to(dev), num_train_timesteps=1000)]
    files = sorted(Path(args.src).glob("lat/video*.pt"), key=lambda f: int(f.stem[5:]))[args.rank::args.world]
    for f in files:
        src = torch.load(f, map_location="cpu", weights_only=False)
        fork = torch.load(src["scene"], map_location="cpu", weights_only=False)
        keys = src.get("keys_applied", src["keys"]).long()  # the keys the replayed run actually applied
        acts = torch.cat([fork["actions"][0, :P].long(), keys])[None].to(dev)  # [1, N]
        cam = camera_tokens_long(src["poses"][None].float(), dev, args.cam_scale)  # [1, N*405, ...]
        prompt = fork["prompt"].to(dev)[None]
        seq = [src["lat"][i].to(dev, torch.bfloat16) for i in range(P)]
        g = torch.Generator(device=dev).manual_seed(int(f.stem[5:]))
        n_tok = cam["viewmats"].shape[1] // acts.shape[1]
        for j in range(P, acts.shape[1], BL):  # one AR block of BL latents: window = 15 prior latents + the block
            lo = max(0, j - args.history)
            n_h = j - lo
            layout = window_layout(1, n_h, 0, t0=lo, block=BL)
            c = {k_: v_[:, lo * n_tok:(j + BL) * n_tok] for k_, v_ in cam.items()}
            a = tokens(acts[:, lo:j + BL].float()).long()
            ctx = torch.stack(seq[lo:j])[None]
            x = torch.randn((1, BL, *ctx.shape[2:]), generator=g, device=dev, dtype=torch.bfloat16)
            for i, t in enumerate(fewstep):
                tt = torch.zeros(1, n_h + BL, device=dev)
                tt[:, n_h:] = t
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = tree_call(root, torch.cat([ctx, x], 1), tokens(tt), prompt, c, layout, action_tokens=a)
                x0 = d.flow_to_x0(x, o, torch.full((1, BL), t, device=dev))
                if i < len(fewstep) - 1:
                    tn = torch.full((BL,), fewstep[i + 1], device=dev)
                    x = d.scheduler.add_noise(x0[0].float(), torch.randn(x0[0].shape, generator=g, device=dev),
                                              tn)[None].to(torch.bfloat16)
                else:
                    x = x0.to(torch.bfloat16)
            seq += [x[0, i] for i in range(BL)]
        torch.save({**src, "lat": torch.stack(seq).cpu(), "mode": "reference", "student": args.student},
                   out / "lat" / f.name)
        print(json.dumps({"video": f.stem, "sid": src["sid"]}), flush=True)


# ----------------------------------------------------------------------------- scoring
def lat_feat(x: torch.Tensor) -> torch.Tensor:  # [n,48,H,W] -> [n,96]
    x = x.float()
    return torch.cat([x.mean((-2, -1)), x.std((-2, -1))], -1)


def frame_stats(frames: np.ndarray):
    import cv2
    sharp, con = [], []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        con.append(float(g.std()))
    return np.array(sharp), np.array(con)


def boundary_ratio(frames: np.ndarray, n_future_lat: int):
    """frames: decoded future frames, 4 per latent (last n_future_lat * 4 frames used)."""
    f = frames[-4 * n_future_lat:].astype(np.float32)[:, ::4, ::4]
    diff = np.abs(f[1:] - f[:-1]).mean((1, 2, 3))  # diff[i] = frame i -> i+1
    idx = np.arange(len(diff))
    cross = diff[idx % 4 == 3]
    within = diff[idx % 4 != 3]
    return float(cross.mean() / max(within.mean(), 1e-6))


def red_jumps(frames):
    f = frames[:, ::8, ::8].astype(np.float32)
    red = (f[..., 0] - f[..., 1:].mean(-1)).mean((1, 2))
    return int((np.abs(np.diff(red)) > 15).sum())


def score(args):
    import imageio.v2 as iio
    from taehv import TAEHV, StreamingTAEHV
    from shownow.eval.idm import MotionIDM, load_idm, windows
    from shownow.eval.long_drift_summary import fd
    dev = torch.device("cuda")
    tae = StreamingTAEHV(TAEHV(str(REPO / "third_party/taehv/taew2_2.pth")).to(dev, torch.float16).eval())

    def decode(lat):  # [n,48,H,W] -> uint8 [T,H,W,3], streamed one latent at a time (as in deployment)
        tae.reset()
        outf = []
        for i in range(lat.shape[0]):
            fr = tae.decode(lat[None, i:i + 1].to(dev).half())
            while fr is not None:
                outf.append((fr[0].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu())
                fr = tae.decode()
        return torch.cat(outf).numpy()

    kidm = load_idm(REPO / "runs/idm/v1/idm.pt", dev)
    midm = MotionIDM().to(dev).eval()
    midm.load_state_dict(torch.load(REPO / "runs/idm/motion_v1/motion_idm.pt", map_location=dev,
                                    weights_only=False)["model"])
    axis = {1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0), 4: (1.0, 0.0)}  # (x, z): W +z, A -x, S -z, D +x
    real_cache = {}

    def real_ref(scene):  # real Doom reference of one scene: the 4 real branches (21 latents each)
        if scene not in real_cache:
            rl = torch.load(scene, map_location="cpu", weights_only=False)["latents"].float()  # [4,21,...]
            rframes = [decode(rl[b]) for b in range(4)]
            st = [frame_stats(fr) for fr in rframes]
            real_cache[scene] = dict(feat=lat_feat(rl.flatten(0, 1)).mean(0), all=lat_feat(rl.flatten(0, 1)),
                                     sharp=float(np.mean([a.mean() for a, _ in st])),
                                     con=float(np.mean([b.mean() for _, b in st])),
                                     bound=float(np.mean([boundary_ratio(fr, 20) for fr in rframes])))
        return real_cache[scene]

    for run in map(Path, args.runs):
        files = sorted(run.glob("lat/video*.pt"), key=lambda f: int(f.stem[5:]))
        scenes = [torch.load(f, map_location="cpu", weights_only=False)["scene"] for f in files]
        refs = [real_ref(sc_) for sc_ in sorted(set(scenes))]
        pool = dict(feat=torch.stack([r["feat"] for r in refs]).mean(0), sharp=np.mean([r["sharp"] for r in refs]),
                    con=np.mean([r["con"] for r in refs]))
        floor = float(np.mean([(r["feat"] - pool["feat"]).norm() for r in refs]))  # real scene-to-scene spread
        per_video, gen_feats = [], []
        (run / "videos").mkdir(exist_ok=True)
        for f in files:
            src = torch.load(f, map_location="cpu", weights_only=False)
            lat, keys = src["lat"].float(), src["keys"].long()
            N = lat.shape[0]
            ref = {**real_ref(src["scene"]), "hist_f": pool["feat"], "floor": floor, "sharp": pool["sharp"],
                   "con": pool["con"]}
            frames = decode(lat)
            fut = frames[-4 * (N - P):]
            sh, co = frame_stats(fut)
            n_win = (N - P) // WIN
            feats = lat_feat(lat[P:])
            gen_feats.append(feats.numpy())
            drift = [float((feats[w * WIN:(w + 1) * WIN].mean(0) - ref["hist_f"]).norm()) for w in range(n_win)]
            sharp = [float(sh[w * 4 * WIN:(w + 1) * 4 * WIN].mean() / ref["sharp"]) for w in range(n_win)]
            con = [float(co[w * 4 * WIN:(w + 1) * 4 * WIN].mean() / ref["con"]) for w in range(n_win)]
            # IDMs on settled future latents with a t+1 neighbour
            ts = [t for t in range(P, N - 1)]
            with torch.no_grad():
                wx = windows(lat, ts).to(dev)
                pk = torch.cat([kidm(wx[i:i + 64]) for i in range(0, len(ts), 64)]).argmax(1).cpu() + 1
                mv = torch.cat([midm(wx[i:i + 64]) for i in range(0, len(ts), 64)]).float().cpu()
            ok_k, ok_m, ok_k_win = [], [], [[] for _ in range(n_win)]
            for i, t in enumerate(ts):
                j = t - P
                key = int(keys[j])
                s = 0
                while j - s - 1 >= 0 and int(keys[j - s - 1]) == key:
                    s += 1
                if s < SETTLE:
                    continue
                ok_k.append(int(pk[i]) == key)
                ax = torch.tensor(axis[key])
                ok_m.append(bool(float(mv[i] @ ax) > 0.5 * float(mv[i].norm()) and float(mv[i].norm()) * 0.5 > 0.05))
                if j // WIN < n_win:
                    ok_k_win[j // WIN].append(ok_k[-1])
            rec = {"video": f.stem, "sid": src["sid"], "key_idm_acc": float(np.mean(ok_k)),
                   "key_idm_acc_win": [float(np.mean(v)) for v in ok_k_win], "along_key": float(np.mean(ok_m)),
                   "drift_win": drift, "drift_floor": ref["floor"], "sharp_win": sharp, "contrast_win": con,
                   "washed_windows": int(sum(1 for a, b in zip(sharp, con) if a < 0.5 or b < 0.5)),
                   "boundary": boundary_ratio(frames, N - P), "boundary_real": ref["bound"],
                   "red_jumps": red_jumps(fut)}
            per_video.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_win")}), flush=True)
            if args.mp4:
                iio.mimwrite(run / "videos" / f"{f.stem}_{src['sid']}.mp4", frames, fps=16, quality=7,
                             macro_block_size=8)
        agg = {"run": str(run), "n_videos": len(per_video)}
        for k in ("key_idm_acc", "along_key", "boundary", "boundary_real", "red_jumps", "washed_windows",
                  "drift_floor"):
            agg[k] = float(np.mean([r[k] for r in per_video]))
        for k in ("drift_win", "sharp_win", "contrast_win", "key_idm_acc_win"):
            agg[k] = [float(v) for v in np.mean([r[k] for r in per_video], 0)]
        agg["drift_last"] = agg["drift_win"][-1]
        # latent Frechet distance per 5 s window: all videos' latents of the window vs all real latents of the scenes;
        # floor = FD between two disjoint halves of the real latents (branches W/A vs S/D)
        real = torch.cat([r["all"] for r in refs]).numpy()
        half = torch.cat([r["all"].unflatten(0, (4, 21)) for r in refs], 1)
        agg["fd_floor"] = fd(half[:2].flatten(0, 1).numpy(), half[2:].flatten(0, 1).numpy())
        g_all = np.stack(gen_feats)  # [videos, latents, 96]
        agg["fd_win"] = [fd(g_all[:, w * WIN:(w + 1) * WIN].reshape(-1, 96), real) for w in range(g_all.shape[1] // WIN)]
        agg["fd_last"] = agg["fd_win"][-1]
        agg["sharp_last"] = agg["sharp_win"][-1]
        json.dump({"agg": agg, "videos": per_video}, open(run / "long_metrics.json", "w"), indent=1)
        print("AGG", json.dumps(agg), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--mode", required=True, choices=["realtime", "reference", "solarwm", "realtime_opt"])
    g.add_argument("--opts", default="exact", help="realtime_opt: tree_opt option set")
    g.add_argument("--denoising-steps", default="1000,750,500,250",
                   help="SGF schedule of reference / realtime_opt (realtime FastTree and solarwm stay 4-step)")
    g.add_argument("--block", type=int, default=1, help="latents per AR block (reference / solarwm; 3 = SolarWM B3)")
    g.add_argument("--student", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--from", dest="src", default="", help="reference: realtime run whose keys / poses are replayed")
    g.add_argument("--videos", type=int, default=10)
    g.add_argument("--seconds", type=int, default=60)
    g.add_argument("--switch-sec", type=float, default=2.0)
    g.add_argument("--history", type=int, default=15)
    g.add_argument("--cam-scale", type=float, default=0.5)
    g.add_argument("--rank", type=int, default=0)
    g.add_argument("--world", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--mp4", action="store_true")
    args = ap.parse_args()
    if args.cmd == "score":
        score(args)
    elif args.mode == "realtime":
        gen_realtime(args)
    elif args.mode == "realtime_opt":
        gen_realtime_opt(args)
    elif args.mode == "solarwm":
        gen_solarwm(args)
    else:
        gen_reference(args)


if __name__ == "__main__":
    main()
