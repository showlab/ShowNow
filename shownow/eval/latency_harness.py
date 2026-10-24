"""E1: input-to-photon latency of ShowNow vs. reactive vs. independent multi-future generation (same checkpoint, same
GPU, same decoder), driven by scripted key traces against a 16 fps display clock.

Policies (one tree B1 Stage2 checkpoint, one H100, FastTree + CUDA graphs, TAEHV decode for all):
  shownow      all 4 candidates of the next latent in ONE shared-history tree forward (graph), the held key is read
               just in time (when `--lead` frames remain on screen), only the chosen candidate is committed + decoded
  independent  the 4 candidates as 4 separate K=1 forwards (no shared history / tree), then the same just-in-time
               selection: isolates the value of the shared-history tree
  reactive     no speculation: the held key is read first, then its single latent is generated (K=1), committed and
               decoded; the key is read early enough (`--reactive-lead`, auto from the measured step cost) that the
               display does not starve when nothing changes
  persist      action-persistence speculation (K=1), same selection time as shownow: one candidate under the currently
               held key is generated ahead, the key is read when `--lead` frames remain; a match is committed, a
               mismatch is regenerated under the read key (reactive fallback) and then committed
  shownow_opt  shownow on the optimised path (shownow/latency/tree_opt.RTStepper, option set --opts, CUDA graphs);
               with fused_commit the chosen latent's commit runs inside the next generation, so after the key is read
               only host bookkeeping remains before decoding
  splice       ActionSplice-style in-flight action editing on block generation (use with --block 3, = the reactive
               block schedule of Matrix-Game 2.0): the key is read as in reactive, then between denoising steps the held
               key is polled; on a change the remaining steps run under the new key's action tokens / camera
               ("condition swap" of ActionSplice, arXiv 2609.08230: its latency equals CST_R without the corrector)
  reactive_opt   reactive on the optimised path (a K = 1 RTStepper with the same --opts as shownow_opt, CUDA graphs,
               fused commit): removes the runtime confound between shownow_opt and the FastTree-runtime reactive;
               key read --reactive-lead frames ahead (auto from its own measured step cost)
  one_future_opt  one_future on the optimised runtime (K = 1 RTStepper, same --opts): the single candidate is
               generated under the held key; when the held key changes while a generation runs, that generation finishes
               (graph replay is not interruptible) and the candidate is regenerated under the new key; a step is
               committed when --lead frames remain AND the candidate matches the held key
  --key-set    action-count scaling: the traces only press keys of this set (make_trace_keyset: identical press /
               release timeline for every set with >= 2 keys) and shownow_opt speculates exactly these K keys
               (no candidate miss, no fallback)
  one_future  action-persistence speculation with its native schedule: the single candidate is regenerated as soon
               as the held key differs from the candidate's key (a running generation finishes first), and a step is
               committed when `--lead` frames remain AND the candidate matches the held key (waits for it otherwise)
Latency per key press (a change of the held key, or a press from idle):
  no_vae  press -> the first latent generated under the new key is committed (ready for decoding)
  vae     press -> the first frame decoded from that latent starts playing on the 16 fps display clock
A press that is superseded by the next press before any latent was generated under it is "overridden"; presses with no
served latent by the end of the trace are "unserved". Also recorded: played fps while a key is held, stall time
(display starved while a key is held), candidates-late steps, per-step GPU times, peak memory.

  python -m shownow.eval.latency_harness --student <tree B1 S2 ckpt> --policies shownow,independent,reactive \
      --traces poisson0.5,poisson1,poisson2,poisson4 --seeds 0,1,2 --seconds 60 --out runs/latency_e1/<name> [--data v3]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import threading
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from shownow.demo.play_server import Engine  # noqa: E402
from shownow.eval.long_drift_eval import KEYS  # noqa: E402
from shownow.latency.tree_fast import FRAME_TOKENS  # noqa: E402
from shownow.train.student_common import tokens  # noqa: E402
import shownow.eval.e1_protocol as e1  # noqa: E402
from shownow.eval.e1_protocol import Display, TraceKeys, make_trace, make_trace_keyset, score  # noqa: E402,F401

FPS = 16
FRAME_S = 1.0 / FPS


# ----------------------------------------------------------------------------- engine
class LatencyEngine(Engine):
    """play_server.Engine + a K=1 generation graph on the shared ring (reactive / independent policies)."""

    @torch.no_grad()
    def build_k1_graph(self):
        dev, P, BL = self.dev, self.args.history, self.BL
        steady, cv1, xa, ctx, fewstep, sig = self.steady, self.cv1, self.xa, self.ctx, self.fewstep, self.sig
        self.g1_x = torch.empty(1, BL, 48, 30, 54, device=dev, dtype=torch.bfloat16)
        self.g1_cam = {k: v.clone() for k, v in self.c1_cam.items()}
        self.g1_act = self.c1_act.clone()
        ts1 = [torch.full((1, BL), t, device=dev) for t in fewstep]
        fr1 = steady.freqs_for_positions(torch.arange(P, P + BL, device=dev))

        def gen1():
            steady.prepare_prefix()
            x = torch.randn_like(self.g1_x)
            for i in range(len(fewstep)):
                flow = cv1.forward_fused(x, ts1[i], self.g1_act, ctx, xa, self.g1_cam, fr1, commit=False)
                x0 = x.float() - sig[i] * flow.float()
                x = x0.to(torch.bfloat16) if i == len(fewstep) - 1 else \
                    ((1 - sig[i + 1]) * x0 + sig[i + 1] * torch.randn_like(x0)).to(torch.bfloat16)
            self.g1_x.copy_(x)

        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    gen1()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                gen1()
        self.graph_g1 = g
        self._graph_inputs_k1 = (gen1, ts1, fr1)

    @torch.no_grad()
    def build_step_graphs(self):
        """one CUDA graph per denoising step of the K=1 generation (policy splice): step i reads / writes self.g1_x and
        uses self.g1_act / self.g1_cam, which may be swapped between steps."""
        dev, P, BL = self.dev, self.args.history, self.BL
        steady, cv1, xa, ctx, fewstep, sig = self.steady, self.cv1, self.xa, self.ctx, self.fewstep, self.sig
        ts1, fr1 = self._graph_inputs_k1[1], self._graph_inputs_k1[2]

        def step(i):
            if i == 0:
                steady.prepare_prefix()
                x = torch.randn_like(self.g1_x)
            else:
                x = self.g1_x
            flow = cv1.forward_fused(x, ts1[i], self.g1_act, ctx, xa, self.g1_cam, fr1, commit=False)
            x0 = x.float() - sig[i] * flow.float()
            x = x0.to(torch.bfloat16) if i == len(fewstep) - 1 else \
                ((1 - sig[i + 1]) * x0 + sig[i + 1] * torch.randn_like(x0)).to(torch.bfloat16)
            self.g1_x.copy_(x)

        self.graph_steps = []
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for i in range(len(fewstep)):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        step(i)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                    step(i)
                self.graph_steps.append(g)
        self._graph_inputs_steps = step

    @torch.no_grad()
    def build_opt(self, sc, opts: str):
        """RTStepper (K = number of keys) on the scene's window + its CUDA graphs (policy shownow_opt)."""
        from shownow.latency.tree_opt import RTStepper, parse_opts
        ks = getattr(self, "key_set", None)
        K = len(ks) if ks else (self.args.k or len(KEYS))
        self.opt = st = RTStepper(self.m, K, self.fewstep, self.sig, P=self.args.history, opts=parse_opts(opts),
                                  device=self.dev)
        st.set_text(sc["ctx"])
        st.load_ring(sc["committed"])
        st.set_inputs(self.cams_for([[sc["poses"][-1]]] * K), ks or list(KEYS)[:K])
        st.capture()

    @torch.no_grad()
    def build_opt1(self, sc, opts: str):
        """K = 1 RTStepper (policy reactive_opt): the reactive policy on the optimised runtime."""
        from shownow.latency.tree_opt import RTStepper, parse_opts
        self.opt1 = st = RTStepper(self.m, 1, self.fewstep, self.sig, P=self.args.history, opts=parse_opts(opts),
                                   device=self.dev)
        st.set_text(sc["ctx"])
        st.load_ring(sc["committed"])
        st.set_inputs(self.cams_for([[sc["poses"][-1]]]), [list(KEYS)[0]])
        st.capture()

    def decode_only(self, lat):
        n = 0
        f = self.tae.decode(lat.half())
        while f is not None:
            n += 1
            f = self.tae.decode()
        return n

    def cand_cam(self, sc, poses, k):
        if sc["stream"] is not None:
            return torch.from_numpy(sc["stream"].peek(k)).float()
        p = poses[-1].clone()
        p[:3, :3] = sc["rot"]
        p[:3, 3] = p[:3, 3] + sc["vel"][k]
        return p

    def cand_cams(self, sc, poses, k):
        """poses of the next AR block (BL latents) if key k is held for all of it."""
        if self.BL == 1:
            return [self.cand_cam(sc, poses, k)]
        if sc["stream"] is not None:
            st = copy.deepcopy(sc["stream"])
            return [torch.from_numpy(st.commit(k)).float() for _ in range(self.BL)]
        out, ps = [], list(poses)
        for _ in range(self.BL):
            out.append(self.cand_cam(sc, ps, k))
            ps.append(out[-1])
        return out

    def advance(self, sc, poses, k):
        """after committing a block under key k: key stream / poses move by BL latents."""
        new = self.cand_cams(sc, poses, k)
        if sc["stream"] is not None:
            for _ in range(self.BL):
                sc["stream"].commit(k)
        poses.extend(new)

    @torch.no_grad()
    def run_trace(self, sc, trace, policy, seconds, lead, reactive_lead):
        dev, P, BL = self.dev, self.args.history, self.BL
        n_tok = FRAME_TOKENS * BL  # tokens of one AR block
        steady, act = self.steady, self.act
        poses = list(sc["poses"])
        new_pos = torch.arange(P - BL, P, device=dev)
        self.install(sc)
        if policy in ("shownow_opt", "reactive_opt", "one_future_opt"):
            o = self.opt if policy == "shownow_opt" else self.opt1
            o.set_text(sc["ctx"])
            o.load_ring(sc["committed"])
        self.tae.reset()
        self.decode_only(sc["lat"])  # decoder state after the real history (intro is not timed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        keys = TraceKeys(trace, time.time() + 0.5)
        disp = Display(keys, seconds)
        steps = []
        sync = torch.cuda.synchronize

        def commit(k, lat, cam_k):
            j = k - 1
            self.c1_x.copy_(lat)
            for c_ in self.c1_cam:
                self.c1_cam[c_].copy_(cam_k[c_])
            self.c1_act.copy_(act[:, j * n_tok:(j + 1) * n_tok])
            self.graph_c1.replay()
            s = self.ring_ptr * n_tok
            steady.rk[:, :, s:s + n_tok].copy_(self.cv1.ck[:, 0:1])
            steady.rv[:, :, s:s + n_tok].copy_(self.cv1.cv[:, 0:1])
            steady.pcam_v[:, s:s + n_tok].copy_(cam_k["viewmats"])
            steady.pcam_K[:, s:s + n_tok].copy_(cam_k["K"])
            steady.ppos.sub_(BL)
            steady.ppos[self.ring_ptr * BL:(self.ring_ptr + 1) * BL].copy_(new_pos)
            self.ring_ptr = (self.ring_ptr + 1) % (P // BL)

        tick_mode = getattr(self.args, "lead_mode", "frames") == "tick" and policy in ("shownow_opt", "one_future_opt")
        guard = getattr(self.args, "tick_guard_ms", 20.0) / 1000.0

        def at_decision(lead_frames):
            """decision point. frames (default): at most lead_frames frames still queued. tick (--lead-mode tick,
            shownow_opt / one_future_opt only): the queue is empty and the display ticked at most FRAME_S - guard
            ago, i.e. the decision is made right after the last queued frame started playing, leaving >= guard for
            commit + decode before the next tick (no frame is skipped or repeated; a late decision shows as a stall)."""
            if not tick_mode:
                return disp.pending() <= lead_frames
            return disp.pending() == 0 and time.time() - disp.last_tick <= FRAME_S - guard

        pre = getattr(self.args, "preempt", 0)

        def wait_key(lead_frames, switch_from=None):
            """until a key is held and the decision point is reached; returns (key, t_decision). --preempt: a key other
            than switch_from (the last committed key) is a decision point at once (its frames replace queued ones)."""
            while not keys.done(seconds):
                k = keys.current()
                if k is not None and (at_decision(lead_frames) or (pre and switch_from is not None and k != switch_from)):
                    return k, time.time()
                time.sleep(0.001)
            return None, time.time()

        def cams1(k):
            c = self.cams_for([self.cand_cams(sc, poses, k)])
            return {c_: v.contiguous() for c_, v in c.items()}

        def gen1(k):
            """one K=1 graph generation under key k into self.g1_x (synchronised); -> its camera dict."""
            c = cams1(k)
            for c_ in self.g1_cam:
                self.g1_cam[c_].copy_(c[c_])
            self.g1_act.copy_(act[:, (k - 1) * n_tok:k * n_tok])
            self.graph_g1.replay()
            sync()
            return c

        cand_key = cand_lat = cand_cam = cand_ready = None
        last_key, n_gen = None, 0
        recent = list(KEYS)  # keys by recency of their last press (candidate order for shownow_opt with --k)
        while not keys.done(seconds):
            rec = {"policy": policy}
            if policy == "one_future_opt":
                gen, gens = 0.0, []
                while not keys.done(seconds):
                    k = keys.current()
                    if k is not None and k != cand_key:
                        tg = time.time()
                        self.opt1.set_inputs(self.cams_for([self.cand_cams(sc, poses, k)]), [k])
                        self.opt1.replay_gen()
                        sync()
                        cand_key, cand_ready = k, time.time()
                        gens.append((tg - keys.t0, cand_ready - keys.t0, [k]))
                        gen += cand_ready - tg
                        n_gen += 1
                        continue
                    if k is not None and k == cand_key and (at_decision(lead) or (pre and last_key is not None
                                                                                  and k != last_key)):
                        break
                    time.sleep(0.0005)
                else:
                    break
                key, t_dec, lat, cam_k = cand_key, time.time(), self.opt1.x[:, 0:1], None
                rec.update(gen_s=gen, t_ready=cand_ready - keys.t0, late=False, hit=True, gens=gens,
                           pend_dec=disp.pending())
                cand_key = None  # the cache moves on: the next candidate is generated after this commit
            elif policy == "one_future":
                gen = 0.0
                while not keys.done(seconds):
                    k = keys.current()
                    if k is not None and k != cand_key:
                        tg = time.time()
                        cand_cam, cand_lat = gen1(k), self.g1_x.clone()
                        cand_key, cand_ready = k, time.time()
                        gen += cand_ready - tg
                        n_gen += 1
                        continue
                    if k is not None and k == cand_key and disp.pending() <= lead:
                        break
                    time.sleep(0.0005)
                else:
                    break
                key, t_dec, lat, cam_k = cand_key, time.time(), cand_lat, {cand_key: cand_cam}
                rec.update(gen_s=gen, t_ready=cand_ready - keys.t0, late=False)
                cand_key = None  # the K/V cache moves on: the next candidate is generated after this commit
            elif policy == "persist":
                t0 = time.time()
                late = disp.pending() == 0 and keys.current() is not None
                guess = keys.current() or last_key or 1
                cam_g, lat_g = gen1(guess), self.g1_x.clone()
                n_gen += 1
                t_ready = time.time()
                key, t_dec = wait_key(lead)
                if key is None:
                    break
                gen = t_ready - t0
                if key == guess:
                    lat, cam_k = lat_g, {key: cam_g}
                else:  # miss: reactive fallback under the read key
                    tg = time.time()
                    cam_k, lat = {key: gen1(key)}, self.g1_x
                    n_gen += 1
                    gen += time.time() - tg
                rec.update(gen_s=gen, t_ready=t_ready - keys.t0, late=late, hit=key == guess)
            elif policy in ("shownow", "independent"):
                t0 = time.time()
                late = disp.pending() == 0 and keys.current() is not None
                cam_k = {k: cams1(k) for k in KEYS}
                if policy == "shownow":
                    cam4 = self.cams_for([self.cand_cams(sc, poses, k) for k in KEYS])
                    for c_ in self.st_cam:
                        self.st_cam[c_].copy_(cam4[c_])
                    self.graph.replay()
                    sync()
                    outs = {k: self.st_x[:, (k - 1) * BL:k * BL] for k in KEYS}
                else:
                    outs = {}
                    for k in KEYS:
                        for c_ in self.g1_cam:
                            self.g1_cam[c_].copy_(cam_k[k][c_])
                        self.g1_act.copy_(act[:, (k - 1) * n_tok:k * n_tok])
                        self.graph_g1.replay()
                        outs[k] = self.g1_x.clone()
                    sync()
                t_ready = time.time()
                key, t_dec = wait_key(lead)
                if key is None:
                    break
                lat = outs[key]
                n_gen += 4
                rec.update(gen_s=t_ready - t0, t_ready=t_ready - keys.t0, late=late)
            elif policy == "splice":
                key, t_dec = wait_key(reactive_lead)
                if key is None:
                    break
                t0 = time.time()
                late = disp.pending() == 0

                def set_key(k):
                    c = cams1(k)
                    for c_ in self.g1_cam:
                        self.g1_cam[c_].copy_(c[c_])
                    self.g1_act.copy_(act[:, (k - 1) * n_tok:k * n_tok])
                    return {k: c}

                cam_k, swaps = set_key(key), []
                for i, g in enumerate(self.graph_steps):
                    g.replay()
                    sync()
                    k2 = keys.current()
                    if i < len(self.graph_steps) - 1 and k2 is not None and k2 != key:
                        key, t_dec = k2, time.time()  # the block now serves the new key (decision = the swap)
                        cam_k = set_key(key)
                        swaps.append(i + 1)
                lat = self.g1_x
                n_gen += 1
                rec.update(gen_s=time.time() - t0, t_ready=time.time() - keys.t0, late=late, swaps=swaps)
            elif policy == "shownow_opt":
                t0 = time.time()
                late = disp.pending() == 0 and keys.current() is not None
                K = self.opt.K if hasattr(self.opt, "K") else len(KEYS)
                if getattr(self, "key_set", None):  # action-count scaling: exactly the supported actions
                    cand = list(self.key_set)
                elif K >= len(KEYS):
                    cand = list(KEYS)
                else:  # --k < number of keys: the held key, then the most recently pressed keys
                    held = keys.current() or last_key or recent[0]
                    cand = [held] + [k for k in recent if k != held][:K - 1]
                self.opt.set_inputs(self.cams_for([self.cand_cams(sc, poses, k) for k in cand]), cand)
                self.opt.replay_gen()
                sync()
                t_ready = time.time()
                gens = [(t0 - keys.t0, t_ready - keys.t0, list(cand))]
                key, t_dec = wait_key(lead, last_key)
                if key is None:
                    break
                pend_dec = disp.pending()
                hit = key in cand
                if not hit:  # reactive fallback: regenerate with the read key in slot 0
                    cand = [key] + cand[1:]
                    tg = time.time()
                    self.opt.set_inputs(self.cams_for([self.cand_cams(sc, poses, k) for k in cand]), cand)
                    self.opt.replay_gen()
                    sync()
                    gens.append((tg - keys.t0, time.time() - keys.t0, list(cand)))
                j = cand.index(key)
                recent.remove(key)
                recent.insert(0, key)
                lat, cam_k = self.opt.x[:, j:j + 1], None
                n_gen += K * (1 if hit else 2)
                rec.update(gen_s=t_ready - t0, t_ready=t_ready - keys.t0, late=late, hit=hit, cand_k=K, gens=gens,
                           pend_dec=pend_dec)
            elif policy == "reactive_opt":
                key, t_dec = wait_key(reactive_lead, last_key)
                if key is None:
                    break
                t0 = time.time()
                self.opt1.set_inputs(self.cams_for([self.cand_cams(sc, poses, key)]), [key])
                self.opt1.replay_gen()
                sync()
                lat, cam_k = self.opt1.x[:, 0:1], None
                n_gen += 1
                rec.update(gen_s=time.time() - t0, t_ready=time.time() - keys.t0, late=disp.pending() == 0)
            else:  # reactive
                key, t_dec = wait_key(reactive_lead)
                if key is None:
                    break
                t0 = time.time()
                cam_k = {key: cams1(key)}
                for c_ in self.g1_cam:
                    self.g1_cam[c_].copy_(cam_k[key][c_])
                self.g1_act.copy_(act[:, (key - 1) * n_tok:key * n_tok])
                self.graph_g1.replay()
                sync()
                lat = self.g1_x
                n_gen += 1
                rec.update(gen_s=time.time() - t0, t_ready=time.time() - keys.t0, late=disp.pending() == 0)
            t1 = time.time()
            if policy == "shownow_opt":
                self.opt.replay_choose(j)
            elif policy in ("reactive_opt", "one_future_opt"):
                self.opt1.replay_choose(0)
            else:
                commit(key, lat, cam_k[key])
            sync()
            t_commit = time.time()
            nf = self.decode_only(lat)
            sync()
            t_dec_done = time.time()
            step = len(steps)
            if pre and policy in ("shownow_opt", "one_future_opt", "reactive_opt") and last_key is not None \
                    and key != last_key:
                rec["dropped"] = disp.replace([(step, key)] * nf, pre)
                rec["pend_push"] = disp.pending()
            else:
                disp.push([(step, key)] * nf)
            self.advance(sc, poses, key)
            last_key = key
            rec.update(step=step, key=key, t_decision=t_dec - keys.t0, t_commit=t_commit - keys.t0,
                       t_decoded=t_dec_done - keys.t0, commit_s=t_commit - t1, decode_s=t_dec_done - t_commit, nf=nf)
            if getattr(self, "save_lat", None) is not None:
                self.save_lat.append(lat[0].detach().to("cpu", torch.bfloat16).clone())  # all BL latents of the step
            steps.append(rec)
        time.sleep(2.0)  # let the display drain
        disp.stop = True
        first_play = {}
        for tp, (step, key) in disp.played:
            first_play.setdefault(step, tp - keys.t0)
        for s in steps:
            s["t_play"] = first_play.get(s["step"])
        self.n_gen = n_gen
        return steps, disp, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--data", default="v1", choices=["v1", "v3", "sw8a"])
    ap.add_argument("--denoising-steps", default="1000,750,500,250", help="SGF schedule (8-action models: 1000,667,333)")
    ap.add_argument("--block", type=int, default=1, help="latents per AR block (1 = B1, 3 = SolarWM's B3)")
    ap.add_argument("--policies", default="shownow,independent,reactive")
    ap.add_argument("--traces", default="poisson0.5,poisson1,poisson2,poisson4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--lead", type=int, default=2, help="frames on screen when the key is read (shownow/independent)")
    ap.add_argument("--nsteps", type=int, default=0, help="TIMING ONLY: use the first n timesteps of the 4-step schedule "
                    "(no student is trained for fewer steps; the frames are not meaningful, the timing is)")
    ap.add_argument("--fps", type=float, default=16.0,
                    help="display clock (frames / s); one latent = 4 frames, so it plays for 4 / fps s. Traces stay in "
                         "wall-clock seconds; --lead stays in frames (1 frame = 1 / fps s)")
    ap.add_argument("--reactive-lead", type=int, default=-1, help="-1 = auto from the measured K=1 step cost "
                    "(0 = read the key only once the queue is empty)")
    ap.add_argument("--opts", default="exact", help="shownow_opt: tree_opt option set (none | exact | exact,fp8 | ...)")
    ap.add_argument("--lead-mode", choices=["frames", "tick"], default="frames",
                    help="decision point of shownow_opt / one_future_opt: frames = at most --lead frames queued "
                         "(paper); tick = queue empty, right after a display tick (sub-frame lead)")
    ap.add_argument("--tick-guard-ms", type=float, default=20.0, help="--lead-mode tick: minimum time left before the "
                    "next display tick for commit + decode")
    ap.add_argument("--timeline", action="store_true", help="also store every step record (candidate generations, "
                    "queue depth at decision, commit / decode / first-play times) in rows.jsonl")
    ap.add_argument("--save-lat", default="", help="comma list of <trace>:<seed> whose committed latents are saved")
    ap.add_argument("--key-set", default="", help="comma-separated action ids the traces press and shownow_opt "
                    "speculates (action-count scaling, e.g. 1,3 / 1,2,3,4); empty = every key")
    ap.add_argument("--k", type=int, default=0, help="shownow_opt candidates per step (0 = every key); with fewer "
                    "candidates than keys: the held key + the most recently pressed keys, a miss is regenerated")
    ap.add_argument("--preempt", type=int, default=0, help="shownow_opt / one_future_opt / reactive_opt: on a key "
                    "switch, decide at once (no waiting for the --lead point) and let the new latent's frames replace up "
                    "to this many still-queued frames of the old action (content is skipped); 0 = off (paper)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-shared", action="store_true", help="skip the idle-machine check (not for reported numbers)")
    args = ap.parse_args()
    args.jpeg_quality = 85
    global FPS, FRAME_S
    FPS, FRAME_S = float(args.fps), 1.0 / float(args.fps)
    e1.FPS, e1.FRAME_S = FPS, FRAME_S  # the display clock lives in e1_protocol
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import subprocess
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=REPO).stdout.strip()  # noqa: E731
    code = {"commit": git("rev-parse", "HEAD"), "dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "harness_sha1": __import__("hashlib").sha1((REPO / "shownow/eval/latency_harness.py").read_bytes()).hexdigest(),
            "protocol_sha1": __import__("hashlib").sha1((REPO / "shownow/eval/e1_protocol.py").read_bytes()).hexdigest()}
    (out / "args.json").write_text(json.dumps({**vars(args), "code_version": code}, indent=1))
    import os
    import subprocess
    other = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.split()
    load1 = os.getloadavg()[0]
    # CPU used by OTHER processes right now (the 1-min load average also counts this process's own import burst)
    ps = subprocess.run(["ps", "-eo", "pid,pcpu,comm", "--no-headers"], capture_output=True, text=True).stdout.split("\n")
    other_cpu = sum(float(l.split()[1]) for l in ps if l.strip() and int(l.split()[0]) != os.getpid()
                    and l.split()[2] not in ("ps",))
    host = {"hostname": os.uname().nodename, "gpu_procs_before": len(other), "loadavg_1min": load1,
            "other_cpu_percent": round(other_cpu, 1), "cpus": os.cpu_count()}
    print("HOST", json.dumps(host), flush=True)
    if (other or other_cpu > 200.0) and not args.allow_shared:
        raise SystemExit(f"machine not idle (gpu procs {other}, other cpu {other_cpu:.0f}%): timing would be contaminated")
    (out / "host.json").write_text(json.dumps(host, indent=1))
    eng = LatencyEngine(args, None, None)
    eng.key_set = [int(x) for x in args.key_set.split(",")] if args.key_set else None
    if eng.key_set:
        assert all(k in KEYS for k in eng.key_set), (eng.key_set, list(KEYS))
    if args.nsteps:
        eng.fewstep, eng.sig = eng.fewstep[:args.nsteps], eng.sig[:args.nsteps]
    sc = eng.load_scene(eng.scenes[0])
    eng.build_graphs(sc)
    eng.build_k1_graph()
    if "splice" in args.policies.split(","):
        eng.build_step_graphs()
    opt_label = None
    if "shownow_opt" in args.policies.split(","):
        from shownow.latency.tree_opt import opts_name
        eng.build_opt(sc, args.opts)
        opt_label = f"shownow_opt[{opts_name(eng.opt.o)}]" + (f"_K{args.k}" if args.k else "")
    if {"reactive_opt", "one_future_opt"} & set(args.policies.split(",")):
        from shownow.latency.tree_opt import opts_name
        eng.build_opt1(sc, args.opts)
    # step costs (for the reactive lead and the compute table)
    cost = {}
    for name, g in (("gen4_tree", eng.graph), ("gen1", eng.graph_g1), ("commit1", eng.graph_c1)):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        cost[name] = (time.time() - t) / 20
    if opt_label:
        for name, g in eng.opt.graphs.items():
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            cost[f"opt_{name}"] = (time.time() - t) / 20
        eng.opt.load_ring(sc["committed"])  # the timing replays wrote into the ring
    lat0 = eng.st_x[:, :eng.BL].clone()  # one AR block
    eng.tae.reset()
    eng.decode_only(sc["lat"])
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        eng.decode_only(lat0)
    torch.cuda.synchronize()
    cost["decode1"] = (time.time() - t) / 10
    auto_rl = args.reactive_lead < 0
    rl = args.reactive_lead if not auto_rl else math.ceil((cost["gen1"] + cost["commit1"] + cost["decode1"]) / FRAME_S) + 1
    cost["reactive_lead_frames"] = rl
    rl_opt = rl
    if {"reactive_opt", "one_future_opt"} & set(args.policies.split(",")):
        for name, g in eng.opt1.graphs.items():
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            cost[f"opt1_{name}"] = (time.time() - t) / 20
        eng.opt1.load_ring(sc["committed"])
        step1 = cost["opt1_fused"] if "opt1_fused" in cost else cost["opt1_plain"] + cost["opt1_commit"]
        rl_opt = args.reactive_lead if not auto_rl else math.ceil((step1 + cost["decode1"]) / FRAME_S) + 1
        cost["reactive_opt_lead_frames"] = rl_opt
    SUFFIX = f"_b{args.block}" if args.block > 1 else ""  # B3 rows / costs are kept apart from the B1 ones
    cost = {k + SUFFIX: v for k, v in cost.items()}
    print("COST", json.dumps(cost), flush=True)
    (out / "cost.json").write_text(json.dumps(cost, indent=1))
    rows = open(out / "rows.jsonl", "a")
    for tr in args.traces.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            if tr.startswith("human:"):  # recorded human play (sim/record_human_trace.py): one session per file
                ev = json.load(open(tr[len("human:"):]))["events"]
                trace = [(round(float(t), 4), None if k is None else int(k)) for t, k in ev if float(t) < args.seconds]
                if eng.key_set:  # keys outside the deployed set fall back to idle
                    trace = [(t, k if k in eng.key_set else None) for t, k in trace]
            elif tr.startswith(("iso", "burst", "hold")):
                from shownow.eval.e1_protocol import make_trace_diag
                trace = make_trace_diag(tr, seed, args.seconds, eng.key_set or list(KEYS))
            else:
                trace = make_trace_keyset(tr, seed, args.seconds, eng.key_set) if eng.key_set else \
                    make_trace(tr, seed, args.seconds)
            for pol in args.policies.split(","):
                sc = eng.load_scene(eng.scenes[seed % len(eng.scenes)])
                eng.save_lat = [] if f"{tr}:{seed}" in args.save_lat.split(",") else None
                steps, disp, mem = eng.run_trace(sc, trace, pol, args.seconds, args.lead,
                                                 rl_opt if pol == "reactive_opt" else rl)
                pr = score(trace, steps, args.seconds)
                held_s = disp.held_ticks * FRAME_S
                pol_name = (opt_label if pol == "shownow_opt" else
                            f"{pol}[{opts_name(eng.opt1.o)}]" if pol in ("reactive_opt", "one_future_opt")
                            else pol + SUFFIX)
                if eng.key_set:
                    pol_name += f"_set{len(eng.key_set)}"
                if args.preempt:
                    pol_name += f"_pre{args.preempt}"
                rec = {"policy": pol_name, "fps": FPS, "nsteps": len(eng.fewstep), "lead_frames": args.lead, "lead_ms": args.lead * FRAME_S * 1000,
                       "trace": tr, "seed": seed, "scene": sc["sid"], "presses": pr, "trace_events": trace,
                       "stall_times": disp.stall_times,
                       "play_times": [round(t_ - disp.keys.t0, 4) for t_, _ in disp.played],
                       "n_steps": len(steps), "frames_played": len(disp.played),
                       "stall_s": disp.stall_ticks * FRAME_S, "held_s": held_s,
                       "fps_while_held": (disp.held_ticks - disp.stall_ticks) / max(1e-6, held_s),
                       "late_steps": sum(bool(s.get("late")) for s in steps),
                       "n_swaps": sum(len(s.get("swaps", [])) for s in steps),
                       "spec_misses": sum(s.get("hit") is False for s in steps),
                       "preempt": args.preempt, "dropped_frames": disp.dropped,
                       "preempt_cuts": sum(s.get("dropped", 0) > 0 for s in steps),
                       "switch_steps": sum("dropped" in s for s in steps),
                       "key_set": eng.key_set, "steps_keys": [s["key"] for s in steps],
                       "spec_steps": sum("hit" in s for s in steps),
                       "gen_s_p50": float(np.median([s["gen_s"] for s in steps])) if steps else None,
                       "commit_s_p50": float(np.median([s["commit_s"] for s in steps])) if steps else None,
                       "decode_s_p50": float(np.median([s["decode_s"] for s in steps])) if steps else None,
                       "peak_mem_gib": mem, "n_gen": eng.n_gen,
                       "gpu_busy_s": float(sum(s["gen_s"] + s["commit_s"] + s["decode_s"] for s in steps))}
                if args.timeline:
                    rec["timeline"] = steps
                if eng.save_lat is not None:
                    (out / "lat").mkdir(exist_ok=True)
                    torch.save({"lat": torch.cat(eng.save_lat), "keys": [s["key"] for s in steps], "trace": trace,
                                "steps": steps, "scene": sc["sid"], "scene_lat": sc["lat"].cpu(), "policy": pol_name,
                                "fps": FPS, "played": [(round(t_ - disp.keys.t0, 4), int(t[0])) for t_, t in disp.played],
                                "stall_times": disp.stall_times},
                               out / "lat" / f"{tr}_s{seed}_{pol}.pt")
                    eng.save_lat = None
                rows.write(json.dumps(rec) + "\n")
                rows.flush()
                srv = [p for p in pr if p["status"] == "served"]
                print(json.dumps({"policy": pol_name, "trace": tr, "seed": seed, "served": len(srv), "presses": len(pr),
                                  "no_vae_p50": float(np.median([p["lat_no_vae"] for p in srv])) if srv else None,
                                  "stall_s": rec["stall_s"], "fps": round(rec["fps_while_held"], 2)}), flush=True)


if __name__ == "__main__":
    main()
