"""Batched-independent baseline: eight candidates as ONE BATCH of eight independent sequences vs. eight serial K=1
generations vs. the shared-history tree, same checkpoint / resolution / sampling steps / CUDA-graph capture.

Steady state (full 15-latent history), B1 (one latent per candidate), 864x480 latents (48x30x54), bf16, one GPU.
One "update" = generation of all K candidates of the next latent (all denoising steps incl. the per-update preparation
of the window's prefix keys); the commit of the chosen latent is timed separately (it is the same K = 1 pass for every
method except the optimised fused tree, where it rides in the first denoising pass).

Legacy runtime (shownow/latency/tree_fast.FastTree, the paper's "reference runtime"):
  serial_k1x{K}     one K = 1 generation graph replayed K times (key's action / camera copied in before each replay,
                    as latency_harness's `independent` policy); == compute_bench `cached_independent{K}_graph`
  tree_k{K}         shared-history tree, one graph (== compute_bench `cached_tree{K}_graph` / Engine.build_graphs)
  batched_kv_k{K}   batch of K independent sequences, each with ITS OWN COPY of the history K/V: a persistent duplicated
                    cache [K, P*405 + 405] per layer; per update the prefix keys are prepared ONCE for the window and
                    broadcast-written into all K copies; per layer the sequence's own-latent K/V are written into its cache
                    tail and one standard batch-K flash-attention runs over [own history copy | own latent]. No tree mask,
                    no shared-prefix cascade. Linear layers / cross-attention / FFN see the same K*405 tokens.
  batched_cat_k{K}  FastTree(share=False): same math, but the per-sequence [history | own latent] K/V are materialised
                    by torch.cat(prefix.expand(K), own) in every layer of every pass (naive batching)
Optimised runtime (shownow/latency/tree_opt.RTStepper, --opts, compiled layer, needs the FA3 env):
  opt_serial_k1x{K} K = 1 RTStepper 'plain' graph replayed K times
  opt_tree_k{K}     RTStepper K: 'plain' (generation), 'commit' (separate commit), 'fused' (commit inside generation =
                    the paper's 228.5 ms row)
  opt_batched_kv_k{K}  batch-K duplicated-cache variant of the compiled layer (same fused QKV / per-frame modulation /
                    GELU epilogue options), 'plain' graph

Sanity check (--sanity): eager runs with the SAME noise (x_T and the per-step eps) and the same keys / cameras: every
K-candidate output vs. the K serial K = 1 outputs (max / mean abs diff, bf16).

  python -m shownow.eval.batched_indep_bench --student <ckpt> --data sw8a --denoising-steps 1000,667,333 \
      --variants serial,tree,batched_kv,batched_cat --sanity --out <json>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from shownow.eval.long_drift_eval import KEYS  # noqa: E402
from shownow.eval.latency_harness import LatencyEngine  # noqa: E402
from shownow.latency.tree_fast import FRAME_TOKENS, FastTree, _rope_math, _tiled_projmat_math  # noqa: E402
from shownow.latency.tree_opt import BF, _fa, _lin, _ln, _prope, _rms  # noqa: E402
from shownow.train.student_common import tokens  # noqa: E402

GIB = 2 ** 30


# ----------------------------------------------------------------------------- legacy runtime: batched duplicated cache
class BatchedKV(FastTree):
    """K independent sequences, each with its own copy of the history K/V (persistent [K, Lp + blk] cache per layer)."""

    def __init__(self, model, K, prefix_latents, h=15, w=27, device="cuda", block=1):
        super().__init__(model, K, prefix_latents=prefix_latents, share=False, h=h, w=w, device=device, block=block)
        del self.pk, self.pv  # replaced by the per-sequence caches
        torch.cuda.empty_cache()
        L, bt = len(model.blocks), self.blk * FRAME_TOKENS
        self.kc = torch.zeros(L, K, self.Lp + bt, self.nh, self.hd, dtype=torch.bfloat16, device=device)
        self.vc = torch.zeros_like(self.kc)
        self.pk, self.pv = self.kc, self.vc  # forward_fused passes self.pk[li] / self.pv[li] -> the layer's caches

    def prepare_prefix(self):
        """final prefix keys / values for the current window, computed once and written into all K copies."""
        fr = self.freqs_for_positions(self.ppos)
        _, fkv, _ = self.prope_fns(self.pcam_v, self.pcam_K)
        for li, blk in enumerate(self.m.blocks):
            k = self.rope(self.rk[li], fr)
            v = self.rv[li]
            if blk.camera_attention_enabled:
                k = fkv(k.transpose(1, 2)).transpose(1, 2)
                v = fkv(v.transpose(1, 2)).transpose(1, 2)
            self.kc[li, :, :self.Lp].copy_(k)  # [1, Lp, H, D] -> broadcast into the K copies
            self.vc[li, :, :self.Lp].copy_(v)

    def _layer_impl(self, blk, tok, e0, freqs, cos, sin, Pq, Pkv, Po, kc, vc, xk, xv):
        """FastTree._layer_impl with standard batched attention over each sequence's own cache."""
        K, n, d = self.K, self.nh, self.hd
        blk_tok = self.blk * FRAME_TOKENS
        ee = (blk.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
        a_in = blk.norm1(tok) * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)
        sa = blk.self_attn
        q = sa.norm_q(sa.q(a_in)).view(1, -1, n, d)
        k_raw = sa.norm_k(sa.k(a_in)).view(1, -1, n, d)
        v_raw = sa.v(a_in).view(1, -1, n, d)
        q, k = _rope_math(q, cos, sin), _rope_math(k_raw, cos, sin)
        v = v_raw
        if blk.camera_attention_enabled:
            q = _tiled_projmat_math(q.transpose(1, 2), Pq).transpose(1, 2)
            k = _tiled_projmat_math(k.transpose(1, 2), Pkv).transpose(1, 2)
            v = _tiled_projmat_math(v.transpose(1, 2), Pkv).transpose(1, 2)
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        qb = q.view(K, blk_tok, n, d)
        kc[:, self.Lp:].copy_(k.view(K, blk_tok, n, d))  # own latent -> the tail of the sequence's cache
        vc[:, self.Lp:].copy_(v.view(K, blk_tok, n, d))
        o = self.fa(qb, kc, vc).reshape(1, -1, n, d)  # batch K, each [own history copy | own latent]
        o = o.to(torch.bfloat16)
        if blk.camera_attention_enabled:
            o = _tiled_projmat_math(o.transpose(1, 2), Po).transpose(1, 2)
        y = sa.o(o.reshape(1, -1, n * d))
        tok = tok + y * ee[2].squeeze(2)
        ca = blk.cross_attn
        xq = ca.norm_q(ca.q(blk.norm3(tok))).view(1, -1, n, d)
        xo = self.fa(xq.to(torch.bfloat16), xk.to(torch.bfloat16), xv.to(torch.bfloat16))
        tok = tok + ca.o(xo.flatten(2))
        y = blk.ffn(blk.norm2(tok) * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2))
        tok = tok + y * ee[5].squeeze(2)
        return tok, k_raw, v_raw


# ----------------------------------------------------------------------------- optimised runtime: batched layer
def _layer_b(L: dict, tok, e0, cos, sin, Pq, Pkv, Po, xk, xv, kc, vc, Fr: int, Lp: int, cam: bool, epi: bool,
             eps_ln: float, eps_rms: float):
    """tree_opt._layer (no commit frame) with standard batch-Fr attention over each sequence's own cache."""
    T = FRAME_TOKENS
    dim = tok.shape[-1]
    n, d = xk.shape[2], xk.shape[3]
    x = tok.view(1, Fr, T, dim)
    ee = (L["mod"].view(1, 1, 1, 6, dim) + e0).unbind(3)
    a = (_ln(x, None, None, eps_ln) * (1 + ee[1]) + ee[0]).view(1, Fr * T, dim)
    if "qkv" in L:
        q, k_raw, v_raw = _lin(a, L["qkv"]).split(dim, -1)
    else:
        q, k_raw, v_raw = _lin(a, L["q"]), _lin(a, L["k"]), _lin(a, L["v"])
    q = _rms(q, L["nq"], eps_rms).view(1, -1, n, d)
    k_raw = _rms(k_raw, L["nk"], eps_rms).view(1, -1, n, d)
    v_raw = v_raw.reshape(1, -1, n, d)
    q, k, v = _rope_math(q, cos, sin), _rope_math(k_raw, cos, sin), v_raw
    if cam:
        q, k, v = _prope(q, Pq), _prope(k, Pkv), _prope(v, Pkv)
    q, k, v = q.to(BF), k.to(BF), v.to(BF)
    kc[:, Lp:].copy_(k.view(Fr, T, n, d))
    vc[:, Lp:].copy_(v.view(Fr, T, n, d))
    o = _fa(q.view(Fr, T, n, d), kc, vc).reshape(1, -1, n, d).to(BF)
    if cam:
        o = _prope(o, Po)
    x = x + _lin(o.reshape(1, -1, n * d), L["o"]).view(1, Fr, T, dim) * ee[2]
    xq = _rms(_lin(_ln(x, L["n3w"], L["n3b"], eps_ln).view(1, -1, dim), L["xq"]), L["xnq"], eps_rms)
    xo = _fa(xq.view(1, -1, n, d).to(BF), xk, xv)
    x = x + _lin(xo.reshape(1, -1, n * d), L["xo"]).view(1, Fr, T, dim)
    h = _lin((_ln(x, None, None, eps_ln) * (1 + ee[4]) + ee[3]).view(1, -1, dim), L["f1"], gelu=True, epi=epi)
    x = x + _lin(h, L["f2"]).view(1, Fr, T, dim) * ee[5]
    return x.view(1, -1, dim)


def make_opt_batched(model, K, fewstep, sig, P, opts, device):
    from shownow.latency.tree_opt import RTStepper, _compiled
    from shownow.train.teacher_action_ft import action_input

    class OptBatched(RTStepper):
        def __init__(self):
            o = dict(opts)
            o["fused_commit"] = False  # independent sequences: no fused commit (the commit is a separate K=1 pass)
            super().__init__(model, K, fewstep, sig, P=P, opts=o, device=device)
            ft = self.ft
            del ft.pk, ft.pv
            torch.cuda.empty_cache()
            nL, T = len(model.blocks), FRAME_TOKENS
            self.kc = torch.zeros(nL, K, P * T + T, self.nh, self.hd, dtype=BF, device=device)
            self.vc = torch.zeros_like(self.kc)
            self.layer_b = _compiled("layer_b", _layer_b)

        def prep_all(self):
            n = self.n_valid
            pcos, psin, PkvP = self._prefix_rot(n)
            ft = self.ft
            for li in range(len(self.m.blocks)):  # prepared once, broadcast-written into the K copies
                self.prep_fn(ft.rk[li], ft.rv[li], self.kc[li], self.vc[li], pcos, psin, PkvP, n * FRAME_TOKENS,
                             self.cam_on[li])

        def _pass(self, x_c, t_c, nc, n_old, n_new, store=False, head=True):
            assert nc == 0 and not store and n_new == self.P
            m, ft, T = self.m, self.ft, FRAME_TOKENS
            Kc = x_c.shape[1]
            af = self.act[:, :Kc]
            pos = torch.full((Kc,), n_new, device=self.dev)
            act_tok = tokens(af)
            tok = m.patch_embedding(x_c.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
            tok = action_input(m, tok, act_tok)
            dim = tok.shape[-1]
            if self.o["per_frame"]:
                m._action_tokens = af
                e = m._time_condition(t_c, tok, None)
                e0 = m.time_projection(e).unflatten(1, (6, dim)).view(1, Kc, 1, 6, dim)
            else:
                m._action_tokens = act_tok
                e = m._time_condition(tokens(t_c), tok, None)
                e0 = m.time_projection(e).unflatten(1, (6, dim)).view(1, Kc, T, 6, dim)
            m._action_tokens = None
            fr = ft.freqs_for_positions(pos)
            cos, sin = fr.real[None].contiguous(), fr.imag[None].contiguous()
            from shownow.latency.tree_fast import prope_mats
            Pq, Pkv, Po = prope_mats(self.cam_v, self.cam_K)
            for li in range(len(m.blocks)):
                tok = self.layer_b(self.L[li], tok, e0, cos, sin, Pq, Pkv, Po, self.xk[li], self.xv[li],
                                   self.kc[li], self.vc[li], Kc, self.P * T, self.cam_on[li], self.o["gelu"],
                                   self.eps_ln, self.eps_rms)
            if self.o["per_frame"]:
                e_tok = e[:, None].expand(Kc, T, dim).reshape(1, Kc * T, dim)
            else:
                e_tok = e.view(1, Kc * T, dim)
            out = m.head(tok, e_tok.unsqueeze(2))
            grid = torch.tensor([[Kc, ft.h, ft.w]])
            return torch.stack(m.unpatchify(out, grid)).permute(0, 2, 1, 3, 4)

        @torch.no_grad()
        def capture(self):
            assert self.n_valid == self.P
            self.graphs = {}
            fn = lambda: self.gen()  # noqa: E731
            with torch.autocast("cuda", dtype=BF, cache_enabled=False):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                    fn()
            self.graphs["plain"] = g
            self._capture_fns = {"plain": fn}

    return OptBatched()


# ----------------------------------------------------------------------------- helpers
def capture(fn):
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
            fn()
    return g


def timed(fn, reps, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out, torch.cuda.max_memory_allocated() / GIB


def summ(ms, peak, base, build_peak, **kw):
    ms = np.array(ms)
    return {"n": len(ms), "mean": float(ms.mean()), "std": float(ms.std()), "p50": float(np.percentile(ms, 50)),
            "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)), "min": float(ms.min()),
            "max": float(ms.max()), "peak_gib": peak, "base_gib": base, "peak_incl_build_gib": build_peak, **kw}


def free():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_max_abs": float(b.abs().max()),
            "ref_rms": float(b.pow(2).mean().sqrt()), "rel_l2": float((a - b).norm() / b.norm())}


# ----------------------------------------------------------------------------- main
@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--data", default="sw8a", choices=["v1", "v3", "sw8a"])
    ap.add_argument("--denoising-steps", default="1000,667,333")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--k", type=int, default=0, help="candidates (0 = number of keys)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--variants", default="serial,tree,batched_kv,batched_cat",
                    help="comma list of serial, tree, batched_kv, batched_cat, opt_serial, opt_tree, opt_batched_kv")
    ap.add_argument("--opts", default="exact", help="optimised-runtime option set (tree_opt.parse_opts)")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--nsteps", type=int, default=0, help="SANITY ONLY: keep the first n denoising steps (n = 1: the "
                    "diff of a single forward pass); timings with n < schedule length are not the reported setting")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True, help="output json")
    args = ap.parse_args()
    args.jpeg_quality, args.lead = 85, 1
    torch.backends.cuda.matmul.allow_tf32 = True
    other = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.split()
    env = {"hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "venv": os.environ.get("VIRTUAL_ENV"), "gpu_procs_before": len(other),
           "SHOWNOW_FA3": os.environ.get("SHOWNOW_FA3", "1"), "SHOWNOW_OPT_DYNAMIC": os.environ.get("SHOWNOW_OPT_DYNAMIC")}
    try:
        from shownow.latency.tree_fast import flash_attn_func_any
        env["attn_backend"] = getattr(flash_attn_func_any(), "backend", "fa2")
    except Exception as e:  # noqa: BLE001
        env["attn_backend"] = repr(e)
    print("ENV", json.dumps(env), flush=True)

    eng = LatencyEngine(args, None, None)
    if args.nsteps:
        eng.fewstep, eng.sig = eng.fewstep[:args.nsteps], eng.sig[:args.nsteps]
    dev, P, BL = eng.dev, args.history, eng.BL
    assert BL == 1, "B1 only"
    K = args.k or len(KEYS)
    n_tok = FRAME_TOKENS * BL
    sc = eng.load_scene(eng.scenes[0])
    xa, ctx, fewstep, sig = sc["xa"], sc["ctx"], eng.fewstep, eng.sig
    keys = [1 + k % len(KEYS) for k in range(K)]
    act = tokens(torch.tensor([[k for k in keys for _ in range(BL)]], device=dev).float()).long()
    cam = {c: v.contiguous() for c, v in eng.cams_for([eng.cand_cams(sc, sc["poses"], k) for k in keys]).items()}
    cam1 = [{c: v[:, j * n_tok:(j + 1) * n_tok].contiguous() for c, v in cam.items()} for j in range(K)]
    act1 = [act[:, j * n_tok:(j + 1) * n_tok].contiguous() for j in range(K)]
    shape = (1, K * BL, 48, 30, 54)
    res = {"env": env, "args": vars(args), "K": K, "keys": keys, "scene": sc["sid"], "nsteps": len(fewstep),
           "fewstep": fewstep, "rows": {}, "sanity": {}}
    free()
    base0 = torch.cuda.memory_allocated() / GIB
    res["base_alloc_gib_model_scene"] = base0
    print("BASE", base0, flush=True)

    def legacy_gen(ft, Kf, act_, cam_, out):
        fr = ft.freqs_for_positions(torch.arange(P, P + BL, device=dev).repeat(Kf))
        ts = [torch.full((1, Kf * BL), t, device=dev) for t in fewstep]

        def gen(noise=None):
            ft.prepare_prefix()
            x = noise[0] if noise is not None else torch.randn_like(out)
            for i in range(len(fewstep)):
                flow = ft.forward_fused(x, ts[i], act_, ctx, xa, cam_, fr, commit=False)
                x0 = x.float() - sig[i] * flow.float()
                if i == len(fewstep) - 1:
                    x = x0.to(torch.bfloat16)
                else:
                    eps = noise[i + 1] if noise is not None else torch.randn_like(x0)
                    x = ((1 - sig[i + 1]) * x0 + sig[i + 1] * eps).to(torch.bfloat16)
            out.copy_(x)
        gen._keep = (fr, ts)
        return gen

    def mk_noise():
        return [torch.randn(shape, device=dev, dtype=torch.bfloat16)] + \
            [torch.randn(shape, device=dev, dtype=torch.float32) for _ in range(len(fewstep) - 1)]

    noise = None
    if args.sanity:
        torch.manual_seed(1234)
        noise = mk_noise()
    sanity_out = {}

    def run_row(name, build, **kw):
        """build() -> (timed fn, eager-sanity fn or None, cleanup objects)."""
        free()
        base = torch.cuda.memory_allocated() / GIB
        torch.cuda.reset_peak_memory_stats()
        fn, san, keep = build()
        build_peak = torch.cuda.max_memory_allocated() / GIB
        ms, peak = timed(fn, args.reps, args.warm)
        res["rows"][name] = summ(ms, peak, base, build_peak, **kw)
        print(name, json.dumps(res["rows"][name]), flush=True)
        if san is not None and noise is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                sanity_out[name] = san().float().cpu()
        del fn, san, keep
        free()

    variants = args.variants.split(",")
    committed = sc["committed"]

    # ---------------- legacy runtime
    if "serial" in variants:
        def build():
            ft = eng.make_ft(committed, P, 1, share=True)
            out = torch.empty(1, BL, 48, 30, 54, device=dev, dtype=torch.bfloat16)
            a_buf, c_buf = act1[0].clone(), {c: v.clone() for c, v in cam1[0].items()}
            gen = legacy_gen(ft, 1, a_buf, c_buf, out)
            g = capture(gen)
            outs = torch.empty(shape, device=dev, dtype=torch.bfloat16)

            def serial():
                for j in range(K):
                    a_buf.copy_(act1[j])
                    for c in c_buf:
                        c_buf[c].copy_(cam1[j][c])
                    g.replay()
                    outs[:, j * BL:(j + 1) * BL].copy_(out)

            def san():
                res_ = []
                for j in range(K):
                    a_buf.copy_(act1[j])
                    for c in c_buf:
                        c_buf[c].copy_(cam1[j][c])
                    gen([z[:, j * BL:(j + 1) * BL] for z in noise])
                    res_.append(out.clone())
                return torch.cat(res_, 1)
            return serial, san, (ft, g, gen, out, outs)
        run_row(f"serial_k1x{K}", build, K=K, runtime="legacy", graph=True)

    if "tree" in variants:
        def build():
            ft = eng.make_ft(committed, P, K, share=True)
            out = torch.empty(shape, device=dev, dtype=torch.bfloat16)
            gen = legacy_gen(ft, K, act, cam, out)
            g = capture(gen)
            return g.replay, (lambda: (gen(noise), out.clone())[1]), (ft, g, gen, out)
        run_row(f"tree_k{K}", build, K=K, runtime="legacy", graph=True)

    if "batched_kv" in variants:
        def build():
            ft = BatchedKV(eng.m, K, P, device=dev, block=BL)
            eng.fill_ring(ft, committed[-P:])
            out = torch.empty(shape, device=dev, dtype=torch.bfloat16)
            gen = legacy_gen(ft, K, act, cam, out)
            g = capture(gen)
            return g.replay, (lambda: (gen(noise), out.clone())[1]), (ft, g, gen, out)
        run_row(f"batched_kv_k{K}", build, K=K, runtime="legacy", graph=True)

    if "batched_cat" in variants:
        def build():
            ft = eng.make_ft(committed, P, K, share=False)
            out = torch.empty(shape, device=dev, dtype=torch.bfloat16)
            gen = legacy_gen(ft, K, act, cam, out)
            g = capture(gen)
            return g.replay, (lambda: (gen(noise), out.clone())[1]), (ft, g, gen, out)
        run_row(f"batched_cat_k{K}", build, K=K, runtime="legacy", graph=True)

    # ---------------- optimised runtime
    if any(v.startswith("opt_") for v in variants):
        from shownow.latency.tree_opt import RTStepper, opts_name, parse_opts
        o = parse_opts(args.opts)
        res["opts"] = opts_name(o)
        cam_o = eng.cams_for([eng.cand_cams(sc, sc["poses"], k) for k in keys])
        cam_o1 = [eng.cams_for([eng.cand_cams(sc, sc["poses"], k)]) for k in keys]

        def stepper(Kf, opts_):
            st = RTStepper(eng.m, Kf, fewstep, sig, P=P, opts=opts_, device=dev)
            st.set_text(ctx)
            st.load_ring(committed)
            return st

        if "opt_serial" in variants:
            def build():
                st = stepper(1, o)
                st.set_inputs(cam_o1[0], [keys[0]])
                st.capture()
                g = st.graphs["plain"]
                outs = torch.empty(shape, device=dev, dtype=torch.bfloat16)
                key_t = [torch.tensor([[k]], device=dev) for k in keys]

                def serial():
                    for j in range(K):
                        st.cam_v.copy_(cam_o1[j]["viewmats"])
                        st.cam_K.copy_(cam_o1[j]["K"])
                        st.act.copy_(key_t[j])
                        g.replay()
                        outs[:, j:j + 1].copy_(st.x)

                def san():
                    st.load_ring(committed)
                    r = []
                    for j in range(K):
                        st.set_inputs(cam_o1[j], [keys[j]])
                        st.gen([z[:, j:j + 1] for z in noise])
                        r.append(st.x.clone())
                    return torch.cat(r, 1)
                return serial, san, (st, outs)
            run_row(f"opt_serial_k1x{K}", build, K=K, runtime="opt", graph=True, opts=res["opts"])

        if "opt_tree" in variants:
            free()
            base = torch.cuda.memory_allocated() / GIB
            torch.cuda.reset_peak_memory_stats()
            st = stepper(K, o)
            st.set_inputs(cam_o, keys)
            st.capture()
            build_peak = torch.cuda.max_memory_allocated() / GIB
            for gname in ("plain", "commit", "fused"):
                if gname not in st.graphs:
                    continue
                ms, peak = timed(st.graphs[gname].replay, args.reps, args.warm)
                nm = f"opt_tree_k{K}_{gname}"
                res["rows"][nm] = summ(ms, peak, base, build_peak, K=K, runtime="opt", graph=True, opts=res["opts"])
                print(nm, json.dumps(res["rows"][nm]), flush=True)
            if noise is not None:
                st.load_ring(committed)
                st.pending = False
                st.set_inputs(cam_o, keys)
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    st.gen(noise)
                sanity_out[f"opt_tree_k{K}_plain"] = st.x.float().cpu()
            del st
            free()

        if "opt_batched_kv" in variants:
            def build():
                st = make_opt_batched(eng.m, K, fewstep, sig, P, o, dev)
                st.set_text(ctx)
                st.load_ring(committed)
                st.set_inputs(cam_o, keys)
                st.capture()

                def san():
                    st.set_inputs(cam_o, keys)
                    st.gen(noise)
                    return st.x.clone()
                return st.graphs["plain"].replay, san, (st,)
            run_row(f"opt_batched_kv_k{K}", build, K=K, runtime="opt", graph=True, opts=res["opts"])

    # ---------------- sanity: every K-candidate output vs. the K serial K=1 outputs (same noise / keys / cameras)
    if noise is not None:
        for ref in (f"serial_k1x{K}", f"opt_serial_k1x{K}"):
            if ref not in sanity_out:
                continue
            for name, x in sanity_out.items():
                if name != ref:
                    d = diff(x, sanity_out[ref])
                    d["per_candidate_max_abs"] = [float((x[:, j] - sanity_out[ref][:, j]).abs().max()) for j in range(K)]
                    res["sanity"][f"{name}__vs__{ref}"] = d
                    print("SANITY", name, "vs", ref, json.dumps(d), flush=True)
        if f"serial_k1x{K}" in sanity_out and f"opt_serial_k1x{K}" in sanity_out:
            res["sanity"]["opt_serial_vs_legacy_serial"] = diff(sanity_out[f"opt_serial_k1x{K}"],
                                                               sanity_out[f"serial_k1x{K}"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
