"""Supplementary compute benchmark: cache-free normal inference vs the cached real-time paths, and
the shared-history tree's cost vs the number of candidates K.

All rows: final B1 Stage2 checkpoint, 864x480 (48x30x54 latents), history 15 latents, 4 SGF steps, bf16, one H100,
steady state (a full 15-latent history), CUDA events around each step after warm-up, peak memory over the timed reps.
  cachefree_k{1,4}   normal inference as in E2/E3 "normal inference" (training-faithful sliding window, no K/V cache):
                     every step re-encodes the 15 history latents with the K noisy candidates (tree layout, K = 1 or
                     4 sharing the history), 4 forwards; nothing to commit. Eager PyTorch (the variable layout is not
                     graph-captured).
  cached_*           K/V cache of the history (the real-time path): K=1 generation, 4 independent K=1 generations,
                     4-candidate tree, commit of the chosen latent; each eager and as a CUDA graph.
  tree_k{K}          the tree forward (graph) for K = 1, 2, 4, 8, 16 candidates (timing only: candidates past 4 repeat
                     the 4 keys; the model is trained with 4 actions).

  python -m shownow.eval.compute_bench --student <ckpt> --data v3 --out runs/compute_bench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from shownow.eval.long_drift_eval import KEYS  # noqa: E402
from shownow.eval.latency_harness import LatencyEngine  # noqa: E402
from shownow.latency.tree_fast import FRAME_TOKENS  # noqa: E402
from shownow.train.student_common import tokens  # noqa: E402


def timed(fn, reps, warm=5):
    """-> (list of ms per call, peak GiB during the timed reps)."""
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
    return out, torch.cuda.max_memory_allocated() / 2**30


def summ(ms, mem, **kw):
    ms = np.array(ms)
    return {"n": len(ms), "mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)),
            "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)), "peak_gib": mem, **kw}


@torch.no_grad()
def cachefree(eng, sc, K, reps):
    """one steady-state step of the cache-free window sampler (student_tree_sample's inner loop, hist = 15)."""
    from shownow.tree.tree import enable_tree, tree_call, window_layout
    dev, P = eng.dev, eng.args.history
    root = eng.m
    if not getattr(root, "_tree_enabled", False):
        enable_tree(root)
        root._tree_enabled = True
    hist = sc["lat"][:, -P:].to(dev, torch.bfloat16)  # [1, 15, C, H, W] the real history latents
    layout = window_layout(K, P, 0, t0=0, block=1)
    cam_h = eng.cams_for([[p for p in sc["poses"][-P:]]])
    cam_n = eng.cams_for([eng.cand_cams(sc, sc["poses"], 1 + k % len(KEYS)) for k in range(K)])
    c = {n: torch.cat([cam_h[n], cam_n[n]], 1) for n in ("viewmats", "K")}
    a = tokens(torch.tensor([[0] * P + [1 + k % len(KEYS) for k in range(K)]], device=dev).float()).long()
    prompt = torch.load(eng.scenes[0], map_location="cpu", weights_only=False)["prompt"].to(dev)[None]
    fewstep, sig = eng.fewstep, eng.sig

    def step():
        x = torch.randn(1, K, *hist.shape[2:], device=dev, dtype=torch.bfloat16)
        for i, t in enumerate(fewstep):
            tt = torch.zeros(1, P + K, device=dev)
            tt[:, P:] = t
            with torch.autocast("cuda", dtype=torch.bfloat16):
                flow = tree_call(root, torch.cat([hist, x], 1), tokens(tt), prompt, c, layout, action_tokens=a)
            x0 = x.float() - sig[i] * flow.float()
            x = x0.to(torch.bfloat16) if i == len(fewstep) - 1 else \
                ((1 - sig[i + 1]) * x0 + sig[i + 1] * torch.randn_like(x0)).to(torch.bfloat16)
        return x

    return timed(step, reps)


@torch.no_grad()
def tree_k(eng, sc, K, reps):
    """graph of the shared-history tree generation with K candidates (same code as Engine.build_graphs)."""
    dev, P, BL = eng.dev, eng.args.history, eng.BL
    steady = eng.make_ft(sc["committed"], P, K, share=True)
    keys = [1 + k % len(KEYS) for k in range(K)]
    act = tokens(torch.tensor([[k for k in keys for _ in range(BL)]], device=dev).float()).long()
    cam = eng.cams_for([eng.cand_cams(sc, sc["poses"], k) for k in keys])
    fr = steady.freqs_for_positions(torch.arange(P, P + BL, device=dev).repeat(K))
    ts = [torch.full((1, K * BL), t, device=dev) for t in eng.fewstep]
    xs = torch.empty(1, K * BL, 48, 30, 54, device=dev, dtype=torch.bfloat16)
    fewstep, sig, ctx, xa = eng.fewstep, eng.sig, eng.ctx, eng.xa

    def gen():
        steady.prepare_prefix()
        x = torch.randn_like(xs)
        for i in range(len(fewstep)):
            flow = steady.forward_fused(x, ts[i], act, ctx, xa, cam, fr, commit=False)
            x0 = x.float() - sig[i] * flow.float()
            x = x0.to(torch.bfloat16) if i == len(fewstep) - 1 else \
                ((1 - sig[i + 1]) * x0 + sig[i + 1] * torch.randn_like(x0)).to(torch.bfloat16)
        xs.copy_(x)

    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                gen()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
            gen()
    res = timed(g.replay, reps)
    del g, steady
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--data", default="v3", choices=["v1", "v3", "sw8a"])
    ap.add_argument("--denoising-steps", default="1000,750,500,250", help="SGF schedule (8-action models: 1000,667,333)")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.jpeg_quality, args.lead = 85, 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = LatencyEngine(args, None, None)
    sc = eng.load_scene(eng.scenes[0])
    eng.build_graphs(sc)
    eng.build_k1_graph()
    eng.install(sc)
    res = {}
    gen_tree, commit_block = eng._graph_inputs[0], eng._graph_inputs[1]
    gen1 = eng._graph_inputs_k1[0]

    NK = len(KEYS)  # candidates = keys

    def indep4():
        for _ in range(NK):
            gen1()

    def indep4_graph():
        for _ in range(NK):
            eng.graph_g1.replay()

    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for name, fn, kw in (("cached_k1_eager", gen1, {"K": 1, "graph": False}),
                             (f"cached_independent{NK}_eager", indep4, {"K": NK, "graph": False}),
                             (f"cached_tree{NK}_eager", gen_tree, {"K": NK, "graph": False}),
                             ("commit1_eager", commit_block, {"graph": False})):
            res[name] = summ(*timed(fn, args.reps), **kw)
            print(name, json.dumps(res[name]), flush=True)
    for name, fn, kw in (("cached_k1_graph", eng.graph_g1.replay, {"K": 1, "graph": True}),
                         (f"cached_independent{NK}_graph", indep4_graph, {"K": NK, "graph": True}),
                         (f"cached_tree{NK}_graph", eng.graph.replay, {"K": NK, "graph": True}),
                         ("commit1_graph", eng.graph_c1.replay, {"graph": True})):
        res[name] = summ(*timed(fn, args.reps), **kw)
        print(name, json.dumps(res[name]), flush=True)
    for K in [int(k) for k in args.ks.split(",")]:
        res[f"tree_k{K}"] = summ(*tree_k(eng, sc, K, args.reps), K=K, graph=True)
        print(f"tree_k{K}", json.dumps(res[f"tree_k{K}"]), flush=True)
    for K in (1, NK):  # last: enable_tree wraps the model's forward
        res[f"cachefree_k{K}"] = summ(*cachefree(eng, sc, K, args.reps), K=K, graph=False)
        print(f"cachefree_k{K}", json.dumps(res[f"cachefree_k{K}"]), flush=True)
    res["student"] = args.student
    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
