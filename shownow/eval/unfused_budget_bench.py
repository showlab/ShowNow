"""Can eight futures fit the 250 ms / 16 fps budget with the deployment optimisations OFF?

Steady-state per-update loop of the 8-candidate tree, exactly as latency_harness runs it, timed per component with
CUDA events in the same repetition: generation (graph) -> commit of the chosen latent -> TAEHV decode of that latent.
  legacy          tree_fast.FastTree tree (Engine.build_graphs): gen graph, then latency_harness's commit (K=1 commit
                  graph + ring update), then decode
  opt:<opts>      tree_opt.RTStepper (K = 8) with option set <opts> ('+'-separated, e.g. none, exact, per_frame,
                  per_frame+qkv+gelu): replay_gen() ('plain', or 'fused' when fused_commit is on and a commit is
                  pending), replay_choose(j) (host bookkeeping + 'commit' graph unless fused), decode
The chosen key cycles over the 8 keys, so the ring advances as in a real run. Total = gen + commit + decode per rep.

  python -m shownow.eval.unfused_budget_bench --student <ckpt> --data sw8a --denoising-steps 1000,667,333 \
      --configs legacy,opt:none,opt:exact --out <json>
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
from shownow.latency.tree_fast import FRAME_TOKENS  # noqa: E402

GIB = 2 ** 30
BUDGET_MS = 250.0


def stats(x):
    x = np.asarray(x)
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std()), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def run_loop(step, reps, warm):
    """step(r, ev) records ev[0..3] around gen / commit / decode."""
    for r in range(warm):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        step(r, ev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    comp = {"gen": [], "commit": [], "decode": [], "total": []}
    for r in range(reps):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        step(warm + r, ev)
        torch.cuda.synchronize()
        g, c, d = ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]), ev[2].elapsed_time(ev[3])
        comp["gen"].append(g)
        comp["commit"].append(c)
        comp["decode"].append(d)
        comp["total"].append(ev[0].elapsed_time(ev[3]))
    out = {k: stats(v) for k, v in comp.items()}
    out["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
    out["total_over_budget_frac"] = float(np.mean(np.asarray(comp["total"]) > BUDGET_MS))
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--data", default="sw8a", choices=["v1", "v3", "sw8a"])
    ap.add_argument("--denoising-steps", default="1000,667,333")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--configs", default="legacy,opt:none,opt:exact")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.jpeg_quality, args.lead = 85, 1
    other = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.split()
    from shownow.latency.tree_fast import flash_attn_func_any
    env = {"hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "venv": os.environ.get("VIRTUAL_ENV"), "gpu_procs_before": len(other),
           "SHOWNOW_OPT_DYNAMIC": os.environ.get("SHOWNOW_OPT_DYNAMIC", "auto"),
           "attn_backend": getattr(flash_attn_func_any(), "backend", "fa2")}
    print("ENV", json.dumps(env), flush=True)
    eng = LatencyEngine(args, None, None)
    dev, P, BL = eng.dev, args.history, eng.BL
    assert BL == 1
    n_tok = FRAME_TOKENS * BL
    keys = list(KEYS)
    K = len(keys)
    sc = eng.load_scene(eng.scenes[0])
    poses = sc["poses"]
    cam8 = eng.cams_for([eng.cand_cams(sc, poses, k) for k in keys])
    cam1 = {k: {c: v.contiguous() for c, v in eng.cams_for([eng.cand_cams(sc, poses, k)]).items()} for k in keys}
    res = {"env": env, "args": vars(args), "scene": sc["sid"], "K": K, "nsteps": len(eng.fewstep), "budget_ms": BUDGET_MS,
           "rows": {}}

    def reset_decoder():
        eng.tae.reset()
        eng.decode_only(sc["lat"])  # decoder state after the real history (as the harness)
        torch.cuda.synchronize()

    for cfg in args.configs.split(","):
        gc.collect()
        torch.cuda.empty_cache()
        if cfg == "legacy":
            eng.build_graphs(sc)
            eng.install(sc)
            for c in eng.st_cam:
                eng.st_cam[c].copy_(cam8[c])
            steady, act = eng.steady, eng.act
            new_pos = torch.arange(P - BL, P, device=dev)
            reset_decoder()

            def step(r, ev):  # == latency_harness run_trace, policy "shownow": gen graph, commit(), decode_only()
                k = keys[r % K]
                ev[0].record()
                eng.graph.replay()
                ev[1].record()
                lat = eng.st_x[:, (k - 1) * BL:k * BL]
                eng.c1_x.copy_(lat)
                for c in eng.c1_cam:
                    eng.c1_cam[c].copy_(cam1[k][c])
                eng.c1_act.copy_(act[:, (k - 1) * n_tok:k * n_tok])
                eng.graph_c1.replay()
                s = eng.ring_ptr * n_tok
                steady.rk[:, :, s:s + n_tok].copy_(eng.cv1.ck[:, 0:1])
                steady.rv[:, :, s:s + n_tok].copy_(eng.cv1.cv[:, 0:1])
                steady.pcam_v[:, s:s + n_tok].copy_(cam1[k]["viewmats"])
                steady.pcam_K[:, s:s + n_tok].copy_(cam1[k]["K"])
                steady.ppos.sub_(BL)
                steady.ppos[eng.ring_ptr * BL:(eng.ring_ptr + 1) * BL].copy_(new_pos)
                eng.ring_ptr = (eng.ring_ptr + 1) % (P // BL)
                ev[2].record()
                eng.decode_only(lat)
                ev[3].record()

            row = run_loop(step, args.reps, args.warm)
            row["runtime"] = "legacy FastTree"
            del eng.graph, eng.graph_c1, eng._graph_inputs, eng.steady, eng.cv1
        else:
            assert cfg.startswith("opt:"), cfg
            from shownow.latency.tree_opt import RTStepper, opts_name, parse_opts
            o = parse_opts(cfg[4:].replace("+", ","))
            st = RTStepper(eng.m, K, eng.fewstep, eng.sig, P=P, opts=o, device=dev)
            st.set_text(sc["ctx"])
            st.load_ring(sc["committed"])
            st.set_inputs(cam8, keys)
            st.capture()
            st.load_ring(sc["committed"])
            reset_decoder()

            def step(r, ev):  # == latency_harness run_trace, policy "shownow_opt"
                j = r % K
                ev[0].record()
                st.replay_gen()
                ev[1].record()
                st.replay_choose(j)
                ev[2].record()
                eng.decode_only(st.x[:, j:j + 1])
                ev[3].record()

            row = run_loop(step, args.reps, args.warm)
            row["runtime"] = "tree_opt RTStepper"
            row["opts"] = opts_name(st.o)
            row["graphs"] = sorted(st.graphs)
            del st
        res["rows"][cfg] = row
        print(cfg, json.dumps({k: (v if not isinstance(v, dict) else {q: round(v[q], 2) for q in ("p50", "p90", "p99", "mean")})
                              for k, v in row.items()}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
