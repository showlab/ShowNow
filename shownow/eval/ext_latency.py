"""External world models under the E1 protocol, latency only (same GPU, same Poisson traces / seeds / 60 s, display clock,
lead, scoring and row format as latency_harness -> fps_compare / latency_paper).

Each model runs its OWN code and settings (weights, resolution, steps, block / frame size), wrapped by an adapter in
shownow/eval/ext_models/<model>.py with: make(seconds) -> Adapter; Adapter.reset(start_image, seed),
.block(key) -> (gen_s, commit_s, decode_s, n_frames) for one native generation unit under the held key (1-4 = W A S D),
.can_continue(); attributes name, nsteps, frames_per_unit; module constants NATIVE_FPS, CODE_DIR.
Pacing = latency_harness `reactive` (no speculation, as these models are deployed): the held key is read just in time,
when <= reactive_lead frames remain on screen (lead from the measured unit cost). Start image of seed s = the last real
history frame of the E1 scene of seed s (mg2_start_frames.py); before each trace one untimed unit is generated and decoded
but not shown (as the harness decodes the real intro without showing it).

  (model env) python -m shownow.eval.ext_latency --model mg2 --out runs/latency_e1/ext/mg2 [--fps <native>]
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import shownow.eval.e1_protocol as lh  # noqa: E402  (E1 traces / display / scoring without model code)


def run_trace(m, trace, img, seed, seconds, reactive_lead):
    """latency_harness.run_trace's reactive branch with the model's native generation unit (block / frame) as the step."""
    m.reset(img, seed)
    m.block(None)  # untimed first block (decoded, not shown)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    keys = lh.TraceKeys(trace, time.time() + 0.5)
    disp = lh.Display(keys, seconds)
    steps = []

    def wait_key(lead_frames):
        while not keys.done(seconds):
            k = keys.current()
            if k is not None and disp.pending() <= lead_frames:
                return k, time.time()
            time.sleep(0.001)
        return None, time.time()

    while not keys.done(seconds) and m.can_continue():
        key, t_dec = wait_key(reactive_lead)
        if key is None:
            break
        late = disp.pending() == 0
        gen_s, commit_s, decode_s, nf = m.block(key)
        t_dd = time.time()
        t_commit = t_dd - decode_s
        step = len(steps)
        disp.push([(step, key)] * nf)
        steps.append({"policy": m.name, "step": step, "key": key, "t_decision": t_dec - keys.t0,
                      "t_ready": t_commit - commit_s - keys.t0, "t_commit": t_commit - keys.t0,
                      "t_decoded": t_dd - keys.t0, "gen_s": gen_s, "commit_s": commit_s, "decode_s": decode_s,
                      "late": late})
    time.sleep(2.0)
    disp.stop = True
    first = {}
    for tp, (step, _) in disp.played:
        first.setdefault(step, tp - keys.t0)
    for s in steps:
        s["t_play"] = first.get(s["step"])
    return steps, disp, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter module in shownow/eval/ext_models")
    ap.add_argument("--start-dir", default=str(REPO / "runs/latency_e1/ext/mg2_start"))
    ap.add_argument("--traces", default="poisson0.5,poisson1,poisson2,poisson4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=0.0, help="display clock; 0 = the model's native frame rate")
    ap.add_argument("--warmup", type=int, default=12, help="untimed units before measuring the unit cost")
    ap.add_argument("--size", default="", help="generation size HxW passed to the adapter (empty = native)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mod = importlib.import_module(f"shownow.eval.ext_models.{args.model}")
    fps = args.fps or mod.NATIVE_FPS
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import hashlib
    import socket
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=REPO).stdout.strip()  # noqa: E731
    files = [Path(__file__), Path(mod.__file__), REPO / "shownow/eval/e1_protocol.py"]
    code = {"commit": git("rev-parse", "HEAD"), "dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "sha1": {str(f.relative_to(REPO)): hashlib.sha1(f.read_bytes()).hexdigest() for f in files},
            "model_code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                cwd=mod.CODE_DIR).stdout.strip()}
    (out / "args.json").write_text(json.dumps(dict(vars(args), fps_used=fps, node=socket.gethostname(),
                                                   gpu=torch.cuda.get_device_name(0), command=" ".join(sys.argv),
                                                   python_O=not __debug__, code_version=code), indent=1))
    lh.FPS, lh.FRAME_S = float(fps), 1.0 / float(fps)
    other = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.split()
    if other:
        raise SystemExit(f"machine not idle (gpu procs {other})")
    imgs = {s: str(Path(args.start_dir).resolve() / f"scene{s}.png") for s in (0, 1, 2)}
    sys.path.insert(0, str(mod.CODE_DIR))
    os.chdir(mod.CODE_DIR)  # external code uses repo-relative paths
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    m = mod.make(args.seconds, size=size) if size else mod.make(args.seconds)
    m.reset(imgs[0], 0)
    costs = [m.block(1 + i % 4) for i in range(args.warmup)]
    c = np.median(np.array([x[:3] for x in costs[3:]]), 0)
    cost = {"gen_unit": float(c[0]), "commit_unit": float(c[1]), "decode_unit": float(c[2]),
            "frames_per_unit": int(costs[-1][3]), "fps": fps, "nsteps": getattr(m, "nsteps", None),
            "unit_play_s": costs[-1][3] / fps}
    rl = math.ceil((c[0] + c[1] + c[2]) / lh.FRAME_S) + 1
    cost["reactive_lead_frames"] = rl
    print("COST", json.dumps(cost), flush=True)
    (out / "cost.json").write_text(json.dumps(cost, indent=1))
    rows = open(out / "rows.jsonl", "a")
    for tr in args.traces.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            trace = lh.make_trace(tr, seed, args.seconds)
            steps, disp, mem = run_trace(m, trace, imgs[seed % 3], seed, args.seconds, rl)
            pr = lh.score(trace, steps, args.seconds)
            held_s = disp.held_ticks * lh.FRAME_S
            rec = {"policy": m.name, "fps": lh.FPS, "lead_frames": rl, "lead_ms": rl * lh.FRAME_S * 1000,
                   "nsteps": getattr(m, "nsteps", None), "trace": tr, "seed": seed, "scene": f"scene{seed % 3}",
                   "presses": pr, "trace_events": trace, "stall_times": disp.stall_times, "n_steps": len(steps),
                   "frames_played": len(disp.played), "stall_s": disp.stall_ticks * lh.FRAME_S, "held_s": held_s,
                   "fps_while_held": (disp.held_ticks - disp.stall_ticks) / max(1e-6, held_s),
                   "late_steps": sum(bool(s["late"]) for s in steps), "n_swaps": 0,
                   "gen_s_p50": float(np.median([s["gen_s"] for s in steps])) if steps else None,
                   "commit_s_p50": float(np.median([s["commit_s"] for s in steps])) if steps else None,
                   "decode_s_p50": float(np.median([s["decode_s"] for s in steps])) if steps else None,
                   "peak_mem_gib": mem, "n_gen": len(steps),
                   "gpu_busy_s": float(sum(s["gen_s"] + s["commit_s"] + s["decode_s"] for s in steps))}
            rows.write(json.dumps(rec) + "\n")
            rows.flush()
            srv = [p for p in pr if p["status"] == "served"]
            print(json.dumps({"policy": m.name, "trace": tr, "seed": seed, "served": len(srv), "presses": len(pr),
                              "no_vae_p50": float(np.median([p["lat_no_vae"] for p in srv])) if srv else None,
                              "stall_s": rec["stall_s"], "fps": round(rec["fps_while_held"], 2)}), flush=True)


if __name__ == "__main__":
    main()
