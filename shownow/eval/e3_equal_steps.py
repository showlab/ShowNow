"""E3 long-horizon quality at an equal number of autoregressive steps.

B1 (ShowNow) commits 1 latent (0.25 s) per AR step, B3 (SolarWM-FT) 3 latents (0.75 s). E3 scores twelve 5 s windows:
a window is 20 B1 steps and 20/3 B3 steps, so B1 window w and B3 window 3w end at the same AR step 20w (w = 1..4).
Per aligned step: latent FD of that window (pooled over the 20 test scenes; mean and range over noise seeds) and key-IDM
accuracy of that window (mean over seeds; scene-level paired bootstrap CI of B1 - B3 on seed-averaged scenes).
Also the full per-window curves against AR steps.

  python -m shownow.eval.e3_equal_steps --b1 rt=<run>,<run>,... --b1 normal=... --b3 SolarWM-FT=<run>,... --out <md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WIN_STEPS = {1: 20.0, 3: 20.0 / 3}  # AR steps per 5 s window


def load(runs):
    out = []
    for r in runs.split(","):
        d = json.load(open(Path(r) / "long_metrics_v2.json"))
        out.append({"fd": d["agg"]["fd_win"], "acc": {v["sid"]: v["key_idm_acc_win"] for v in d["videos"]}})
    return out


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", action="append", default=[], help="label=run,run,... (seeds)")
    ap.add_argument("--b3", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = {}
    for blk, specs in ((1, a.b1), (3, a.b3)):
        for s in specs:
            lab, runs = s.split("=", 1)
            cfg[lab] = (blk, load(runs))
    ref = next(l for l, (b, _) in cfg.items() if b == 3)
    sids = sorted(set.intersection(*[set(r["acc"]) for _, rs in cfg.values() for r in rs]))

    def at(lab, step):  # window index (0-based) ending at this AR step
        blk = cfg[lab][0]
        w = round(step / WIN_STEPS[blk]) - 1
        return w

    def scene_acc(lab, w):
        return np.array([np.mean([r["acc"][s][w] for r in cfg[lab][1]]) for s in sids])

    L = ["## E3 test split at equal autoregressive steps", "",
         f"{len(sids)} scenes. B1 window w and B3 window 3w end at the same AR step (20 B1 steps or 6.7 B3 steps per 5 s "
         "window). FD: pooled latent FD of the window, mean over noise seeds (range). Accuracy: key-IDM v2 accuracy of the "
         f"window, mean over seeds; difference to {ref}: seed-averaged per scene, paired scene bootstrap 95 % CI.", "",
         "| AR step | time B1 / B3 | " + " | ".join(f"{l} FD" for l in cfg) + " | " + " | ".join(f"{l} acc" for l in cfg) +
         " | " + " | ".join(f"{l} - {ref} acc" for l in cfg if l != ref) + " |",
         "|---|---|" + "---|" * (2 * len(cfg) + len(cfg) - 1)]
    res = {"aligned": []}
    for step in (20, 40, 60, 80):
        row, rec = [], {"step": step}
        for lab in cfg:
            fd = [r["fd"][at(lab, step)] for r in cfg[lab][1]]
            row.append(f"{np.mean(fd):.1f} ({min(fd):.1f}-{max(fd):.1f})" if len(fd) > 1 else f"{fd[0]:.1f}")
            rec[f"{lab}_fd"] = fd
        for lab in cfg:
            acc = scene_acc(lab, at(lab, step))
            row.append(f"{acc.mean():.3f}")
            rec[f"{lab}_acc"] = float(acc.mean())
        for lab in cfg:
            if lab == ref:
                continue
            dd = scene_acc(lab, at(lab, step)) - scene_acc(ref, at(ref, step))
            lo, hi = boot(dd)
            row.append(f"{dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            rec[f"{lab}_minus_ref"] = [float(dd.mean()), lo, hi]
        res["aligned"].append(rec)
        L.append(f"| {step} | {step * 0.25:.0f} s / {step * 0.75:.0f} s | " + " | ".join(row) + " |")
    L += ["", "Full curves (window end AR step: FD seed-mean, accuracy):", ""]
    res["curves"] = {}
    for lab, (blk, rs) in cfg.items():
        pts = [(round((w + 1) * WIN_STEPS[blk], 1), float(np.mean([r["fd"][w] for r in rs])),
                float(np.mean([np.mean([r["acc"][s][w] for s in sids]) for r in rs]))) for w in range(len(rs[0]["fd"]))]
        res["curves"][lab] = pts
        L.append(f"- {lab} (block {blk}, {len(rs)} seeds): " + ", ".join(f"{s:g}: {f:.1f}/{c:.2f}" for s, f, c in pts))
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
