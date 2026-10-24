"""Summarise runs/long_drift/<name> (shownow/eval/long_drift_eval.py) -> summary.md / summary.json.

Per model: latent FD vs real ViZDoom latents per 5 s window, pixel-stat log-ratios vs the fork's real history and
motion ratio vs the real futures in the late window (last 10 s), and a drift score = late latent FD / real noise floor
+ 10 x mean |log ratio| (sharpness, contrast, saturation). Lower = less drift. A model whose late motion ratio < 0.3 is
flagged as frozen (static video has trivially low drift).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WIN = 20  # latents per FD window (5 s)
LATE_SEC = 10


def fd(a: np.ndarray, b: np.ndarray) -> float:
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr sqrtm(s1 s2) = sum sqrt(eig(s1 s2)) (eigenvalues real >= 0 for PSD s1, s2; no scipy in the venv)
    tr_cs = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2) - 2 * tr_cs)


def main() -> None:
    run = Path(sys.argv[1])
    ref = np.load(run / "real_ref.npy")
    rows = [json.loads(l) for p in sorted(run.glob("rows_rank*.jsonl")) for l in open(p)]
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    rng = np.random.default_rng(0)
    res = {}
    for model, rs in sorted(by.items()):
        feats = np.stack([np.load(run / "feats" / f"{model}__{r['sample_id']}.npy") for r in rs])  # [n,4,N,96]
        N = feats.shape[2]
        nwin = N // WIN
        fds = [fd(feats[:, :, w * WIN:(w + 1) * WIN].reshape(-1, 96), ref) for w in range(nwin)]
        if "floor" not in res:  # FD of a real subset of the same size vs the rest
            n = feats.shape[0] * 4 * WIN
            fl = []
            for _ in range(5):
                idx = rng.permutation(len(ref))
                fl.append(fd(ref[idx[:n]], ref[idx[n:]]))
            res["floor"] = float(np.mean(fl))
        late = slice(-LATE_SEC, None)
        lr = {k: [] for k in ("sharp", "contrast", "sat", "bright")}
        motion = []
        curves = defaultdict(list)
        for r in rs:
            for st in r["stats"]:
                for k in lr:
                    v = np.array(st[k])
                    if k == "bright":
                        lr[k].append(np.mean(v[late]) - r["ref"][k])
                    else:
                        lr[k].append(np.log(np.mean(v[late]) / r["ref"][k]))
                    curves[k].append(np.log(v / r["ref"][k]) if k != "bright" else v - r["ref"][k])
                motion.append(np.mean(np.array(st["motion"])[late]) / r["ref"]["motion_future"])
        late_fd = float(np.mean(fds[-(LATE_SEC * 4) // WIN:]))
        abs_log = float(np.mean([np.mean(np.abs(lr[k])) for k in ("sharp", "contrast", "sat")]))
        res[model] = {
            "forks": len(rs), "fd_windows": fds, "fd_late": late_fd, "fd_first": fds[0],
            "log_sharp": float(np.mean(lr["sharp"])), "log_contrast": float(np.mean(lr["contrast"])),
            "log_sat": float(np.mean(lr["sat"])), "d_bright": float(np.mean(lr["bright"])),
            "abs_log_pix": abs_log, "motion_ratio": float(np.mean(motion)),
            "sec_gen": float(np.mean([r["sec_gen"] for r in rs])),
            "curve_sharp": np.mean(curves["sharp"], 0).round(3).tolist(),
            "curve_sat": np.mean(curves["sat"], 0).round(3).tolist(),
        }
        res[model]["drift_score"] = late_fd / res["floor"] + 10 * abs_log
        res[model]["frozen"] = res[model]["motion_ratio"] < 0.3
    floor = res.pop("floor")
    models = sorted(res, key=lambda m: (res[m]["frozen"], res[m]["drift_score"]))
    secs = len(next(iter(res.values()))["curve_sharp"])
    hdr = " | ".join(f"FD {w * 5}-{w * 5 + 5}s" for w in range(secs // 5))
    lines = [f"Real-vs-real latent FD noise floor (same sample size): {floor:.3f}. Late window = last {LATE_SEC} s.",
             "drift_score = FD_late / floor + 10 * mean|log ratio| (sharp, contrast, sat); lower = less drift. "
             "motion = late motion / real future motion.", "",
             f"| rank | model | forks | {hdr} | log sharp | log contrast | log sat | d bright | motion | drift_score |",
             "|" + "---|" * (9 + secs // 5)]
    for i, m in enumerate(models, 1):
        r = res[m]
        lines.append(f"| {i} | {m}{' (FROZEN)' if r['frozen'] else ''} | {r['forks']} | "
                     + " | ".join(f"{v:.2f}" for v in r["fd_windows"])
                     + f" | {r['log_sharp']:+.3f} | {r['log_contrast']:+.3f} | {r['log_sat']:+.3f} | "
                       f"{r['d_bright']:+.1f} | {r['motion_ratio']:.2f} | **{r['drift_score']:.2f}** |")
    lines += ["", "Per-second log sharpness ratio vs history (mean over rollouts), every 5 s:"]
    for m in models:
        lines.append(f"- {m}: " + " ".join(f"{v:+.2f}" for v in res[m]["curve_sharp"][::5]))
    (run / "summary.md").write_text("\n".join(lines) + "\n")
    (run / "summary.json").write_text(json.dumps({"floor": floor, "models": res, "ranking": models}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
