"""Per-press latency breakdown of harness runs (where the commit-time gain of preparing candidates goes).

For every press served directly (its own key is displayed; successor-timed presses excluded) the input-to-display
latency splits into: wait for the next decision point (wait_decision), generation after the decision (after_decision;
~0 when a prepared candidate is selected), decoding (decode_s) and display-queue wait (queue_s). Also: fraction of
presses selected from a prepared candidate (hit). Means and P50/P90 per policy, pooled over traces and seeds.

  python -m shownow.eval.press_breakdown --run ShowNow-Reactive=<dir> --run ShowNow-8F=<dir> --out <md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PARTS = ("wait_decision", "after_decision", "decode_s", "queue_s", "lat_vae")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=run_dir; repeat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = ["## Per-press input-to-display breakdown (ms), directly served presses", "",
         "decision wait + generation after decision + decode + display-queue wait = presentation latency. "
         "Mean (P50 / P90).", "",
         "| method | presses | candidate ready at decision % | decision wait | generation after decision | decode | queue wait | presentation |",
         "|---|---|---|---|---|---|---|---|"]
    res = {}
    for spec in a.run:
        lab, d = spec.split("=", 1)
        ps = [p for f in Path(d).glob("rows*.jsonl") for l in open(f) for p in json.loads(l)["presses"]
              if p.get("status") == "served" and all(p.get(k) is not None for k in PARTS)]
        hits = [p["hit"] for p in ps if p.get("hit") is not None]
        col = {k: np.array([p[k] for p in ps]) * 1000 for k in PARTS}
        res[lab] = {"n": len(ps), "hit_pct": 100 * float(np.mean(hits)) if hits else None,
                    **{k: {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90))}
                       for k, v in col.items()}}
        c = lambda k: f"{res[lab][k]['mean']:.0f} ({res[lab][k]['p50']:.0f} / {res[lab][k]['p90']:.0f})"  # noqa: E731
        h = "-" if res[lab]["hit_pct"] is None else f"{res[lab]['hit_pct']:.1f}"
        L.append(f"| {lab} | {len(ps)} | {h} | " + " | ".join(c(k) for k in PARTS) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
