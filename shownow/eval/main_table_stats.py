"""Latency statistics for the paper tables (protocol fixed before the 6-seed main-table runs).

Per press (a key different from the held one, incl. idle -> action): all-press successor latency = time until this or
a later press is served (presentation: first frame played; primary: latent committed); a press with no served
successor in the observation window is +inf (kept, never dropped).

Equal-rate mixture quantiles: one distribution F_mix(t) = 1/4 sum_r F_r(t) over the input rates r; a press of rate r
weighs 1 / (R * N_r). P50 / P90 / P99 are the generalised inverse of F_mix (nearest rank on the weighted CDF). If the
quantile falls on +inf it is reported as not finitely estimable ("n/e"), never imputed.

Uncertainty: paired seed-cluster bootstrap (resample seeds with replacement; a seed keeps all its rates, inputs and the
method pairing), 2000 resamples, percentile 95 % CI of the difference A - B. Every effect and interval is reported;
an interval containing 0 is not evidence of equivalence. Trace-level CI (resample (rate, seed) within rates) is a
supplementary sensitivity check only.

Also: event-state counts (direct / served via a successor / successor across an idle gap / no successor), stall ticks
split into idle-start (restart) and mid-motion (ongoing), fps while held, GPU busy share; and, for transparency,
unweighted pooled and per-rate quantiles (supplement).

  python -m shownow.eval.main_table_stats --run Reactive=<dir> --run ShowNow-1F=<dir> --run ShowNow=<dir> \
      --ref Reactive --out <md> [--metric presentation|primary]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

QS = (0.50, 0.90, 0.99)


def press_events(r, vae: bool, t_max: float = math.inf):
    """-> list of dicts (rate, seed, latency s or inf, state, idle) for one harness row. Presses later than t_max are
    not scored (end-of-trace censoring rule); they still serve as successors of earlier presses."""
    pr = sorted(r["presses"], key=lambda p: p["t"])
    trace = r["trace_events"]
    out = []
    for i, p in enumerate(pr):
        if p["t"] > t_max:
            continue
        ok = lambda q: q["status"] == "served" and (not vae or q.get("lat_vae") is not None)  # noqa: E731
        q = next((q for q in pr[i:] if ok(q)), None)
        if q is None:
            lat, state = math.inf, "no_successor"
        else:
            lat = q["t"] + (q["lat_vae"] if vae else q["lat_no_vae"]) - p["t"]
            if q is p:
                state = "direct"
            else:  # served through a later press: did the key go idle in between?
                idle_gap = any(k is None and p["t"] < t < q["t"] for t, k in trace)
                state = "successor_across_idle" if idle_gap else "successor"
        out.append({"rate": r["trace"], "seed": r["seed"], "lat": lat, "state": state, "idle": bool(p.get("from_idle"))})
    return out


def mixture_quantiles(events, qs=QS):
    """equal-rate mixture quantiles (ms; inf -> None) of a list of events."""
    by_rate = defaultdict(list)
    for e in events:
        by_rate[e["rate"]].append(e["lat"])
    R = len(by_rate)
    vals, w = [], []
    for v in by_rate.values():
        vals += v
        w += [1.0 / (R * len(v))] * len(v)
    order = np.argsort(vals, kind="stable")
    v = np.asarray(vals)[order]
    cw = np.cumsum(np.asarray(w)[order])
    out = []
    for q in qs:
        i = int(np.searchsorted(cw, q - 1e-12))
        x = v[min(i, len(v) - 1)]
        out.append(None if not math.isfinite(x) else float(x) * 1000)
    return out


def plain_quantiles(lats, qs=QS):
    a = sorted(lats)
    return [None if not math.isfinite(a[max(0, math.ceil(q * len(a)) - 1)]) else a[max(0, math.ceil(q * len(a)) - 1)] * 1000
            for q in qs]


def boot_ci(ev_a, ev_b, unit="seed", n=2000, seed=0, qs=QS):
    """paired bootstrap of mixture-quantile differences A - B -> per q (lo, hi, n_finite)."""
    rng = np.random.default_rng(seed)
    key = (lambda e: e["seed"]) if unit == "seed" else (lambda e: (e["rate"], e["seed"]))
    ga, gb = defaultdict(list), defaultdict(list)
    for e in ev_a:
        ga[key(e)].append(e)
    for e in ev_b:
        gb[key(e)].append(e)
    units = sorted(set(ga) & set(gb))
    strata = defaultdict(list)
    for u in units:
        strata[u[0] if unit == "trace" else 0].append(u)
    diffs = [[] for _ in qs]
    for _ in range(n):
        sa, sb = [], []
        for us in strata.values():
            for j, i in enumerate(rng.integers(0, len(us), len(us))):
                u = us[i]
                tag = f"{u}#{j}"
                sa += [{**e, "seed": tag} for e in ga[u]]
                sb += [{**e, "seed": tag} for e in gb[u]]
        qa, qb = mixture_quantiles(sa, qs), mixture_quantiles(sb, qs)
        for k in range(len(qs)):
            if qa[k] is not None and qb[k] is not None:
                diffs[k].append(qa[k] - qb[k])
    return [(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)) if d else (None, None, 0)
            for d in diffs]


def stall_split(rows):
    fs = 1.0 / rows[0]["fps"]
    held = sum(r["held_s"] for r in rows)
    idle = mid = 0
    for r in rows:
        pr = sorted(r["presses"], key=lambda p: p["t"])
        ts = [p["t"] for p in pr]
        for s in r["stall_times"]:
            i = int(np.searchsorted(ts, s)) - 1
            if i >= 0 and pr[i].get("from_idle"):
                idle += 1
            else:
                mid += 1
    return idle * fs / held * 100, mid * fs / held * 100


def fmt(v):
    return "n/e" if v is None else f"{v:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=run_dir (rows.jsonl); repeat")
    ap.add_argument("--policy", action="append", default=[], help="Label=policy prefix, if a run dir holds several")
    ap.add_argument("--ref", required=True, help="label of the reference method for paired differences")
    ap.add_argument("--metric", choices=["presentation", "primary"], default="presentation")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--end-margin", type=float, default=0.0,
                    help="score only presses up to (trace seconds - margin); paper convention: 10 s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pol = dict(x.split("=", 1) for x in a.policy)
    runs = {}
    for spec in a.run:
        lab, d = spec.split("=", 1)
        rows = [json.loads(l) for l in open(Path(d) / "rows.jsonl")]
        if lab in pol:
            rows = [r for r in rows if r["policy"].startswith(pol[lab])]
        runs[lab] = (d, rows)
    vae = a.metric == "presentation"
    def t_max(d):
        try:
            sec = float(json.load(open(Path(d) / "args.json")).get("seconds", 60.0))
        except (OSError, ValueError):
            sec = 60.0
        return sec - a.end_margin if a.end_margin > 0 else math.inf
    ev = {lab: [e for r in rows for e in press_events(r, vae, t_max(d))] for lab, (d, rows) in runs.items()}
    res = {}
    L = [f"## {a.metric.capitalize()} latency, equal-rate mixture quantiles (ms)", "",
         "Frozen convention: all presses, successor timing, no-successor = +inf (n/e if a quantile hits it), nearest rank "
         "on the equal-rate mixture CDF; paired seed-cluster bootstrap, " + f"{a.n_boot} resamples." +
         (f" Presses in the final {a.end_margin:g} s of each trace are not scored (end-of-trace censoring)."
          if a.end_margin > 0 else ""), "",
         "| method | presses | seeds | P50 | P90 | P99 | direct / successor / across idle / no successor | "
         "stall idle-start % | stall mid-motion % | fps | GPU busy % | source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, (d, rows) in runs.items():
        q = mixture_quantiles(ev[lab])
        st = defaultdict(int)
        for e in ev[lab]:
            st[e["state"]] += 1
        si, sm = stall_split(rows)
        held = sum(r["held_s"] for r in rows)
        fps = sum(r["fps_while_held"] * r["held_s"] for r in rows) / held
        busy = (sum(r["gpu_busy_s"] for r in rows) / (60.0 * len(rows)) * 100) if all("gpu_busy_s" in r for r in rows) \
            else float("nan")  # latency_mg rows (multi-GPU) do not record GPU busy time
        seeds = sorted({r["seed"] for r in rows})
        res[lab] = {"q": q, "states": dict(st), "stall_idle": si, "stall_mid": sm, "fps": fps, "busy": busy,
                    "n": len(ev[lab]), "seeds": seeds, "source": d}
        L.append(f"| {lab} | {len(ev[lab])} | {len(seeds)} | " + " | ".join(fmt(x) for x in q) +
                 f" | {st['direct']} / {st['successor']} / {st['successor_across_idle']} / {st['no_successor']} | "
                 f"{si:.2f} | {sm:.2f} | {fps:.2f} | {busy:.0f} | `{d}` |")
    L += ["", f"Paired differences to {a.ref} (ms), seed-cluster bootstrap 95 % CI; trace-level CI in brackets "
          "(supplementary sensitivity only):", "", "| method | ΔP50 | ΔP90 | ΔP99 |", "|---|---|---|---|"]
    for lab in runs:
        if lab == a.ref:
            continue
        ci_s = boot_ci(ev[lab], ev[a.ref], "seed", a.n_boot)
        ci_t = boot_ci(ev[lab], ev[a.ref], "trace", a.n_boot)
        cells = []
        for k in range(len(QS)):
            qa, qb = res[lab]["q"][k], res[a.ref]["q"][k]
            dd = None if qa is None or qb is None else qa - qb
            (lo, hi, n), (lo2, hi2, n2) = ci_s[k], ci_t[k]
            c = f"{fmt(dd) if dd is None else f'{dd:+.0f}'} [{fmt(lo) if lo is None else f'{lo:+.0f}'}, " \
                f"{fmt(hi) if hi is None else f'{hi:+.0f}'}]"
            if n < a.n_boot:
                c += f" ({n}/{a.n_boot} finite)"
            c += f" (trace: [{fmt(lo2) if lo2 is None else f'{lo2:+.0f}'}, {fmt(hi2) if hi2 is None else f'{hi2:+.0f}'}])"
            cells.append(c)
            res[lab].setdefault("diff", []).append({"d": dd, "ci_seed": ci_s[k], "ci_trace": ci_t[k]})
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "Supplementary sensitivity: unweighted pooled and per-rate nearest-rank quantiles (ms):", "",
          "| method | pooled P50/P90/P99 | " + " | ".join(sorted({e['rate'] for e in ev[a.ref]})) + " |",
          "|---|---|" + "---|" * len({e['rate'] for e in ev[a.ref]})]
    for lab in runs:
        pooled = plain_quantiles([e["lat"] for e in ev[lab]])
        per = []
        for rt in sorted({e["rate"] for e in ev[a.ref]}):
            per.append("/".join(fmt(x) for x in plain_quantiles([e["lat"] for e in ev[lab] if e["rate"] == rt])))
        L.append(f"| {lab} | {'/'.join(fmt(x) for x in pooled)} | " + " | ".join(per) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"metric": a.metric, "ref": a.ref, "results": res}, indent=1,
                                                           default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
