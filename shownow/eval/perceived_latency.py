"""Perceived (action-to-effect) latency of harness runs saved with --save-lat.

Effect time of a press of key k at trace time t: the first displayed latent whose first frame is played at or after t
(and before the next press of the same key k) that the frozen 8-action IDM v2 reads as k (window t-1, t, t+1; argmax
over the run's key set). Presses later than (run seconds - margin) are not scored. Perceived latency of a press (successor timing, as main_table_stats): time until the effect of
this or a later press is first shown; never shown = +inf (kept). States: direct (its own effect first) / successor /
not shown. The detector sees only
the displayed latents (black box: the committed key is not used). Resolution: one latent = 4 frames (250 ms at 16 fps);
calibration on 1808 real key switches: detected 96.2 %, in the switch latent itself 91 % (shownow/eval/content_latency.py).

Blocks of BL latents per step (B3 = SolarWM-FT) are handled: latent i of step s plays from frame 4 * i of that step.
Statistics: the frozen main-table convention (equal-rate mixture P50/P90/P99, paired seed-cluster bootstrap).

  python -m shownow.eval.perceived_latency --run ShowNow-Reactive=<dir> --run ShowNow-8F=<dir> --ref ShowNow-8F --out <md>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from shownow.eval.idm import windows
from shownow.eval.idm_8a_v2 import scorer
from shownow.eval.main_table_stats import boot_ci, fmt, mixture_quantiles

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_LATENT = 4


def presses(trace):
    """(t, key, t_end, from_idle) for every press; its effect window ends at the next press of the same key."""
    out, held = [], None
    for t, k in trace:
        if k is not None and k != held:
            out.append([t, k, math.inf, held is None])
        held = k
    for i, a in enumerate(out):
        a[2] = next((b[0] for b in out[i + 1:] if b[1] == a[1]), math.inf)
    return out


@torch.no_grad()
def run_file(path, idm, keyset, t_score):
    d = torch.load(path, map_location="cpu", weights_only=False)
    lat = d["lat"].float()
    n_steps = len(d["steps"])
    BL = lat.shape[0] // n_steps
    assert BL * n_steps == lat.shape[0], (path, lat.shape, n_steps)
    seq = torch.cat([d["scene_lat"][0].float(), lat])  # intro latents + committed latents
    P = d["scene_lat"].shape[1]
    n = lat.shape[0]
    ts = list(range(P, P + n - 1))  # the centre latent needs a successor
    wx = windows(seq, ts)
    logit = torch.cat([idm(wx[i:i + 128].to(DEV)).float().cpu() for i in range(0, len(ts), 128)])
    mask = torch.full((logit.shape[1],), -1e9)
    mask[[k - 1 for k in keyset]] = 0
    pred = (logit + mask).argmax(1).numpy() + 1  # pred[i] = key read at committed latent i
    frames = defaultdict(list)
    for t, s in d["played"]:
        frames[s].append(t)
    t_play = np.full(n, np.nan)
    for i in range(n):
        f = sorted(frames.get(i // BL, []))
        j = (i % BL) * FRAMES_PER_LATENT
        if j < len(f):
            t_play[i] = f[j]
    pr = presses(d["trace"])
    shown = []  # absolute effect time of each press (inf if never read)
    for t, k, t_end, _ in pr:
        ok = [t_play[i] for i in range(n - 1) if not np.isnan(t_play[i]) and t <= t_play[i] < t_end and pred[i] == k]
        shown.append(min(ok) if ok else math.inf)
    out = []
    for i, (t, _, _, idle) in enumerate(pr):
        if t > t_score:  # too close to the end of the run for a slow system to show it
            continue
        j = min(range(i, len(pr)), key=lambda j: shown[j])
        lat_c = shown[j] - t
        out.append({"lat": lat_c, "idle": idle, "state": "not_shown" if not math.isfinite(lat_c) else
                    "direct" if shown[j] == shown[i] else "successor"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=run_dir[#policy] (with lat/*.pt); repeat")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--idm", default="runs/idm_8a/v2_frozen.pt")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=2.0, help="score presses up to (run seconds - margin)")
    ap.add_argument("--qs", default="0.5,0.9,0.99", help="quantiles reported")
    ap.add_argument("--switch-only", action="store_true", help="drop idle -> action presses")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs = tuple(float(x) for x in a.qs.split(","))
    qn = [f"P{round(q * 100)}" for q in qs]
    idm = scorer(a.idm, DEV)
    ev, src = {}, {}
    for spec in a.run:
        lab, d = spec.split("=", 1)
        d, _, pol = d.partition("#")  # Label=dir#policy when a run dir holds several policies
        args = json.load(open(Path(d) / "args.json"))
        keyset = [int(x) for x in args["key_set"].split(",")] if args.get("key_set") else list(range(1, 9))
        ev[lab], src[lab] = [], d
        for f in sorted((Path(d) / "lat").glob(f"*_{pol}.pt" if pol else "*.pt")):
            tr, seed = f.stem.split("_s")[0], int(f.stem.split("_s")[1].split("_")[0])
            ev[lab] += [{**e, "rate": tr, "seed": seed} for e in run_file(f, idm, keyset, args["seconds"] - a.margin)
                        if not (a.switch_only and e["idle"])]
        print("scored", lab, len(ev[lab]), flush=True)
    L = [f"## Perceived (action-to-effect) latency, equal-rate mixture quantiles (ms){' — switches only' if a.switch_only else ''}", "",
         "First displayed latent read by the frozen IDM v2 as the pressed key, successor timing; never shown = +inf "
         f"(n/e if a quantile hits it). Latent resolution (4 frames). Paired seed-cluster bootstrap, {a.n_boot} resamples.",
         "", "| method | presses | seeds | " + " | ".join(qn) + " | own effect first % | not shown % | source |",
         "|---|---|---|" + "---|" * len(qs) + "---|---|---|"]
    res = {}
    for lab, e in ev.items():
        q = mixture_quantiles(e, qs)
        ns = sum(x["state"] == "not_shown" for x in e) / max(len(e), 1) * 100
        nd = sum(x["state"] == "direct" for x in e) / max(len(e), 1) * 100
        res[lab] = {"q": q, "direct_pct": nd, "not_shown_pct": ns, "n": len(e), "source": src[lab]}
        L.append(f"| {lab} | {len(e)} | {len({x['seed'] for x in e})} | " + " | ".join(fmt(x) for x in q) +
                 f" | {nd:.1f} | {ns:.1f} | `{src[lab]}` |")
    L += ["", f"Paired differences to {a.ref} (ms), seed-cluster bootstrap 95 % CI:", "",
          "| method | " + " | ".join("Δ" + x for x in qn) + " |", "|---|" + "---|" * len(qs)]
    for lab in ev:
        if lab == a.ref:
            continue
        ci = boot_ci(ev[lab], ev[a.ref], "seed", a.n_boot, qs=qs)
        cells = []
        for k in range(len(qs)):
            qa, qb = res[lab]["q"][k], res[a.ref]["q"][k]
            dd = None if qa is None or qb is None else qa - qb
            lo, hi, nf = ci[k]
            c = f"{'n/e' if dd is None else f'{dd:+.0f}'} [{'n/e' if lo is None else f'{lo:+.0f}'}, " \
                f"{'n/e' if hi is None else f'{hi:+.0f}'}]" + (f" ({nf}/{a.n_boot} finite)" if nf < a.n_boot else "")
            cells.append(c)
            res[lab].setdefault("diff", []).append({"d": dd, "ci_seed": ci[k]})
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"ref": a.ref, "results": res}, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
