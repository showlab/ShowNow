"""E2 scoring for the 8-action models: the 8-action latent IDM (shownow/eval/idm_8a.py, runs/idm_8a/v1) reads the key
from generated and real latents (t-1, t, t+1) of every branch of the key-switch forks (e2_8a.py <run>/lat).

  action accuracy   labelled future latents (idm_8a.label_mask: the REAL branch's camera confirms the key during that
                    latent: W/A/S/D move along the key axis, turns rotate > 5 deg the right way, JUMP leaves the
                    ground, ATTACK always), IDM(generated) == scheduled key; same latents for the real video
  transition        labelled latents within the first 2 latents of a key segment; settled = the rest
  response          per key segment: latents until the IDM first reads the new key (0 = in its first latent), on the
                    generated and the real video of the same branch; extra = generated - real (negative = faster than
                    the game, whose momentum delays motion); not slower = share with extra <= 0
  per key           action accuracy per scheduled key
  coverage          labelled / all scored future latents (the rest: walls, momentum)
Also mean LPIPS / PSNR / SSIM and red jumps from the generation rows (rows_rank*.jsonl).

  python -m shownow.eval.e2_8a_score runs/e2_8a/<run> [more runs] [--out-json x.json] [--table]
  IDM v2 (shownow/eval/idm_8a_v2.py): --idm runs/idm_8a/<v2>/idm.pt --tag v2 -> the v2 identifiable mask (engine state
  rules, labels_v2.pt) replaces label_mask and the result goes to e2_score_v2.json (the v1 e2_score.json is kept).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from shownow.eval.idm import windows  # noqa: E402
from shownow.eval.idm_8a import BRANCHES, label_mask  # noqa: E402

H = 9
TS = list(range(H, 20))  # future latents with a t+1 neighbour


def after_switch(keys: torch.Tensor) -> np.ndarray:
    """keys [21] -> latents since the current key segment started, per future latent (segment 0 starts at the fork)."""
    out = np.zeros(21, dtype=int)
    for t in range(H + 1, 21):
        out[t] = out[t - 1] + 1 if int(keys[t]) == int(keys[t - 1]) else 0
    return out


@torch.no_grad()
def score_fork(model, gen, fork, cam_true, dev, lab=None, why=None):
    acts = fork["actions"]
    lab = label_mask(acts, cam_true) if lab is None else lab
    rec = {k: [] for k in ("key", "pg", "pr", "lab", "after", "branch", "t", "why")}
    resp = []
    for b in range(gen.shape[0]):
        pg = model(windows(gen[b].to(dev), TS)).argmax(1).cpu().numpy() + 1
        pr = model(windows(fork["latents"][b].to(dev), TS)).argmax(1).cpu().numpy() + 1
        aft = after_switch(acts[b])
        for i, t in enumerate(TS):
            rec["key"].append(int(acts[b, t]))
            rec["pg"].append(int(pg[i]))
            rec["pr"].append(int(pr[i]))
            rec["lab"].append(bool(lab[b, t]))
            rec["after"].append(int(aft[t]))
            rec["branch"].append(b)
            rec["t"].append(t)
            rec["why"].append(int(why[b, t]) if why is not None else -1)
        for i, t in enumerate(TS):  # response per segment start
            if aft[t] != 0:
                continue
            key = int(acts[b, t])
            seg = [i2 for i2, t2 in enumerate(TS) if t2 >= t and int(acts[b, t2]) == key and aft[t2] == t2 - t]
            hit = lambda p: next((n for n, i2 in enumerate(seg) if p[i2] == key), None)  # noqa: E731
            resp.append({"key": key, "len": len(seg), "gen": hit(pg), "real": hit(pr)})
    return rec, resp


def summarize(recs, resps):
    key, pg, pr = (np.array(sum((r[k] for r in recs), [])) for k in ("key", "pg", "pr"))
    lab, aft = np.array(sum((r["lab"] for r in recs), []), bool), np.array(sum((r["after"] for r in recs), []))
    tr, st = lab & (aft < 2), lab & (aft >= 2)
    pairs = [(x["gen"], x["real"]) for x in resps if x["gen"] is not None and x["real"] is not None]
    extra = np.array([g - r for g, r in pairs]) if pairs else np.zeros(0)
    acc = lambda p, m: float((p == key)[m].mean()) if m.any() else None  # noqa: E731
    why = np.array(sum((r["why"] for r in recs), []))
    allm = np.array([t != 20 for r in recs for t in r["t"]])
    sens = {"acc_all_latents_gen": acc(pg, allm), "acc_all_latents_real": acc(pr, allm)}
    if (why >= 0).any():  # v2: accuracy per exclusion reason (sensitivity of the score to the exclusion rules)
        from shownow.eval.idm_8a_v2 import REASONS
        for i, nm in enumerate(REASONS):
            m = why == i
            if m.any():
                sens[f"why_{nm}"] = {"n": int(m.sum()), "gen": acc(pg, m), "real": acc(pr, m)}
    return {**sens, "acc_gen": acc(pg, lab), "acc_real": acc(pr, lab), "transition_gen": acc(pg, tr), "settled_gen": acc(pg, st),
            "transition_real": acc(pr, tr), "settled_real": acc(pr, st),
            "per_key_gen": {b: acc(pg, lab & (key == i + 1)) for i, b in enumerate(BRANCHES)},
            "per_key_real": {b: acc(pr, lab & (key == i + 1)) for i, b in enumerate(BRANCHES)},
            "n_labelled": int(lab.sum()), "coverage": float(lab.mean()),
            "response_extra_mean": float(extra.mean()) if len(extra) else None,
            "not_slower": float((extra <= 0).mean()) if len(extra) else None, "n_pairs": len(extra),
            "never_gen": float(np.mean([x["gen"] is None for x in resps])),
            "never_real": float(np.mean([x["real"] is None for x in resps])), "n_segments": len(resps)}


def score_run(run: Path, model, dev, v2: bool = False):
    from shownow.tree.cond_v3 import cam_dir_for
    groups = {}
    for f in sorted((run / "lat").glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fp = Path(d["fork"])
        fork = torch.load(fp, map_location="cpu", weights_only=False)
        cam = torch.load(cam_dir_for(fp.parent) / fp.name, weights_only=True)["true"]
        lab = why = None
        if v2:
            from shownow.eval.idm_8a_v2 import reasons
            why = reasons(fp.stem).clone()
            why[:, :H + 1] = -1  # latent 9 (first future latent) is never scored, as in v1 and the IDM training
            lab = why == 0
        rec, resp = score_fork(model, d["gen"], fork, cam, dev, lab, why)
        groups.setdefault(d["cond"], ([], [], []))
        groups[d["cond"]][0].append(rec)
        groups[d["cond"]][1].extend(resp)
        groups[d["cond"]][2].append(fp.stem)
    rows = [json.loads(l) for f in run.glob("rows_rank*.jsonl") for l in open(f)]
    res = {}
    for cond, (recs, resps, sids) in groups.items():
        q = [r for r in rows if r["cond"] == cond and r["sample_id"] in set(sids)]
        qual = {k: float(np.mean([r[k] for r in q])) for k in q[0] if k.startswith(("lpips", "psnr", "ssim"))} if q else {}
        qual["red_jumps_per_fork"] = float(np.mean([r["red_jumps"] for r in q])) if q else None
        res[cond] = {"forks": len(recs), **summarize(recs, resps), **qual}
    return res


def table(results: dict) -> str:
    head = ("| run | cond | forks | action acc (real) | transition / settled | extra response / not slower | " +
            " | ".join(BRANCHES) + " | LPIPS 0-0.5/0-1 s | PSNR 0-1 s | SSIM 0-1 s |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for run, res in results.items():
        for cond, r in res.items():
            f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
            lines.append(f"| {run} | {cond} | {r['forks']} | {f(r['acc_gen'])} ({f(r['acc_real'])}) | "
                         f"{f(r['transition_gen'])} / {f(r['settled_gen'])} | "
                         f"{f(r['response_extra_mean'], 2)} / {f(r['not_slower'], 2)} | " +
                         " | ".join(f(r["per_key_gen"][b], 2) for b in BRANCHES) +
                         f" | {f(r.get('lpips_0-0.5s'))}/{f(r.get('lpips_0-1s'))} | {f(r.get('psnr_0-1s'), 2)} | "
                         f"{f(r.get('ssim_0-1s'))} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--idm", default=str(REPO / "runs/idm_8a/v1/idm.pt"))
    ap.add_argument("--out-json", default="")
    ap.add_argument("--tag", default="", help="writes e2_score_<tag>.json (v2* tags use the v2 identifiable mask)")
    ap.add_argument("--mask", choices=["v1", "v2"], default="", help="override the scoring mask (default: v2 for v2* tags)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    from shownow.eval.idm_8a_v2 import scorer
    model = scorer(args.idm, dev)
    results = {}
    for run in map(Path, args.runs):
        v2 = (args.mask or ("v2" if args.tag.startswith("v2") else "v1")) == "v2"
        results[str(run)] = score_run(run, model, dev, v2=v2)
        name = f"e2_score_{args.tag}.json" if args.tag else "e2_score.json"
        (run / name).write_text(json.dumps({**results[str(run)], "_idm": args.idm, "_mask": "v2" if v2 else "v1"}, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=1))
    print(table(results))


if __name__ == "__main__":
    main()
