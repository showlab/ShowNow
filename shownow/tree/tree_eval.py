"""Evaluate branch-tree models (all 4 futures of a fork in one sequence) with the same metrics as the baseline evals.

--kind teacher : bidirectional teacher, tree sampler (UniPC 30)
--kind s1      : Stage1 AnyFlow student, tree window AR sampler (flow map, --nfe)
--kind s2      : Stage2 SGF student, tree window AR sampler (4-step schedule)
Launch: sbatch slurm/tree_eval.sbatch --kind teacher --ckpts a.pt,b.pt --out runs/tree_eval/<name>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
from shownow.eval.camera_prior_eval import LAT, SRC  # noqa: E402
from shownow.eval.student_eval import gt_features, record  # noqa: E402
from shownow.train.teacher_action_ft import camera_tokens  # noqa: E402
from shownow.tree.tree import enable_tree  # noqa: E402
from shownow.tree.tree_samplers import student_tree_sample, teacher_tree_sample  # noqa: E402


def load(kind, ckpt, field, device):
    if kind == "teacher":
        from shownow.train.teacher_action_ft import build_teacher
        d, _ = build_teacher(ckpt)
    elif kind == "s1":
        from shownow.train.student_stage1 import build_student
        d, _ = build_student(ckpt, from_teacher=False, field=field)
    else:
        from shownow.train.student_stage2 import build_stage2_student
        d, _, _ = build_stage2_student(ckpt, field=field)
    d.module.to(device, dtype=torch.bfloat16).eval()
    enable_tree(d.module)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["teacher", "s1", "s2"])
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--field", default="generator_ema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--forks", type=int, default=48)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--videos-per-rank", type=int, default=1)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--history", type=int, default=15)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    split = json.load(open(REPO / "runs/teacher_ft/wasd_v1/split.json"))
    val = sorted(split["val_ood"])[: args.forks][args.rank::args.world]
    from solarwm.backends.wan22.runtime.components import Wan5BVAE
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    vae = Wan5BVAE(REPO / "models/SolarWM-5B-base/vae/Wan2.2_VAE.pth").to(device, dtype=torch.bfloat16)
    rows_path = out / f"rows_rank{args.rank}.jsonl"
    done = {(r["model"], r["sample_id"]) for r in map(json.loads, open(rows_path))} if rows_path.exists() else set()
    gt_cache = {}
    for ckpt in args.ckpts.split(","):
        stem = Path(ckpt).stem
        name = f"tree_{args.kind}_{stem.split('_step')[0].split('_')[-1]}{stem.split('_step')[-1]}"
        if args.kind != "teacher":
            name += "_" + args.field.replace("generator", "g")
        if args.kind == "s1":
            name += f"_nfe{args.nfe}"
        if args.block != 3:
            name += f"_b{args.block}"
        if all((name, sid) in done for sid in val):
            continue
        d = load(args.kind, ckpt, args.field, device)
        root = d.module
        steps_raw = [float(s) for s in warp_denoising_steps([1000, 750, 500, 250], d.scheduler.timesteps.to(device),
                                                            num_train_timesteps=1000)]
        for fi, sid in enumerate(val):
            if (name, sid) in done:
                continue
            t0 = time.time()
            fork = torch.load(LAT / f"{sid}.pt", map_location="cpu", weights_only=False)
            cam_rel = torch.load(REPO / "data/cf_vizdoom_10k_cam" / f"{sid}.pt", weights_only=True)["nominal"]
            m = json.load(open(SRC / sid / "meta.json"))
            clean = fork["latents"].to(device)
            prompt = fork["prompt"].to(device)[None]
            cam = camera_tokens(cam_rel, 4, device, args.cam_scale)
            acts = fork["actions"].to(device)
            if sid not in gt_cache:
                gt_cache[sid] = gt_features(vae, clean)
            torch.cuda.synchronize()
            tg = time.time()
            with torch.no_grad():
                if args.kind == "teacher":
                    x = teacher_tree_sample(d, root, clean, prompt, cam, acts, steps=30, seed=fi)
                elif args.kind == "s1":
                    x = student_tree_sample(d, root, clean, prompt, cam, acts, solver="flowmap", nfe=args.nfe, seed=fi,
                                            max_prior_chunks=args.history // args.block, block=args.block)
                else:
                    x = student_tree_sample(d, root, clean, prompt, cam, acts, solver="fewstep", fewstep=steps_raw,
                                            seed=fi, x0_of=d.flow_to_x0,
                                            max_prior_chunks=args.history // args.block, block=args.block)
                torch.cuda.synchronize()
                gen_sec = time.time() - tg
                pix = vae.decode(x)
            record(out, rows_path, name, "nom_s%g_act" % args.cam_scale, sid, m, x, pix, clean, gt_cache[sid], fi,
                   args.videos_per_rank, t0)
            with open(out / f"timing_rank{args.rank}.jsonl", "a") as fh:
                fh.write(json.dumps({"model": name, "sample_id": sid, "gen_sec": gen_sec}) + "\n")
        del d, root
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
