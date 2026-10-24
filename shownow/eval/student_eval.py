"""Evaluate AR student checkpoints (Stage1 AnyFlow / Stage2 SGF) on held-out ViZDoom forks, same metrics as the teacher
camera-prior eval (shownow/eval/camera_prior_eval.py): 4-way counterfactual retrieval (latent / flow / pixel MSE per
time window), PSNR vs own vs other GT branch, ego-motion direction; plus side-by-side videos (top: W A S D generated,
bottom: GT).

Launch: sbatch slurm/student_eval.sbatch --stage 1 --ckpts a.pt,b.pt --out runs/student_eval/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
from shownow.eval.camera_prior_eval import (LAT, N_HIST, SRC, flow_features, mse_matrices, retrieval,  # noqa: E402
                                            to_uint8)
from shownow.eval.ego_motion import ego_motion  # noqa: E402
from shownow.train.student_common import ar_flowmap_sample, history_rollout  # noqa: E402
from shownow.train.teacher_action_ft import FUTURE0, camera_tokens  # noqa: E402

BRANCHES = ["W", "A", "S", "D"]


def load_model(stage: int, ckpt: str, field: str, device, block: int = 3, history: int = 15):
    if stage == 1:
        from shownow.train.student_stage1 import build_student
        diffusion, _ = build_student(ckpt, from_teacher=False, field=field)
    else:
        from shownow.train.student_stage2 import build_stage2_student
        diffusion, _, _ = build_stage2_student(ckpt, field=field, block=block, history=history)
    diffusion.module.to(device, dtype=torch.bfloat16).eval()
    return diffusion


def boundary_jump(frames: np.ndarray) -> float:
    """frames uint8 [48,H,W,3] (future, 4 frames per latent). Mean |diff| across latent boundaries divided by the mean
    |diff| between adjacent frames inside a latent (1.0 = no visible seam at latent boundaries)."""
    f = frames.astype(np.float32)
    d = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))  # d[i] = |f[i+1]-f[i]|
    idx = np.arange(len(d))
    at_boundary = (idx % 4) == 3
    return float(d[at_boundary].mean() / (d[~at_boundary].mean() + 1e-6))


def record(out, rows_path, name, cond, sid, m, x, pix, clean, gt, fi, videos_per_rank, t0):
    """Metrics vs GT for one fork (4 branches) + optional side-by-side video. gt = (gt_pix, gt_flow, gt_ego)."""
    gt_pix, gt_flow, gt_ego = gt
    frames = [to_uint8(pix[i, N_HIST:]) for i in range(4)]
    gen_flow = np.stack([flow_features(f) for f in frames])
    gen_lat = x[:, FUTURE0:].float().flatten(1).cpu().numpy()
    gt_lat = clean[:, FUTURE0:].float().flatten(1).cpu().numpy()
    row = {"model": name, "cond": cond, "sample_id": sid, "map": m["environment"]["map"],
           "free": [m["futures"][k]["summary"]["displacement_m"] >= 12.0 for k in BRANCHES],
           "acc_latent": retrieval(gen_lat, gt_lat, False), "acc_flow": retrieval(gen_flow, gt_flow, True),
           "flow_cos_own": [float(np.dot(gen_flow[i], gt_flow[i]) / (np.linalg.norm(gen_flow[i]) *
                                                                     np.linalg.norm(gt_flow[i]) + 1e-8))
                            for i in range(4)],
           "ego_pred": [ego_motion(to_uint8(pix[i, N_HIST - 1:N_HIST + 16]))["pred"] for i in range(4)],
           "gt_ego_pred": gt_ego, "mse": mse_matrices(pix, gt_pix, x, clean), "sec": round(time.time() - t0, 1),
           "boundary_jump": [boundary_jump(f) for f in frames],
           "boundary_jump_gt": [boundary_jump(to_uint8(gt_pix[i, N_HIST:])) for i in range(4)]}
    with open(rows_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("model", "sample_id", "acc_latent", "acc_flow", "sec")}), flush=True)
    if fi < videos_per_rank:
        import cv2
        import imageio.v2 as iio2

        def small(f):
            return np.stack([cv2.resize(z, (432, 240), interpolation=cv2.INTER_AREA) for z in f])
        top = np.concatenate([small(f) for f in frames], axis=2)
        bot = np.concatenate([small(to_uint8(gt_pix[i, N_HIST:])) for i in range(4)], axis=2)
        iio2.mimwrite(out / "videos" / f"{sid}_{name}.mp4", np.concatenate([top, bot], 1), fps=16, quality=7,
                      macro_block_size=8)


def gt_features(vae, clean):
    gt_pix = vae.decode(clean)
    return (gt_pix, np.stack([flow_features(to_uint8(gt_pix[i, N_HIST:])) for i in range(4)]),
            [ego_motion(to_uint8(gt_pix[i, N_HIST - 1:N_HIST + 16]))["pred"] for i in range(4)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--field", default="generator_ema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--forks", type=int, default=48)
    ap.add_argument("--nfe", type=int, default=4, help="Stage1 flow-map steps")
    ap.add_argument("--block", type=int, default=3, help="latents per AR step (Stage1 sampler / Stage2 rollout)")
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--videos-per-rank", type=int, default=1)
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
        name = f"s{args.stage}_{Path(ckpt).stem.split('_step')[-1]}_{args.field.replace('generator', 'g')}"
        if args.stage == 1:
            name += f"_nfe{args.nfe}"
        name += f"_b{args.block}" if args.block != 3 else ""
        if all((name, sid) in done for sid in val):
            continue
        diffusion = load_model(args.stage, ckpt, args.field, device, args.block, args.history)
        inner = diffusion.module
        steps_raw = [float(s) for s in warp_denoising_steps([1000, 750, 500, 250], diffusion.scheduler.timesteps.to(device),
                                                            num_train_timesteps=1000)]
        for fi, sid in enumerate(val):
            if (name, sid) in done:
                continue
            t0 = time.time()
            fork = torch.load(LAT / f"{sid}.pt", map_location="cpu", weights_only=False)
            cam_rel = torch.load(REPO / "data/cf_vizdoom_10k_cam" / f"{sid}.pt", weights_only=True)["nominal"]
            m = json.load(open(SRC / sid / "meta.json"))
            clean = fork["latents"].to(device)
            cond = {"prompt_embeds": fork["prompt"].to(device)[None].expand(4, -1, -1)}
            cam = camera_tokens(cam_rel, 4, device, args.cam_scale)
            acts = fork["actions"].to(device)
            if sid not in gt_cache:
                gt_cache[sid] = gt_features(vae, clean)
            with torch.no_grad():
                if args.stage == 1:
                    x = ar_flowmap_sample(diffusion, inner, clean, cond, cam, acts, nfe=args.nfe, seed=fi,
                                          block=args.block, history=args.history)
                else:
                    g = torch.Generator(device=device).manual_seed(fi)
                    x = history_rollout(diffusion, inner, inner, clean, cond, cam, acts, steps_raw,
                                        args.history + args.block, require_grad=False,
                                        exit_index=len(steps_raw) - 1, generator=g, block=args.block).output
                pix = vae.decode(x)
            record(out, rows_path, name, "nom_s%g_act" % args.cam_scale, sid, m, x, pix, clean, gt_cache[sid], fi,
                   args.videos_per_rank, t0)
        del diffusion, inner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
