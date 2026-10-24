"""Long-horizon drift eval for AR students: from the real 2 s history of held-out forks, roll out --seconds of video under
random W/A/S/D key scripts (4 scripts per fork, seeded by fork) and measure how far the video drifts from ViZDoom.

No GT exists beyond 3 s, so drift is measured against distributions:
- latent FD per time window: Frechet distance between per-latent features (48 channel means + 48 channel stds) of the
  generated latents in that window (all forks x scripts) and real future latents of all 48 val forks (`real_ref.npy`).
- pixel stats per 1 s window relative to the fork's own real history frames: sharpness (Laplacian var), contrast (gray
  std), saturation, brightness; and motion (mean |frame diff|) relative to the real futures (a frozen video has no drift
  but motion ratio -> 0).
Camera: nominal free-motion trajectory, per-key per-latent velocity = mean nominal future velocity of that fork's branch.
Stage1 uses the AnyFlow sliding-window sampler, Stage2 the KV-cache few-step rollout.

Launch: sbatch slurm/long_drift_eval.sbatch --stage 2 --block 1 --ckpts a.pt,b.pt --out runs/long_drift/<name>
Summary: python -m shownow.eval.long_drift_summary runs/long_drift/<name>
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
from shownow.eval.camera_prior_eval import LAT, N_HIST, to_uint8  # noqa: E402
from shownow.eval.student_eval import load_model  # noqa: E402
from shownow.train.student_common import HIST_LATENTS, ar_flowmap_sample, history_rollout  # noqa: E402
from shownow.train.teacher_action_ft import FRAME_TOKENS, FX_NORM, FY_NORM  # noqa: E402

# key id -> name; SHOWNOW_N_ACTIONS=9 (8-action models) adds the turn / fire / jump keys
KEYS = dict(enumerate(["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
                      [:int(__import__("os").environ.get("SHOWNOW_N_ACTIONS", "5")) - 1], 1))
LAT_PER_SEC = 4  # 1 latent = 4 frames @ 16 fps


def key_script(n: int, seed: int) -> np.ndarray:
    """n latents of W/A/S/D (1-4): segments of 0.5-2 s, W 40 %, A/S/D 20 % each."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        out += [int(rng.choice([1, 2, 3, 4], p=[0.4, 0.2, 0.2, 0.2]))] * int(rng.integers(2, 9))
    return np.array(out[:n])


def long_camera(nominal: torch.Tensor, keys: np.ndarray) -> torch.Tensor:
    """nominal [4,21,4,4] (relative c2w of the W/A/S/D branches) + keys [N] -> [9+N,4,4]."""
    vel = {k: (nominal[k - 1, 20, :3, 3] - nominal[k - 1, HIST_LATENTS - 1, :3, 3]) / (21 - HIST_LATENTS)
           for k in KEYS}
    out = [nominal[0, i] for i in range(HIST_LATENTS)]
    pose = nominal[0, 20].clone()  # fork view direction, fixed during the future
    pose[:3, 3] = nominal[0, HIST_LATENTS - 1, :3, 3]
    for k in keys:
        pose = pose.clone()
        pose[:3, 3] += vel[int(k)]
        out.append(pose)
    return torch.stack(out)


def camera_tokens_long(c2w: torch.Tensor, device, scale: float) -> dict:
    """[B,F,4,4] relative c2w -> PRoPE viewmats / K per token (teacher_action_ft.camera_tokens for any F)."""
    c2w = c2w.float().clone()
    c2w[..., :3, 3] *= scale
    w2c = torch.linalg.inv(c2w)
    B, F = c2w.shape[:2]
    view = w2c[:, :, None].expand(B, F, FRAME_TOKENS, 4, 4).reshape(B, -1, 4, 4)
    K = torch.zeros(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2] = FX_NORM, FY_NORM, 0.5, 0.5, 1.0
    return {"viewmats": view.contiguous().to(device), "K": K.expand(B, view.shape[1], 3, 3).contiguous().to(device)}


def lat_features(x: torch.Tensor) -> np.ndarray:
    """[..., 48, H, W] -> [..., 96] channel means + stds."""
    x = x.float()
    return torch.cat([x.mean((-2, -1)), x.std((-2, -1))], -1).cpu().numpy()


def frame_stats(frames: np.ndarray) -> dict:
    """uint8 [T,H,W,3] -> per-frame stats (downscaled 2x)."""
    import cv2
    out = {"sharp": [], "contrast": [], "sat": [], "bright": []}
    for f in frames:
        f = cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        out["sharp"].append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        out["contrast"].append(float(g.std()))
        out["bright"].append(float(g.mean()))
        out["sat"].append(float(cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 1].mean()))
    fl = frames.astype(np.float32)
    out["motion"] = [0.0] + list(np.abs(fl[1:] - fl[:-1]).mean(axis=(1, 2, 3)).astype(float))
    return {k: np.array(v) for k, v in out.items()}


def window_means(v: np.ndarray, per: int) -> list:
    n = len(v) // per
    return [float(v[i * per:(i + 1) * per].mean()) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--field", default="generator_ema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--forks", type=int, default=24)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--videos-per-rank", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "feats").mkdir(exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    n_fut = args.seconds * LAT_PER_SEC
    F = HIST_LATENTS + n_fut

    split = json.load(open(REPO / "runs/teacher_ft/wasd_v1/split.json"))
    val48 = sorted(split["val_ood"])[:48]
    if args.rank == 0 and not (out / "real_ref.npy").exists():  # real future-latent features of all 48 val forks
        ref = [lat_features(torch.load(LAT / f"{s}.pt", map_location="cpu", weights_only=False)["latents"]
                            [:, HIST_LATENTS:]).reshape(-1, 96) for s in val48]
        np.save(out / "real_ref.npy", np.concatenate(ref))
    val = val48[: args.forks][args.rank::args.world]

    from solarwm.backends.wan22.runtime.components import Wan5BVAE
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    vae = Wan5BVAE(REPO / "models/SolarWM-5B-base/vae/Wan2.2_VAE.pth").to(device, dtype=torch.bfloat16)
    rows_path = out / f"rows_rank{args.rank}.jsonl"
    done = {(r["model"], r["sample_id"]) for r in map(json.loads, open(rows_path))} if rows_path.exists() else set()
    ref_stats = {}

    for ckpt in args.ckpts.split(","):
        name = f"s{args.stage}_{Path(ckpt).stem.split('_step')[-1]}_{args.field.replace('generator', 'g')}_b{args.block}"
        if all((name, sid) in done for sid in val):
            continue
        diffusion = load_model(args.stage, ckpt, args.field, device, args.block, args.history)
        inner = diffusion.module
        steps_raw = [float(s) for s in warp_denoising_steps([1000, 750, 500, 250],
                                                            diffusion.scheduler.timesteps.to(device),
                                                            num_train_timesteps=1000)]
        for fi, sid in enumerate(val):
            if (name, sid) in done:
                continue
            t0 = time.time()
            fork = torch.load(LAT / f"{sid}.pt", map_location="cpu", weights_only=False)
            nominal = torch.load(REPO / "data/cf_vizdoom_10k_cam" / f"{sid}.pt", weights_only=True)["nominal"]
            seed = int(sid.split("_")[1])
            keys = np.stack([key_script(n_fut, seed * 4 + b) for b in range(4)])  # [4, N]
            acts = torch.cat([fork["actions"][:, :HIST_LATENTS], torch.from_numpy(keys)], 1).to(device)
            cam = camera_tokens_long(torch.stack([long_camera(nominal, keys[b]) for b in range(4)]), device,
                                     args.cam_scale)
            clean = torch.zeros((4, F, *fork["latents"].shape[2:]), dtype=torch.bfloat16, device=device)
            clean[:, :HIST_LATENTS] = fork["latents"][:, :HIST_LATENTS].to(device)
            cond = {"prompt_embeds": fork["prompt"].to(device)[None].expand(4, -1, -1)}
            with torch.no_grad():
                if args.stage == 1:
                    x = ar_flowmap_sample(diffusion, inner, clean, cond, cam, acts, nfe=args.nfe, seed=fi,
                                          block=args.block, history=args.history)
                else:
                    g = torch.Generator(device=device).manual_seed(fi)
                    x = history_rollout(diffusion, inner, inner, clean, cond, cam, acts, steps_raw,
                                        args.history + args.block, require_grad=False,
                                        exit_index=len(steps_raw) - 1, generator=g, block=args.block).output
                t_gen = time.time() - t0
                frames = [to_uint8(vae.decode(x[b:b + 1])[0]) for b in range(4)]  # [1+4(F-1),H,W,3] each
            if sid not in ref_stats:  # real history frames + real futures of this fork
                with torch.no_grad():
                    real = to_uint8(vae.decode(fork["latents"][:1].to(device))[0])
                    real_f = [to_uint8(vae.decode(fork["latents"][b:b + 1].to(device))[0])[N_HIST:] for b in range(4)]
                hs = frame_stats(real[:N_HIST])
                ref_stats[sid] = {k: float(v[1:].mean()) for k, v in hs.items()}
                ref_stats[sid]["motion_future"] = float(np.mean([frame_stats(f)["motion"][1:].mean() for f in real_f]))
            np.save(out / "feats" / f"{name}__{sid}.npy", lat_features(x[:, HIST_LATENTS:]).astype(np.float32))
            per_sec = 16
            row = {"model": name, "sample_id": sid, "keys": keys.tolist(), "ref": ref_stats[sid],
                   "sec_gen": round(t_gen, 1), "sec": round(time.time() - t0, 1), "stats": []}
            for b in range(4):
                st = frame_stats(frames[b][N_HIST:])
                row["stats"].append({k: window_means(v, per_sec) for k, v in st.items()})
            with open(rows_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({"model": name, "sample_id": sid, "sec_gen": row["sec_gen"], "sec": row["sec"]}),
                  flush=True)
            if fi < args.videos_per_rank:
                save_video(out / "videos" / f"{sid}_{name}.mp4", frames, keys)
        del diffusion, inner
        torch.cuda.empty_cache()


def save_video(path: Path, frames: list, keys: np.ndarray) -> None:
    """2x2 grid of the 4 scripted rollouts (history included), current key in the corner."""
    import cv2
    import imageio.v2 as iio
    tiles = []
    for b, f in enumerate(frames):
        t = np.stack([cv2.resize(z, (432, 240), interpolation=cv2.INTER_AREA) for z in f])
        for i in range(len(t)):
            li = (i - 1) // 4 + 1 if i > 0 else 0  # latent index of frame i
            lab = "history" if li < HIST_LATENTS else f"{KEYS[int(keys[b][li - HIST_LATENTS])]}  {i / 16:5.1f}s"
            cv2.rectangle(t[i], (0, 0), (130, 22), (0, 0, 0), -1)
            cv2.putText(t[i], lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    grid = np.concatenate([np.concatenate(tiles[:2], 2), np.concatenate(tiles[2:], 2)], 1)
    iio.mimwrite(path, grid, fps=16, quality=7, macro_block_size=8)


if __name__ == "__main__":
    main()
