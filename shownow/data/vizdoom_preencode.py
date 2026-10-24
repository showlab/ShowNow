"""Pre-encode ViZDoom counterfactual forks for action-conditioned teacher fine-tuning (W/A/S/D only).

Per fork -> <out>/<sample_id>.pt with
  latents   [4, 21, 48, 30, 54] bf16   (branch order W, A, S, D; 81 frames @16 fps, 864x480)
  prompt    [512, 4096] bf16           action-neutral caption, T5 (shared by the 4 branches)
  c2w_rot   [21, 4, 4] f32             rotation-only camera per latent, relative to latent 0 (View channel)
  actions   [4, 21] int64              per-latent action id (0 none, 1 W, 2 A, 3 S, 4 D)
  meta      dict                        map, weapon, effect flags, source frame indices, caption

Frame layout (docs/phase2_teacher_finetune_design.md §3): pixel frames 0-32 = history (times (n-33)/16 s before the
fork), 33-80 = future (times (n-32)/16 s after the fork). Latent 9 is the first future latent.
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
sys.path.insert(0, str(REPO / "SolarWM/src"))
BRANCHES = ["W", "A", "S", "D"]
ACTION_ID = {"none": 0, "W": 1, "A": 2, "S": 3, "D": 4}
SRC_FPS, DST_FPS = 35.0, 16.0
N_HIST, N_FUT = 33, 48  # pixel frames
H, W = 480, 864


def history_src_index(n: int, n_src: int = 105) -> int:
    t = (n - N_HIST) / DST_FPS  # seconds before the fork (negative)
    # source history frame i shows time (i - n_src + 1) / SRC_FPS (last frame = fork state)
    return int(np.clip(round(t * SRC_FPS + n_src - 1), 0, n_src - 1))


def future_src_index(n: int, n_src: int = 105) -> int:
    t = (n - N_HIST + 1) / DST_FPS  # seconds after the fork
    return int(np.clip(round(t * SRC_FPS) - 1, 0, n_src - 1))


def read_video(path: Path) -> np.ndarray:
    import imageio.v2 as iio2

    r = iio2.get_reader(str(path))
    fr = np.stack([f for f in r])
    r.close()
    return fr


def to_model_frames(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T,720,1280,3] -> [3,T,480,864] in [-1,1] (resize to 864x486, centre crop)."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(x, size=(486, 864), mode="bilinear", antialias=True, align_corners=False)
    x = x[:, :, 3:483, :]
    return (x / 127.5 - 1.0).permute(1, 0, 2, 3)


def dominant_key(btn_rows: list) -> int:
    """history button rows [MF, MB, ML, MR, ...] -> dominant single movement key (diagonals/none -> 0)."""
    votes = {1: 0, 2: 0, 3: 0, 4: 0}
    for b in btn_rows:
        f, bk, l, r = (b[0] > 0), (b[1] > 0), (b[2] > 0), (b[3] > 0)
        keys = [k for k, on in ((1, f and not bk), (3, bk and not f), (2, l and not r), (4, r and not l)) if on]
        if len(keys) == 1:
            votes[keys[0]] += 1
    best = max(votes, key=votes.get)
    return best if votes[best] >= 2 else 0


def rot_only_relative(c2w_list: list) -> np.ndarray:
    R = np.array(c2w_list, dtype=np.float64)[:, :3, :3]
    rel = np.einsum("ij,njk->nik", R[0].T, R)  # R0^-1 R_n
    out = np.tile(np.eye(4), (len(R), 1, 1))
    out[:, :3, :3] = rel
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data/cf_vizdoom_10k"))
    ap.add_argument("--out", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    forks = sorted({json.loads(l)["sample_id"] for l in open(src / "train_manifest.jsonl")})
    mine = forks[args.shard::args.num_shards]

    from solarwm.backends.wan22.runtime.components import Wan5BVAE, WanTextEncoder

    base = REPO / "models/SolarWM-5B-base"
    dev = torch.device("cuda")
    vae = Wan5BVAE(base / "vae/Wan2.2_VAE.pth").to(dev, dtype=torch.bfloat16)
    t5 = WanTextEncoder(base / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth", base / "tokenizer").to(dev, dtype=torch.bfloat16)
    lat_pix = np.array([0, *(1 + 4 * k for k in range(20))])
    t0 = time.time()
    done = 0
    for sid in mine:
        dst = out / f"{sid}.pt"
        if dst.exists():
            continue
        m = json.load(open(src / sid / "meta.json"))
        hist = read_video(src / sid / "history.mp4")
        h_idx = [history_src_index(n, len(hist)) for n in range(N_HIST)]
        f_idx = [future_src_index(n) for n in range(N_HIST, N_HIST + N_FUT)]
        pix_all, acts = [], []
        for k in BRANCHES:
            fut = read_video(src / sid / f"future_{k}.mp4")
            frames = np.concatenate([hist[h_idx], fut[f_idx]])
            pix_all.append(to_model_frames(frames))
            # per-latent action ids: history latents from the explorer's buttons, future latents = branch action
            ids = []
            for li in range(21):
                if li < 9:
                    pf = [0] if li == 0 else list(range(1 + 4 * (li - 1), 1 + 4 * li))
                    rows = [m["history"]["actions"][h_idx[p]] for p in pf]
                    ids.append(dominant_key(rows))
                else:
                    ids.append(ACTION_ID[k])
            acts.append(ids)
        with torch.no_grad():
            pix = torch.stack(pix_all).to(dev, dtype=torch.bfloat16)  # [4,3,81,480,864]
            lat = vae.encode(pix).to(torch.bfloat16).cpu()  # [4,21,48,30,54]
            caption = (f"First-person Doom-style shooter in a {m['environment']['map']} level, "
                       f"the player holds a {m['weapon']['name']}.")
            prompt = t5([caption])["prompt_embeds"][0].to(torch.bfloat16).cpu()
        # camera: rotation-only per latent (history c2w for latents 0-8, branch W's c2w for the future:
        # the view direction is fixed during futures, identical across branches)
        c2w_frames = [m["history"]["c2w"][h_idx[n]] for n in range(N_HIST)] + \
                     [m["futures"]["W"]["c2w"][f_idx[n - N_HIST]] for n in range(N_HIST, N_HIST + N_FUT)]
        c2w_rot = rot_only_relative([c2w_frames[p] for p in lat_pix])
        torch.save({"latents": lat, "prompt": prompt, "c2w_rot": torch.from_numpy(c2w_rot),
                    "actions": torch.tensor(acts, dtype=torch.long),
                    "meta": {"sample_id": sid, "map": m["environment"]["map"], "weapon": m["weapon"]["name"],
                             "caption": caption, "history_src_idx": h_idx, "future_src_idx": f_idx,
                             "effect_observed": {k: m["futures"][k]["effect"]["effect_observed"] for k in BRANCHES}}},
                   dst)
        done += 1
        if done % 20 == 0:
            print(f"shard {args.shard}: {done}/{len(mine)} forks, {(time.time() - t0) / done:.2f} s/fork", flush=True)
    print(f"shard {args.shard} finished {done} forks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
