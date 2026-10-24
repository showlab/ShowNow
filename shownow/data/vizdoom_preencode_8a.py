"""Pre-encode the 8-action ViZDoom forks (sim/gen_vizdoom_cf8.py, data/cf_vizdoom_8a_32fps) for training at 16 fps.

Branches W A S D TURN_L TURN_R ATTACK JUMP (action ids 1-8, 0 = none). The 32 fps videos are subsampled by exactly 2
(no resampling): history frame n (t = (n-32)/16 s, n = 0..32) = 32 fps history frame 2n+31; future frame n
(t = (n+1)/16 s, n = 0..47) = 32 fps future frame 2n+1. Same 81-frame / 21-latent layout as the 4-action data
(shownow/data/vizdoom_preencode.py, docs/phase2_teacher_finetune_design.md §3).

Outputs (new directories, the 4-action data is untouched):
  <out>/<sid>.pt      latents [8,21,48,30,54] bf16, prompt [512,4096] bf16, actions [8,21] int64,
                      c2w_rot [8,21,4,4] f32 (rotation-only camera per latent, relative to latent 0, per branch:
                      the TURN branches rotate), meta
  <cam>/<sid>.pt      {"true": [8,21,4,4], "nominal": [8,21,4,4]} relative c2w (metres, OpenCV): history = real
                      explorer path; future = the branch's real path ("true") or the action's nominal free motion
                      ("nominal": median per-frame displacement + yaw over training-map forks, see `curves`)
  <out>/index_shard<i>.json   {sid: map} (teacher_action_ft.split_files reads it -> map-disjoint split)

  python -m shownow.data.vizdoom_preencode_8a curves                  # once: nominal curves from training maps
  python -m shownow.data.vizdoom_preencode_8a encode --shard i --num-shards n --forks <json list>
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
sys.path.insert(0, str(REPO))
from shownow.data.vizdoom_preencode import read_video, to_model_frames  # noqa: E402

BRANCHES = ["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
ACTION_ID = {"none": 0, **{k: i + 1 for i, k in enumerate(BRANCHES)}}
N_HIST, N_FUT = 33, 48
HIST_IDX = [2 * n + 31 for n in range(N_HIST)]  # into the 96-frame 32 fps history
FUT_IDX = [2 * n + 1 for n in range(N_FUT)]  # into the 96-frame 32 fps future
LAT_PIX = np.array([0, *(1 + 4 * k for k in range(20))])
VAL_MAPS = {"MAP29", "MAP30", "MAP31", "MAP32"}
SRC = REPO / "data/cf_vizdoom_8a_32fps"


def dominant_key(rows: list) -> int:
    """history button rows [MF, MB, ML, MR, JUMP, CROUCH, ATTACK, USE, SPEED, TURN_DELTA] -> one action id, else 0.
    Movement keys as in the 4-action data; a turn-only row counts as TURN_L/TURN_R (|delta| > 1 deg/tic)."""
    votes = {}
    for b in rows:
        f, bk, l, r = (b[0] > 0), (b[1] > 0), (b[2] > 0), (b[3] > 0)
        keys = [k for k, on in ((1, f and not bk), (3, bk and not f), (2, l and not r), (4, r and not l)) if on]
        if not keys and abs(b[9]) > 1.0:
            keys = [6 if b[9] > 0 else 5]
        if len(keys) == 1:
            votes[keys[0]] = votes.get(keys[0], 0) + 1
    if not votes:
        return 0
    best = max(votes, key=votes.get)
    return best if votes[best] >= 2 else 0


def rel_pose(fork: np.ndarray, c: np.ndarray):
    """c2w c relative to the fork camera: displacement (fork camera frame) and yaw change (rad, about camera -y)."""
    d = fork[:3, :3].T @ (c[:3, 3] - fork[:3, 3])
    R = fork[:3, :3].T @ c[:3, :3]
    yaw = np.arctan2(R[0, 2], R[2, 2])  # rotation of the forward axis in the camera x-z plane
    return d, yaw


def yaw_rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def curves(out: Path, n: int = 800) -> None:
    """Nominal free-motion curves per branch: per future frame (32 fps, 96) median displacement [3] and yaw, over
    training-map forks whose branch was effective (and, for W/A/S/D, never stuck)."""
    rng = np.random.default_rng(0)
    clean = json.load(open(REPO / "runs/action_scaling/doom8_clean.json"))
    ids = [r["sid"] for r in clean if r.get("keep") and r["map"] not in VAL_MAPS]
    ids = rng.choice(ids, size=min(n, len(ids)), replace=False)
    acc = {k: [] for k in BRANCHES}
    for sid in ids:
        m = json.load(open(SRC / sid / "meta.json"))
        fork = np.array(m["history"]["c2w"][-1])
        for k in BRANCHES:
            f = m["futures"][k]
            if not f["effect"]["effect_observed"] or (k in "WASD" and f["effect"]["stuck_s"] > 0):
                continue
            rows = np.array([[*d, y] for d, y in (rel_pose(fork, np.array(c)) for c in f["c2w"])])
            rows[:, 3] = np.unwrap(rows[:, 3])  # TURN branches reach +-180 deg: no wrap before the median
            acc[k].append(rows)
    cur = {k: np.median(np.stack(v), 0) for k, v in acc.items()}
    np.save(out / "nominal_curves.npy", np.stack([cur[k] for k in BRANCHES]))  # [8, 96, 4] (dx, dy, dz, dyaw)
    (out / "nominal_curves.json").write_text(json.dumps({"branches": BRANCHES, "n_forks": {k: len(v) for k, v in acc.items()},
                                                        "end": {k: cur[k][-1].round(3).tolist() for k in BRANCHES},
                                                        "mid": {k: cur[k][47].round(3).tolist() for k in BRANCHES}}))
    print({k: (len(v), cur[k][47].round(2).tolist(), cur[k][-1].round(2).tolist()) for k, v in acc.items()}, flush=True)


def latent_c2w(m: dict, k: str) -> np.ndarray:
    frames = [m["history"]["c2w"][i] for i in HIST_IDX] + [m["futures"][k]["c2w"][i] for i in FUT_IDX]
    return np.array(frames, dtype=np.float64)[LAT_PIX]  # [21,4,4]


def nominal_c2w(m: dict, k: str, cur: np.ndarray) -> np.ndarray:
    out = latent_c2w(m, k).copy()
    fork = np.array(m["history"]["c2w"][-1], dtype=np.float64)
    for li in range(9, 21):
        fi = FUT_IDX[LAT_PIX[li] - N_HIST]
        dx, dy, dz, yaw = cur[fi]
        out[li] = fork.copy()
        out[li][:3, :3] = fork[:3, :3] @ yaw_rot(yaw)
        out[li][:3, 3] = fork[:3, 3] + fork[:3, :3] @ np.array([dx, dy, dz])
    return out


def relative(c2w: np.ndarray, rot_only: bool = False) -> np.ndarray:
    rel = np.einsum("ij,njk->nik", np.linalg.inv(c2w[0]), c2w)
    if rot_only:
        rel[:, :3, 3] = 0.0
    return rel.astype(np.float32)


def encode(args) -> None:
    out, cam_out = Path(args.out), Path(args.cam_out)
    out.mkdir(parents=True, exist_ok=True)
    cam_out.mkdir(parents=True, exist_ok=True)
    forks = json.load(open(args.forks))
    mine = forks[args.shard::args.num_shards]
    cur = np.load(cam_out / "nominal_curves.npy")
    from solarwm.backends.wan22.runtime.components import Wan5BVAE, WanTextEncoder

    base = REPO / "models/SolarWM-5B-base"
    dev = torch.device("cuda")
    vae = Wan5BVAE(base / "vae/Wan2.2_VAE.pth").to(dev, dtype=torch.bfloat16)
    t5 = WanTextEncoder(base / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth", base / "tokenizer").to(dev, dtype=torch.bfloat16)
    maps, t0, done = {}, time.time(), 0
    for sid in mine:
        dst = out / f"{sid}.pt"
        m = json.load(open(SRC / sid / "meta.json"))
        maps[sid] = m["environment"]["map"]
        if dst.exists() and (cam_out / f"{sid}.pt").exists():
            continue
        hist = read_video(SRC / sid / "history.mp4")
        assert len(hist) == 96, (sid, len(hist))
        pix, acts = [], []
        hrows = m["history"]["actions"]  # per 32 fps history frame
        for k in BRANCHES:
            fut = read_video(SRC / sid / f"future_{k}.mp4")
            assert len(fut) == 96, (sid, k, len(fut))
            pix.append(to_model_frames(np.concatenate([hist[HIST_IDX], fut[FUT_IDX]])))
            ids = []
            for li in range(21):
                if li < 9:  # pixel frames covered by the latent (16 fps index) -> their 32 fps action rows
                    pf = [0] if li == 0 else list(range(1 + 4 * (li - 1), 1 + 4 * li))
                    ids.append(dominant_key([hrows[HIST_IDX[p]] for p in pf]))
                else:
                    ids.append(ACTION_ID[k])
            acts.append(ids)
        with torch.no_grad():
            lat = torch.cat([vae.encode(torch.stack(pix[i:i + 4]).to(dev, dtype=torch.bfloat16)).to(torch.bfloat16).cpu()
                             for i in (0, 4)])  # [8,21,48,30,54]
            caption = (f"First-person Doom-style shooter in a {m['environment']['map']} level, "
                       f"the player holds a {m['weapon']['name']}.")
            prompt = t5([caption])["prompt_embeds"][0].to(torch.bfloat16).cpu()
        true = np.stack([relative(latent_c2w(m, k)) for k in BRANCHES])
        nom = np.stack([relative(nominal_c2w(m, k, cur[i])) for i, k in enumerate(BRANCHES)])
        rot = np.stack([relative(latent_c2w(m, k), rot_only=True) for k in BRANCHES])
        hist_diff = float((lat[:, :9].float() - lat[0:1, :9].float()).abs().max())
        rec = {"latents": lat, "prompt": prompt, "c2w_rot": torch.from_numpy(rot),
               "actions": torch.tensor(acts, dtype=torch.long),
               "meta": {"sample_id": sid, "map": m["environment"]["map"], "weapon": m["weapon"]["name"],
                        "caption": caption, "branches": BRANCHES, "fps": 16, "src_fps": 32,
                        "history_src_idx": HIST_IDX, "future_src_idx": FUT_IDX, "hist_latent_max_abs_diff": hist_diff,
                        "effect_observed": {k: m["futures"][k]["effect"]["effect_observed"] for k in BRANCHES}}}
        for path, obj in ((dst, rec), (cam_out / f"{sid}.pt", {"true": torch.from_numpy(true), "nominal": torch.from_numpy(nom)})):
            tmp = path.with_suffix(".pt.part")
            torch.save(obj, tmp)
            tmp.rename(path)
        done += 1
        if done % 20 == 0 or done <= 2:
            print(f"shard {args.shard}: {done}/{len(mine)}, {(time.time() - t0) / done:.2f} s/fork, hist diff {hist_diff:.4f}",
                  flush=True)
    (out / f"index_shard{args.shard:03d}.json").write_text(json.dumps(maps))
    print(f"shard {args.shard} finished {done} forks in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["curves", "encode"])
    ap.add_argument("--out", default=str(REPO / "data/cf_vizdoom_8a_lat"))
    ap.add_argument("--cam-out", default=str(REPO / "data/cf_vizdoom_8a_cam"))
    ap.add_argument("--forks", default=str(REPO / "data/cf_vizdoom_8a_lat_forks.json"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "curves":
        Path(args.cam_out).mkdir(parents=True, exist_ok=True)
        curves(Path(args.cam_out))
    else:
        encode(args)


if __name__ == "__main__":
    main()
