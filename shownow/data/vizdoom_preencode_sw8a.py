"""Pre-encode the 8-action key-switch evaluation forks (sim/gen_vizdoom_cf8.py --mode switch --maps MAP29-32,
data/cf_vizdoom_sw8a_32fps) at 16 fps. Evaluation only (held-out maps); same per-fork format as
vizdoom_preencode_8a.py (latents [8,21,48,30,54], prompt, actions [8,21], c2w_rot, meta; <cam>/<sid>.pt "true" and
"nominal" [8,21,4,4] relative c2w, "fork" [4,4] fork pose in the same frame), with:
  * future actions per latent from the generator's switch schedule (keys change on the latent grid);
  * nominal future camera following the key sequence (shownow/data/nominal_8a.py, segments chained from the fork);
  * forks SKIPPED when the player takes any damage / health change in the used history or any future (red flash);
  * meta["blocked"] [8,12] bool: future latent of a W/A/S/D segment, >= 1 latent after the switch, whose real camera
    moves < MIN_MOVE_M along the key axis during the latent (wall / corner); meta["move_free_frac"] = 1 - blocked
    share of those latents (fork selection: shownow/eval/sw8a_split.py keeps forks with few blocked latents).

  python -m shownow.data.vizdoom_preencode_sw8a --shard i --num-shards n
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
from shownow.data.nominal_8a import nominal_future_c2w  # noqa: E402
from shownow.data.vizdoom_preencode import read_video, to_model_frames  # noqa: E402
from shownow.data.vizdoom_preencode_8a import (ACTION_ID, BRANCHES, FUT_IDX, HIST_IDX, dominant_key,  # noqa: E402
                                               latent_c2w, relative)

AXIS = {1: (2, 1.0), 2: (0, -1.0), 3: (2, -1.0), 4: (0, 1.0)}  # key id -> (OpenCV camera axis, sign)
MIN_MOVE_M = 0.1  # free walking ~0.33 m per latent


def damage_free(m: dict) -> bool:
    h = m["history"]["state"][HIST_IDX[0]:]
    for st in [h] + [m["futures"][k]["state"] for k in BRANCHES]:
        d = [s["damage_taken"] for s in st]
        hp = [s["health"] for s in st]
        if max(d) > min(d) or max(hp) > min(hp):
            return False
    return True


def blocked_mask(c2w: np.ndarray, keys: list) -> tuple[np.ndarray, int]:
    """c2w [21,4,4] true latent poses of one branch, keys [21] -> blocked [12] (future latents), n scored."""
    out, n = np.zeros(12, dtype=bool), 0
    for j in range(12):
        li, k = 9 + j, keys[9 + j]
        if k not in AXIS or j == 0 or keys[li - 1] != k:  # not a move key, or the first latent of its segment
            continue
        a, b = (li, li + 1) if li < 20 else (li - 1, li)
        d = c2w[a][:3, :3].T @ (c2w[b][:3, 3] - c2w[a][:3, 3])
        ax, sg = AXIS[k]
        out[j] = sg * d[ax] < MIN_MOVE_M
        n += 1
    return out, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data/cf_vizdoom_sw8a_32fps"))
    ap.add_argument("--out", default=str(REPO / "data/cf_vizdoom_sw8a_lat"))
    ap.add_argument("--cam-out", default=str(REPO / "data/cf_vizdoom_sw8a_cam"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sids", default="", help="json list of sample ids to encode (default: all passing forks)")
    ap.add_argument("--index-name", default="index_shard{:03d}.json", help="map index file name (format: shard)")
    args = ap.parse_args()
    src, out, cam_out = Path(args.src), Path(args.out), Path(args.cam_out)
    out.mkdir(parents=True, exist_ok=True)
    cam_out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for f in sorted(src.glob("index_*.jsonl")) for l in open(f)]
    forks = sorted({r["sample_id"] for r in rows if r.get("pass")})
    if args.sids:  # restrict to a list (e.g. the checked forks still missing)
        want = set(json.load(open(args.sids)))
        forks = [f for f in forks if f in want]
    forks = forks[args.shard::args.num_shards]
    from solarwm.backends.wan22.runtime.components import Wan5BVAE, WanTextEncoder

    base = REPO / "models/SolarWM-5B-base"
    dev = torch.device("cuda")
    vae = Wan5BVAE(base / "vae/Wan2.2_VAE.pth").to(dev, dtype=torch.bfloat16)
    t5 = WanTextEncoder(base / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth", base / "tokenizer").to(dev, dtype=torch.bfloat16)
    t0, done, skipped = time.time(), 0, {"damage": 0}
    for sid in forks:
        dst = out / f"{sid}.pt"
        if dst.exists() and (cam_out / f"{sid}.pt").exists():
            continue
        m = json.load(open(src / sid / "meta.json"))
        if not damage_free(m):
            skipped["damage"] += 1
            continue
        sched = m["action_space"]["switch_schedules"]
        hist = read_video(src / sid / "history.mp4")
        assert len(hist) == 96, (sid, len(hist))
        hrows = m["history"]["actions"]
        h_ids = [dominant_key([hrows[HIST_IDX[p]] for p in ([0] if li == 0 else range(1 + 4 * (li - 1), 1 + 4 * li))])
                 for li in range(9)]
        fork_c2w = np.array(m["history"]["c2w"][-1], dtype=np.float64)
        pix, acts, true, nom, rot, blocked, n_scored = [], [], [], [], [], [], 0
        for k in BRANCHES:
            fut = read_video(src / sid / f"future_{k}.mp4")
            assert len(fut) == 96, (sid, k, len(fut))
            pix.append(to_model_frames(np.concatenate([hist[HIST_IDX], fut[FUT_IDX]])))
            keys = [ACTION_ID[s["key"]] for s in sched[k] for _ in range(s["n_latents"])]
            assert len(keys) == 12, (sid, k, len(keys))
            acts.append(h_ids + keys)
            c2w = latent_c2w(m, k)
            true.append(relative(c2w))
            rot.append(relative(c2w, rot_only=True))
            nc = c2w.copy()
            nc[9:] = nominal_future_c2w(fork_c2w, keys)
            nom.append(relative(nc))
            b, n = blocked_mask(c2w, acts[-1])
            blocked.append(b)
            n_scored += n
        blocked = np.stack(blocked)
        fork_rel = relative(np.stack([c2w[0], fork_c2w]))[1]  # fork pose in the latent-0 frame (KeyStream8 start)
        with torch.no_grad():
            lat = torch.cat([vae.encode(torch.stack(pix[i:i + 4]).to(dev, dtype=torch.bfloat16)).to(torch.bfloat16).cpu()
                             for i in (0, 4)])  # [8,21,48,30,54]
            caption = (f"First-person Doom-style shooter in a {m['environment']['map']} level, "
                       f"the player holds a {m['weapon']['name']}.")
            prompt = t5([caption])["prompt_embeds"][0].to(torch.bfloat16).cpu()
        rec = {"latents": lat, "prompt": prompt, "c2w_rot": torch.from_numpy(np.stack(rot)),
               "actions": torch.tensor(acts, dtype=torch.long),
               "meta": {"sample_id": sid, "map": m["environment"]["map"], "weapon": m["weapon"]["name"],
                        "caption": caption, "branches": BRANCHES, "fps": 16, "src_fps": 32, "switch_schedules": sched,
                        "blocked": blocked.tolist(), "move_free_frac": 1.0 - float(blocked.sum()) / max(n_scored, 1),
                        "n_move_scored": n_scored, "fork_center_depth": m["environment"]["fork_center_depth"]}}
        for path, obj in ((dst, rec), (cam_out / f"{sid}.pt", {"true": torch.from_numpy(np.stack(true)),
                                                               "nominal": torch.from_numpy(np.stack(nom)),
                                                               "fork": torch.from_numpy(fork_rel)})):
            tmp = path.with_suffix(".pt.part")
            torch.save(obj, tmp)
            tmp.rename(path)
        done += 1
        if done % 20 == 0 or done <= 2:
            print(f"shard {args.shard}: {done}, {(time.time() - t0) / done:.2f} s/fork, skipped {skipped}", flush=True)
    maps = {sid: json.load(open(src / sid / "meta.json"))["environment"]["map"] for sid in forks if (out / f"{sid}.pt").exists()}
    (out / args.index_name.format(args.shard)).write_text(json.dumps(maps))  # sid -> map (teacher_action_ft.split_files)
    print(f"shard {args.shard} finished {done} forks in {time.time() - t0:.0f}s, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
