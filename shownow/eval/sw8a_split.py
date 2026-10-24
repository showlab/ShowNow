"""Fixed splits of the 8-action key-switch evaluation forks. Two sets (env SHOWNOW_SW8_DATA):
  data/cf_vizdoom_sw8i   (default) the TRAINING maps MAP01-28: new states and key schedules (generator seed 1)
  data/cf_vizdoom_sw8a   the held-out maps MAP29-32 (new-map supplement)

Eligible forks: encoded (damage-free, see vizdoom_preencode_sw8a.py) and not stuck in a corner: of the first 3
latents of every W/A/S/D key segment (all 8 branches; the fork latent itself excluded), at most MAX_BLOCKED move
< 0.1 m along the key axis in the real game. (Later in a segment the player legitimately reaches walls: Doom moves
~0.8 m per latent, so long segments end at a wall in most rooms; E2 excludes those latents with the IDM label mask.)
They are shuffled once with SEED and cut into disjoint sets, fixed before any model is evaluated on them:
  e2_val   48  E2 + checkpoint selection
  scenes   10  E1 (first 3) / E3 / demo videos
  test_e2  96  held-out test E2
  test_e3  20  held-out test E3
The split is created once, after all forks are encoded (python -m shownow.eval.sw8a_split), written to
<set>_lat/split.json and only read from there afterwards.

  python -m shownow.eval.sw8a_split        # create (once) / print
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
# default: the training-map set once its split exists, the held-out-map set before that
ROOT = os.environ.get("SHOWNOW_SW8_DATA") or ("data/cf_vizdoom_sw8i" if (REPO / "data/cf_vizdoom_sw8i_lat/split.json").exists()
                                              else "data/cf_vizdoom_sw8a")
LAT, CAM = REPO / f"{ROOT}_lat", REPO / f"{ROOT}_cam"
SEED = 20261018
MAX_BLOCKED = 0.15
AXIS = {1: (2, 1.0), 2: (0, -1.0), 3: (2, -1.0), 4: (0, 1.0)}  # key id -> (OpenCV camera axis, sign)
SIZES = (("e2_val", 48), ("scenes", 10), ("test_e2", 96), ("test_e3", 20))


def early_blocked(actions: torch.Tensor, cam_true: torch.Tensor) -> float:
    """share of the first 3 latents of W/A/S/D segments (future latents 10-20) that move < 0.1 m along the key."""
    cam = cam_true.double()
    n = bl = 0
    for b in range(actions.shape[0]):
        aft = 0
        for li in range(9, 21):
            k = int(actions[b, li])
            aft = aft + 1 if li > 9 and k == int(actions[b, li - 1]) else 0
            if k not in AXIS or aft > 2 or li == 9:
                continue
            a, c = (li, li + 1) if li < 20 else (li - 1, li)
            d = cam[b, a, :3, :3].T @ (cam[b, c, :3, 3] - cam[b, a, :3, 3])
            ax, sg = AXIS[k]
            n += 1
            bl += int(sg * float(d[ax]) < 0.1)
    return bl / max(n, 1)


def split(create: bool = False) -> dict:
    path = LAT / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    assert create, f"{path} missing: create it once after encoding (python -m shownow.eval.sw8a_split)"
    files = sorted(f for f in LAT.glob("*.pt") if f.name != "split.json")
    ok, stats = [], []
    for f in files:
        acts = torch.load(f, map_location="cpu", weights_only=False)["actions"]
        eb = early_blocked(acts, torch.load(CAM / f.name, weights_only=True)["true"])
        stats.append(eb)
        if eb <= MAX_BLOCKED:
            ok.append(str(f))
    need = sum(n for _, n in SIZES)
    assert len(ok) >= need, f"only {len(ok)} eligible forks of {len(files)}, need {need}"
    random.Random(SEED).shuffle(ok)
    out, i = {}, 0
    for name, n in SIZES:
        out[name] = ok[i:i + n]
        i += n
    out["info"] = {"encoded": len(files), "eligible": len(ok), "max_blocked": MAX_BLOCKED, "seed": SEED,
                   "early_blocked_median": sorted(stats)[len(stats) // 2]}
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    s = split(create=True)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in s.items()}))
