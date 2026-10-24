"""v3 training data + conditioning: roaming / key-switch Doom forks mixed 1:1 with a SolarWM-Data subset,
and a camera condition that is sometimes the real trajectory, sometimes the action-derived one and sometimes static.

Data
  Doom   data/cf_vizdoom_sw_lat (Freedoom 2) + data/cf_vizdoom_fd1_lat (Freedoom 1): K = 4 counterfactual branches,
         no monsters / no firing / damage-free (no red flashes), keys switch every 0.25-1.5 s; cameras in <root>_cam
         ("true" = engine trajectory, "nominal" = from the key sequence, shownow/data/nominal_switch.py).
  SolarWM data/swm_sub/lat (shownow/data/swm_subset.py): K = 1 real / game videos with their real camera and pseudo
         W/A/S/D labels from the camera motion; translation already in SolarWM units (cam_scale 1.0).
  MixSampler alternates the two pools 1:1 (the smaller pool is cycled).

Camera condition per sample (future latents 9-20; the history always keeps its real trajectory)
  true     real trajectory: condition == content, exactly SolarWM's camera-control contract
  nominal  what inference can provide (action-derived; Doom only, == true for SolarWM samples)
  static   future frozen at the last history pose: the action embedding has to carry the motion alone
An extra p_uncond drops action + camera motion together (= the CFG unconditional input used by the students).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
H = 9
DOOM_ROOTS = ["data/cf_vizdoom_sw_lat", "data/cf_vizdoom_fd1_lat"]
SWM_ROOT = "data/swm_sub/lat"


def cam_dir_for(lat_root: Path) -> Path:
    if lat_root.name == "lat":  # data/swm_sub/lat -> data/swm_sub/cam
        return lat_root.with_name("cam")
    return lat_root.with_name(lat_root.name[:-4] + "_cam")  # <name>_lat -> <name>_cam


class V3Dataset(torch.utils.data.Dataset):
    def __init__(self, files: list):
        self.files = [Path(f) for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        fork = torch.load(f, map_location="cpu", weights_only=False)
        cam = torch.load(cam_dir_for(f.parent) / f.name, weights_only=True)
        fork["cam_true"], fork["cam_nom"] = cam["true"].float(), cam["nominal"].float()
        fork["is_swm"] = "cam_scale" in fork
        return fork


def v3_split(doom_roots=DOOM_ROOTS, swm_root=SWM_ROOT, seed: int = 0):
    """-> (train_doom, train_swm, val_doom, val_swm) file lists. Doom: map-disjoint MAP29-32 / E4M7-9 validation
    (split_files on the per-root map index); SolarWM: 2 % hashed hold-out."""
    from shownow.train.teacher_action_ft import VAL_MAPS, split_files

    import os

    VAL_MAPS.update({"E4M7", "E4M8", "E4M9"})  # Freedoom 1 hold-out maps
    if os.environ.get("SHOWNOW_V3_DOOM_ROOTS"):  # smoke tests: comma-separated lat roots
        doom_roots = os.environ["SHOWNOW_V3_DOOM_ROOTS"].split(",")
    td, vd = [], []
    for r in doom_roots:
        root = REPO / r if not Path(r).is_absolute() else Path(r)
        if not root.exists():
            continue
        tr, vo, vi = split_files(root, seed)
        td += tr + vi
        vd += vo
    if os.environ.get("SHOWNOW_V3_SWM_ROOT"):  # "none" = no SolarWM-Data subset (Doom roots only)
        swm_root = os.environ["SHOWNOW_V3_SWM_ROOT"]
    sw = sorted((REPO / swm_root).glob("*.pt")) if swm_root != "none" and (REPO / swm_root).exists() else []
    vs = [f for f in sw if zlib.crc32(f.name.encode()) % 50 == 0]
    ts = [f for f in sw if zlib.crc32(f.name.encode()) % 50 != 0]
    return td, ts, vd, vs


TEST_SEED = 20261012


def test_forks(val_sorted, n, offset=0):
    """Held-out TEST forks (fixed after all model / checkpoint / schedule choices): validation forks never
    used for any selection or tuning -- not the first 48 (E2 / checkpoint selection) and not the val[::7] scenes
    (E1 / E3 / demo videos) -- drawn with a fixed seed. offset: start in the drawn order (E2 forks 0..95, E3 scenes
    96..115), so the sets do not overlap."""
    import random

    pool = [f for i, f in enumerate(val_sorted) if i >= 48 and i % 7 != 0]
    order = random.Random(TEST_SEED).sample(pool, len(pool))
    return order[offset:offset + n]


def split_name() -> str:
    """SHOWNOW_SPLIT=test selects the held-out test forks / scenes in the evaluations (default: validation)."""
    import os

    return os.environ.get("SHOWNOW_SPLIT", "val")


class MixSampler(torch.utils.data.Sampler):
    """Indices into V3Dataset(doom + swm): alternating doom / swm (1:1), per-epoch shuffle, rank-sharded."""

    def __init__(self, n_doom: int, n_swm: int, rank: int, world: int, seed: int = 0):
        self.nd, self.ns, self.rank, self.world, self.seed, self.epoch = n_doom, n_swm, rank, world, seed, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        n = 2 * max(self.nd, self.ns) if self.ns else self.nd
        return n // self.world

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        pd = torch.randperm(self.nd, generator=g).tolist()
        if not self.ns:
            seq = pd
        else:
            ps = torch.randperm(self.ns, generator=g).tolist()
            n = max(self.nd, self.ns)
            flip = torch.randint(0, 2, (n,), generator=g).tolist()  # random order inside each pair: ranks see both pools
            seq = [x for i in range(n) for x in ((pd[i % self.nd], self.nd + ps[i % self.ns]) if not flip[i]
                                                  else (self.nd + ps[i % self.ns], pd[i % self.nd]))]
        seq = seq[: len(seq) // self.world * self.world]
        return iter(seq[self.rank::self.world])


def static_future(cam_rel: torch.Tensor) -> torch.Tensor:
    """[K,21,4,4] relative c2w -> future latents frozen at the last history pose."""
    out = cam_rel.clone()
    out[:, H:] = cam_rel[:, H - 1:H]
    return out


def pick_condition(fork: dict, gen: torch.Generator, p_true: float, p_static: float, p_uncond: float):
    """Sets fork["cam_rel"] (and zeroes future actions for the unconditional case). Returns the mode name."""
    r = torch.rand(1, generator=gen).item()
    if r < p_uncond:
        fork["cam_rel"] = static_future(fork["cam_true"])
        fork["actions"] = fork["actions"].clone()
        fork["actions"][:, H:] = 0
        return "uncond"
    r = torch.rand(1, generator=gen).item()
    if r < p_static:
        fork["cam_rel"] = static_future(fork["cam_true"])
        return "static"
    if fork["is_swm"] or r < p_static + p_true:
        fork["cam_rel"] = fork["cam_true"]
        return "true"
    # nominal, but only for branches the engine did not block: a branch whose real path is shorter than half the
    # action-derived path (walked into a wall) keeps its real trajectory (no "camera moves, picture does not" pairs)
    fork["cam_rel"] = torch.where(unblocked(fork)[:, None, None, None], fork["cam_nom"], fork["cam_true"])
    return "nominal"


def path_len(cam_rel: torch.Tensor) -> torch.Tensor:
    """[K,21,4,4] -> [K] travelled distance over the future latents (from the last history latent)."""
    p = cam_rel[:, H - 1:, :3, 3]
    return (p[:, 1:] - p[:, :-1]).norm(dim=-1).sum(1)


def unblocked(fork: dict, ratio: float = 0.5) -> torch.Tensor:
    return path_len(fork["cam_true"]) >= ratio * path_len(fork["cam_nom"])
