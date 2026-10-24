"""Action-derived ("nominal") camera trajectories for arbitrary W/A/S/D key sequences on the latent grid.

The hold data used one free-motion displacement curve per key (median of unblocked training branches,
data/cf_vizdoom_10k_cam/nominal_curves.npy: metres along the key's axis vs. source frame after the fork, 35 fps).
For key switches each segment restarts its key's curve at the switch time from the position reached so far
(view direction fixed). Beyond the curve's 3 s the last velocity is extrapolated, so the same function also serves
unbounded interactive key streams at inference.

Latent j of the future (0-based) shows time (4 j + 1) / 16 s after the fork; a key segment that starts at latent s
starts at time s * 0.25 s (the generator switches on that grid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AXIS = {1: (2, 1.0), 3: (2, -1.0), 2: (0, -1.0), 4: (0, 1.0)}  # key id -> (OpenCV camera axis, sign); W=1 A=2 S=3 D=4
SRC_FPS = 35.0
_CURVES = None


def curves() -> np.ndarray:
    global _CURVES
    if _CURVES is None:
        _CURVES = np.load(REPO / "data/cf_vizdoom_10k_cam/nominal_curves.npy").astype(np.float64)  # [4 (W A S D), 105]
    return _CURVES


def displacement(key: int, tau: float) -> float:
    """Metres travelled along the key's axis tau seconds after the key went down (free motion)."""
    c = curves()[key - 1]
    x = tau * SRC_FPS - 1.0  # curve index i shows time (i + 1) / 35 s
    if x <= 0:
        return float(c[0] * max(0.0, tau * SRC_FPS))
    if x >= len(c) - 1:
        slope = c[-1] - c[-11]
        return float(c[-1] + slope / 10.0 * (x - (len(c) - 1)))
    i = int(np.floor(x))
    return float(c[i] + (c[i + 1] - c[i]) * (x - i))


def future_offsets(keys: list) -> np.ndarray:
    """keys: per future latent key id (1-4) -> [n, 3] camera-frame offsets (metres) of each latent from the fork."""
    keys = [int(k) for k in keys]
    out = np.zeros((len(keys), 3))
    base = np.zeros(3)  # position at the start of the current segment
    seg_key, seg_t0 = keys[0], 0.0
    for j, k in enumerate(keys):
        if k != seg_key:  # switch at latent j: freeze the previous segment at the switch time
            t_sw = j * 0.25
            ax, sg = AXIS[seg_key]
            base = base.copy()
            base[ax] += sg * displacement(seg_key, t_sw - seg_t0)
            seg_key, seg_t0 = k, t_sw
        t = (4 * j + 1) / 16.0
        ax, sg = AXIS[k]
        p = base.copy()
        p[ax] += sg * displacement(k, t - seg_t0)
        out[j] = p
    return out


def nominal_future_c2w(fork_c2w: np.ndarray, keys: list) -> np.ndarray:
    """fork_c2w [4,4] absolute (metres, OpenCV) -> [n,4,4] absolute c2w of the future latents (view direction fixed)."""
    off = future_offsets(keys)
    out = np.repeat(fork_c2w[None].astype(np.float64), len(keys), 0)
    out[:, :3, 3] = fork_c2w[:3, 3] + off @ fork_c2w[:3, :3].T
    return out


class KeyStream:
    """Interactive version of future_offsets: one latent at a time, candidates can be peeked before the key is known.

    fork_c2w: [4,4] pose (any frame, e.g. first-latent-relative) of the fork; latents after it follow the key stream
    with the view direction of the fork fixed. Positions are in the same units as fork_c2w (scale applied by caller).
    """

    def __init__(self, fork_c2w, unit_scale: float = 1.0):
        self.fork = np.asarray(fork_c2w, dtype=np.float64)
        self.unit = float(unit_scale)  # metres -> units of fork_c2w (1.0 = metres)
        self.j = 0  # next future latent index
        self.base = np.zeros(3)
        self.seg_key, self.seg_t0 = None, 0.0

    def _offset(self, key: int, j: int):
        base, t0 = self.base, self.seg_t0
        if key != self.seg_key and self.seg_key is not None:
            ax, sg = AXIS[self.seg_key]
            base = base.copy()
            base[ax] += sg * displacement(self.seg_key, j * 0.25 - self.seg_t0)
            t0 = j * 0.25
        elif self.seg_key is None:
            t0 = 0.0
        ax, sg = AXIS[key]
        p = base.copy()
        p[ax] += sg * displacement(key, (4 * j + 1) / 16.0 - t0)
        return p, base, t0

    def peek(self, key: int) -> np.ndarray:
        """[4,4] pose of the next latent if `key` is held for it."""
        p, _, _ = self._offset(int(key), self.j)
        out = self.fork.copy()
        out[:3, 3] = self.fork[:3, 3] + self.fork[:3, :3] @ (p * self.unit)
        return out

    def commit(self, key: int) -> np.ndarray:
        key = int(key)
        pose = self.peek(key)
        _, self.base, self.seg_t0 = self._offset(key, self.j)
        self.seg_key = key
        self.j += 1
        return pose
