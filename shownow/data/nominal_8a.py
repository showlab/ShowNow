"""Action-derived ("nominal") camera trajectories for arbitrary 8-action key sequences (W A S D TURN_L TURN_R ATTACK
JUMP, ids 1-8) on the latent grid. 8-action version of nominal_switch.py (that file is unchanged).

The 8-action hold data uses one free-motion curve per key (data/cf_vizdoom_8a_cam/nominal_curves.npy, [8, 96, 4]:
camera-frame displacement dx, dy, dz in metres and yaw in rad relative to the fork, per 32 fps future frame; frame n
shows time (n + 1) / 32 s). For key switches each segment restarts its key's curve at the switch time from the pose
reached so far (so a W after a turn walks along the new view direction). Beyond the curve's 3 s, translation and yaw
are extrapolated with the last velocity (JUMP repeats its curve, a jump cycle), so the same function serves unbounded
interactive key streams.

Latent j of the future (0-based) shows time (4 j + 1) / 16 s after the fork; a segment starting at latent s starts at
s * 0.25 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FPS = 32.0
JUMP = 8
_CURVES = None


def curves() -> np.ndarray:
    global _CURVES
    if _CURVES is None:
        _CURVES = np.load(REPO / "data/cf_vizdoom_8a_cam/nominal_curves.npy").astype(np.float64)  # [8, 96, 4]
    return _CURVES


def yaw_rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def motion(key: int, tau: float) -> np.ndarray:
    """[dx, dy, dz, yaw] of free motion tau seconds after `key` went down (camera frame at key-down)."""
    c = curves()[key - 1]
    n = len(c)
    x = tau * FPS - 1.0  # curve index i shows time (i + 1) / 32 s
    if x <= 0:
        return c[0] * max(0.0, tau * FPS)
    if x >= n - 1:
        if key == JUMP:  # repeated jumps: replay the cycle
            x = x % (n - 1)
        else:
            return c[-1] + (c[-1] - c[-11]) / 10.0 * (x - (n - 1))
    i = int(np.floor(x))
    return c[i] + (c[i + 1] - c[i]) * (x - i)


def apply(base: np.ndarray, m: np.ndarray, unit: float = 1.0) -> np.ndarray:
    """base [4,4] c2w, m = motion(...) -> [4,4] c2w (translation scaled by unit)."""
    out = base.copy()
    out[:3, :3] = base[:3, :3] @ yaw_rot(m[3])
    out[:3, 3] = base[:3, 3] + base[:3, :3] @ (m[:3] * unit)
    return out


class KeyStream8:
    """One future latent at a time; candidates can be peeked before the key is known (as nominal_switch.KeyStream).
    fork_c2w [4,4]: pose of the fork (history end) in any frame; translations in metres x unit_scale."""

    def __init__(self, fork_c2w, unit_scale: float = 1.0):
        self.base = np.asarray(fork_c2w, dtype=np.float64).copy()  # pose at the start of the current segment
        self.unit = float(unit_scale)
        self.j = 0
        self.seg_key, self.seg_t0 = None, 0.0

    def _seg(self, key: int, j: int):
        if self.seg_key is None:
            return self.base, 0.0
        if key == self.seg_key:
            return self.base, self.seg_t0
        t_sw = j * 0.25  # switch at latent j: freeze the previous segment at the switch time
        return apply(self.base, motion(self.seg_key, t_sw - self.seg_t0), self.unit), t_sw

    def peek(self, key: int) -> np.ndarray:
        key = int(key)
        base, t0 = self._seg(key, self.j)
        return apply(base, motion(key, (4 * self.j + 1) / 16.0 - t0), self.unit)

    def commit(self, key: int) -> np.ndarray:
        key = int(key)
        pose = self.peek(key)
        self.base, self.seg_t0 = self._seg(key, self.j)
        self.seg_key = key
        self.j += 1
        return pose


def nominal_future_c2w(fork_c2w: np.ndarray, keys) -> np.ndarray:
    """fork_c2w [4,4] -> [n,4,4] c2w of the future latents for per-latent key ids `keys` (metres)."""
    ks = KeyStream8(fork_c2w)
    return np.stack([ks.commit(int(k)) for k in keys])
