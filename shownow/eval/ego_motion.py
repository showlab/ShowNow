"""Ego-motion direction from video: which of W/A/S/D does the camera motion look like?

Dense Farneback flow over the first ~1 s of the future (before most wall hits), HUD/weapon region masked, fitted per frame
pair with a translation + isotropic expansion model  u = tx + d (x - cx),  v = ty + d (y - cy)  (the view direction is
fixed during futures, so no rotation term). Forward motion -> expansion d > 0, backward -> d < 0, strafe left -> scene
moves right (tx > 0), strafe right -> tx < 0.
"""

from __future__ import annotations

import numpy as np

W_SMALL, H_SMALL = 432, 240
DIV_RADIUS = 0.5 * W_SMALL  # px: converts expansion rate to an edge displacement comparable to tx


def ego_motion(frames: np.ndarray, step: int = 2) -> dict:
    """uint8 [T,H,W,3] (T ~ 17 frames at 16 fps) -> summed tx, ty, d over the clip and a W/A/S/D prediction."""
    import cv2

    g = [cv2.cvtColor(cv2.resize(f, (W_SMALL, H_SMALL), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
         for f in frames]
    ys, xs = np.mgrid[0:H_SMALL, 0:W_SMALL].astype(np.float32)
    cx, cy = (W_SMALL - 1) / 2, (H_SMALL - 1) / 2
    keep = (ys < 0.72 * H_SMALL)  # drop weapon sprite / status bar
    keep &= ~((ys > 0.55 * H_SMALL) & (np.abs(xs - cx) < 0.18 * W_SMALL))
    X = np.stack([np.ones(keep.sum()), (xs - cx)[keep]], 1)  # u = tx + d x
    Y = np.stack([np.ones(keep.sum()), (ys - cy)[keep]], 1)  # v = ty + d y
    A = np.block([[X[:, :1], np.zeros_like(X[:, :1]), X[:, 1:]], [np.zeros_like(Y[:, :1]), Y[:, :1], Y[:, 1:]]])
    tx = ty = d = 0.0
    for i in range(0, len(g) - step, step):
        fl = cv2.calcOpticalFlowFarneback(g[i], g[i + step], None, 0.5, 4, 21, 3, 5, 1.2, 0)
        b = np.concatenate([fl[..., 0][keep], fl[..., 1][keep]])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        tx, ty, d = tx + sol[0], ty + sol[1], d + sol[2]
    fwd, side = d * DIV_RADIUS, tx
    pred = ("W" if fwd > 0 else "S") if abs(fwd) >= abs(side) else ("A" if side > 0 else "D")
    return {"tx": float(tx), "ty": float(ty), "d": float(d), "mag": float(max(abs(fwd), abs(side))), "pred": pred}
