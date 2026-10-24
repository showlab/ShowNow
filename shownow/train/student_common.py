"""AR student samplers with a clean history prefix (shared by Stage1/Stage2 training and evaluation).

Layout (81 frames @16 fps = 21 latents, 3-latent blocks): blocks 0-2 (latents 0-8) = real 2 s history, given clean;
blocks 3-6 (latents 9-20) = the 3 s future generated autoregressively under one action (W/A/S/D).

- `ar_flowmap_sample`: Stage1 AnyFlow flow-map sampler, cache-free sliding window (SolarWM Stage1 validation path).
- `history_rollout`:   Stage2 SGF student rollout with KV cache (SolarWM `stage2_camera_rollout`): history blocks are
  committed to the cache as clean context, future blocks are denoised with the few-step schedule; optional
  gradient-carrying teacher-forcing replay at a random exit step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

FRAME_TOKENS = 405
BLOCK = 3
HIST_LATENTS = 9


def tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, F] per-latent values -> [B, F*405] per-token."""
    return x[..., None].expand(*x.shape, FRAME_TOKENS).reshape(x.shape[0], -1)


def slice_camera(camera: dict, start: int, end: int) -> dict:
    return {k: v[:, start * FRAME_TOKENS:end * FRAME_TOKENS] for k, v in camera.items()}


def flowmap_schedule(nfe: int, shift: float = 5.0, ntt: int = 1000, device=None):
    base = torch.linspace(1.0, 0.0, nfe + 1, device=device, dtype=torch.float32)
    s = shift * base / (1.0 + (shift - 1.0) * base)
    s[-1] = 0.0
    raw = s * ntt
    return raw[:-1], raw[1:]


@torch.no_grad()
def ar_flowmap_sample(diffusion, inner, clean: torch.Tensor, cond: dict, camera: dict, acts: torch.Tensor,
                      nfe: int = 4, max_prior_chunks: int = 5, seed: int = 0, block: int = BLOCK,
                      history: int | None = None, start: int = HIST_LATENTS) -> torch.Tensor:
    """clean [B,T,C,H,W] (only the history latents [0, start) are used), acts [B,T] -> latents with generated future.
    block = latents generated per AR step; history = max clean latents in the window (default max_prior_chunks*3)."""
    B, F = clean.shape[:2]
    history = max_prior_chunks * BLOCK if history is None else history
    out = clean.clone()
    g = torch.Generator(device=clean.device).manual_seed(seed)
    ts, rs = flowmap_schedule(nfe, device=clean.device)
    act_tok = tokens(acts.float()).long()
    assert (F - start) % block == 0, (F, start, block)
    for s0 in range(start, F, block):
        end = s0 + block
        hs = max(0, s0 - history)
        hist = out[:, hs:s0]
        x = torch.randn((B, block, *clean.shape[2:]), generator=g, device=clean.device, dtype=clean.dtype)
        cam = slice_camera(camera, hs, end)
        inner._action_tokens = act_tok[:, hs * FRAME_TOKENS:end * FRAME_TOKENS]
        for t, r in zip(ts, rs):
            tt = torch.full((B, block), float(t), device=clean.device)
            rt = torch.full((B, block), float(r), device=clean.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                u = diffusion.forward_inference_window(
                    x, hist, cond, cam, tokens(tt), num_frame_per_block=block,
                    sequence_length=block * FRAME_TOKENS, clean_history_timestep=0.0, r_timestep_tokens=tokens(rt))
            x = (x.float() - (float(t) - float(r)) / 1000.0 * u.float()).to(clean.dtype)
        inner._action_tokens = None
        out[:, s0:end] = x
    return out


def allocate_caches(root, batch: int, cache_frames: int, dtype, device):
    heads, dim = int(root.num_heads), int(root.dim)
    hd = dim // heads
    n = cache_frames * FRAME_TOKENS
    kv = [{"k": torch.zeros(batch, n, heads, hd, dtype=dtype, device=device),
           "v": torch.zeros(batch, n, heads, hd, dtype=dtype, device=device),
           "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
           "local_end_index": torch.zeros(1, dtype=torch.long, device=device)} for _ in root.blocks]
    kv[0]["_fused_prope_camera_metadata"] = {
        "viewmats": torch.zeros(batch, n, 4, 4, dtype=torch.float32, device=device),
        "K": torch.zeros(batch, n, 3, 3, dtype=torch.float32, device=device)}
    xa = [{"k": torch.zeros(batch, 512, heads, hd, dtype=dtype, device=device),
           "v": torch.zeros(batch, 512, heads, hd, dtype=dtype, device=device), "is_init": False}
          for _ in root.blocks]
    return kv, xa


@dataclass
class Rollout:
    output: torch.Tensor  # [B,21,...] x0 (history = clean), grad-carrying if requested
    committed: torch.Tensor  # [B,21,...] per-block exit x0 (no grad)
    loss_mask: torch.Tensor  # [B,21,1,1,1] future latents only
    exit_index: int


def history_rollout(student, inner, root, clean: torch.Tensor, cond: dict, camera: dict, acts: torch.Tensor,
                    steps: list, cache_frames: int, *, require_grad: bool, exit_index: int | None = None,
                    generator: torch.Generator | None = None, block: int = BLOCK) -> Rollout:
    """SGF rollout with the history prefix. steps: warped denoising timesteps (e.g. 1000/750/500/250 -> raw).
    block = latents per AR step (cache_frames = history window + block). Any length F with (F - 9) % block == 0
    (long rollouts: pass zeros for the future part of `clean`, camera/acts for all F latents)."""
    B, F = clean.shape[:2]
    dev, dt = clean.device, clean.dtype
    if exit_index is None:
        exit_index = int(torch.randint(0, len(steps), (1,), device=dev).item())
    t_exit = float(steps[exit_index])
    act_tok = tokens(acts.float()).long()
    kv, xa = allocate_caches(root, B, cache_frames, dt, dev)
    committed = clean.clone()
    noisy_at_exit = clean.clone()

    def call(x, t_frames, start, policy):
        inner._action_tokens = act_tok[:, start * FRAME_TOKENS:(start + block) * FRAME_TOKENS]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return student(x, cond, slice_camera(camera, start, start + block), tokens(t_frames),
                               sequence_length=block * FRAME_TOKENS, kv_cache=kv, crossattn_cache=xa,
                               current_start=start * FRAME_TOKENS, cache_start=0, cache_update_policy=policy)
        finally:
            inner._action_tokens = None

    assert HIST_LATENTS % block == 0 and (F - HIST_LATENTS) % block == 0, (F, block)
    zeros = torch.zeros(B, block, device=dev)
    with torch.no_grad():
        for start in range(0, F, block):
            if start < HIST_LATENTS:  # clean history -> cache
                call(clean[:, start:start + block], zeros, start, "commit_detached")
                continue
            x = torch.randn((B, block, *clean.shape[2:]), device=dev, dtype=dt, generator=generator)
            x0_exit = None
            for i, t in enumerate(steps):
                tf = torch.full((B, block), float(t), device=dev)
                if i == exit_index:
                    noisy_at_exit[:, start:start + block] = x
                flow = call(x, tf, start, "none")
                x0 = student.flow_to_x0(x, flow, tf)
                if i == exit_index:
                    x0_exit = x0
                if i < len(steps) - 1:
                    tn = torch.full((B * block,), float(steps[i + 1]), device=dev)
                    x = student.scheduler.add_noise(x0.flatten(0, 1).float(), torch.randn_like(x0).flatten(0, 1).float(),
                                                    tn).unflatten(0, (B, block)).to(dt)
            committed[:, start:start + block] = x0_exit
            call(x0_exit, zeros, start, "commit_detached")
    loss_mask = torch.zeros(B, F, 1, 1, 1, dtype=torch.bool, device=dev)
    loss_mask[:, HIST_LATENTS:] = True
    if not require_grad:
        return Rollout(committed, committed, loss_mask, exit_index)
    t2 = torch.zeros(B, F, device=dev)
    t2[:, HIST_LATENTS:] = t_exit
    inner._action_tokens = act_tok
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flow = student.forward_train_tf(noisy_at_exit, committed, cond, camera, tokens(t2),
                                            tokens(torch.zeros(B, F, device=dev)), num_frame_per_block=block,
                                            sequence_length=F * FRAME_TOKENS)
    finally:
        inner._action_tokens = None
    out = student.flow_to_x0(noisy_at_exit, flow, t2)
    out = torch.cat([clean[:, :HIST_LATENTS], out[:, HIST_LATENTS:]], 1)
    return Rollout(out, committed, loss_mask, exit_index)
