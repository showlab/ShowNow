"""Branch-tree forward for SolarWM/Wan: one sequence = shared history + K action branches.

A layout is a list of latent-frame "slots". Each slot has a lineage (branch id, -1 = shared history), a time index
(position on the video timeline, used for RoPE), a chunk index and a kind (clean/noisy). A frame-level visibility
matrix decides which slots attend to which; branches never see each other, all branches share the RoPE time
positions of a plain single-future sequence. Every token carries its own timestep / AnyFlow r / action id / camera.

With exact visibility and positions, each branch's outputs are identical (up to kernel numerics) to running the
original model on [history + that branch] alone - `tests/test_tree_equivalence.py` checks this - while the history is
embedded, projected and attended once instead of K times.

Layouts:
  bidir_layout   bidirectional teacher: history <-> history; branch k <-> history + branch k
  tf_layout      causal teacher forcing ([clean | noisy] per lineage, SolarWM Stage1 rules incl. prior window)
  window_layout  cache-free AR inference: [shared clean history | per-branch clean generated | K noisy blocks]
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch

FRAME_TOKENS = 405
BLOCK = 3


@dataclass
class Layout:
    branch: torch.Tensor  # [N] long, -1 = shared
    time: torch.Tensor  # [N] long, RoPE frame index
    chunk: torch.Tensor  # [N] long, time // BLOCK (for window rules)
    noisy: torch.Tensor  # [N] bool
    vis: torch.Tensor  # [N, N] bool, q-slot x kv-slot
    out: torch.Tensor  # [N'] long, slots returned by the forward (through the head)
    key: tuple

    @property
    def n(self) -> int:
        return int(self.branch.numel())


def _same_lineage(bq, bk):
    return (bk[None, :] == -1) | (bk[None, :] == bq[:, None])


def bidir_layout(K: int, H: int = 9, F: int = 12) -> Layout:
    branch = torch.tensor([-1] * H + [k for k in range(K) for _ in range(F)])
    time = torch.tensor(list(range(H)) + list(range(H, H + F)) * K)
    hist_q = branch[:, None] == -1
    vis = torch.where(hist_q, (branch[None, :] == -1).expand(len(branch), -1), _same_lineage(branch, branch))
    return Layout(branch, time, time // BLOCK, torch.zeros(len(branch), dtype=torch.bool), vis,
                  torch.arange(len(branch)), ("bidir", K, H, F))


def tf_layout(K: int, H: int = 9, F: int = 12, max_prior: int = 5, block: int = BLOCK) -> Layout:
    """[clean H | clean branch 0..K-1 | noisy H | noisy branch 0..K-1], SolarWM teacher-forcing rules per lineage:
    clean q sees clean kv of its lineage with chunk <= own; noisy q sees clean kv of its lineage with chunk in
    [own - max_prior, own - 1] plus the noisy kv of its own lineage and chunk. Chunks = `block` latents (B3: 3, B1: 1;
    max_prior is counted in chunks, e.g. B1 with a 15-latent history window -> max_prior 15)."""
    b1 = [-1] * H + [k for k in range(K) for _ in range(F)]
    t1 = list(range(H)) + list(range(H, H + F)) * K
    branch = torch.tensor(b1 + b1)
    time = torch.tensor(t1 + t1)
    noisy = torch.tensor([False] * len(b1) + [True] * len(b1))
    chunk = time // block
    lin = _same_lineage(branch, branch)
    cq, ck = chunk[:, None], chunk[None, :]
    nq, nk = noisy[:, None], noisy[None, :]
    clean_rule = ~nq & ~nk & (ck <= cq)
    noisy_prior = nq & ~nk & (ck <= cq - 1) & (ck >= cq - max_prior)
    noisy_own = nq & nk & (ck == cq) & (branch[None, :] == branch[:, None])
    vis = lin & (clean_rule | noisy_prior | noisy_own)
    return Layout(branch, time, chunk, noisy, vis, torch.nonzero(noisy).flatten(), ("tf", K, H, F, max_prior, block))


def window_layout(K: int, hist_frames: int, gen_frames: int, t0: int = 0, block: int = BLOCK) -> Layout:
    """Cache-free AR window: [shared clean history (hist_frames) | branch k clean generated (gen_frames) each |
    K noisy blocks]. Chunk-causal within a lineage (SolarWM inference-window rule); RoPE indices are window-relative
    (0 at the first window frame), t0 = timeline index of the first window frame (only used for chunk alignment)."""
    b = [-1] * hist_frames + [k for k in range(K) for _ in range(gen_frames)] + [k for k in range(K) for _ in range(block)]
    rel = list(range(hist_frames)) + list(range(hist_frames, hist_frames + gen_frames)) * K + \
        list(range(hist_frames + gen_frames, hist_frames + gen_frames + block)) * K
    branch, time = torch.tensor(b), torch.tensor(rel)
    noisy = torch.tensor([False] * (hist_frames + K * gen_frames) + [True] * (K * block))
    chunk = (time + t0) // block
    vis = _same_lineage(branch, branch) & (chunk[None, :] <= chunk[:, None])
    return Layout(branch, time, chunk, noisy, vis, torch.nonzero(noisy).flatten(),
                  ("win", K, hist_frames, gen_frames, t0 % block, block))


# ----------------------------------------------------------------------------- block mask
_MASKS: dict = {}


def block_mask_for(layout: Layout, device) -> tuple:
    from torch.nn.attention.flex_attention import create_block_mask

    key = (layout.key, str(device))
    if key in _MASKS:
        return _MASKS[key]
    n_tok = layout.n * FRAME_TOKENS
    pad = math.ceil(n_tok / 128) * 128 - n_tok
    vis = torch.zeros(layout.n + 1, layout.n + 1, dtype=torch.bool)
    vis[:layout.n, :layout.n] = layout.vis
    vis = vis.to(device)
    n_frames = layout.n

    def mask_mod(b, h, q_idx, kv_idx):
        qf = torch.clamp(q_idx // FRAME_TOKENS, max=n_frames)
        kf = torch.clamp(kv_idx // FRAME_TOKENS, max=n_frames)
        return vis[qf, kf] | (q_idx == kv_idx)

    bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=n_tok + pad, KV_LEN=n_tok + pad, device=device,
                           _compile=True)
    _MASKS[key] = (bm, pad)
    return bm, pad


# ----------------------------------------------------------------------------- forward
def tree_forward_impl(self, x, t, context, seq_len, y_camera=None, r=None, layout: Layout = None,
                      action_tokens=None):
    """x: [B, C, N, H, W] latents of all slots (layout order); t, r: [B, N*405]; y_camera: per-token viewmats/K
    [B, N*405, 4|3, 4|3]; action_tokens: [B, N*405] or None. Returns [B, C_out, N_out, H, W] for layout.out slots."""
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    B = x.shape[0]
    pe = self.patch_embedding(x)  # [B, dim, N, h, w] (temporal patch 1: frame-independent)
    N, h, w = pe.shape[2:]
    s = h * w
    assert s == FRAME_TOKENS and N == layout.n, (s, N, layout.n)
    tok = pe.flatten(2).transpose(1, 2)  # [B, N*s, dim]
    from shownow.train.teacher_action_ft import action_input
    tok = action_input(self, tok, action_tokens)  # token-path action conditioning
    seq_lens = torch.full((B,), N * s, dtype=torch.long)

    self._action_tokens = action_tokens
    e_full = self._time_condition(t, tok, r if self.uses_anyflow else None)
    e0 = self.time_projection(e_full).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
    out_tok_idx = (layout.out.to(device)[:, None] * s + torch.arange(s, device=device)[None]).flatten()
    t_out = t[:, out_tok_idx]
    r_out = r[:, out_tok_idx] if (r is not None and self.uses_anyflow) else None
    self._action_tokens = action_tokens[:, out_tok_idx] if action_tokens is not None else None
    e_head = self._time_condition(t_out, tok, r_out).unflatten(dim=0, sizes=t_out.shape)
    self._action_tokens = None

    ctx = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                                           for u in context]))
    cam_v = cam_K = None
    if y_camera is not None:
        cam_v, cam_K = y_camera["viewmats"], y_camera["K"]
    bm, pad = block_mask_for(layout, device)
    if pad:
        tok = torch.cat([tok, tok.new_zeros(B, pad, tok.shape[2])], 1)
        e0 = torch.cat([e0, e0.new_zeros(B, pad, *e0.shape[2:])], 1)
        if cam_v is not None:
            cam_v = torch.cat([cam_v, torch.eye(4, device=device, dtype=cam_v.dtype).expand(B, pad, 4, 4)], 1)
            cam_K = torch.cat([cam_K, torch.eye(3, device=device, dtype=cam_K.dtype).expand(B, pad, 3, 3)], 1)
    grid = torch.tensor([[N, h, w]] * B, dtype=torch.long)
    kw = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid, freqs=self.freqs, context=ctx, context_lens=None,
              cam_viewmats=cam_v, cam_K=cam_K, cache_update_policy="none", block_mask=bm,
              frame_indices=layout.time.to(device), kv_cache=None, crossattn_cache=None, current_start=0,
              cache_start=0)
    for blk in self.blocks:
        tok, _ = blk(tok, **kw)
    tok = tok[:, :N * s][:, out_tok_idx]
    tok = self.head(tok, e_head.unsqueeze(2))
    return torch.stack(self.unpatchify(tok, torch.tensor([[len(layout.out), h, w]] * B, dtype=torch.long)))


def enable_tree(module) -> None:
    """Route module(..., training_mode='tree', layout=..., action_tokens=...) to the tree forward, so FSDP root
    pre/post-forward hooks stay active (same trick SolarWM uses for teacher forcing)."""
    orig = module.forward

    def forward(self, *args, **kwargs):
        if kwargs.get("training_mode") == "tree":
            kwargs.pop("training_mode")
            return tree_forward_impl(self, *args, **kwargs)
        return orig(*args, **kwargs)

    module.forward = types.MethodType(forward, module)


def tree_call(diffusion_module, x_btchw, t_tok, prompt_embeds, camera, layout, r_tok=None, action_tokens=None):
    """Convenience wrapper with SolarWM's [B,T,C,H,W] latent layout; diffusion_module = root (possibly FSDP)."""
    out = diffusion_module(x_btchw.permute(0, 2, 1, 3, 4), t=t_tok, context=prompt_embeds,
                           seq_len=x_btchw.shape[1] * FRAME_TOKENS, y_camera=camera, r=r_tok,
                           training_mode="tree", layout=layout, action_tokens=action_tokens)
    return out.permute(0, 2, 1, 3, 4)


def full_layout(N: int) -> Layout:
    """All slots see all slots, times 0..N-1 (for checking the plumbing against the native bidirectional forward)."""
    b = torch.full((N,), -1)
    return Layout(b, torch.arange(N), torch.arange(N) // BLOCK, torch.zeros(N, dtype=torch.bool),
                  torch.ones(N, N, dtype=torch.bool), torch.arange(N), ("full", N))
