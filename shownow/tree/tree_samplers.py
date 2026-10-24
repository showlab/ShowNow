"""Branch-tree samplers: all K futures of a fork generated in ONE sequence (shared history computed once).

- teacher_tree_sample: bidirectional teacher, UniPC over the K future slots, history fixed.
- student_tree_sample: causal AR student, cache-free sliding window per block, window_layout
  [shared clean history in window | per-branch clean generated blocks in window | K noisy blocks];
  solver "flowmap" (Stage1 AnyFlow, (t,r) pairs) or "fewstep" (Stage2 SGF: x0 -> re-noise schedule).
Inputs follow the fork layout: clean [K,21,C,H,W] (history taken from branch 0), acts [K,21], camera per branch.
"""

from __future__ import annotations

import torch

from shownow.train.student_common import flowmap_schedule, tokens
from shownow.tree.tree import BLOCK, FRAME_TOKENS, bidir_layout, tree_call, window_layout

H = 9


def _cat_cam(cam, parts):
    """cam: per-branch dict [K, 21*405, ...]; parts: list of (branch, lo, hi) latent ranges -> [1, n*405, ...]."""
    return {n: torch.cat([cam[n][k:k + 1, lo * FRAME_TOKENS:hi * FRAME_TOKENS] for k, lo, hi in parts], 1)
            for n in ("viewmats", "K")}


def _cat_act(acts, parts):
    return torch.cat([acts[k:k + 1, lo:hi] for k, lo, hi in parts], 1)


@torch.no_grad()
def teacher_tree_sample(diffusion, root, clean, prompt, cam, acts, steps=30, seed=0):
    from solarwm.backends.wan22.runtime.scheduler import build_wan_flow_unipc_scheduler

    K, T = clean.shape[:2]
    F = T - H
    layout = bidir_layout(K, H, F)
    parts = [(0, 0, H)] + [(k, H, T) for k in range(K)]
    c = _cat_cam(cam, parts)
    a = tokens(_cat_act(acts, parts).float()).long()
    g = torch.Generator().manual_seed(seed)
    n = torch.randn((F, *clean.shape[2:]), generator=g).to(clean.device, clean.dtype)
    x = torch.cat([clean[0:1, :H]] + [n[None]] * K, 1)
    sched = build_wan_flow_unipc_scheduler(num_train_timesteps=1000, shift=5.0, num_inference_steps=steps,
                                           device=clean.device)
    for t in sched.timesteps:
        ts = torch.zeros(1, x.shape[1], device=clean.device)
        ts[:, H:] = float(t.item())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flow = tree_call(root, x, tokens(ts), prompt, c, layout, action_tokens=a)
        flow[:, :H] = 0
        x = sched.step(flow.to(x.dtype), t, x, return_dict=False)[0].to(clean.dtype)
        x[:, :H] = clean[0:1, :H]
    out = torch.cat([torch.cat([clean[0:1, :H], x[:, H + F * k:H + F * (k + 1)]], 1) for k in range(K)])
    return out  # [K, 21, ...]


@torch.no_grad()
def student_tree_sample(diffusion, root, clean, prompt, cam, acts, solver="flowmap", nfe=4, fewstep=None,
                        max_prior_chunks=5, seed=0, x0_of=None, block=BLOCK):
    """Returns [K,21,...] with generated futures. fewstep: list of raw timesteps for the SGF sampler;
    x0_of(noisy, flow, t_frames) converts flow to x0 (diffusion.flow_to_x0)."""
    K, T = clean.shape[:2]
    dev, dt = clean.device, clean.dtype
    branch_gen = [[] for _ in range(K)]  # generated blocks per branch
    g = torch.Generator(device=dev).manual_seed(seed)
    for blk in range(H // block, T // block):
        start = blk * block
        ws = max(0, blk - max_prior_chunks) * block  # window start (timeline)
        hist_lo, hist_hi = ws, min(H, start)  # shared history frames in window
        gen_lo = max(ws, H)  # generated frames in window: [gen_lo, start)
        gen_frames = start - gen_lo
        hist_frames = max(0, hist_hi - hist_lo)
        layout = window_layout(K, hist_frames, gen_frames, t0=ws, block=block)
        parts = ([(0, hist_lo, hist_hi)] if hist_frames else []) + \
            ([(k, gen_lo, start) for k in range(K)] if gen_frames else []) + [(k, start, start + block) for k in range(K)]
        c = _cat_cam(cam, parts)
        a = tokens(_cat_act(acts, parts).float()).long()
        ctx = ([clean[0:1, hist_lo:hist_hi]] if hist_frames else []) + \
            ([torch.cat(branch_gen[k], 1)[:, gen_lo - H:] for k in range(K)] if gen_frames else [])
        n_ctx = hist_frames + K * gen_frames
        x = torch.randn((1, K * block, *clean.shape[2:]), generator=g, device=dev, dtype=dt)
        if solver == "flowmap":
            ts, rs = flowmap_schedule(nfe, device=dev)
            pairs = list(zip(ts.tolist(), rs.tolist()))
        else:
            pairs = [(float(t), None) for t in fewstep]
        for i, (t, r) in enumerate(pairs):
            tt = torch.zeros(1, n_ctx + K * block, device=dev)
            tt[:, n_ctx:] = t
            rr = None
            if r is not None:
                rr = torch.zeros_like(tt)
                rr[:, n_ctx:] = r
            inp = torch.cat(ctx + [x], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = tree_call(root, inp, tokens(tt), prompt, c, layout, r_tok=None if rr is None else tokens(rr),
                                action_tokens=a)  # outputs = noisy slots [1, K*3, ...]
            if solver == "flowmap":
                x = (x.float() - (t - r) / 1000.0 * out.float()).to(dt)
            else:
                tf = torch.full((1, K * block), t, device=dev)
                x0 = x0_of(x, out, tf)
                if i < len(pairs) - 1:
                    tn = torch.full((K * block,), pairs[i + 1][0], device=dev)
                    x = diffusion.scheduler.add_noise(x0[0].float(), torch.randn_like(x0[0]).float(),
                                                      tn)[None].to(dt)
                else:
                    x = x0.to(dt)
        for k in range(K):
            branch_gen[k].append(x[:, k * block:(k + 1) * block])
    return torch.cat([torch.cat([clean[0:1, :H]] + branch_gen[k], 1) for k in range(K)])


@torch.no_grad()
def sgf_tree_rollout(diffusion, root, clean, prompt, cam, acts, fewstep, exit_index, max_prior_chunks=5,
                     generator=None, block=BLOCK):
    """SGF pass 1 in the tree layout (cache-free window). For every future block: run the few-step schedule for all
    K branches at once, record the input at the exit step and the exit x0 (committed as context, as SolarWM's
    self-gradient-forcing does). Returns (committed [K,21,...], noisy_at_exit [K,21,...]); history = clean."""
    K, T = clean.shape[:2]
    dev, dt = clean.device, clean.dtype
    committed = clean.clone()
    noisy_exit = clean.clone()
    for blk in range(H // block, T // block):
        start = blk * block
        ws = max(0, blk - max_prior_chunks) * block
        hist_lo, hist_hi = ws, min(H, start)
        gen_lo = max(ws, H)
        gen_frames, hist_frames = start - gen_lo, max(0, hist_hi - hist_lo)
        layout = window_layout(K, hist_frames, gen_frames, t0=ws, block=block)
        parts = ([(0, hist_lo, hist_hi)] if hist_frames else []) + \
            ([(k, gen_lo, start) for k in range(K)] if gen_frames else []) + [(k, start, start + block) for k in range(K)]
        c = _cat_cam(cam, parts)
        a = tokens(_cat_act(acts, parts).float()).long()
        ctx = ([clean[0:1, hist_lo:hist_hi]] if hist_frames else []) + \
            ([committed[k:k + 1, gen_lo:start] for k in range(K)] if gen_frames else [])
        n_ctx = hist_frames + K * gen_frames
        x = torch.randn((1, K * block, *clean.shape[2:]), generator=generator, device=dev, dtype=dt)
        x0_exit = None
        for i, t in enumerate(fewstep):
            if i == exit_index:
                for k in range(K):
                    noisy_exit[k:k + 1, start:start + block] = x[:, k * block:(k + 1) * block]
            tt = torch.zeros(1, n_ctx + K * block, device=dev)
            tt[:, n_ctx:] = float(t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = tree_call(root, torch.cat(ctx + [x], 1), tokens(tt), prompt, c, layout, action_tokens=a)
            tf = torch.full((1, K * block), float(t), device=dev)
            x0 = diffusion.flow_to_x0(x, out, tf)
            if i == exit_index:
                x0_exit = x0
            if i < len(fewstep) - 1:
                tn = torch.full((K * block,), float(fewstep[i + 1]), device=dev)
                x = diffusion.scheduler.add_noise(x0[0].float(), torch.randn_like(x0[0]).float(), tn)[None].to(dt)
        for k in range(K):
            committed[k:k + 1, start:start + block] = x0_exit[:, k * block:(k + 1) * block]
    return committed, noisy_exit


def pack_tree(x, H=9):
    """[K,21,...] per-branch -> [1, H + K*(21-H), ...] tree slots (history from branch 0)."""
    K = x.shape[0]
    return torch.cat([x[0:1, :H]] + [x[k:k + 1, H:] for k in range(K)], 1)


def unpack_tree(s, K, H=9):
    F = (s.shape[1] - H) // K
    return torch.cat([torch.cat([s[:, :H], s[:, H + F * k:H + F * (k + 1)]], 1) for k in range(K)])
