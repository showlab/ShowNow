"""Graph-capturable fast path for real-time speculative AR inference (steady state, re-rooting every block).

State: a ring buffer holding the last `prefix` committed latents (shared by all candidates), stored per layer as
FINAL attention keys/values - RoPE with absolute timeline positions and PRoPE key transform applied once at commit
(RoPE/PRoPE are relative, so this equals the window-relative formulation). One step forward for the K candidate
blocks (K*3 latents in one sequence):
    Q/K/V projections, RoPE, PRoPE for the current tokens only;
    attention = cascade: [all queries vs shared prefix]  merged by log-sum-exp with  [each branch vs its own block]
    (share=False: every branch gets its own copy of the prefix, i.e. what batching does - for comparison);
    cross-attention with cached text K/V; FFN.
Everything has static shapes and no host syncs -> the whole 4-step block + commit is captured in one CUDA graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fnn

from shownow.train.teacher_action_ft import action_input

FRAME_TOKENS = 405
BLOCK = 3


def _graph_safe_sinusoidal(dim, position):
    """Same as SolarWM's sinusoidal_embedding_1d, but builds the frequency vector on the position's device
    (the original creates it on the CPU and copies, which CUDA graph capture does not allow)."""
    half = dim // 2
    position = position.type(torch.float64)
    freq = torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64).div(half))
    sinusoid = torch.outer(position, freq)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def flash_attn_func_any():
    """flash_attn_func with FA2's calling convention: FA3 (Hopper, `flash_attn_interface`) when it is installed and
    SHOWNOW_FA3 != 0, else FA2; SHOWNOW_FA3=math: exact fp32 attention (ablation only, slow). return_attn_probs=True -> (out, softmax_lse [B, H, L], None) for both."""
    import os
    if os.environ.get("SHOWNOW_FA3", "1") == "math":  # exact fp32 reference attention (ablation of the FA kernels)
        def fa_math(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                    deterministic=False, return_attn_probs=False):
            assert dropout_p == 0.0 and not causal and tuple(window_size) == (-1, -1)
            scale = softmax_scale or q.shape[-1] ** -0.5
            out, lse = [], []
            for h in range(q.shape[2]):  # per head: [B, Lq, Lk] fp32 scores
                s_ = torch.einsum("bqd,bkd->bqk", q[:, :, h].float(), k[:, :, h].float()) * scale
                l_ = torch.logsumexp(s_, -1)
                out.append(torch.einsum("bqk,bkd->bqd", torch.exp(s_ - l_[..., None]), v[:, :, h].float()))
                lse.append(l_)
            o = torch.stack(out, 2).to(q.dtype)
            return (o, torch.stack(lse, 1), None) if return_attn_probs else o
        fa_math.backend = "math"
        return fa_math
    if os.environ.get("SHOWNOW_FA3", "1") == "1":
        try:
            from flash_attn_interface import flash_attn_func as fa3
        except ModuleNotFoundError:
            fa3 = None
        if fa3 is not None:
            def fa(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                   deterministic=False, return_attn_probs=False):
                assert dropout_p == 0.0
                r = fa3(q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=window_size,
                        deterministic=deterministic, return_attn_probs=return_attn_probs)
                return (r[0], r[1], None) if return_attn_probs else r
            fa.backend = "fa3"
            return fa
    from flash_attn import flash_attn_func
    return flash_attn_func


def _dense_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                           causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """SolarWM's flash_attention without padding (q_lens = k_lens = None) as a plain dense flash_attn_func call.
    The original builds q_lens/k_lens on the CPU and copies them to the GPU on every call: inside a CUDA graph that
    copy keeps pointing at a freed pageable host buffer -> random segfaults / illegal memory access on replay."""
    flash_attn_func = flash_attn_func_any()

    if q_lens is not None or k_lens is not None:
        return _ORIG_FLASH_ATTENTION(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                                     softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
                                     window_size=window_size, deterministic=deterministic, dtype=dtype, version=version)
    out_dtype = q.dtype
    half = (torch.float16, torch.bfloat16)
    q, k, v = [x if x.dtype in half else x.to(dtype) for x in (q, k, v)]
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                           window_size=window_size, deterministic=deterministic).type(out_dtype)


_ORIG_FLASH_ATTENTION = None
_FUSED = {}


def _tiled_projmat_math(feats, matrix):
    """Same as camera_prope._apply_tiled_projmat (per-ray branch): out[..., i] = sum_j M[t, i, j] x[..., j] for every
    4-dim group, written as a broadcast multiply-sum so inductor emits ONE fused kernel instead of tiny batched GEMMs."""
    b, n, t, f = feats.shape
    D = matrix.shape[-1]
    x = feats.view(b, n, t, f // D, 1, D).float()
    m = matrix.to(feats.dtype).float()[:, None, :, None, :, :]  # [b, 1, t, 1, D, D] (original casts M to feats dtype)
    return (x * m).sum(-1).reshape(b, n, t, f).to(feats.dtype)


def _fused_tiled_projmat(feats, matrix):
    from solarwm.backends.wan22.runtime.modeling import camera_prope
    if matrix.shape[1] != feats.shape[2]:  # per-camera branch: not used by FastTree, keep the original
        return _FUSED["orig_projmat"](feats, matrix)
    return _FUSED["projmat"](feats, matrix)


def prope_mats(viewmats, Ks):
    """(P^T, P^-1, P) per token [1, L, 4, 4] fp32, exactly as camera_prope._prepare_apply_fns_all_dim (with Ks)."""
    from solarwm.backends.wan22.runtime.camera import transform_relative_viewmats
    from solarwm.backends.wan22.runtime.modeling import camera_prope as cp
    vm = transform_relative_viewmats(viewmats, "linear")
    Kn = torch.zeros_like(Ks)
    Kn[..., 0, 0], Kn[..., 1, 1], Kn[..., 2, 2] = Ks[..., 0, 0], Ks[..., 1, 1], 1.0
    P = torch.einsum("...ij,...jk->...ik", cp._lift_K(Kn), vm)
    P_inv = torch.einsum("...ij,...jk->...ik", cp._invert_SE3(vm), cp._lift_K(cp._invert_K(Kn))).to(vm.dtype)
    return P.transpose(-1, -2).to(vm.dtype), P_inv, P


class FP8Linear(torch.nn.Module):
    """e4m3 weights (per-output-channel scale) x e4m3 activations (per-token dynamic scale), bf16 out, torch._scaled_mm."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        ws = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
        self.register_buffer("w8", (w / ws).to(torch.float8_e4m3fn).contiguous())
        self.register_buffer("ws", ws.t().contiguous())  # [1, out]
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().to(torch.bfloat16),
                                                                     requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xs = (x2.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp(min=1e-12)
        xq = (x2.float() / xs).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.w8.t(), scale_a=xs, scale_b=self.ws, bias=self.bias,
                             out_dtype=torch.bfloat16, use_fast_accum=True)
        return y.reshape(*shp[:-1], self.out_features)


def fp8_quantize_blocks(model):
    """Replace the per-token Linear layers of every transformer block with FP8Linear (cross-attention k/v act on the
    cached text context once per video and stay bf16)."""
    n = 0
    for blk in model.blocks:
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, torch.nn.Linear) and name not in ("cross_attn.k", "cross_attn.v"):
                parent = blk.get_submodule(name.rsplit(".", 1)[0]) if "." in name else blk
                setattr(parent, name.rsplit(".", 1)[-1], FP8Linear(mod).to(mod.weight.device))
                n += 1
    return n


def commit_view(ft, K: int = 1):
    """FastTree with K candidates that SHARES ft's prefix ring / prepared keys (commit only the chosen block)."""
    c = FastTree(ft.m, K, prefix_latents=ft.P, share=ft.share, h=ft.h, w=ft.w, device=ft.dev, block=ft.blk)
    for name in ("rk", "rv", "pk", "pv", "pcam_v", "pcam_K", "ppos"):
        setattr(c, name, getattr(ft, name))
    return c


def _rope_math(x, cos, sin):
    """Real-valued RoPE on interleaved pairs, == view_as_complex(x) * (cos + i sin) (fp32 math, cast back)."""
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).type_as(x)


def make_graph_safe():
    global _ORIG_FLASH_ATTENTION
    from solarwm.backends.wan22.runtime.modeling import causal_model
    from solarwm.backends.wan22.runtime.modeling import model as wan_model
    causal_model.sinusoidal_embedding_1d = _graph_safe_sinusoidal
    if _ORIG_FLASH_ATTENTION is None:  # cross-attention (text) path used by FastTree.forward
        _ORIG_FLASH_ATTENTION = wan_model.flash_attention
        wan_model.flash_attention = _dense_flash_attention
    import os
    if os.environ.get("SHOWNOW_FUSED", "1") == "1" and "projmat" not in _FUSED:  # fused PRoPE / RoPE kernels
        from solarwm.backends.wan22.runtime.modeling import camera_prope
        _FUSED["orig_projmat"] = camera_prope._apply_tiled_projmat
        _FUSED["projmat"] = torch.compile(_tiled_projmat_math, dynamic=False)
        _FUSED["rope"] = torch.compile(_rope_math, dynamic=False)
        camera_prope._apply_tiled_projmat = _fused_tiled_projmat


def _lse_merge(o1, l1, o2, l2):
    """o: [B, L, H, D], l: [B, H, L] (natural-log sum-exp) -> merged attention output."""
    l1t, l2t = l1.transpose(1, 2)[..., None], l2.transpose(1, 2)[..., None]
    mx = torch.maximum(l1t, l2t)
    w1, w2 = torch.exp(l1t - mx), torch.exp(l2t - mx)
    return (o1.float() * w1 + o2.float() * w2) / (w1 + w2)


class FastTree:
    def __init__(self, model, K: int, prefix_latents: int = 15, share: bool = True, h: int = 15, w: int = 27,
                 device="cuda", block: int = BLOCK):
        make_graph_safe()
        self.fa = flash_attn_func_any()
        self.m, self.K, self.P, self.share = model, K, prefix_latents, share
        self.blk = int(block)  # latents generated per step (SolarWM: 3; 1 = per-latent AR)
        self.dev = device
        self.nh = model.num_heads
        self.hd = model.dim // model.num_heads
        self.h, self.w = h, w
        self.Lq = K * self.blk * FRAME_TOKENS
        self.Lp = prefix_latents * FRAME_TOKENS
        L = len(model.blocks)
        bf = torch.bfloat16
        # prefix ring buffers: RAW keys (post qk-norm, pre-RoPE) / values (pre-PRoPE) + cameras + timeline index;
        # final keys pk/pv are rebuilt once per block for the current window positions (prepare_prefix) because
        # PRoPE does not commute with RoPE, so positions must stay window-relative exactly as SolarWM uses them.
        self.rk = torch.zeros(L, 1, self.Lp, self.nh, self.hd, dtype=bf, device=device)
        self.rv = torch.zeros_like(self.rk)
        self.pk = torch.zeros_like(self.rk)
        self.pv = torch.zeros_like(self.rk)
        self.pcam_v = torch.eye(4, device=device).expand(1, self.Lp, 4, 4).contiguous()
        self.pcam_K = torch.eye(3, device=device).expand(1, self.Lp, 3, 3).contiguous()
        self.ppos = torch.zeros(prefix_latents, dtype=torch.long, device=device)  # window-relative position/slot
        self.ccam_v = torch.eye(4, device=device).expand(1, self.Lq, 4, 4).contiguous()  # candidate cams
        self.ccam_K = torch.eye(3, device=device).expand(1, self.Lq, 3, 3).contiguous()
        self.ring = 0  # next block slot (python-side; slot index is baked per captured graph -> use copy offsets)
        # per-branch committed block (candidates), final keys / values: [L, K, 1215, H, D]
        self.ck = torch.zeros(L, K, self.blk * FRAME_TOKENS, self.nh, self.hd, dtype=bf, device=device)
        self.cv = torch.zeros_like(self.ck)
        self._rope_cache = {}

    # --------------------------------------------------------------------------------------------- helpers
    def rope_freqs_times(self, times):
        """complex freqs [n_frames*405, 1, D/2] for arbitrary per-frame timeline positions."""
        f = self.m.freqs.to(self.dev)
        c = self.hd // 2
        ft, fh, fw = f.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        t = torch.as_tensor(times, device=self.dev)
        nf = t.numel()
        full = torch.cat([ft[t].view(nf, 1, 1, -1).expand(nf, self.h, self.w, -1),
                          fh[:self.h].view(1, self.h, 1, -1).expand(nf, self.h, self.w, -1),
                          fw[:self.w].view(1, 1, self.w, -1).expand(nf, self.h, self.w, -1)], -1)
        return full.reshape(nf * self.h * self.w, 1, -1).to(torch.complex64)

    def rope_freqs(self, t0: int):
        """complex freqs [Lq, 1, D/2] for K branches x 3 latents at absolute timeline t0..t0+2."""
        key = int(t0)
        if key not in self._rope_cache:
            f = self.m.freqs.to(self.dev)
            c = self.hd // 2
            ft, fh, fw = f.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            t = torch.arange(t0, t0 + self.blk, device=self.dev).repeat(self.K)
            nf = t.numel()
            full = torch.cat([ft[t].view(nf, 1, 1, -1).expand(nf, self.h, self.w, -1),
                              fh[:self.h].view(1, self.h, 1, -1).expand(nf, self.h, self.w, -1),
                              fw[:self.w].view(1, 1, self.w, -1).expand(nf, self.h, self.w, -1)], -1)
            self._rope_cache[key] = full.reshape(nf * self.h * self.w, 1, -1).to(torch.complex64)
        return self._rope_cache[key]

    def freqs_for_positions(self, pos):
        """pos: [n_frames] long GPU tensor of window-relative positions -> complex [n*405, 1, D/2] (graph-safe)."""
        if not hasattr(self, "_ft"):
            f = self.m.freqs.to(self.dev)
            c = self.hd // 2
            ft, fh, fw = f.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            self._ft = ft.to(torch.complex64)
            self._fhw = torch.cat([fh[:self.h].view(self.h, 1, -1).expand(self.h, self.w, -1),
                                   fw[:self.w].view(1, self.w, -1).expand(self.h, self.w, -1)], -1).reshape(
                self.h * self.w, -1).to(torch.complex64)
        nf = pos.numel()
        t = self._ft[pos].view(nf, 1, -1).expand(nf, self.h * self.w, -1)
        return torch.cat([t, self._fhw[None].expand(nf, -1, -1)], -1).reshape(nf * self.h * self.w, 1, -1)

    def prepare_prefix(self):
        fr = self.freqs_for_positions(self.ppos)
        _, fkv, _ = self.prope_fns(self.pcam_v, self.pcam_K)
        for li, blk in enumerate(self.m.blocks):
            k = self.rope(self.rk[li], fr)
            v = self.rv[li]
            if blk.camera_attention_enabled:
                k = fkv(k.transpose(1, 2)).transpose(1, 2)
                v = fkv(v.transpose(1, 2)).transpose(1, 2)
            self.pk[li].copy_(k)
            self.pv[li].copy_(v)

    @staticmethod
    def rope(x, freqs):
        """x [1, L, H, D] -> rotated (fp32 complex math, matches SolarWM's echorope up to fp64 vs fp32)."""
        if "rope" in _FUSED:
            return _FUSED["rope"](x, freqs.real[None].contiguous(), freqs.imag[None].contiguous())
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * freqs[None]).flatten(-2).type_as(x)

    def prope_fns(self, viewmats, Ks):
        from solarwm.backends.wan22.runtime.camera import transform_relative_viewmats
        from solarwm.backends.wan22.runtime.modeling.camera_prope import _prepare_apply_fns_all_dim

        vm = transform_relative_viewmats(viewmats, "linear")
        return _prepare_apply_fns_all_dim(head_dim=self.hd, viewmats=vm, Ks=Ks, patches_x=None, patches_y=None,
                                          image_width=None, image_height=None)

    # --------------------------------------------------------------------------------------------- forward
    def forward(self, x_lat, t_frames, act_tok, ctx, xa_cache, cam, freqs, commit: bool):
        """x_lat [1, K*3, C, H, W]; t_frames [1, K*3]; act_tok [1, Lq]; cam: dict viewmats/K [1, Lq, ...].
        commit=True: t must be 0; stores final K/V of every branch block into self.ck/self.cv."""
        m = self.m
        xb = x_lat.permute(0, 2, 1, 3, 4)
        tok = m.patch_embedding(xb).flatten(2).transpose(1, 2)  # [1, Lq, dim]
        tok = action_input(m, tok, act_tok)
        t = t_frames[..., None].expand(*t_frames.shape, FRAME_TOKENS).reshape(1, -1)
        m._action_tokens = act_tok
        e = m._time_condition(t, tok, None)
        m._action_tokens = None
        e0 = m.time_projection(e).unflatten(1, (6, m.dim)).unflatten(0, t.shape)  # [1, Lq, 6, dim]
        fq, fkv, fo = self.prope_fns(cam["viewmats"], cam["K"])  # q/out and current-kv transforms (same cams)
        if commit:
            self.ccam_v.copy_(cam["viewmats"])
            self.ccam_K.copy_(cam["K"])
        K, n, d = self.K, self.nh, self.hd
        blk_tok = self.blk * FRAME_TOKENS
        for li, blk in enumerate(m.blocks):
            ee = (blk.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
            a_in = blk.norm1(tok) * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)
            sa = blk.self_attn
            q = sa.norm_q(sa.q(a_in)).view(1, -1, n, d)
            k = sa.norm_k(sa.k(a_in)).view(1, -1, n, d)
            v = sa.v(a_in).view(1, -1, n, d)
            if commit:
                self.ck[li].copy_(k.view(K, blk_tok, n, d))
                self.cv[li].copy_(v.view(K, blk_tok, n, d))
            q, k = self.rope(q, freqs), self.rope(k, freqs)
            use_cam = blk.camera_attention_enabled
            if use_cam:
                q = fq(q.transpose(1, 2)).transpose(1, 2)
                k = fkv(k.transpose(1, 2)).transpose(1, 2)
                v = fkv(v.transpose(1, 2)).transpose(1, 2)
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            # own-block attention: batch the K branches
            qb, kb, vb = q.view(K, blk_tok, n, d), k.view(K, blk_tok, n, d), v.view(K, blk_tok, n, d)
            if self.share:
                o1, l1, _ = self.fa(q, self.pk[li], self.pv[li], return_attn_probs=True)  # all queries vs prefix
                o2, l2, _ = self.fa(qb, kb, vb, return_attn_probs=True)
                # l2 [K, H, 1215] -> [1, H, K*1215] in the same branch-major token order as q
                o = _lse_merge(o1, l1, o2.reshape(1, -1, n, d), l2.permute(1, 0, 2).reshape(1, n, -1))
            else:  # batching: each branch attends [its own copy of the prefix | own block]
                kk = torch.cat([self.pk[li].expand(K, -1, -1, -1), kb], 1)
                vv = torch.cat([self.pv[li].expand(K, -1, -1, -1), vb], 1)
                o = self.fa(qb, kk, vv).reshape(1, -1, n, d)
            o = o.to(torch.bfloat16)
            if use_cam:
                o = fo(o.transpose(1, 2)).transpose(1, 2)
            y = sa.o(o.reshape(1, -1, n * d))
            tok = tok + y * ee[2].squeeze(2)
            tok = tok + blk.cross_attn(blk.norm3(tok), ctx, None, crossattn_cache=xa_cache[li])
            y = blk.ffn(blk.norm2(tok) * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2))
            tok = tok + y * ee[5].squeeze(2)
        out = m.head(tok, e.unflatten(0, t.shape).unsqueeze(2))
        grid = torch.tensor([[K * self.blk, self.h, self.w]])
        return torch.stack(m.unpatchify(out, grid)).permute(0, 2, 1, 3, 4)

    def _layer_impl(self, blk, tok, e0, freqs, cos, sin, Pq, Pkv, Po, pk, pv, xk, xv):
        """One transformer block for the K candidate blocks. Returns (tok, raw k, raw v) - raw = post-norm pre-RoPE."""
        K, n, d = self.K, self.nh, self.hd
        blk_tok = self.blk * FRAME_TOKENS
        ee = (blk.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
        a_in = blk.norm1(tok) * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)
        sa = blk.self_attn
        q = sa.norm_q(sa.q(a_in)).view(1, -1, n, d)
        k_raw = sa.norm_k(sa.k(a_in)).view(1, -1, n, d)
        v_raw = sa.v(a_in).view(1, -1, n, d)
        q, k = _rope_math(q, cos, sin), _rope_math(k_raw, cos, sin)
        v = v_raw
        if blk.camera_attention_enabled:
            q = _tiled_projmat_math(q.transpose(1, 2), Pq).transpose(1, 2)
            k = _tiled_projmat_math(k.transpose(1, 2), Pkv).transpose(1, 2)
            v = _tiled_projmat_math(v.transpose(1, 2), Pkv).transpose(1, 2)
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        qb, kb, vb = q.view(K, blk_tok, n, d), k.view(K, blk_tok, n, d), v.view(K, blk_tok, n, d)
        if self.share:
            o1, l1, _ = self.fa(q, pk, pv, return_attn_probs=True)
            o2, l2, _ = self.fa(qb, kb, vb, return_attn_probs=True)
            o = _lse_merge(o1, l1, o2.reshape(1, -1, n, d), l2.permute(1, 0, 2).reshape(1, n, -1))
        else:
            kk = torch.cat([pk.expand(K, -1, -1, -1), kb], 1)
            vv = torch.cat([pv.expand(K, -1, -1, -1), vb], 1)
            o = self.fa(qb, kk, vv).reshape(1, -1, n, d)
        o = o.to(torch.bfloat16)
        if blk.camera_attention_enabled:
            o = _tiled_projmat_math(o.transpose(1, 2), Po).transpose(1, 2)
        y = sa.o(o.reshape(1, -1, n * d))
        tok = tok + y * ee[2].squeeze(2)
        ca = blk.cross_attn  # text cross-attention with the cached context K/V (cache initialised beforehand)
        xq = ca.norm_q(ca.q(blk.norm3(tok))).view(1, -1, n, d)
        xo = self.fa(xq.to(torch.bfloat16), xk.to(torch.bfloat16), xv.to(torch.bfloat16))
        tok = tok + ca.o(xo.flatten(2))
        y = blk.ffn(blk.norm2(tok) * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2))
        tok = tok + y * ee[5].squeeze(2)
        return tok, k_raw, v_raw

    def forward_fused(self, x_lat, t_frames, act_tok, ctx, xa_cache, cam, freqs, commit: bool):
        """Same as forward(), with the block body in _layer_impl (torch.compile'd when SHOWNOW_COMPILE_LAYER=1)."""
        import os
        m = self.m
        if not hasattr(self, "_layer_fn"):
            self._layer_fn = torch.compile(self._layer_impl, dynamic=False) \
                if os.environ.get("SHOWNOW_COMPILE_LAYER", "0") == "1" else self._layer_impl
        if not xa_cache[0]["is_init"]:  # initialise the text K/V cache with the reference path once
            return self.forward(x_lat, t_frames, act_tok, ctx, xa_cache, cam, freqs, commit)
        xb = x_lat.permute(0, 2, 1, 3, 4)
        tok = m.patch_embedding(xb).flatten(2).transpose(1, 2)
        tok = action_input(m, tok, act_tok)
        t = t_frames[..., None].expand(*t_frames.shape, FRAME_TOKENS).reshape(1, -1)
        m._action_tokens = act_tok
        e = m._time_condition(t, tok, None)
        m._action_tokens = None
        e0 = m.time_projection(e).unflatten(1, (6, m.dim)).unflatten(0, t.shape)
        Pq, Pkv, Po = prope_mats(cam["viewmats"], cam["K"])
        cos, sin = freqs.real[None].contiguous(), freqs.imag[None].contiguous()
        if commit:
            self.ccam_v.copy_(cam["viewmats"])
            self.ccam_K.copy_(cam["K"])
        K, n, d = self.K, self.nh, self.hd
        blk_tok = self.blk * FRAME_TOKENS
        for li, blk in enumerate(m.blocks):
            tok, k_raw, v_raw = self._layer_fn(blk, tok, e0, freqs, cos, sin, Pq, Pkv, Po, self.pk[li], self.pv[li],
                                               xa_cache[li]["k"], xa_cache[li]["v"])
            if commit:
                self.ck[li].copy_(k_raw.view(K, blk_tok, n, d))
                self.cv[li].copy_(v_raw.view(K, blk_tok, n, d))
        out = m.head(tok, e.unflatten(0, t.shape).unsqueeze(2))
        grid = torch.tensor([[K * self.blk, self.h, self.w]])
        return torch.stack(m.unpatchify(out, grid)).permute(0, 2, 1, 3, 4)

    def reroot(self, branch: int, slot: int):
        """chosen candidate block -> prefix ring slot (the oldest block is overwritten); window positions of the
        ring slots are shifted by one block (the new block is the most recent: positions P-3..P-1)."""
        n = self.blk * FRAME_TOKENS
        s = slot * n
        self.rk[:, :, s:s + n].copy_(self.ck[:, branch:branch + 1])
        self.rv[:, :, s:s + n].copy_(self.cv[:, branch:branch + 1])
        self.pcam_v[:, s:s + n].copy_(self.ccam_v[:, branch * n:(branch + 1) * n])
        self.pcam_K[:, s:s + n].copy_(self.ccam_K[:, branch * n:(branch + 1) * n])
        self.ppos.sub_(self.blk)
        self.ppos[slot * self.blk:(slot + 1) * self.blk] = torch.arange(self.P - self.blk, self.P, device=self.dev)
