"""Optimised real-time speculative AR path (opt-in; the paper's numbers use tree_fast.FastTree unchanged).

OptTree keeps FastTree's state layout (raw K/V ring + cameras + window positions, cascade attention over the shared
prefix) and adds, each switchable for A/B checks (`parse_opts`):
  per_frame     time/action modulation computed per latent frame ([1, F, 6, dim]) and broadcast over its 405 tokens
                instead of per token (all tokens of a frame share t and the key) - exact up to GEMM rounding
  qkv           one fused Q|K|V GEMM per layer - exact
  gelu          FFN GELU(tanh) in the cuBLASLt GEMM epilogue - exact up to one bf16 rounding
  fused_commit  the chosen latent's commit pass (t = 0) rides in the NEXT step's first denoising pass as one more frame:
                it attends the old window + itself (= the separate commit pass); at every layer its raw K/V go into the
                ring slot and that layer's prefix keys are re-rotated for the new window before the candidates attend
                it (= prepare_prefix after the commit) - exact; no separate commit graph after the key is read
  fp8           per-token linears in e4m3 (per-output-channel weight scale, per-token dynamic activation scale) - lossy
  fp8ffn        only the FFN linears in e4m3 (61 % of the GEMM FLOPs), attention projections stay bf16 - lossy
  fp8static     the e4m3 linears use a STATIC per-tensor activation scale (calibrated max |x| / 448 per linear, file
                $SHOWNOW_FP8_CALIB, see calibrate_fp8) instead of the per-token amax: quantisation becomes a pointwise
                op that fuses into its producer (no row reduction) - lossy (outliers above the calibrated max clip)
The compiled layer takes plain tensors (no module), so one compilation serves all 30 layers.

RTStepper drives one OptTree: load / commit history, gen() -> K candidates of the next latent, choose(j).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shownow.latency.tree_fast import (FRAME_TOKENS, FastTree, _lse_merge, _rope_math, _tiled_projmat_math,
                                       flash_attn_func_any, prope_mats)
from shownow.train.student_common import tokens
from shownow.train.teacher_action_ft import action_input

BF = torch.bfloat16
E4M3 = torch.float8_e4m3fn
OPT_NAMES = ("per_frame", "qkv", "gelu", "fused_commit", "fp8", "fp8ffn", "fp8static")
EXACT = ("per_frame", "qkv", "gelu", "fused_commit")


def parse_opts(s: str) -> dict:
    """"none" | "exact" | "exact,fp8" | comma list of OPT_NAMES -> {name: bool}."""
    on = set()
    for p in filter(None, s.split(",")):
        if p == "exact":
            on |= set(EXACT)
        elif p != "none":
            assert p in OPT_NAMES, p
            on.add(p)
    return {k: k in on for k in OPT_NAMES}


def opts_name(o: dict) -> str:
    on = [k for k in OPT_NAMES if o[k]]
    return "+".join(on) if on else "none"


# ----------------------------------------------------------------------------------------------- functional pieces
def _pack(lins, fp8: bool) -> dict:
    w = torch.cat([l.weight.detach() for l in lins]).to(BF).contiguous()
    b = torch.cat([l.bias.detach() for l in lins]).to(BF).contiguous()
    if not fp8:
        return {"w": w, "b": b}
    wf = w.float()
    s = (wf.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    return {"w8": (wf / s).to(E4M3).contiguous(), "ws": s.t().contiguous(), "b": b}


@torch.library.custom_op("shownow::fp8_mm", mutates_args=())
def fp8_mm(a: torch.Tensor, w8: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """e4m3 a [M, K] x w8 [N, K]^T -> bf16 [M, N] ROW-MAJOR. An opaque op for inductor: as a plain _scaled_mm it may pick a
    column-major output, and the fused row-wise LayerNorm / residual / quantisation kernels that read it then load with a
    stride of M along the row (uncoalesced, ~0.25 TB/s)."""
    return torch._scaled_mm(a, w8.t(), scale_a=sa, scale_b=sb, bias=bias, out_dtype=BF, use_fast_accum=True)


@fp8_mm.register_fake
def _(a, w8, sa, sb, bias):
    return a.new_empty((a.shape[0], w8.shape[0]), dtype=BF)


_CALIB = None  # {id(W["w8"]): max |x|} while recording (eager only, see calibrate_fp8)
DEFAULT_CALIB = "runs/fp8_calib/b1_s2_step1500.pt"


def _lin(x, W: dict, gelu: bool = False, epi: bool = False):
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).to(BF)
    if _CALIB is not None and "w8" in W:
        k = id(W["w8"])
        _CALIB[k] = max(_CALIB.get(k, 0.0), float(x2.abs().amax()))
    if "sa" in W:  # static per-tensor activation scale (fp8static)
        y = fp8_mm((x2.float() * W["inv"]).clamp(-448.0, 448.0).to(E4M3), W["w8"], W["sa"][:x2.shape[0]], W["ws"],
                   W["b"])
        if gelu:
            y = F.gelu(y, approximate="tanh")
    elif "w8" in W:
        xs = (x2.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp(min=1e-12)
        y = fp8_mm((x2.float() / xs).to(E4M3), W["w8"], xs, W["ws"], W["b"])
        if gelu:
            y = F.gelu(y, approximate="tanh")
    elif gelu and epi:
        y = torch._addmm_activation(W["b"], x2, W["w"].t(), use_gelu=True)
    else:
        y = F.linear(x2, W["w"], W["b"])
        if gelu:
            y = F.gelu(y, approximate="tanh")
    return y.view(*shp[:-1], y.shape[-1])


def _rms(x, w, eps: float):
    """WanRMSNorm.forward (same ops; runs under the same autocast)."""
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(x) * w


def _ln(x, w, b, eps: float):
    """WanLayerNorm.forward."""
    return F.layer_norm(x, (x.shape[-1],), w, b, eps).type_as(x)


def _prope(x, M):
    return _tiled_projmat_math(x.transpose(1, 2), M).transpose(1, 2)


_FA = None


def _fa(*a, **k):
    global _FA
    if _FA is None:
        _FA = flash_attn_func_any()
    return _FA(*a, **k)


def _prep(rk, rv, pk, pv, pcos, psin, PkvP, n: int, cam: bool):
    """final prefix keys/values of the first n ring tokens for the current window positions (prepare_prefix)."""
    k = _rope_math(rk[:, :n], pcos, psin)
    v = rv[:, :n]
    if cam:
        k, v = _prope(k, PkvP), _prope(v, PkvP)
    pk[:, :n].copy_(k.to(BF))
    pv[:, :n].copy_(v.to(BF))


def _layer(L: dict, tok, e0, cos, sin, Pq, Pkv, Po, xk, xv, pk, pv, rk, rv, slot, pcos, psin, PkvP,
           Fr: int, nc: int, n_old: int, n_new: int, cam: bool, epi: bool, eps_ln: float, eps_rms: float):
    """One transformer block for Fr frames (nc leading commit frames at t = 0 + candidates). Returns (tok, raw k, raw v).
    e0 [1, Fr, Tm, 6, dim] with Tm = 1 (per-frame) or 405 (per-token). Prefix: commit frames attend the first n_old ring
    tokens as prepared before this pass; with nc the commit frame's raw K/V are written into the ring at `slot` and the
    first n_new ring tokens re-prepared (pcos/psin/PkvP = new window) before the candidates attend them."""
    T = FRAME_TOKENS
    dim = tok.shape[-1]
    n, d = xk.shape[2], xk.shape[3]
    x = tok.view(1, Fr, T, dim)
    ee = (L["mod"].view(1, 1, 1, 6, dim) + e0).unbind(3)  # 6 x [1, Fr, Tm, dim]
    a = (_ln(x, None, None, eps_ln) * (1 + ee[1]) + ee[0]).view(1, Fr * T, dim)
    if "qkv" in L:
        q, k_raw, v_raw = _lin(a, L["qkv"]).split(dim, -1)
    else:
        q, k_raw, v_raw = _lin(a, L["q"]), _lin(a, L["k"]), _lin(a, L["v"])
    q = _rms(q, L["nq"], eps_rms).view(1, -1, n, d)
    k_raw = _rms(k_raw, L["nk"], eps_rms).view(1, -1, n, d)
    v_raw = v_raw.reshape(1, -1, n, d)
    q, k, v = _rope_math(q, cos, sin), _rope_math(k_raw, cos, sin), v_raw
    if cam:
        q, k, v = _prope(q, Pq), _prope(k, Pkv), _prope(v, Pkv)
    q, k, v = q.to(BF), k.to(BF), v.to(BF)
    o2, l2, _ = _fa(q.view(Fr, T, n, d), k.view(Fr, T, n, d), v.view(Fr, T, n, d), return_attn_probs=True)
    o2, l2 = o2.reshape(1, -1, n, d), l2.permute(1, 0, 2).reshape(1, n, -1)
    if nc:
        Tc = nc * T
        if n_old:
            o1c, l1c, _ = _fa(q[:, :Tc], pk[:, :n_old], pv[:, :n_old], return_attn_probs=True)
        rk.index_copy_(1, slot, k_raw[:, :Tc])
        rv.index_copy_(1, slot, v_raw[:, :Tc])
        _prep(rk, rv, pk, pv, pcos, psin, PkvP, n_new, cam)
        o1x, l1x, _ = _fa(q[:, Tc:], pk[:, :n_new], pv[:, :n_new], return_attn_probs=True)
        if n_old:
            o = torch.cat([_lse_merge(o1c, l1c, o2[:, :Tc], l2[..., :Tc]),
                           _lse_merge(o1x, l1x, o2[:, Tc:], l2[..., Tc:])], 1)
        else:
            o = torch.cat([o2[:, :Tc].float(), _lse_merge(o1x, l1x, o2[:, Tc:], l2[..., Tc:])], 1)
    elif n_new:
        o1, l1, _ = _fa(q, pk[:, :n_new], pv[:, :n_new], return_attn_probs=True)
        o = _lse_merge(o1, l1, o2, l2)
    else:  # empty prefix (first history latent)
        o = o2
    o = o.to(BF)
    if cam:
        o = _prope(o, Po)
    x = x + _lin(o.reshape(1, -1, n * d), L["o"]).view(1, Fr, T, dim) * ee[2]
    xq = _rms(_lin(_ln(x, L["n3w"], L["n3b"], eps_ln).view(1, -1, dim), L["xq"]), L["xnq"], eps_rms)
    xo = _fa(xq.view(1, -1, n, d).to(BF), xk, xv)
    x = x + _lin(xo.reshape(1, -1, n * d), L["xo"]).view(1, Fr, T, dim)
    h = _lin((_ln(x, None, None, eps_ln) * (1 + ee[4]) + ee[3]).view(1, -1, dim), L["f1"], gelu=True, epi=epi)
    x = x + _lin(h, L["f2"]).view(1, Fr, T, dim) * ee[5]
    return x.view(1, -1, dim), k_raw, v_raw


_COMPILED = {}


def _compiled(name, fn):
    """SHOWNOW_OPT_DYNAMIC=0: static shapes (one compilation per shape - timing runs); default: automatic dynamic
    (growing windows in evaluation rollouts would otherwise recompile for every prefix length)."""
    import os
    if name not in _COMPILED:
        dyn = None if os.environ.get("SHOWNOW_OPT_DYNAMIC", "auto") == "auto" else False
        _COMPILED[name] = torch.compile(fn, dynamic=dyn)
    return _COMPILED[name]


class record_amax:
    """with record_amax() as rec: <eager (compile=False) rowwise-fp8 stepper runs>  -> rec = {id(W["w8"]): max |x|}"""

    def __enter__(self):
        global _CALIB
        _CALIB = {}
        return _CALIB

    def __exit__(self, *exc):
        global _CALIB
        _CALIB = None


# ----------------------------------------------------------------------------------------------- stepper
class RTStepper:
    """K candidates of the next latent per step from a P-latent window; ring + cameras + positions as in FastTree."""

    _weights = {}  # (id(model), qkv, fp8) -> per-layer tensor dicts, shared by steppers of different K

    def __init__(self, model, K: int, fewstep, sig, P: int = 15, opts: dict | None = None, h: int = 15, w: int = 27,
                 device="cuda", compile: bool = True):
        global _FA
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
        _FA = _FA or flash_attn_func_any()
        self.o = opts or parse_opts("none")
        self.m, self.K, self.P, self.dev = model, K, P, device
        self.fewstep, self.sig = list(fewstep), list(sig)
        self.ft = FastTree(model, K, prefix_latents=P, share=True, h=h, w=w, device=device, block=1)
        del self.ft.ck, self.ft.cv  # unused (raw K/V of the commit go straight into the ring)
        self.nh, self.hd = self.ft.nh, self.ft.hd
        blk0 = model.blocks[0]
        self.eps_ln, self.eps_rms = float(blk0.norm1.eps), float(blk0.self_attn.norm_q.eps)
        self.cam_on = [bool(b.camera_attention_enabled) for b in model.blocks]
        self.L = self._layer_weights()
        self.layer_fn = _compiled("layer", _layer) if compile else _layer
        self.prep_fn = _compiled("prep", _prep) if compile else _prep
        T, C = FRAME_TOKENS, 48
        nL = len(model.blocks)
        self.xk = torch.zeros(nL, 1, 512, self.nh, self.hd, dtype=BF, device=device)
        self.xv = torch.zeros_like(self.xk)
        # static per-step inputs (graph-safe): candidates and the pending commit frame
        self.x = torch.zeros(1, K, C, 2 * h, 2 * w, dtype=BF, device=device)
        self.cam_v = torch.eye(4, device=device).expand(1, K * T, 4, 4).contiguous()
        self.cam_K = torch.eye(3, device=device).expand(1, K * T, 3, 3).contiguous()
        self.act = torch.zeros(1, K, dtype=torch.long, device=device)  # per-frame key ids
        self.c_x = torch.zeros(1, 1, C, 2 * h, 2 * w, dtype=BF, device=device)
        self.c_cam_v = torch.eye(4, device=device).expand(1, T, 4, 4).contiguous()
        self.c_cam_K = torch.eye(3, device=device).expand(1, T, 3, 3).contiguous()
        self.c_act = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.slot_tok = torch.arange(T, device=device)
        self.ts = [torch.full((1, K), float(t), device=device) for t in self.fewstep]
        self.zero1 = torch.zeros(1, 1, device=device)
        self.reset()

    def _layer_weights(self):
        key = (id(self.m), self.o["qkv"], self.o["fp8"], self.o["fp8ffn"], self.o["fp8static"])
        if key not in RTStepper._weights:
            fp8, out = self.o["fp8"], []
            fp8f = fp8 or self.o["fp8ffn"]
            for blk in self.m.blocks:
                sa, ca, ffn = blk.self_attn, blk.cross_attn, blk.ffn
                L = {"mod": blk.modulation.detach().to(BF), "nq": sa.norm_q.weight.detach().to(BF),
                     "nk": sa.norm_k.weight.detach().to(BF), "xnq": ca.norm_q.weight.detach().to(BF),
                     "n3w": blk.norm3.weight.detach().to(BF), "n3b": blk.norm3.bias.detach().to(BF),
                     "o": _pack([sa.o], fp8), "xq": _pack([ca.q], fp8), "xo": _pack([ca.o], fp8),
                     "f1": _pack([ffn[0]], fp8f), "f2": _pack([ffn[2]], fp8f)}
                if self.o["qkv"]:
                    L["qkv"] = _pack([sa.q, sa.k, sa.v], fp8)
                else:
                    L["q"], L["k"], L["v"] = _pack([sa.q], fp8), _pack([sa.k], fp8), _pack([sa.v], fp8)
                out.append(L)
            if self.o["fp8static"]:
                import os
                path = os.environ.get("SHOWNOW_FP8_CALIB", DEFAULT_CALIB)
                amax = torch.load(path, weights_only=True)["amax"]  # {"<layer>.<linear>": max |x|}
                rows = FRAME_TOKENS * 17  # >= tokens of any pass (K <= 16 candidates + 1 commit frame)
                for li, L in enumerate(out):
                    for n, W in L.items():
                        if isinstance(W, dict) and "w8" in W:
                            sc = max(float(amax[f"{li}.{n}"]), 1e-6) / 448.0
                            W["inv"] = torch.tensor(1.0 / sc, device=self.dev)
                            W["sa"] = torch.full((rows, 1), sc, device=self.dev)
            RTStepper._weights[key] = out
        return RTStepper._weights[key]

    def linear_names(self):
        """{id(W["w8"]): "<layer>.<linear>"} of this stepper's e4m3 linears (calibration bookkeeping)."""
        return {id(W["w8"]): f"{li}.{n}" for li, L in enumerate(self.L) for n, W in L.items()
                if isinstance(W, dict) and "w8" in W}

    # ------------------------------------------------------------------ state
    def reset(self):
        self.n_valid, self.ring_ptr, self.pending, self.c_pos = 0, 0, False, 0

    @torch.no_grad()
    def set_text(self, ctx):
        """text cross-attention K/V of the prompt context (as WanCrossAttention computes them)."""
        n, d = self.nh, self.hd
        with torch.autocast("cuda", dtype=BF):
            for li, blk in enumerate(self.m.blocks):
                ca = blk.cross_attn
                self.xk[li].copy_(ca.norm_k(ca.k(ctx)).view(1, -1, n, d))
                self.xv[li].copy_(ca.v(ctx).view(1, -1, n, d))

    def load_ring(self, committed):
        """committed: chronological list of (raw k [L,1,405,H,D], raw v, cam [1,405,25]) -> ring (last P)."""
        T, ft = FRAME_TOKENS, self.ft
        committed = committed[-self.P:]
        for i, (k_, v_, c_) in enumerate(committed):
            s = i * T
            ft.rk[:, :, s:s + T].copy_(k_)
            ft.rv[:, :, s:s + T].copy_(v_)
            ft.pcam_v[:, s:s + T].copy_(c_[..., :16].unflatten(2, (4, 4)))
            ft.pcam_K[:, s:s + T].copy_(c_[..., 16:].unflatten(2, (3, 3)))
        ft.ppos.zero_()
        ft.ppos[:len(committed)] = torch.arange(len(committed), device=self.dev)
        self.n_valid, self.ring_ptr, self.pending = len(committed), 0, False

    def set_inputs(self, cam: dict, keys):
        """candidate cameras (per token, K frames) and key ids [K]."""
        self.cam_v.copy_(cam["viewmats"])
        self.cam_K.copy_(cam["K"])
        self.act.copy_(torch.as_tensor(keys, device=self.dev).view(1, -1))

    def _advance(self):
        """ring bookkeeping for committing self.c_*: slot, cameras, window positions (host side, before the pass that
        computes the commit's raw K/V). Returns n_old (prefix latents the commit frame attends)."""
        ft, T, P = self.ft, FRAME_TOKENS, self.P
        n_old = self.n_valid
        slot = n_old if n_old < P else self.ring_ptr
        ft.pcam_v[:, slot * T:(slot + 1) * T].copy_(self.c_cam_v)
        ft.pcam_K[:, slot * T:(slot + 1) * T].copy_(self.c_cam_K)
        self.slot_tok.copy_(torch.arange(slot * T, (slot + 1) * T, device=self.dev))
        if n_old < P:
            ft.ppos[n_old] = n_old
            self.n_valid = n_old + 1
        else:
            ft.ppos.sub_(1)
            ft.ppos[slot] = P - 1
            self.ring_ptr = (self.ring_ptr + 1) % P
        self.c_pos = n_old
        return n_old

    # ------------------------------------------------------------------ passes
    def _prefix_rot(self, n_frames):
        fr = self.ft.freqs_for_positions(self.ft.ppos[:n_frames])
        n = n_frames * FRAME_TOKENS
        PkvP = prope_mats(self.ft.pcam_v[:, :n], self.ft.pcam_K[:, :n])[1]
        return fr.real[None].contiguous(), fr.imag[None].contiguous(), PkvP

    def prep_all(self):
        n = self.n_valid
        if not n:
            return
        pcos, psin, PkvP = self._prefix_rot(n)
        ft = self.ft
        for li in range(len(self.m.blocks)):
            self.prep_fn(ft.rk[li], ft.rv[li], ft.pk[li], ft.pv[li], pcos, psin, PkvP, n * FRAME_TOKENS,
                         self.cam_on[li])

    def _pass(self, x_c, t_c, nc: int, n_old: int, n_new: int, store: bool = False, head: bool = True):
        """x_c [1, Kc, C, H, W] frames after the nc commit frame(s) (taken from c_*); t_c [1, Kc]. Positions: commit frame
        at n_old, the others at n_new (latents). store: write the raw K/V of the (single) frame into the ring slot."""
        m, ft, T = self.m, self.ft, FRAME_TOKENS
        Kc = x_c.shape[1]
        Fr = nc + Kc
        if nc:
            xin = torch.cat([self.c_x, x_c], 1)
            tf = torch.cat([self.zero1, t_c], 1)
            af = torch.cat([self.c_act, self.act[:, :Kc]], 1)
            cv = torch.cat([self.c_cam_v, self.cam_v[:, :Kc * T]], 1)
            cK = torch.cat([self.c_cam_K, self.cam_K[:, :Kc * T]], 1)
            pos = torch.cat([torch.full((1,), n_old, device=self.dev), torch.full((Kc,), n_new, device=self.dev)])
        else:
            xin, tf = x_c, t_c
            af = self.c_act if store else self.act[:, :Kc]
            cv, cK = (self.c_cam_v, self.c_cam_K) if store else (self.cam_v[:, :Kc * T], self.cam_K[:, :Kc * T])
            pos = torch.full((Kc,), n_old if store else n_new, device=self.dev)
        act_tok = tokens(af)
        tok = m.patch_embedding(xin.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        tok = action_input(m, tok, act_tok)
        dim = tok.shape[-1]
        if self.o["per_frame"]:
            m._action_tokens = af
            e = m._time_condition(tf, tok, None)  # [Fr, dim]
            e0 = m.time_projection(e).unflatten(1, (6, dim)).view(1, Fr, 1, 6, dim)
        else:
            m._action_tokens = act_tok
            e = m._time_condition(tokens(tf), tok, None)  # [Fr*T, dim]
            e0 = m.time_projection(e).unflatten(1, (6, dim)).view(1, Fr, T, 6, dim)
        m._action_tokens = None
        fr = ft.freqs_for_positions(pos)
        cos, sin = fr.real[None].contiguous(), fr.imag[None].contiguous()
        Pq, Pkv, Po = prope_mats(cv, cK)
        pcos = psin = PkvP = None
        if nc:
            pcos, psin, PkvP = self._prefix_rot(n_new)
        n_o, n_n = n_old * T, (n_old if store else n_new) * T
        for li in range(len(m.blocks)):
            tok, kr, vr = self.layer_fn(self.L[li], tok, e0, cos, sin, Pq, Pkv, Po, self.xk[li], self.xv[li],
                                        ft.pk[li], ft.pv[li], ft.rk[li], ft.rv[li], self.slot_tok, pcos, psin, PkvP,
                                        Fr, nc, n_o, n_n, self.cam_on[li], self.o["gelu"], self.eps_ln, self.eps_rms)
            if store:
                ft.rk[li].index_copy_(1, self.slot_tok, kr)
                ft.rv[li].index_copy_(1, self.slot_tok, vr)
        if not head:
            return None
        Tc = nc * T
        tok = tok[:, Tc:]
        if self.o["per_frame"]:
            e_tok = e[nc:, None].expand(Kc, T, dim).reshape(1, Kc * T, dim)
        else:
            e_tok = e[Tc:].view(1, Kc * T, dim)
        out = m.head(tok, e_tok.unsqueeze(2))
        grid = torch.tensor([[Kc, ft.h, ft.w]])
        return torch.stack(m.unpatchify(out, grid)).permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def commit_frame(self, lat, cam: dict, key: int):
        """commit a given latent (history) with a separate K=1 pass at t = 0."""
        self.c_x.copy_(lat)
        self.c_cam_v.copy_(cam["viewmats"])
        self.c_cam_K.copy_(cam["K"])
        self.c_act.fill_(int(key))
        assert not self.pending
        self._commit_pass()

    def _commit_pass(self):
        n_old = self._advance()
        self._pass(self.c_x, self.zero1, 0, n_old, n_old, store=True, head=False)

    @torch.no_grad()
    def gen(self, noise=None):
        """K candidates of the next latent -> self.x. noise: None (sampled on the GPU, graph-safe) or the list
        [x_T (bf16, [1,K,...]), eps_1, ..., eps_{n-1} (fp32)] (reproducible eager runs)."""
        nc = 1 if self.pending else 0
        n_new = self.n_valid
        if not nc:
            self.prep_all()
        x = noise[0] if noise is not None else torch.randn_like(self.x)
        ns = len(self.fewstep)
        for i in range(ns):
            flow = self._pass(x, self.ts[i], nc if i == 0 else 0, self.c_pos, n_new)
            x0 = x.float() - self.sig[i] * flow.float()
            if i == ns - 1:
                x = x0.to(BF)
            else:
                eps = noise[i + 1] if noise is not None else torch.randn_like(x0)
                x = ((1 - self.sig[i + 1]) * x0 + self.sig[i + 1] * eps).to(BF)
        self.x.copy_(x)
        self.pending = False
        return self.x

    @torch.no_grad()
    def choose(self, j: int, run_commit: bool = True):
        """candidate j becomes the committed latent. fused_commit: its commit pass runs inside the next gen()
        (call _advance now - host bookkeeping only); else a separate commit pass now (run_commit=False: the caller
        replays a captured commit graph after this bookkeeping)."""
        T = FRAME_TOKENS
        self.c_x.copy_(self.x[:, j:j + 1])
        self.c_cam_v.copy_(self.cam_v[:, j * T:(j + 1) * T])
        self.c_cam_K.copy_(self.cam_K[:, j * T:(j + 1) * T])
        self.c_act.copy_(self.act[:, j:j + 1])
        if self.o["fused_commit"]:
            self._advance()
            self.pending = True
        elif run_commit:
            self._commit_pass()
        else:
            self._advance()

    # ------------------------------------------------------------------ CUDA graphs (steady state: full window)
    @torch.no_grad()
    def capture(self):
        """graphs: 'plain' (prepare + n passes), 'fused' (commit frame in pass 1), 'commit' (separate commit pass)."""
        assert self.n_valid == self.P
        self.graphs = {}
        T, P = FRAME_TOKENS, self.P

        def plain():
            self.pending = False
            self.gen()

        def fused():
            self.pending = True
            self.c_pos = P
            self.gen()

        def commit():
            self._pass(self.c_x, self.zero1, 0, P, P, store=True, head=False)

        fns = {"plain": plain, "commit": commit}
        if self.o["fused_commit"]:
            fns["fused"] = fused
        snap = [t.clone() for t in (self.ft.rk, self.ft.rv, self.ft.pk, self.ft.pv)]
        with torch.autocast("cuda", dtype=BF, cache_enabled=False):
            for name, fn in fns.items():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                    fn()
                self.graphs[name] = g
        for t, s in zip((self.ft.rk, self.ft.rv, self.ft.pk, self.ft.pv), snap):  # warm-up wrote into the ring
            t.copy_(s)
        del snap
        self.pending = False
        self._capture_fns = fns  # keep closures (and their buffers) alive

    def replay_gen(self):
        """steady-state gen() via the captured graphs (pending commit -> 'fused')."""
        self.graphs["fused" if self.pending else "plain"].replay()
        self.pending = False
        return self.x

    def replay_choose(self, j: int):
        self.choose(j, run_commit=False)
        if not self.o["fused_commit"]:
            self.graphs["commit"].replay()
