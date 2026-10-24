"""Matrix-Game 2.0 adapter for ext_latency (Skywork, arXiv 2508.13009; universal distilled model, native code and weights
in third_party/Matrix-Game-2 + models/Matrix-Game-2.0, env .venv-mg2).

Native schedule (pipeline.CausalInferenceStreamingPipeline, unchanged math): 3-latent blocks (first block 9 frames, then
12), the action is read at the block start, 3 denoising steps (1000 / 666 / 333), a t = 0 rerun updates the KV cache
(= commit), streaming Wan2.1 VAE block decoder (compiled max-autotune as in their inference script). Keys W / A / S / D ->
its keyboard one-hot, mouse 0. Native output 16 fps (Wan2.1).

Generation size (`size=(H, W)`, default the native 352x640): the image condition, the noise / latent grid
(H/8 x W/8, 16 channels) and with it the DiT token grid are built at that size, and the per-frame token count
(pipeline.frame_seq_length, the KV caches and each self-attention's max_attention_size, all derived from the fixed 880 =
22x40 tokens of the native size in their code) is set to (H/16)(W/16). Weights, layers and the analytic 3D RoPE are
unchanged. Their action module additionally asserts the 880-token native grid; non-native sizes therefore run under
`python -O` (assertions off). The model was trained at 352x640 only.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
MG = REPO / "third_party/Matrix-Game-2"
WEIGHTS = REPO / "models/Matrix-Game-2.0"
KEYBOARD = {1: [1, 0, 0, 0], 2: [0, 0, 1, 0], 3: [0, 1, 0, 0], 4: [0, 0, 0, 1]}  # ours W A S D -> MG2 [w, s, a, d]
NFB = 3  # latents per block (configs/inference_yaml/inference_universal.yaml)
NATIVE_FPS = 16.0
CODE_DIR = MG


class Adapter:
    name = "mg2_native"
    frames_per_unit = 4 * NFB  # pixel frames per generation unit (steady state)
    keep_video = False  # validation: keep the last decoded block in self.last_video

    def __init__(self, max_latents: int, weights: Path = WEIGHTS, size: tuple[int, int] = (352, 640)):
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from torchvision.transforms import v2

        from demo_utils.vae_block3 import VAEDecoderWrapper
        from pipeline import CausalInferenceStreamingPipeline
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper

        self.dev, self.dt, self.L = torch.device("cuda"), torch.bfloat16, max_latents
        self.H, self.W = size
        assert self.H % 16 == 0 and self.W % 16 == 0, size
        self.last_video = None
        self.cfg = OmegaConf.load(MG / "configs/inference_yaml/inference_universal.yaml")
        self.cfg.pop("mode")
        gen = WanDiffusionWrapper(**getattr(self.cfg, "model_kwargs", {}), is_causal=True)
        dec = VAEDecoderWrapper()
        sd = torch.load(weights / "Wan2.1_VAE.pth", map_location="cpu")
        dec.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k})
        dec.to(self.dev, torch.float16).requires_grad_(False).eval()
        dec.compile(mode="max-autotune-no-cudagraphs")
        pipe = CausalInferenceStreamingPipeline(self.cfg, generator=gen, vae_decoder=dec)
        pipe.generator.load_state_dict(load_file(str(weights / "base_distilled_model/base_distill.safetensors")))
        self.pipe = pipe.to(device=self.dev, dtype=self.dt)
        self.pipe.vae_decoder.to(torch.float16)
        vae = get_wanx_vae_wrapper(str(weights), torch.float16)  # not a full nn.Module: call as their script does
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.dev, self.dt)
        fsl = (self.H // 16) * (self.W // 16)  # DiT tokens per latent frame (880 at the native 352x640)
        if fsl != self.pipe.frame_seq_length:
            assert not __debug__, "non-native size: run with python -O (their action module asserts 880 tokens)"
            self.pipe.frame_seq_length = fsl
            for blk in self.pipe.generator.model.blocks:
                a = blk.self_attn
                a.max_attention_size = (15 if a.local_attn_size == -1 else a.local_attn_size) * fsl
        self.proc = v2.Compose([v2.Resize(size=(self.H, self.W), antialias=True), v2.ToTensor(),
                                v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        self._cond = {}

    @torch.no_grad()
    def scene_cond(self, img_path: str):
        """image -> cond_concat / visual_context (as inference_streaming.generate_videos), cached per image."""
        if img_path not in self._cond:
            from diffusers.utils import load_image
            img = load_image(img_path)
            w, h = img.size  # center crop to H:W (their _resizecrop)
            if h / w > self.H / self.W:
                nw, nh = w, int(w * self.H / self.W)
            else:
                nh, nw = h, int(h * self.W / self.H)
            img = img.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2))
            x = self.proc(img)[None, :, None].to(self.dev, self.dt)
            pad = torch.zeros_like(x).repeat(1, 1, 4 * (self.L - 1), 1, 1)
            ic = self.vae.encode(torch.cat([x, pad], 2), device=self.dev, tiled=True, tile_size=[44, 80],
                                 tile_stride=[23, 38]).to(self.dev)
            mask = torch.ones_like(ic)
            mask[:, :, 1:] = 0
            self._cond[img_path] = {"cond_concat": torch.cat([mask[:, :4], ic], 1).to(self.dt),
                                    "visual_context": self.vae.clip.encode_video(x).to(self.dt)}
        return self._cond[img_path]

    @torch.no_grad()
    def reset(self, img_path: str, seed: int):
        from demo_utils.constant import ZERO_VAE_CACHE
        p = self.pipe
        nf = (self.L - 1) * 4 + 1
        base = self.scene_cond(img_path)
        self.cond = {"cond_concat": base["cond_concat"], "visual_context": base["visual_context"],
                     "keyboard_cond": torch.zeros(1, nf, 4, device=self.dev, dtype=self.dt),
                     "mouse_cond": torch.zeros(1, nf, 2, device=self.dev, dtype=self.dt)}
        g = torch.Generator(device=self.dev).manual_seed(seed)
        self.noise = torch.randn([1, 16, self.L, self.H // 8, self.W // 8], device=self.dev, dtype=self.dt, generator=g)
        p.kv_cache1 = p.kv_cache_keyboard = p.kv_cache_mouse = p.crossattn_cache = None
        p._initialize_kv_cache(batch_size=1, dtype=self.dt, device=self.dev)
        p._initialize_kv_cache_mouse_and_keyboard(batch_size=1, dtype=self.dt, device=self.dev)
        p._initialize_crossattn_cache(batch_size=1, dtype=self.dt, device=self.dev)
        self.vae_cache = [None] * len(copy.deepcopy(ZERO_VAE_CACHE))
        self.csf = 0

    def can_continue(self):
        return self.csf + NFB <= self.L

    @torch.no_grad()
    def block(self, key):
        """one native block under `key` (None = no movement) -> (gen_s, commit_s, decode_s, n_frames)."""
        from einops import rearrange

        from pipeline.causal_inference import cond_current
        p, sync = self.pipe, torch.cuda.synchronize
        kb = torch.tensor(KEYBOARD.get(key, [0, 0, 0, 0]), device=self.dev)
        replace = {"keyboard": kb, "mouse": torch.zeros(2, device=self.dev)}
        t0 = time.time()
        new_act, self.cond = cond_current(self.cond, self.csf, NFB, replace=replace, mode="universal")
        x = self.noise[:, :, self.csf:self.csf + NFB]
        steps = p.denoising_step_list
        start = self.csf * p.frame_seq_length
        caches = dict(kv_cache=p.kv_cache1, kv_cache_mouse=p.kv_cache_mouse, kv_cache_keyboard=p.kv_cache_keyboard,
                      crossattn_cache=p.crossattn_cache, current_start=start)
        for i, t in enumerate(steps):
            ts = torch.ones([1, NFB], device=self.dev, dtype=torch.int64) * t
            _, den = p.generator(noisy_image_or_video=x, conditional_dict=new_act, timestep=ts, **caches)
            if i < len(steps) - 1:
                flat = rearrange(den, "b c f h w -> (b f) c h w")
                x = p.scheduler.add_noise(flat, torch.randn_like(flat),
                                          steps[i + 1] * torch.ones([NFB], device=self.dev, dtype=torch.long))
                x = rearrange(x, "(b f) c h w -> b c f h w", b=den.shape[0])
        sync()
        t1 = time.time()
        p.generator(noisy_image_or_video=den, conditional_dict=new_act,
                    timestep=torch.ones_like(ts) * self.cfg.context_noise, **caches)  # KV update (commit)
        sync()
        t2 = time.time()
        video, self.vae_cache = p.vae_decoder(den.transpose(1, 2).half(), *self.vae_cache)
        sync()
        t3 = time.time()
        if self.keep_video:
            self.last_video = video
        self.csf += NFB
        return t1 - t0, t2 - t1, t3 - t2, int(video.shape[1])


def make(seconds: float, size: tuple[int, int] | None = None) -> Adapter:
    import math
    n_lat = int(math.ceil(seconds * NATIVE_FPS / 4 / NFB) + 6) * NFB  # blocks for the trace + slack
    a = Adapter(n_lat, size=size or (352, 640))
    if size and tuple(size) != (352, 640):
        a.name = f"mg2_{size[1]}x{size[0]}"
    a.nsteps = len(a.pipe.denoising_step_list)
    return a
