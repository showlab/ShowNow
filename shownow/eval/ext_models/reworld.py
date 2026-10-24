"""ReWorld-5B adapter for ext_latency (Wan2.2-TI2V-5B based, keyboard + mouse, CC BY-NC-SA 4.0; native code in
third_party/ReWorld, weights models/ReWorld-5B + models/Wan2.2-TI2V-5B, env .venv-reworld).

Generation runs THEIR interactive path unchanged in a worker thread: pipeline.inference() of
CausalDiffusionInferenceCompressedPipeline with the real-time settings of their docs (configs/plucker720p_dmd_infer.yaml,
704x1280, 4-step DMD LoRA, CFG off, KV policy v15b: budget 12 chunks = 1 sink + 5 recent + 6 landmarks from a 30-entry
pose-indexed bank; chunks of 4 latents = 16 frames, 13 for the first). Per chunk it reads mouse + keyboard with two
input() calls in utils/camera.py; that module's `input` is replaced by a hand-shake with ext_latency (mouse 'u' = no
rotation, keys W / A / S / D -> w / a / s / d). _call_generator_step33 (the clean t = 0 rerun that writes the chunk's
K/V) and _compress_post_step33 (landmark bank / cache rewrite) are wrapped only to time them and to capture the chunk's
clean latents.
Decoding: their code decodes the whole rollout once at the end (the 5B VAE wrapper has no cached / streaming decode), so
for a streaming display the adapter decodes each finished chunk with the SAME Wan2.2 VAE decoder loop
(wan_5b/modules/vae2_2.py WanVAE_.decode) but keeps its causal feature cache across chunks (first_chunk only for latent
0) - identical math to their final full decode, split per chunk. Display 24 fps (their write_video fps).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
CODE_DIR = REPO / "third_party/ReWorld"
WEIGHTS = REPO / "models/ReWorld-5B"
WAN_DIR = REPO / "models/Wan2.2-TI2V-5B"
NATIVE_FPS = 24.0
PROMPT = "A first-person view walking through the dark corridors and rooms of a retro first-person shooter game level."
KEYS = {1: "w", 2: "a", 3: "s", 4: "d"}
_STOP = object()


class _Stop(Exception):
    pass


class Adapter:
    name = "reworld_native"
    frames_per_unit = 16
    nsteps = 4

    def __init__(self, n_latents: int):
        import os
        os.environ.setdefault("WAN_MODEL_DIR", str(WAN_DIR))
        import inference_action as IA
        import utils.camera as cam
        from omegaconf import OmegaConf

        sys.argv = ["inference_action.py"]  # their module parses nothing at import; keep argv clean
        config = IA.load_config(str(CODE_DIR / "configs/plucker720p_dmd_infer.yaml"))
        config.model_kwargs.model_name = str(WAN_DIR)
        shape = list(config.image_or_video_shape)
        shape[1] = n_latents
        config.image_or_video_shape = shape
        kvc = config.kv_compression if "kv_compression" in config else OmegaConf.create({})
        kvc.enabled, kvc.policy, kvc.budget_chunks, kvc.n_sink, kvc.recent_w = True, "v15b", 12, 1, 5
        config.kv_compression = kvc
        self.config, self.dev = config, torch.device("cuda:0")
        gen, te, vae = IA.build_models(config, self.dev)
        IA.load_checkpoint(gen, str(WEIGHTS / "reworld_5b_ar_ema.pt"))
        IA.load_lora_checkpoint(gen, config, str(WEIGHTS / "reworld_5b_dmd_lora.pt"))
        pipe = IA.CausalDiffusionInferenceCompressedPipeline(args=config, device=self.dev, generator=gen,
                                                             text_encoder=te, vae=vae).to(dtype=torch.bfloat16)
        pipe.text_encoder.to(device=self.dev)
        pipe.generator.to(device=self.dev)
        pipe.vae.to(device=self.dev)
        self.pipe, self.IA = pipe, IA
        cam.input = self._input_hook  # shadows the builtin inside utils/camera.py only
        s33, post = pipe._call_generator_step33, pipe._compress_post_step33

        def step33(latents, *a, **k):
            torch.cuda.synchronize()
            self._t33 = time.time()
            self._chunk_lat = latents.detach().clone()
            return s33(latents, *a, **k)

        def post33(*a, **k):
            r = post(*a, **k)
            torch.cuda.synchronize()
            self._t_post = time.time()
            return r

        pipe._call_generator_step33, pipe._compress_post_step33 = step33, post33
        self.n_lat = n_latents
        self.req, self.res, self.worker, self._t_start, self._pending_key = queue.Queue(), queue.Queue(), None, None, None

    # ------------------------------------------------------------------ streaming decode (their decoder loop, cache kept)
    @torch.no_grad()
    def _decode_chunk(self, lat):
        """lat [1, F, C, h, w] clean latents of one chunk -> number of pixel frames (frames stay on the GPU)."""
        from wan_5b.modules.vae2_2 import unpatchify
        w = self.pipe.vae
        m = w.model
        z = lat.permute(0, 2, 1, 3, 4)
        mean, inv_std = w.mean.to(z.device, z.dtype), 1.0 / w.std.to(z.device, z.dtype)
        z = z / inv_std.view(1, m.z_dim, 1, 1, 1) + mean.view(1, m.z_dim, 1, 1, 1)
        x = m.conv2(z)
        outs = []
        for i in range(x.shape[2]):
            m._conv_idx = [0]
            outs.append(m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx,
                                  first_chunk=self._first_decode))
            self._first_decode = False
        out = unpatchify(torch.cat(outs, 2), patch_size=2).float().clamp_(-1, 1)
        torch.cuda.synchronize()
        return int(out.shape[2])

    # ------------------------------------------------------------------ worker side
    def _input_hook(self, prompt=""):
        if "keyboard" in prompt.lower():
            return self._pending_key
        # mouse prompt = start of a chunk: hand back the finished chunk (decoded here, streaming), wait for the next key
        if self._t_start is not None:
            t0 = time.time()
            nf = self._decode_chunk(self._chunk_lat)
            dec_s = time.time() - t0
            self.res.put((self._t33 - self._t_start, self._t_post - self._t33, dec_s, nf))
        else:
            self.res.put("ready")
        key = self.req.get()
        if key is _STOP:
            raise _Stop
        self._pending_key = KEYS.get(key, "q")
        self._t_start = time.time()
        return "u"

    def _run(self, seed):
        from utils.control_manager import ControlSignalManager
        cfg = self.config
        try:
            p = cfg.eval_datasets[0].params
            ctrl = ControlSignalManager(action_mode=p.get("target_action_mode", "plucker_720p"),
                                        chunk_size=cfg.num_frame_per_block, device=str(self.dev), dtype=torch.bfloat16,
                                        delta_t=p.get("delta_t", 0.35), delta_r=p.get("delta_r", 0.0625),
                                        prope_fixed_divisor=p.get("prope_fixed_divisor", None))
            shape = list(cfg.image_or_video_shape)
            g = torch.Generator(device=self.dev).manual_seed(seed)
            noise = torch.randn([1] + shape[1:], device=self.dev, dtype=torch.bfloat16, generator=g)
            m = self.pipe.vae.model
            m.clear_cache()
            self._first_decode = True
            with torch.no_grad():
                self.pipe.inference(noise=noise, text_prompts=[PROMPT], control_manager=ctrl, return_latents=True,
                                    num_inference_steps=4)
        except _Stop:
            pass
        except Exception as e:
            self.res.put(e)

    # ------------------------------------------------------------------ ext_latency side
    def reset(self, img_path: str, seed: int):
        """text-to-world (their interactive mode has no image input); img_path unused."""
        if self.worker is not None:
            self.req.put(_STOP)
            self.worker.join()
        self.req, self.res, self._t_start = queue.Queue(), queue.Queue(), None
        self.worker = threading.Thread(target=self._run, args=(seed,), daemon=True)
        self.worker.start()
        r = self.res.get()
        if isinstance(r, Exception):
            raise r
        self.n_done = 0

    def can_continue(self):
        return self.n_done + 1 < self.n_lat // 4

    def block(self, key):
        self.req.put(key)
        r = self.res.get()
        if isinstance(r, Exception):
            raise r
        self.n_done += 1
        return r


def make(seconds: float) -> Adapter:
    import math
    n_lat = (int(math.ceil(seconds * NATIVE_FPS / 16)) + 8) * 4
    return Adapter(n_lat)
