"""Matrix-Game 3.0 adapter for ext_latency (Skywork, arXiv 2604.08995; Wan2.2-TI2V-5B based, native code in
third_party/Matrix-Game-3, distilled weights models/Matrix-Game-3.0, env .venv-mg3).

Its interactive inference is ONE function, MatrixGame3Pipeline.generate() (pipeline/inference_interactive_pipeline.py),
that loops over chunks and reads the next action with a blocking get_current_action() at the start of every chunk. We
run that function UNCHANGED in a worker thread and only replace three module-level hooks:
  get_current_action  -> hand the finished chunk back to ext_latency and wait for its next key (W / A / S / D -> its
                         keyboard one-hot [w, s, a, d, 0, 0], mouse 0)
  vae.stream_decode   -> the same call, timed (CUDA synchronised) and its frame count recorded
  process_video       -> no-op (it writes an mp4 per chunk, not part of a streaming deployment)
Native real-time settings of their test.sh on ONE GPU: 704x1280, 3 denoising steps, INT8 DiT linears, FlashAttention 3,
MG-LightVAE (pruning 0.5) with compiled decoder; no async VAE (that needs an extra GPU). Chunks: the first generates 57
frames, then 40 frames each (10 new latents, 4 re-injected, 5 memory latents, no KV cache). Display 16 fps
(wan shared config sample_fps).

Generation size (`size=(H, W)`, default the native 704x1280): their own --size / max_area arguments (generate() sizes
the latent grid from max_area and the image aspect, interpolates the input image to H x W and builds the camera
intrinsics for it); MAX_AREA_CONFIGS only lists 704*1280, so max_area is passed as H * W. The model was trained at
704x1280.
"""

from __future__ import annotations

import queue
import threading
import time
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
CODE_DIR = REPO / "third_party/Matrix-Game-3"
WEIGHTS = REPO / "models/Matrix-Game-3.0"
NATIVE_FPS = 16.0
PROMPT = "A first-person view walking through the dark corridors and rooms of a retro first-person shooter game level."
KEYBOARD = {1: [1, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0], 3: [0, 1, 0, 0, 0, 0], 4: [0, 0, 0, 1, 0, 0]}  # ours W A S D
_STOP = object()


class _Stop(Exception):
    pass


class Adapter:
    name = "mg3_native"
    frames_per_unit = 40
    nsteps = 3

    keep_video = False  # validation: keep the last decoded chunk in self.last_video

    def __init__(self, ulysses: int = 1, size: tuple[int, int] = (704, 1280)):
        import pipeline.inference_interactive_pipeline as IP
        from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

        self.IP, self.cfg = IP, WAN_CONFIGS["matrix_game3"]
        self.args = types.SimpleNamespace(
            size=f"{size[0]}*{size[1]}", ckpt_dir=str(WEIGHTS), ulysses_size=ulysses, t5_fsdp=False, t5_cpu=False, dit_fsdp=False,
            prompt=PROMPT, seed=0, image=None, num_iterations=100000, convert_model_dtype=False,
            output_dir=str(REPO / "runs/latency_e1/ext/mg3_scratch"), save_name="ext", sample_shift=self.cfg.sample_shift,
            sample_guide_scale=5.0, num_inference_steps=3, lightvae_pruning_rate=0.5, use_async_vae=False,
            async_vae_warmup_iters=0, compile_vae=True, vae_type="mg_lightvae", use_int8=True, verify_quant=False,
            fa_version="3", interactive=True, use_base_model=False)
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        self.max_area = MAX_AREA_CONFIGS.get(self.args.size, size[0] * size[1])
        self.last_video = None
        IP.get_current_action = self._action_hook
        IP.process_video = lambda *a, **k: None
        self.pipe = IP.MatrixGame3Pipeline(config=self.cfg, checkpoint_dir=self.args.ckpt_dir, device_id=0, rank=0,
                                           t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
                                           convert_model_dtype=False, args=self.args, fa_version="3",
                                           use_base_model=False)
        dec = self.pipe.vae.stream_decode

        def timed_decode(*a, **k):
            torch.cuda.synchronize()
            t = time.time()
            video, cache = dec(*a, **k)
            torch.cuda.synchronize()
            self._dec = (t, time.time() - t, int(video.shape[2]))
            if self.keep_video:
                self.last_video = video
            return video, cache

        self.pipe.vae.stream_decode = timed_decode
        self.req, self.res, self.worker, self._t_start = queue.Queue(), queue.Queue(), None, None

    # ------------------------------------------------------------------ worker side (inside their generate())
    def _action_hook(self):
        if self._t_start is not None:  # the chunk started at _t_start has just been decoded
            t_dec, dec_s, nf = self._dec
            self.res.put((t_dec - self._t_start, 0.0, dec_s, nf))
        else:
            self.res.put("ready")  # setup (text / image encoding) done, the first chunk is about to start
        key = self.req.get()
        if key is _STOP:
            raise _Stop
        self._t_start = time.time()
        return {"mouse": torch.tensor([0.0, 0.0]), "keyboard": torch.tensor(KEYBOARD.get(key, [0] * 6))}

    def _run(self, img_path, seed):
        from PIL import Image
        try:
            with torch.no_grad():
                self.pipe.generate(PROMPT, Image.open(img_path).convert("RGB"), max_area=self.max_area,
                                   shift=self.args.sample_shift, num_inference_steps=3,
                                   guide_scale=self.args.sample_guide_scale, seed=seed, use_base_model=False,
                                   args=self.args)
        except _Stop:
            pass
        except Exception as e:  # surface worker errors to the timing thread
            self.res.put(e)

    # ------------------------------------------------------------------ ext_latency side
    def reset(self, img_path: str, seed: int):
        if self.worker is not None:
            self.req.put(_STOP)
            self.worker.join()
        self.req, self.res, self._t_start = queue.Queue(), queue.Queue(), None
        self.args.image, self.args.seed = img_path, seed
        self.worker = threading.Thread(target=self._run, args=(img_path, seed), daemon=True)
        self.worker.start()
        r = self.res.get()
        if isinstance(r, Exception):
            raise r

    def can_continue(self):
        return True

    def block(self, key):
        """one native chunk under `key` -> (gen_s, commit_s, decode_s, n_frames)."""
        self.req.put(key)
        r = self.res.get()
        if isinstance(r, Exception):
            raise r
        return r


def make(seconds: float, size: tuple[int, int] | None = None) -> Adapter:
    a = Adapter(size=size or (704, 1280))
    if size and tuple(size) != (704, 1280):
        a.name = f"mg3_{size[1]}x{size[0]}"
    return a  # single GPU (their multi-GPU sequence-parallel mode needs torchrun ranks, not wrapped here)
