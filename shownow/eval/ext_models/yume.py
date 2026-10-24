"""Yume-1.5 (stdstu12/YUME, Yume-5B-720P on Wan2.2-TI2V-5B) adapter for ext_latency (native code in third_party/YUME,
weights models/Yume-5B-720P, env .venv-yume).

Driven exactly through its single-GPU web app's own continuation API: webapp_single_gpu.load_wan(), then one
long_generate(LongGenArgs(...)) call per chunk with sample_num=1 and continue_from_last=True (the first call starts
from the E1 scene image, I2V). The held key becomes its camera_movement1 (W / A / S / D, "None" when released; the web
app turns it into the text "The camera pushes forward (W)." etc., re-encoded by T5 every call), camera_movement2 '·'.
Web-app defaults (16 fps, frame_zero 32 = 8 latents per chunk, shift 5.0, 704x1280) except sampling steps: 4 (the
fewest its README allows; web-app default 50; the repo has no real-time configuration). No KV cache: every chunk
re-encodes the whole (multi-rate compressed) history, so chunks get slower as the video grows. _postprocess_video (the
per-call mp4 writer) is replaced by a frame counter. gen_s = the whole long_generate call (text encode + denoise +
decode of the chunk's 8 latents).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
CODE_DIR = REPO / "third_party/YUME"
WEIGHTS = REPO / "models/Yume-5B-720P"
NATIVE_FPS = 16.0
PROMPT = "A first-person view walking through the dark corridors and rooms of a retro first-person shooter game level."
KEYS = {1: "W", 2: "A", 3: "S", 4: "D"}


class Adapter:
    name = "yume_native"
    frames_per_unit = 32
    nsteps = 4

    def __init__(self):
        import webapp_single_gpu as W
        W.CKPT_DIR = str(WEIGHTS)
        self.W = W
        self.out_dir = REPO / "runs/latency_e1/ext/yume_scratch"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._nf = 0

        def count_frames(video, fps, out_path):
            self._nf = int(video.shape[1])

        W._postprocess_video = count_frames
        W.load_wan()

    def _args(self, key, first):
        return self.W.LongGenArgs(
            prompt=PROMPT, jpg_path=self.img if first else None, output_dir=str(self.out_dir), fps=16,
            sample_steps=4, sample_num=1, frame_zero=32, shift=5.0, seed=self.seed, continue_from_last=not first,
            refine_from_image=False, caption_path=None, mode="I2V", resolution="704x1280", memory_optimization=False,
            vae_memory_optimization=False, camera_movement1=KEYS.get(key, "None"), camera_movement2="·")

    def reset(self, img_path: str, seed: int):
        self.img, self.seed, self.first = img_path, seed, True
        self.W.LAST["last_model_input_de"] = None

    def can_continue(self):
        return True

    @torch.no_grad()
    def block(self, key):
        torch.cuda.synchronize()
        t0 = time.time()
        self.W.long_generate(self._args(key, self.first))
        torch.cuda.synchronize()
        self.first = False
        return time.time() - t0, 0.0, 0.0, self._nf


def make(seconds: float) -> Adapter:
    return Adapter()
