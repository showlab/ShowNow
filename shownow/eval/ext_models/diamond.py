"""DIAMOND (CS:GO world model, Alonso et al., NeurIPS 2024; csgo branch, native code in third_party/diamond, weights
models/diamond/csgo from HF eloialonso/diamond, env .venv-diamond) adapter for ext_latency.

Native streaming schedule (play.py / envs.WorldModelEnv, unchanged math): one frame per step in pixel space; the held
key is written into the last action slot, the next 30x56 frame is denoised from the 4 previous frames + actions
(config world_model_env/fast.yaml: 1 Euler step), then upsampled to 150x280 by the second diffusion model (1 step);
buffers roll. Optional torch.compile of both samplers (play.py --compile) is ON (fastest native setting). Keys
W / A / S / D -> its keys w / a / s / d, mouse 0, no clicks. Start = its own CS:GO spawn (it has no encoder for other
frames), cycled by seed. Native display rate 15 fps (play.py). No decoder / no cache commit: gen = both samplers.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
CODE_DIR = REPO / "third_party/diamond/src"
WEIGHTS = REPO / "models/diamond/csgo"
NATIVE_FPS = 15.0
KEY_IDX = {1: 0, 2: 1, 3: 2, 4: 3}  # ours W A S D -> DIAMOND keys w a s d


class Adapter:
    name = "diamond_native"
    frames_per_unit = 1
    nsteps = 2  # 1 denoising + 1 upsampling step

    def __init__(self):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from agent import Agent
        from csgo.action_processing import MOUSE_X_POSSIBLES, MOUSE_Y_POSSIBLES, N_KEYS
        from envs import WorldModelEnv

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        with initialize_config_dir(version_base="1.3", config_dir=str(CODE_DIR.parent / "config")):
            cfg = compose(config_name="trainer")
        cfg.agent = OmegaConf.load(WEIGHTS / "config/agent/csgo.yaml")
        cfg.env = OmegaConf.load(WEIGHTS / "config/env/csgo.yaml")
        self.dev = torch.device("cuda:0")
        agent = Agent(instantiate(cfg.agent, num_actions=cfg.env.num_actions)).to(self.dev).eval()
        agent.load(WEIGHTS / "model/csgo.pt")
        sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
        if agent.upsampler is not None:
            sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
        wm_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
        self.wm = WorldModelEnv(agent.denoiser, agent.upsampler, agent.rew_end_model, WEIGHTS / "spawn", 1, sl, wm_cfg)
        self.wm.predict_next_obs = torch.compile(self.wm.predict_next_obs, mode="reduce-overhead")
        self.wm.upsample_next_obs = torch.compile(self.wm.upsample_next_obs, mode="reduce-overhead")
        self.n_keys, self.mx0, self.my0 = N_KEYS, MOUSE_X_POSSIBLES.index(0), MOUSE_Y_POSSIBLES.index(0)
        self.n_mx, self.n_my = len(MOUSE_X_POSSIBLES), len(MOUSE_Y_POSSIBLES)
        self.spawn_seed = None

    def action(self, key):
        """51-dim encode_csgo_action layout: keys(11) | l_click | r_click | mouse_x one-hot | mouse_y one-hot."""
        a = torch.zeros(self.n_keys + 2 + self.n_mx + self.n_my, device=self.dev)
        if key in KEY_IDX:
            a[KEY_IDX[key]] = 1
        a[self.n_keys + 2 + self.mx0] = 1
        a[self.n_keys + 2 + self.n_mx + self.my0] = 1
        return a

    @torch.no_grad()
    def reset(self, img_path: str, seed: int):
        for _ in range(seed + 1 if self.spawn_seed is None else 1):  # spawn dirs cycle; one spawn per reset
            self.wm.reset()
        self.spawn_seed = seed

    def can_continue(self):
        return True

    @torch.no_grad()
    def block(self, key):
        """one native frame under `key` -> (gen_s, commit_s, decode_s, n_frames)."""
        wm, sync = self.wm, torch.cuda.synchronize
        t0 = time.time()
        wm.act_buffer[:, -1] = self.action(key)
        next_obs, _ = wm.predict_next_obs()
        full, _ = wm.upsample_next_obs(next_obs)
        wm.obs_buffer = wm.obs_buffer.roll(-1, dims=1)
        wm.act_buffer = wm.act_buffer.roll(-1, dims=1)
        wm.obs_buffer[:, -1] = next_obs
        wm.obs_full_res_buffer = wm.obs_full_res_buffer.roll(-1, dims=1)
        wm.obs_full_res_buffer[:, -1] = full
        _ = (full[0].add(1).mul(127.5).clamp(0, 255).byte()).cpu()  # display-ready frame on the host
        sync()
        return time.time() - t0, 0.0, 0.0, 1


def make(seconds: float) -> Adapter:
    return Adapter()
