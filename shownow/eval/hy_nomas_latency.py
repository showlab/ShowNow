"""Hy-Nomas (Tencent-Hunyuan, 8B, text + camera) under the external-model E1 protocol.

The released interactive path is reused as in the official Reactor integration (reactor/hy_nomas_backend.py):
CPTCausalInferencePipeline with preserve_kv_cache=True, one native segment per call, stable-prefix PixelUMM decoding,
keyboard cameras from reactor/hy_nomas_camera.plan_keyboard (W/A/S/D = ids 1-4, translation 1.0, rotation 2.5).
Reactor targets SM100 (FA4) / SM89; on H100 (SM90) the native magi backend is used and the SM100-only FA4 kernels and
cache staging are not installed. Everything else follows reactor/hy_nomas{,_fast}.yaml.

Protocol = shownow/eval/ext_latency.py (reactive pacing, no speculation, as deployed): the held key is read when
<= lead frames remain on screen, lead from the measured unit cost; display at the native 24 fps; same Poisson traces,
scoring and row format. A trace starts a fresh world from the example text prompt (image initialization is not exposed
by the released path); its first, text-only segment is generated and decoded but not shown (as ext_latency's untimed
first unit). The unit cost includes text conditioning, denoising and the stable-prefix decode of the segment
(reported as gen_s; commit_s = decode_s = 0).

  torchrun --nproc_per_node 8 -m shownow.eval.hy_nomas_latency --preset main --segment-frames 120 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import shownow.eval.e1_protocol as lh  # noqa: E402

PRESETS = {"main": {"denoising_steps": 8, "history_frames": None},  # reactor/hy_nomas.yaml
           "fast": {"denoising_steps": 4, "history_frames": 36}}    # reactor/hy_nomas_fast.yaml
KEY_NAMES = {1: ("w",), 2: ("a",), 3: ("s",), 4: ("d",)}


class Backend:
    """reactor/hy_nomas_backend.HyNomasBackend for SM90 (magi backend, no FA4 / SM100 cache staging)."""

    def __init__(self, source, dit, decoder, text_encoder, steps, history, rank, world):
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(source / "reactor"))
        os.chdir(source)
        torch.set_grad_enabled(False)
        torch.cuda.set_device(rank)
        from configs.config import _flatten_cfg
        from hydra import compose, initialize_config_dir
        from hymm.distributed.parallel_states import ParallelDims, initialize_sequence_parallel_state_v2
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
        self.rank, self.steps, self.installed = rank, steps, []
        ov = ["+experiment=inference", "model.dit.seg_ar_attention_backend=magi", "inference.torch_compile=true",
              "model.vae.compile_decoder=false", f"parallelism.sp_size={world}", f"dit_weight={dit}",
              f"model.vae.checkpoint={decoder}", f"model.text_encoder.model_name={text_encoder}",
              "model.vae.stable_prefix_streaming=true", "inference.seg_ar_segment_vae_decode=true",
              "parallelism.text_encoder_cpu_offload=false", "parallelism.dit_cpu_offload=false",
              "parallelism.dit_cpu_offload_for_decode=false", "model.vae.cpu_offload=false"]
        if world == 1:
            ov += ["model.vae.parallel_mode=none", "parallelism.shard_text_encoder=false"]
        if history is not None:
            ov += ["inference.sliding_window.enabled=true", f"inference.sliding_window.past_frames={history}",
                   "inference.sliding_window.sampling=recent"]
        with initialize_config_dir(config_dir=str(source / "configs"), version_base="1.3"):
            config = _flatten_cfg(compose(config_name="config", overrides=ov))
        mesh = None
        if world > 1:
            par = ParallelDims(dp_replicate=1, dp_shard=1, sp=world, tp=1, pp=1, world_size=world)
            mesh = par.build_mesh(device_type="cuda")
            initialize_sequence_parallel_state_v2(par, mesh, False)
        self.pipeline = CPTCausalInferencePipeline(args=config, device=torch.device("cuda", rank),
                                                   dit_model_path=str(dit), world_mesh=mesh)
        # Reactor's portable deployment helpers; each is optional on SM90 and recorded when installed.
        for name, fn in (("qk_fusion", lambda: __import__("hy_nomas_fusions").install_qk_fusion(self.pipeline.vae.decoder)),
                         ("window_batching", lambda: __import__("hy_nomas_infra").install_window_batching(
                             self.pipeline.vae, 3, quantized=False)),
                         ("gpu_output", lambda: __import__("hy_nomas_infra").install_gpu_output(self.pipeline))):
            try:
                fn()
                self.installed.append(name)
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    print(f"[hy_nomas] {name} not installed: {e!r}", flush=True)
        self.exchange = None
        if world > 1:
            try:
                from hy_nomas_exchange import SequenceExchange
                self.exchange = SequenceExchange(packed_qkv=True)
                self.installed.append("sequence_exchange")
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    print(f"[hy_nomas] sequence exchange not installed: {e!r}", flush=True)
        for layer in self.pipeline.vae.decoder.layers:
            layer.compile(fullgraph=True, dynamic=False)
        from functools import lru_cache
        self.text = lru_cache(maxsize=2)(self.pipeline._encode_single_prompt)
        self.pipeline._encode_single_prompt = self.text
        self.scratch = Path(f"/tmp/hy_nomas_cam_{rank}.npz")
        self.seed, self.seg, self.prompt, self.last_prompt, self.generated, self.emitted = 42, 120, "", None, 0, 0

    def reset(self, seed, seg, prompt):
        p = self.pipeline
        if self.generated:
            p._finish_interactive_decode_phase()
        if self.exchange is not None:
            self.exchange.reset()
        self.text.cache_clear()
        p.kv_cache, p.cache_mask_list, p.cache_valid_len, p.global_frame_counter = None, None, 0, 0
        p.segment_token_lengths = []
        p._interactive_cached_action_tails, p._interactive_cached_seg_video_ranges = [], []
        p._interactive_segment_count, p._interactive_segment_cache_tokens = 0, []
        p._interactive_segment_latent_lengths = []
        p._reset_interactive_noise_state()
        p._reset_interactive_pixelumm_decode_state()
        self.seed, self.seg, self.prompt, self.last_prompt, self.generated, self.emitted = seed, seg, prompt, None, 0, 0

    def step(self, key):
        """one native segment under the held key (None = text only) -> number of newly stable frames (rank 0)."""
        from hy_nomas_camera import plan_keyboard
        from hy_nomas_prompts import shape_prompt
        cam = None
        if key is not None:
            poses, intr = plan_keyboard(KEY_NAMES[key], self.seg)
            np.savez(self.scratch, c2ws=poses, intrinsics=intr)
            cam = str(self.scratch)
        if self.exchange is not None:
            self.exchange.begin_step()
        out = self.pipeline.inference(
            prompts=[shape_prompt(self.prompt, self.pipeline.args,
                                  camera_continuation=cam is not None and self.prompt == self.last_prompt)],
            num_frames=self.seg, height=720, width=1280, guidance_scale=1.0, camera_guidance_scale=1.0,
            num_inference_steps=self.steps, flow_shift=5.0, seed=self.seed,
            negative_prompt=self.pipeline.args.inference.neg_prompt, segment_frame_boundaries=[0, self.seg],
            camera_trajectory=cam, segment_has_camera=[cam is not None], preserve_kv_cache=True,
            trajectory_translation_scale=1.0, trajectory_rotation_scale=2.5)
        self.generated += self.seg
        self.last_prompt = self.prompt
        if self.rank != 0:
            return 0
        meta = self.pipeline._last_pixelumm_streaming_metadata
        start, stable_end = int(meta["replace_pixel_start"]), int(meta["stable_pixel_frames"])
        assert start == self.emitted, (start, self.emitted)
        n = max(0, min(stable_end - start, int(out.shape[2])))
        self.emitted += n
        return n


class Adapter:
    """rank-0 view used by the ext_latency loop; every call is mirrored on all ranks by broadcast."""
    name = "hy_nomas"

    def __init__(self, be, prompt, seg, steps, seconds):
        self.be, self.prompt, self.seg, self.nsteps, self.seconds = be, prompt, seg, steps, seconds
        self.frames_per_unit = seg

    def _cmd(self, op, *a):
        obj = [(op, a)]
        dist.broadcast_object_list(obj, src=0)
        return run(self.be, obj[0], self.prompt, self.seg)

    def reset(self, img, seed):
        self._cmd("reset", seed)

    def block(self, key):
        torch.cuda.synchronize()
        t0 = time.time()
        nf = self._cmd("step", key)
        torch.cuda.synchronize()
        return time.time() - t0, 0.0, 0.0, nf

    def can_continue(self):
        return self.be.generated + self.seg <= 2880  # MAX_WORLD_FRAMES of the released runtime

    def stop(self):
        dist.broadcast_object_list([("stop", ())], src=0)


def run(be, cmd, prompt, seg):
    op, a = cmd
    if op == "reset":
        be.reset(a[0], seg, prompt)
        return 0
    if op == "step":
        return be.step(a[0])
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="main")
    ap.add_argument("--segment-frames", type=int, default=120, help="48-240, multiple of 24 (released range)")
    ap.add_argument("--source", default="/workspace/Hy-Nomas")
    ap.add_argument("--weights", default=str(REPO / "models/hy_nomas"))
    ap.add_argument("--text-encoder", default=str(REPO / "models/qwen2.5-vl-7b"))
    ap.add_argument("--traces", default="poisson0.5,poisson1,poisson2,poisson4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=4, help="timed-for-cost units after the untimed first segment")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    pre = PRESETS[args.preset]
    w = Path(args.weights)
    be = Backend(Path(args.source), w / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
                 w / "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors", args.text_encoder,
                 pre["denoising_steps"], pre["history_frames"], rank, world)
    prompt = json.dumps(json.load(open(w / "examples/prompts.json"))[0])
    if rank != 0:
        while True:
            obj = [None]
            dist.broadcast_object_list(obj, src=0)
            if obj[0][0] == "stop":
                break
            run(be, obj[0], prompt, args.segment_frames)
        dist.destroy_process_group()
        return
    from shownow.eval.ext_latency import run_trace
    fps = 24.0
    lh.FPS, lh.FRAME_S = fps, 1.0 / fps
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    m = Adapter(be, prompt, args.segment_frames, pre["denoising_steps"], args.seconds)
    m.name = f"hy_nomas_{args.preset}_seg{args.segment_frames}"
    src_rev = os.popen(f"git -C {args.source} rev-parse HEAD 2>/dev/null").read().strip()
    (out / "args.json").write_text(json.dumps(dict(vars(args), fps_used=fps, node=socket.gethostname(), world=world,
                                                   gpu=torch.cuda.get_device_name(0), installed=be.installed,
                                                   source_revision=src_rev, preset_values=pre), indent=1))
    m.reset(None, 0)
    m.block(None)
    costs = [m.block(1 + i % 4) for i in range(args.warmup)]
    c = float(np.median([x[0] for x in costs[1:]]))
    rl = math.ceil(c / lh.FRAME_S) + 1
    cost = {"gen_unit": c, "commit_unit": 0.0, "decode_unit": 0.0, "frames_per_unit": int(costs[-1][3]),
            "fps": fps, "nsteps": pre["denoising_steps"], "unit_play_s": costs[-1][3] / fps,
            "warmup_unit_s": [x[0] for x in costs], "reactive_lead_frames": rl}
    print("COST", json.dumps(cost), flush=True)
    (out / "cost.json").write_text(json.dumps(cost, indent=1))
    rows = open(out / "rows.jsonl", "a")
    for tr in args.traces.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            trace = lh.make_trace(tr, seed, args.seconds)
            steps, disp, mem = run_trace(m, trace, None, seed, args.seconds, rl)
            pr = lh.score(trace, steps, args.seconds)
            held_s = disp.held_ticks * lh.FRAME_S
            rec = {"policy": m.name, "fps": lh.FPS, "lead_frames": rl, "lead_ms": rl * lh.FRAME_S * 1000,
                   "nsteps": pre["denoising_steps"], "trace": tr, "seed": seed, "scene": "forest_world",
                   "presses": pr, "trace_events": trace, "stall_times": disp.stall_times, "n_steps": len(steps),
                   "frames_played": len(disp.played), "stall_s": disp.stall_ticks * lh.FRAME_S, "held_s": held_s,
                   "fps_while_held": (disp.held_ticks - disp.stall_ticks) / max(1e-6, held_s),
                   "late_steps": sum(bool(s["late"]) for s in steps), "n_swaps": 0,
                   "gen_s_p50": float(np.median([s["gen_s"] for s in steps])) if steps else None,
                   "commit_s_p50": 0.0, "decode_s_p50": 0.0, "peak_mem_gib": mem, "n_gen": len(steps),
                   "gpu_busy_s": float(sum(s["gen_s"] for s in steps)), "timeline": steps}
            rows.write(json.dumps(rec) + "\n")
            rows.flush()
            srv = [p for p in pr if p["status"] == "served"]
            print(json.dumps({"policy": m.name, "trace": tr, "seed": seed, "served": len(srv), "presses": len(pr),
                              "no_vae_p50": float(np.median([p["lat_no_vae"] for p in srv])) if srv else None,
                              "stall_s": rec["stall_s"], "fps": round(rec["fps_while_held"], 2)}), flush=True)
    m.stop()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
