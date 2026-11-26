"""Generate an eight-future rollout from the public inference package on one GPU.

python -m shownow.demo.generate --out runs/demo
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SHOWNOW_N_ACTIONS", "9")
os.environ.setdefault("SHOWNOW_OPT_DYNAMIC", "0")

import cv2
import numpy as np
import torch

from shownow.demo.play_server import Engine, REPO
from shownow.latency.tree_opt import RTStepper, parse_opts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=REPO / "checkpoints/ShowNow-5B-8F")
    ap.add_argument("--out", type=Path, default=REPO / "runs/demo")
    ap.add_argument("--steps", type=int, default=32, help="Generated latents; four frames each")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys", default="1,2,3,4,5,6,7,8", help="Chosen actions; each held for four latents")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("--steps must be positive")
    keys = [int(k) for k in args.keys.split(",")]
    if not keys or any(k not in range(1, 9) for k in keys):
        ap.error("--keys must contain action IDs 1..8")
    config = json.loads((args.model_dir / "inference_config.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = argparse.Namespace(student=str(args.model_dir / config['weights']),
        scene_dir=str(args.model_dir / 'examples/lat'), history=config['history_latents'],
        denoising_steps=','.join(map(str, config['denoising_steps'])),
        cam_scale=config['camera_scale'], block=1, device='cuda', jpeg_quality=95, data='sw8a')
    engine = Engine(cfg, None, None)
    scene = engine.load_scene(engine.scenes[args.scene])
    stepper = RTStepper(engine.m, 8, engine.fewstep, engine.sig,
                        P=cfg.history, opts=parse_opts('exact'), device=engine.dev)
    stepper.set_text(scene['ctx'])
    stepper.load_ring(scene['committed'])
    stepper.set_inputs(engine.cams_for([[scene['poses'][-1]]] * 8), list(range(1, 9)))
    print('Capturing optimized eight-future CUDA graphs...', flush=True)
    stepper.capture()
    stepper.load_ring(scene['committed'])
    engine.tae.reset()
    engine.dec_push(scene['lat'])  # prime the streaming decoder with the observed prefix
    frames, timings, chosen = [], [], []
    for i in range(args.steps):
        cams = [torch.from_numpy(scene['stream'].peek(k)).float() for k in range(1, 9)]
        stepper.set_inputs(engine.cams_for([[c] for c in cams]), list(range(1, 9)))
        torch.cuda.synchronize()
        start = time.perf_counter()
        stepper.replay_gen()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        assert torch.isfinite(stepper.x).all(), f'Non-finite candidate at step {i}'
        key = keys[(i // 4) % len(keys)]
        lat = stepper.x[:, key-1:key].clone()
        stepper.replay_choose(key-1)
        scene['stream'].commit(key)
        frames.extend(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
                      for j in engine.dec_push(lat))
        chosen.append(key)
        print(f'{i+1}/{args.steps}: eight candidates, select {key}, {timings[-1]:.1f} ms', flush=True)
    assert len(frames) == args.steps * 4, (len(frames), args.steps)
    video = args.out / 'rollout.mp4'
    import imageio.v2 as iio
    with iio.get_writer(video, fps=config['fps'], codec='libx264', macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(args.out / 'first_frame.jpg'), frames[0])
    report = dict(model='ShowNow-5B-8F', gpu=torch.cuda.get_device_name(), seed=args.seed,
        scene=Path(engine.scenes[args.scene]).stem, candidates_per_step=8,
        steps=args.steps, frames=len(frames), width=config['width'], height=config['height'],
        fps=config['fps'], chosen_actions=chosen, finite_candidates=True,
        generation_ms=timings, peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
        note='Release smoke test. Generation timings exclude display scheduling and are not input latency.')
    (args.out / 'smoke_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved {video}', flush=True)


if __name__ == '__main__':
    main()
