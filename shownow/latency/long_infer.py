"""Long-video multi-GPU speculative inference with the Stage1 B1 student (1 latent = 0.25 s per AR step), real weights
and real content, the same AnyFlow sliding-window sampler as training/eval (history 15 latents, 4 NFE).

Ranks 0-3 = generation GPUs, one per key (W A S D). Every step all four generate the next latent from the same shared
history (same noise), each under its own key: the next frame is ready for whatever key the player presses. The
pressed key (script: a new key every --switch-sec s) picks the owner; its latent is broadcast and appended to every
rank's history. Rank 4 = streaming VAE decode of the chosen latents (cached decoder state, 4 frames per latent).
--mode single runs the non-speculative baseline: rank 0 generates only the pressed key's latent.

Reported per video (steady state): step interval (real time needs <= 0.25 s), generation / broadcast / decode time,
and key-press latency = key press -> first frame of the new action decoded. With speculation the press is served from
an already generated candidate, so latency = broadcast + decode (+ waiting for the step in flight if generation is
slower than real time); without speculation it adds the generation of the latent.

  torchrun --nproc-per-node 5 -m shownow.latency.long_infer --out runs/long_infer/b1_s1_3000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SOLARWM_COMPILE_FLEX", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from shownow.eval.camera_prior_eval import LAT  # noqa: E402
from shownow.eval.long_drift_eval import KEYS, camera_tokens_long  # noqa: E402
from shownow.train.student_common import (FRAME_TOKENS, HIST_LATENTS, allocate_caches, flowmap_schedule,  # noqa: E402
                                           tokens)

LAT_PER_SEC = 4


def key_schedule(n: int, per: int, seed: int) -> np.ndarray:
    """a new key every `per` latents, never the same key twice in a row; W twice as likely."""
    rng = np.random.default_rng(seed)
    keys, prev = [], 0
    while len(keys) < n:
        choices = [k for k in (1, 1, 2, 3, 4) if k != prev]
        prev = int(rng.choice(choices))
        keys += [prev] * per
    return np.array(keys[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(REPO / "runs/student_stage1/b1_base/student_stage1_step003000.pt"))
    ap.add_argument("--field", default="generator_ema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--switch-sec", type=float, default=2.0)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--mode", default="spec", choices=["spec", "single"])
    ap.add_argument("--fork-offset", type=int, default=0, help="first val fork index")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2],
                    help="1 = AnyFlow sliding-window sampler; 2 = SGF/DMD KV-cache rollout (4 steps, commit chosen)")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    G = 4 if args.mode == "spec" else 1
    assert world == G + 1, (world, G)
    dec_rank = G
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    is_gen = rank < G
    if is_gen and args.stage == 1:
        from shownow.train.student_stage1 import build_student
        diffusion, _ = build_student(args.student, from_teacher=False, field=args.field)
        diffusion.module.to(dev, dtype=torch.bfloat16).eval()
        inner = diffusion.module
        ts, rs = flowmap_schedule(args.nfe, device=dev)
    elif is_gen:
        from shownow.train.student_stage2 import build_stage2_student
        from solarwm.backends.wan22.sgf import warp_denoising_steps
        diffusion, _, _ = build_stage2_student(args.student, field=args.field, block=1, history=args.history)
        diffusion.module.to(dev, dtype=torch.bfloat16).eval()
        inner = diffusion.module
        steps_raw = [float(v) for v in warp_denoising_steps([1000, 750, 500, 250],
                                                            diffusion.scheduler.timesteps.to(dev),
                                                            num_train_timesteps=1000)]
    else:
        from solarwm.backends.wan22.runtime.components import Wan5BVAE
        vae = Wan5BVAE(REPO / "models/SolarWM-5B-base/vae/Wan2.2_VAE.pth").to(dev, dtype=torch.bfloat16)

    split = json.load(open(REPO / "runs/teacher_ft/wasd_v1/split.json"))
    val = sorted(split["val_ood"])
    n_fut = args.seconds * LAT_PER_SEC
    per = int(round(args.switch_sec * LAT_PER_SEC))
    C, H, W = 48, 30, 54
    lat_buf = torch.empty(1, 1, C, H, W, dtype=torch.bfloat16, device=dev)

    for vi in range(args.videos):
        sid = val[args.fork_offset + vi * 7]  # spread over maps
        fork = torch.load(LAT / f"{sid}.pt", map_location="cpu", weights_only=False)
        nominal = torch.load(REPO / "data/cf_vizdoom_10k_cam" / f"{sid}.pt", weights_only=True)["nominal"]
        keys = key_schedule(n_fut, per, 1000 + vi)
        vel = {k: (nominal[k - 1, 20, :3, 3] - nominal[k - 1, HIST_LATENTS - 1, :3, 3]) / (21 - HIST_LATENTS)
               for k in KEYS}
        hist = fork["latents"][:1, :HIST_LATENTS].to(dev)  # [1,9,C,H,W] shared real history (branch 0)
        poses = [nominal[0, i] for i in range(HIST_LATENTS)]
        rot = nominal[0, 20, :3, :3]
        acts = [int(a) for a in fork["actions"][0, :HIST_LATENTS]]
        cond = {"prompt_embeds": fork["prompt"].to(dev)[None]}
        my_key = rank + 1 if args.mode == "spec" else None
        rec = {"gen": [], "bcast": [], "interval": [], "decode": [], "latency": [], "switch_latency": [], "wait": []}
        frames = []
        if rank == dec_rank:  # warm the streaming decoder with the real history
            vae.module.clear_cache()
            z = hist.permute(0, 2, 1, 3, 4)
            frames.append(vae.module.cached_decode(z, vae._scale(z)).float().clamp(-1, 1)[0].cpu())
        if is_gen and args.stage == 2:  # fresh KV cache; commit the real history latents one by one
            kv, xa = allocate_caches(inner, 1, args.history + 1, torch.bfloat16, dev)
            with torch.no_grad():
                for i in range(HIST_LATENTS):
                    kv_call(diffusion, inner, hist[:, i:i + 1], 0.0, i, poses[i], acts[i], cond, kv, xa, args,
                            dev, "commit_detached")
        dist.barrier()
        torch.cuda.synchronize()
        t_prev = time.time()
        with torch.no_grad():
            for s in range(n_fut):
                key = int(keys[s])
                t0 = time.time()
                if is_gen:
                    k = my_key if args.mode == "spec" else key
                    pose = poses[-1].clone()
                    pose[:3, :3] = rot
                    pose[:3, 3] = poses[-1][:3, 3] + vel[k]
                    g = torch.Generator(device=dev).manual_seed(vi * 100000 + s)  # same noise on every rank
                    if args.stage == 2:
                        x = torch.randn((1, 1, C, H, W), generator=g, device=dev, dtype=torch.bfloat16)
                        start = HIST_LATENTS + s
                        for i, t in enumerate(steps_raw):
                            flow = kv_call(diffusion, inner, x, t, start, pose, k, cond, kv, xa, args, dev, "none")
                            x0 = diffusion.flow_to_x0(x, flow, torch.full((1, 1), float(t), device=dev))
                            if i < len(steps_raw) - 1:
                                nz = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32)
                                tn = torch.full((1,), float(steps_raw[i + 1]), device=dev)
                                x = diffusion.scheduler.add_noise(x0.flatten(0, 1).float(), nz.flatten(0, 1),
                                                                  tn).unflatten(0, (1, 1)).to(torch.bfloat16)
                        x = x0.to(torch.bfloat16)
                    n_h = min(args.history, hist.shape[1])
                    win_poses = torch.stack(poses[-n_h:] + [pose])[None]
                    cam = camera_tokens_long(win_poses, dev, args.cam_scale)
                    inner._action_tokens = tokens(torch.tensor([acts[-n_h:] + [k]], device=dev).float()).long()
                    if args.stage == 1:
                        x = torch.randn((1, 1, C, H, W), generator=g, device=dev, dtype=torch.bfloat16)
                    h = hist[:, -n_h:]
                    for t, r in (zip(ts, rs) if args.stage == 1 else []):
                        tt = torch.full((1, 1), float(t), device=dev)
                        rt = torch.full((1, 1), float(r), device=dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            u = diffusion.forward_inference_window(
                                x, h, cond, cam, tokens(tt), num_frame_per_block=1, sequence_length=405,
                                clean_history_timestep=0.0, r_timestep_tokens=tokens(rt))
                        x = (x.float() - (float(t) - float(r)) / 1000.0 * u.float()).to(torch.bfloat16)
                    inner._action_tokens = None
                    owner = key - 1 if args.mode == "spec" else 0
                    if rank == owner:
                        lat_buf.copy_(x)
                torch.cuda.synchronize()
                t1 = time.time()
                owner = key - 1 if args.mode == "spec" else 0
                dist.broadcast(lat_buf, src=owner)  # the key press is resolved here
                torch.cuda.synchronize()
                t2 = time.time()
                if is_gen:
                    pose = poses[-1].clone()
                    pose[:3, :3] = rot
                    pose[:3, 3] = poses[-1][:3, 3] + vel[key]
                    poses.append(pose)
                    acts.append(key)
                    hist = torch.cat([hist, lat_buf.clone()], 1)[:, -args.history:]
                    if args.stage == 2:  # chosen latent -> KV cache (clean, t = 0)
                        kv_call(diffusion, inner, lat_buf, 0.0, HIST_LATENTS + s, pose, key, cond, kv, xa, args,
                                dev, "commit_detached")
                if rank == dec_rank:
                    z = lat_buf.permute(0, 2, 1, 3, 4)
                    frames.append(vae.module.cached_decode(z, vae._scale(z)).float().clamp(-1, 1)[0].cpu())
                    torch.cuda.synchronize()
                    t3 = time.time()
                    rec["decode"].append(t3 - t2)
                    # dec rank does not generate: its t1 ~ t0 and t2 - t1 is the wait for the step in flight.
                    # speculative -> served from the ready candidate (bcast + decode, bcast added in the summary);
                    # single -> generated after the press (gen + bcast + decode)
                    rec["wait"].append(t2 - t1)
                    lat = (t3 - t2) if args.mode == "spec" else (t3 - t0)
                    rec["latency"].append(lat)
                    if s > 0 and keys[s] != keys[s - 1]:
                        rec["switch_latency"].append(lat)
                rec["gen"].append(t1 - t0)
                rec["bcast"].append(t2 - t1)
                t_now = time.time()
                rec["interval"].append(t_now - t_prev)
                t_prev = t_now
                if rank == 0 and s % 40 == 0:
                    print(json.dumps({"video": vi, "sid": sid, "step": s, "key": KEYS[key],
                                      "interval": round(rec["interval"][-1], 3)}), flush=True)
        # timing summary: decode-side numbers live on dec_rank, generation on rank 0
        med = lambda v: float(np.median(v[8:])) if len(v) > 8 else float("nan")  # noqa: E731
        med_all = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731  (switches: keep warm-up)
        p95 = float(np.percentile(rec["latency"][8:] or rec["latency"], 95)) if rank == dec_rank else 0.0
        stats = torch.tensor([med(rec["gen"]), med(rec["bcast"]), med(rec["interval"]), med(rec["decode"]),
                              med(rec["latency"]), med_all(rec["switch_latency"]), p95, med(rec["wait"])],
                             device=dev)
        all_stats = [torch.zeros_like(stats) for _ in range(world)]
        dist.all_gather(all_stats, stats)
        if rank == dec_rank:
            g0, dd = all_stats[0].tolist(), stats.tolist()
            res = {"video": vi, "sample_id": sid, "mode": args.mode, "stage": args.stage, "gen_gpus": G, "seconds": args.seconds,
                   "switch_sec": args.switch_sec, "keys": "".join(KEYS[int(k)] for k in keys[::per]),
                   "gen_s_per_latent": g0[0], "bcast_s": g0[1], "step_interval_s": g0[2],
                   "realtime_factor": 0.25 / g0[2], "decode_s_per_latent": dd[3],
                   # served = key press -> new frame decoded when the candidate is ready (spec: + bcast);
                   # wait = time the decoder waits for the step in flight (> 0 when generation < real time)
                   "keypress_latency_s": dd[4] + (g0[1] if args.mode == "spec" else 0.0),
                   "keypress_latency_p95_s": dd[6] + (g0[1] if args.mode == "spec" else 0.0),
                   "switch_latency_s": dd[5] + (g0[1] if args.mode == "spec" else 0.0),
                   "wait_for_step_s": dd[7],
                   "wall_s": float(sum(rec["interval"]))}
            print("RESULT", json.dumps(res), flush=True)
            with open(out / "timing.jsonl", "a") as fh:
                fh.write(json.dumps(res) + "\n")
            save_video(out / f"video{vi}_{sid}.mp4", torch.cat(frames, 1), keys, per)
        dist.barrier()
    dist.destroy_process_group()


def kv_call(diffusion, inner, x, t: float, start: int, pose, key: int, cond, kv, xa, args, dev, policy: str):
    """One Stage2 KV-cache forward for a single latent at timeline index `start` (as student_common.history_rollout,
    block = 1). policy "none" = denoise without touching the cache, "commit_detached" = write clean K/V."""
    cam = camera_tokens_long(pose[None, None], dev, args.cam_scale)
    inner._action_tokens = tokens(torch.tensor([[int(key)]], device=dev).float()).long()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return diffusion(x, cond, cam, tokens(torch.full((1, 1), float(t), device=dev)),
                             sequence_length=FRAME_TOKENS, kv_cache=kv, crossattn_cache=xa,
                             current_start=start * FRAME_TOKENS, cache_start=0, cache_update_policy=policy)
    finally:
        inner._action_tokens = None


def save_video(path: Path, pix: torch.Tensor, keys: np.ndarray, per: int) -> None:
    """pix [3,T,H,W] in [-1,1] -> mp4 (16 fps) with the current key and time overlaid."""
    import cv2
    import imageio.v2 as iio
    n_hist = 1 + 4 * (HIST_LATENTS - 1)
    fr = ((pix.permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    out = []
    for i, f in enumerate(fr):
        f = np.ascontiguousarray(f)
        if i < n_hist:
            lab = "real history"
        else:
            li = (i - n_hist) // 4
            lab = f"key {KEYS[int(keys[min(li, len(keys) - 1)])]}   t={(i - n_hist) / 16:5.2f}s"
        cv2.rectangle(f, (0, 0), (250, 30), (0, 0, 0), -1)
        cv2.putText(f, lab, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(f)
    iio.mimwrite(path, np.stack(out), fps=16, quality=7, macro_block_size=8)


if __name__ == "__main__":
    main()
