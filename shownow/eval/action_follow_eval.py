"""Action-following evaluation on the v3 key-switch Doom validation forks (map-disjoint, damage-free, no monsters).

For each fork the 4 counterfactual branches (each with its own W/A/S/D key schedule that switches every 0.25-1.5 s) are
generated in one tree from the real 2 s history, and every key segment of every branch is scored with the optical-flow
ego-motion classifier (shownow/eval/ego_motion.py): does the generated motion over that segment look like its key?

Conditions (what each channel can do on its own):
  both     nominal camera from the key sequence + action embedding   (= what real-time inference provides)
  action   static future camera + action embedding                   (action channel alone)
  camera   nominal camera, no action                                 (camera channel alone)
Reported per condition: segment accuracy over all segments, and over "valid" segments where the REAL branch video
moves the way its key says (the engine can block motion at walls; those segments are not the model's fault).
Also the real-video accuracy itself (upper bound of the classifier) and the red-flash count of the generated frames.

  python -m shownow.eval.action_follow_eval --kind teacher|s1|s2 --ckpt <pt> --out runs/action_eval/<name> \
      [--rank r --world n] [--forks 48] [--conds both,action,camera]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
from shownow.eval.ego_motion import ego_motion  # noqa: E402
from shownow.train.teacher_action_ft import camera_tokens  # noqa: E402
from shownow.tree.cond_v3 import V3Dataset, static_future, v3_split  # noqa: E402
from shownow.tree.tree import enable_tree  # noqa: E402
from shownow.tree.tree_samplers import student_tree_sample, teacher_tree_sample  # noqa: E402

N_HIST = 33  # pixel frames of the history
H_LAT = 9  # history latents
KEYS = "_WASD"


def to_uint8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def load(kind, ckpt, device, block, history):
    if kind == "teacher":
        from shownow.train.teacher_action_ft import build_teacher
        d, _ = build_teacher(ckpt)
    elif kind == "s1":
        from shownow.tree.student_stage1_tree import build_student
        d, _ = build_student(ckpt, from_teacher=False, field="generator_ema")
    else:
        from shownow.train.student_stage2 import build_stage2_student
        d, _, _ = build_stage2_student(ckpt, field="generator_ema", block=block, history=history)
    d.module.to(device, dtype=torch.bfloat16).eval()
    enable_tree(d.module)
    return d


def segments(fork):
    """[(branch, key, start_latent, n_latents)] of the future from the generator's switch schedules."""
    sched = fork["meta"]["switch_schedules"]
    return [(b, s["key"], s["start_latent"], s["n_latents"]) for b, k in enumerate("WASD") for s in sched[k]]


def seg_frames(frames, s0, n):
    """future latent j shows pixel frames 33 + 4 j - 3 .. 33 + 4 j (Wan causal VAE); one frame of context before."""
    lo = N_HIST + 4 * s0 - 1
    return frames[lo: N_HIST + 4 * (s0 + n)]


@torch.no_grad()
def realtime_rollout(d, m, fork, camf, cam_scale, dev, seed, history=15):
    """The deployed ShowNow path (FastTree, shared-prefix K/V ring, cascade attention, 4 denoising steps, eager
    reference kernels): for every branch b, at each future latent all 4 key candidates are generated in one tree from
    the committed prefix and the candidate of b's scheduled key is committed (its commit-time K/V enter the ring).
    Candidate cameras come from the same key stream as the training data's nominal trajectory. -> latents [4,21,...]."""
    from shownow.data.nominal_switch import KeyStream
    from shownow.latency.tree_fast import FRAME_TOKENS, FastTree
    from shownow.train.student_common import allocate_caches, tokens
    from shownow.eval.long_drift_eval import camera_tokens_long
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    n_tok, H = FRAME_TOKENS, 9
    fewstep = [float(v) for v in warp_denoising_steps([1000, 750, 500, 250], d.scheduler.timesteps.to(dev),
                                                      num_train_timesteps=1000)]
    tab_t, tab_s = d.scheduler.timesteps.float(), d.scheduler.sigmas.float()
    sig = [float(tab_s[(tab_t - t).abs().argmin()]) for t in fewstep]
    lat = fork["latents"].to(dev)
    nominal = camf["nominal"].float()
    acts_hist = [int(a) for a in fork["actions"][0, :H]]
    sched = fork["meta"]["switch_schedules"]
    out = lat.clone()
    g = torch.Generator(device=dev).manual_seed(seed)

    def cams(poses):
        return camera_tokens_long(torch.stack(poses)[None], dev, cam_scale)

    for b, bk in enumerate("WASD"):
        keys = [{"W": 1, "A": 2, "S": 3, "D": 4}[x["key"]] for x in sched[bk] for _ in range(x["n_latents"])]
        stream = KeyStream(camf["fork"].double().numpy())
        xa = allocate_caches(m, 1, 1, torch.bfloat16, dev)[1]
        committed = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = m.text_embedding(fork["prompt"].to(dev)[None])

            def make_ft(K):
                n_prefix = min(len(committed), history)
                ft = FastTree(m, K, prefix_latents=n_prefix, share=K > 1, h=15, w=27, block=1)
                for i, (k_, v_, c_) in enumerate(committed[-n_prefix:] if n_prefix else []):
                    s0 = i * n_tok
                    ft.rk[:, :, s0:s0 + n_tok].copy_(k_)
                    ft.rv[:, :, s0:s0 + n_tok].copy_(v_)
                    ft.pcam_v[:, s0:s0 + n_tok].copy_(c_[..., :16].unflatten(2, (4, 4)))
                    ft.pcam_K[:, s0:s0 + n_tok].copy_(c_[..., 16:].unflatten(2, (3, 3)))
                ft.ppos.copy_(torch.arange(n_prefix, device=dev))
                return ft, n_prefix

            for c0 in range(H):  # commit the real history
                ft, n_prefix = make_ft(1)
                cam = cams([nominal[0, c0]])
                fr = ft.freqs_for_positions(torch.arange(n_prefix, n_prefix + 1, device=dev))
                if n_prefix:
                    ft.prepare_prefix()
                ft.forward(lat[0:1, c0:c0 + 1], torch.zeros(1, 1, device=dev),
                           tokens(torch.tensor([[acts_hist[c0]]], device=dev).float()).long(), ctx, xa, cam, fr,
                           commit=True)
                committed.append((ft.ck[:, 0:1].clone(), ft.cv[:, 0:1].clone(),
                                  torch.cat([cam["viewmats"].flatten(2), cam["K"].flatten(2)], 2)))
            for j, key in enumerate(keys):  # 12 future latents, 4 candidates each, commit the scheduled key
                ft, n_prefix = make_ft(4)
                cam = cams([torch.from_numpy(stream.peek(k)).float() for k in (1, 2, 3, 4)])
                act = tokens(torch.tensor([[1, 2, 3, 4]], device=dev).float()).long()
                fr = ft.freqs_for_positions(torch.arange(n_prefix, n_prefix + 1, device=dev).repeat(4))
                ft.prepare_prefix()
                x = torch.randn((1, 4, *lat.shape[2:]), generator=g, device=dev, dtype=torch.bfloat16)
                for i in range(len(fewstep)):
                    flow = ft.forward(x, torch.full((1, 4), fewstep[i], device=dev), act, ctx, xa, cam, fr, commit=False)
                    x0 = x.float() - sig[i] * flow.float()
                    x = x0.to(torch.bfloat16) if i == len(fewstep) - 1 else \
                        ((1 - sig[i + 1]) * x0 + sig[i + 1] * torch.randn(x0.shape, generator=g, device=dev)).to(torch.bfloat16)
                ft.forward(x, torch.zeros(1, 4, device=dev), act, ctx, xa, cam, fr, commit=True)
                kk = key - 1
                committed.append((ft.ck[:, kk:kk + 1].clone(), ft.cv[:, kk:kk + 1].clone(),
                                  torch.cat([cam["viewmats"][:, kk * n_tok:(kk + 1) * n_tok].flatten(2),
                                             cam["K"][:, kk * n_tok:(kk + 1) * n_tok].flatten(2)], 2)))
                out[b, H + j] = x[0, kk]
                stream.commit(key)
                del ft
    return out


def realtime_opt_rollout(d, m, fork, camf, cam_scale, dev, seed, stepper):
    """realtime_rollout on the optimised path (tree_opt.RTStepper, chosen-only commit, its option set): same forks,
    key schedules, cameras and noise stream as realtime_rollout, eager (compiled layers, no CUDA graph)."""
    from shownow.data.nominal_switch import KeyStream
    from shownow.eval.long_drift_eval import camera_tokens_long
    H = 9
    lat = fork["latents"].to(dev)
    nominal = camf["nominal"].float()
    acts_hist = [int(a) for a in fork["actions"][0, :H]]
    sched = fork["meta"]["switch_schedules"]
    out = lat.clone()
    g = torch.Generator(device=dev).manual_seed(seed)
    ns = len(stepper.fewstep)

    def cams(poses):
        return camera_tokens_long(torch.stack(poses)[None], dev, cam_scale)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = m.text_embedding(fork["prompt"].to(dev)[None])
        for b, bk in enumerate("WASD"):
            keys = [{"W": 1, "A": 2, "S": 3, "D": 4}[x["key"]] for x in sched[bk] for _ in range(x["n_latents"])]
            stream = KeyStream(camf["fork"].double().numpy())
            stepper.reset()
            stepper.set_text(ctx)
            for c0 in range(H):  # commit the real history
                stepper.commit_frame(lat[0:1, c0:c0 + 1], cams([nominal[0, c0]]), acts_hist[c0])
            for j, key in enumerate(keys):
                stepper.set_inputs(cams([torch.from_numpy(stream.peek(k)).float() for k in (1, 2, 3, 4)]), [1, 2, 3, 4])
                noise = [torch.randn((1, 4, *lat.shape[2:]), generator=g, device=dev, dtype=torch.bfloat16)]
                noise += [torch.randn((1, 4, *lat.shape[2:]), generator=g, device=dev) for _ in range(ns - 1)]
                x = stepper.gen(noise)
                out[b, H + j] = x[0, key - 1]
                stepper.choose(key - 1)
                stream.commit(key)
    return out


def held_key_inputs(fork, camf, block):
    """SolarWM-FT as deployed reads the key once per AR block: every latent of a future block gets the key held at
    the block start, and its nominal camera follows that applied key sequence (KeyStream, as in realtime_rollout).
    -> (actions [4,21], cam_nom [4,21,4,4]); the history is unchanged and scoring stays against the real branch."""
    from shownow.data.nominal_switch import KeyStream
    acts, cam = fork["actions"].clone(), fork["cam_nom"].clone()
    for b in range(acts.shape[0]):
        sched = [int(k) for k in acts[b, H_LAT:]]
        stream = KeyStream(camf["fork"].double().numpy())
        ref = torch.stack([torch.from_numpy(stream.commit(k)).float() for k in sched])
        assert torch.allclose(ref, cam[b, H_LAT:], atol=1e-4), "KeyStream does not reproduce the nominal camera"
        held = [sched[(j // block) * block] for j in range(len(sched))]
        stream = KeyStream(camf["fork"].double().numpy())
        cam[b, H_LAT:] = torch.stack([torch.from_numpy(stream.commit(k)).float() for k in held])
        acts[b, H_LAT:] = torch.tensor(held, dtype=acts.dtype)
    return acts, cam


_LPIPS = None


def paired_quality(gen, gt, windows=((0, 8), (0, 16))):
    """gen/gt: per-branch uint8 [T,H,W,3] lists; future frames [N_HIST + a, N_HIST + b) -> LPIPS / PSNR / SSIM means."""
    global _LPIPS
    import lpips
    import torch.nn.functional as Fnn
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    def ssim(x, y):  # x, y [N,3,H,W] in [0,1], gaussian 11x11, sigma 1.5
        g1 = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
        w = (g1[:, None] * g1[None]) / g1.sum() ** 2
        w = w.expand(3, 1, 11, 11)
        f = lambda z: Fnn.conv2d(z, w, groups=3)  # noqa: E731
        mx, my = f(x), f(y)
        vx, vy, cxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).mean((1, 2, 3))

    res = {}
    for a, b in windows:
        x = torch.cat([torch.from_numpy(g[N_HIST + a:N_HIST + b]) for g in gen]).cuda().permute(0, 3, 1, 2).float() / 255
        y = torch.cat([torch.from_numpy(g[N_HIST + a:N_HIST + b]) for g in gt]).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            lp = torch.cat([_LPIPS(x[i:i + 16] * 2 - 1, y[i:i + 16] * 2 - 1).flatten() for i in range(0, len(x), 16)])
        mse = ((x - y) ** 2).mean((1, 2, 3))
        tag = f"{a / 16:g}-{b / 16:g}s"
        res[f"lpips_{tag}"] = float(lp.mean())
        res[f"psnr_{tag}"] = float((10 * torch.log10(1 / mse.clamp_min(1e-10))).mean())
        res[f"ssim_{tag}"] = float(torch.cat([ssim(x[i:i + 16], y[i:i + 16]) for i in range(0, len(x), 16)]).mean())
    return res


def red_jumps(frames):
    f = frames[:, ::8, ::8].astype(np.float32)
    red = (f[..., 0] - f[..., 1:].mean(-1)).mean((1, 2))
    return int((np.abs(np.diff(red)) > 15).sum())


def retrieval(gen_lat, gt_lat, gen, gt):
    """4-way counterfactual retrieval (chance 0.25): is generated branch b closest to the REAL future of branch b
    (same state, its own key schedule) among the 4 real futures? Distances over the future only: latent MSE, and
    LPIPS on every 4th future frame. Also returns the 4x4 distance matrices."""
    import torch.nn.functional as Fnn  # noqa: F401
    dl = ((gen_lat[:, H_LAT:].float()[:, None] - gt_lat[:, H_LAT:].float()[None]) ** 2).mean((2, 3, 4, 5)).cpu()
    idx = list(range(N_HIST, N_HIST + 48, 4))
    x = torch.stack([torch.from_numpy(g[idx]) for g in gen]).cuda().permute(0, 1, 4, 2, 3).float() / 127.5 - 1
    y = torch.stack([torch.from_numpy(g[idx]) for g in gt]).cuda().permute(0, 1, 4, 2, 3).float() / 127.5 - 1
    dp = torch.zeros(4, 4)
    with torch.no_grad():
        for i in range(4):
            for j in range(4):
                dp[i, j] = _LPIPS(x[i], y[j]).mean().cpu()
    r = torch.arange(4)
    return {"ret_latent": float((dl.argmin(1) == r).float().mean()), "ret_lpips": float((dp.argmin(1) == r).float().mean()),
            "d_latent": dl.tolist(), "d_lpips": dp.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["teacher", "s1", "s2"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--forks", type=int, default=48)
    ap.add_argument("--conds", default="both,action,camera")
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--videos", type=int, default=2, help="side-by-side videos saved per rank")
    ap.add_argument("--save-lat", action="store_true", help="save generated latents to <out>/lat (for the IDM)")
    ap.add_argument("--opts", default="exact", help="realtime_opt: tree_opt option set (none | exact | exact,fp8 | ...)")
    ap.add_argument("--denoising-steps", default="1000,750,500,250",
                    help="SGF schedule of reference / realtime_opt (the deployed FastTree path stays 4-step)")
    ap.add_argument("--mode", default="reference", choices=["reference", "realtime", "realtime_opt"],
                    help="reference = training-faithful cache-free tree window sampler (non-speculative); realtime = "
                         "the deployed ShowNow path (FastTree, 4 candidates per latent, commit the scheduled key)")
    ap.add_argument("--key-per-block", action="store_true",
                    help="SolarWM-FT deployment input: the key is read once per --block latents (held_key_inputs)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    _, _, val, _ = v3_split()
    from shownow.tree.cond_v3 import split_name, test_forks
    val = sorted(val, key=lambda f: f.name)
    val = (test_forks(val, args.forks) if split_name() == "test" else val[: args.forks])[args.rank::args.world]
    ds = V3Dataset(val)
    from solarwm.backends.wan22.runtime.components import Wan5BVAE
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    vae = Wan5BVAE(REPO / "models/SolarWM-5B-base/vae/Wan2.2_VAE.pth").to(dev, dtype=torch.bfloat16)
    d = load(args.kind, args.ckpt, dev, args.block, args.history)
    root = d.module
    fewstep = [float(s) for s in warp_denoising_steps([int(v) for v in args.denoising_steps.split(",")],
                                                      d.scheduler.timesteps.to(dev), num_train_timesteps=1000)]
    rows = open(out / f"rows_rank{args.rank}.jsonl", "a")
    stepper = None
    if args.mode == "realtime_opt":
        from shownow.latency.tree_opt import RTStepper, opts_name, parse_opts
        tab_t, tab_s = d.scheduler.timesteps.float(), d.scheduler.sigmas.float()
        sig = [float(tab_s[(tab_t - t).abs().argmin()]) for t in fewstep]
        stepper = RTStepper(root, 4, fewstep, sig, P=args.history, opts=parse_opts(args.opts), device=dev)
        args.mode = f"realtime_opt[{opts_name(stepper.o)}]"
    n_vid = 0
    for fi in range(len(ds)):
        fork = ds[fi]
        sid = val[fi].stem
        clean = fork["latents"].to(dev)
        prompt = fork["prompt"].to(dev)[None]
        acts = fork["actions"].to(dev)
        if args.key_per_block:
            from shownow.tree.cond_v3 import cam_dir_for
            camf = torch.load(cam_dir_for(val[fi].parent) / val[fi].name, weights_only=True)
            a_held, fork["cam_nom"] = held_key_inputs(fork, camf, args.block)
            acts = a_held.to(dev)
        with torch.no_grad():
            gt = [to_uint8(p) for p in vae.decode(clean)]
        segs = segments(fork)
        gt_pred = {(b, s0): ego_motion(seg_frames(gt[b], s0, n), step=2)["pred"] for b, _, s0, n in segs}
        for cond in args.conds.split(","):
            cam_rel = fork["cam_nom"] if cond not in ("action", "none") else static_future(fork["cam_nom"])
            a = acts.clone()
            if cond in ("camera", "none"):  # none = negative control: no action, static camera
                a[:, 9:] = 0
            cam = camera_tokens(cam_rel, 4, dev, args.cam_scale)
            t0 = time.time()
            with torch.no_grad():
                if stepper is not None:
                    assert args.kind == "s2" and cond == "both", "realtime mode: final student, deployment input"
                    from shownow.tree.cond_v3 import cam_dir_for
                    camf = torch.load(cam_dir_for(val[fi].parent) / val[fi].name, weights_only=True)
                    x = realtime_opt_rollout(d, root, fork, camf, args.cam_scale, dev, seed=fi, stepper=stepper)
                elif args.mode == "realtime":
                    assert args.kind == "s2" and cond == "both", "realtime mode: final student, deployment input"
                    from shownow.tree.cond_v3 import cam_dir_for
                    camf = torch.load(cam_dir_for(val[fi].parent) / val[fi].name, weights_only=True)
                    x = realtime_rollout(d, root, fork, camf, args.cam_scale, dev, seed=fi, history=args.history)
                elif args.kind == "teacher":
                    x = teacher_tree_sample(d, root, clean, prompt, cam, a, steps=30, seed=fi)
                elif args.kind == "s1":
                    x = student_tree_sample(d, root, clean, prompt, cam, a, solver="flowmap", nfe=args.nfe, seed=fi,
                                            max_prior_chunks=args.history // args.block, block=args.block)
                else:
                    x = student_tree_sample(d, root, clean, prompt, cam, a, solver="fewstep", fewstep=fewstep, seed=fi,
                                            x0_of=d.flow_to_x0, max_prior_chunks=args.history // args.block,
                                            block=args.block)
                gen = [to_uint8(p) for p in vae.decode(x)]
            res = []
            for b, key, s0, n in segs:
                pred = ego_motion(seg_frames(gen[b], s0, n), step=2)["pred"]
                res.append({"branch": b, "key": key, "start": s0, "n": n, "pred": pred, "gt_pred": gt_pred[(b, s0)]})
            ok = [r["pred"] == r["key"] for r in res]
            valid = [r["pred"] == r["key"] for r in res if r["gt_pred"] == r["key"]]
            rec = {"model": f"{Path(args.ckpt).parent.name}/{Path(args.ckpt).stem}", "kind": args.kind, "cond": cond,
                   "mode": args.mode, "key_per_block": args.key_per_block, "sample_id": sid, **paired_quality(gen, gt),
                   "seg_acc": float(np.mean(ok)), "seg_acc_valid": float(np.mean(valid)) if valid else None,
                   "n_valid": len(valid), "gt_acc": float(np.mean([r["gt_pred"] == r["key"] for r in res])),
                   "red_jumps": sum(red_jumps(g[N_HIST:]) for g in gen), **retrieval(x, clean, gen, gt),
                   "sec": time.time() - t0, "segments": res}
            if args.save_lat:
                (out / "lat").mkdir(exist_ok=True)
                torch.save({"gen": x.to(torch.bfloat16).cpu(), "fork": str(val[fi]), "cond": cond, "mode": args.mode},
                           out / "lat" / f"{sid}__{cond}__{args.mode}.pt")
            rows.write(json.dumps(rec) + "\n")
            rows.flush()
            print(json.dumps({k: v for k, v in rec.items() if k not in ("segments", "d_latent", "d_lpips")}), flush=True)
            if n_vid < args.videos and cond == "both":
                import imageio.v2 as iio
                top = np.concatenate([g[:, ::2, ::2] for g in gen], 2)
                bot = np.concatenate([g[:, ::2, ::2] for g in gt], 2)
                iio.mimwrite(out / "videos" / f"{sid}_{Path(args.ckpt).stem}.mp4", np.concatenate([top, bot], 1),
                             fps=16, quality=7)
        n_vid += 1


if __name__ == "__main__":
    main()
