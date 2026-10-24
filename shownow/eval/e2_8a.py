"""E2 for the 8-action models: generation on the 8-action key-switch evaluation forks (shownow/eval/sw8a_split.py:
training maps by default, SHOWNOW_SW8_DATA=data/cf_vizdoom_sw8a for the held-out maps), saving the generated latents and paired picture quality.
8-action version of action_follow_eval.py (that file is unchanged); scoring with the 8-action IDM: e2_8a_score.py.

All 8 counterfactual branches of a fork (each with its own key schedule over the 8 keys, switching every 0.25-1.5 s)
are generated from the real 2 s history.
  --mode reference      training-faithful cache-free tree window sampler (normal, non-speculative inference)
  --mode realtime_opt   the deployed ShowNow path (tree_opt.RTStepper, --opts): at every future latent all 8 key
                        candidates are generated in one shared-history tree pass and the scheduled key's candidate is
                        committed; candidate cameras from the KeyStream8 nominal trajectory (= the data's nominal camera)
  --kind teacher        bidirectional teacher, 30 UniPC steps
Conditions: both (nominal camera + action = deployment input), camera (nominal camera, no action),
none (no action, static camera). --key-per-block: SolarWM-FT deployment input (the key is read once per block).

  SHOWNOW_N_ACTIONS=9 python -m shownow.eval.e2_8a --kind s2 --ckpt <pt> --out runs/e2_8a/<name> [--rank r --world n]
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
from shownow.eval.action_follow_eval import load, paired_quality, red_jumps, to_uint8  # noqa: E402
from shownow.train.teacher_action_ft import camera_tokens  # noqa: E402
from shownow.tree.cond_v3 import cam_dir_for, static_future  # noqa: E402
from shownow.tree.tree_samplers import student_tree_sample, teacher_tree_sample  # noqa: E402

BRANCHES = ["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
KEY_ID = {k: i + 1 for i, k in enumerate(BRANCHES)}
N_HIST, H = 33, 9


def load_fork(f: Path) -> tuple[dict, dict]:
    fork = torch.load(f, map_location="cpu", weights_only=False)
    camf = torch.load(cam_dir_for(f.parent) / f.name, weights_only=True)
    return fork, camf


def keys_of(fork, b) -> list:
    return [int(k) for k in fork["actions"][b, H:]]


@torch.no_grad()
def realtime_opt_rollout(m, fork, camf, cam_scale, dev, seed, stepper):
    """-> latents [8,21,...]: per branch, the deployed path commits the scheduled key's candidate at every latent."""
    from shownow.data.nominal_8a import KeyStream8
    from shownow.eval.long_drift_eval import camera_tokens_long
    lat = fork["latents"].to(dev)
    nominal = camf["nominal"].float()
    acts_hist = [int(a) for a in fork["actions"][0, :H]]
    out = lat.clone()
    g = torch.Generator(device=dev).manual_seed(seed)
    ns, K = len(stepper.fewstep), len(BRANCHES)

    def cams(poses):
        return camera_tokens_long(torch.stack(poses)[None], dev, cam_scale)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = m.text_embedding(fork["prompt"].to(dev)[None])
        for b in range(K):
            stream = KeyStream8(camf["fork"].double().numpy())
            stepper.reset()
            stepper.set_text(ctx)
            for c0 in range(H):  # commit the real history
                stepper.commit_frame(lat[0:1, c0:c0 + 1], cams([nominal[0, c0]]), acts_hist[c0])
            for j, key in enumerate(keys_of(fork, b)):
                stepper.set_inputs(cams([torch.from_numpy(stream.peek(k)).float() for k in range(1, K + 1)]),
                                   list(range(1, K + 1)))
                noise = [torch.randn((1, K, *lat.shape[2:]), generator=g, device=dev, dtype=torch.bfloat16)]
                noise += [torch.randn((1, K, *lat.shape[2:]), generator=g, device=dev) for _ in range(ns - 1)]
                x = stepper.gen(noise)
                out[b, H + j] = x[0, key - 1]
                stepper.choose(key - 1)
                stream.commit(key)
    return out


def held_key_inputs(fork, camf, block):
    """SolarWM-FT as deployed reads the key once per AR block: every latent of a future block gets the key held at the
    block start; its nominal camera follows that applied key sequence. -> (actions [8,21], cam_nom [8,21,4,4])."""
    from shownow.data.nominal_8a import KeyStream8
    acts, cam = fork["actions"].clone(), camf["nominal"].clone().float()
    for b in range(acts.shape[0]):
        sched = keys_of(fork, b)
        held = [sched[(j // block) * block] for j in range(len(sched))]
        stream = KeyStream8(camf["fork"].double().numpy())
        cam[b, H:] = torch.stack([torch.from_numpy(stream.commit(k)).float() for k in held])
        acts[b, H:] = torch.tensor(held, dtype=acts.dtype)
    return acts, cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["teacher", "s1", "s2"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="e2_val", help="sw8a_split key (e2_val | test_e2) or a .json fork list (dev)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--conds", default="both")
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--nfe", type=int, default=3)
    ap.add_argument("--denoising-steps", default="1000,667,333")
    ap.add_argument("--teacher-steps", type=int, default=30)
    ap.add_argument("--videos", type=int, default=1, help="side-by-side videos saved per rank")
    ap.add_argument("--opts", default="exact", help="realtime_opt: tree_opt option set")
    ap.add_argument("--mode", default="reference", choices=["reference", "realtime_opt"])
    ap.add_argument("--key-per-block", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "lat").mkdir(exist_ok=True)
    (out / f"args_rank{args.rank}.json").write_text(json.dumps(vars(args), indent=1))
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.split.endswith(".json"):  # dev: explicit fork list
        files = [Path(f) for f in json.load(open(args.split))][args.rank::args.world]
    else:
        from shownow.eval.sw8a_split import split
        files = [Path(f) for f in split()[args.split]][args.rank::args.world]
    from solarwm.backends.wan22.runtime.components import Wan5BVAE
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    vae = Wan5BVAE(REPO / "models/SolarWM-5B-base/vae/Wan2.2_VAE.pth").to(dev, dtype=torch.bfloat16)
    d = load(args.kind, args.ckpt, dev, args.block, args.history)
    root = d.module
    fewstep = [float(s) for s in warp_denoising_steps([int(v) for v in args.denoising_steps.split(",")],
                                                      d.scheduler.timesteps.to(dev), num_train_timesteps=1000)]
    stepper, mode = None, args.mode
    if args.mode == "realtime_opt":
        from shownow.latency.tree_opt import RTStepper, opts_name, parse_opts
        tab_t, tab_s = d.scheduler.timesteps.float(), d.scheduler.sigmas.float()
        sig = [float(tab_s[(tab_t - t).abs().argmin()]) for t in fewstep]
        stepper = RTStepper(root, len(BRANCHES), fewstep, sig, P=args.history, opts=parse_opts(args.opts), device=dev)
        mode = f"realtime_opt[{opts_name(stepper.o)}]"
    tag = f"{args.kind}_{mode}" + ("_kpb" if args.key_per_block else "")
    rows = open(out / f"rows_rank{args.rank}.jsonl", "a")
    done = {(r["sample_id"], r["cond"]) for f in out.glob("rows_rank*.jsonl") for r in map(json.loads, open(f))}
    n_vid = 0
    for fi, f in enumerate(files):
        fork, camf = load_fork(f)
        sid = f.stem
        seed = int(sid.split("_")[-1])
        clean = fork["latents"].to(dev)
        prompt = fork["prompt"].to(dev)[None]
        acts, cam_nom = fork["actions"], camf["nominal"].float()
        if args.key_per_block:
            acts, cam_nom = held_key_inputs(fork, camf, args.block)
        gt = None
        for cond in args.conds.split(","):
            if (sid, cond) in done:
                continue
            if gt is None:
                with torch.no_grad():
                    gt = [to_uint8(p) for p in vae.decode(clean)]
            cam_rel = cam_nom if cond != "none" else static_future(cam_nom)
            a = acts.clone().to(dev)
            if cond in ("camera", "none"):
                a[:, H:] = 0
            cam = camera_tokens(cam_rel, len(BRANCHES), dev, args.cam_scale)
            t0 = time.time()
            with torch.no_grad():
                if stepper is not None:
                    assert args.kind == "s2" and cond == "both", "realtime: final student, deployment input"
                    x = realtime_opt_rollout(root, fork, camf, args.cam_scale, dev, seed, stepper)
                elif args.kind == "teacher":
                    x = teacher_tree_sample(d, root, clean, prompt, cam, a, steps=args.teacher_steps, seed=seed)
                elif args.kind == "s1":
                    x = student_tree_sample(d, root, clean, prompt, cam, a, solver="flowmap", nfe=args.nfe, seed=seed,
                                            max_prior_chunks=args.history // args.block, block=args.block)
                else:
                    x = student_tree_sample(d, root, clean, prompt, cam, a, solver="fewstep", fewstep=fewstep,
                                            seed=seed, x0_of=d.flow_to_x0,
                                            max_prior_chunks=args.history // args.block, block=args.block)
                gen = [to_uint8(p) for p in vae.decode(x)]
            rec = {"model": f"{Path(args.ckpt).parent.name}/{Path(args.ckpt).stem}", "tag": tag, "kind": args.kind,
                   "cond": cond, "mode": mode, "key_per_block": args.key_per_block, "sample_id": sid,
                   **paired_quality(gen, gt), "red_jumps": sum(red_jumps(g_[N_HIST:]) for g_ in gen),
                   "sec": time.time() - t0}
            torch.save({"gen": x.to(torch.bfloat16).cpu(), "fork": str(f), "cond": cond, "mode": mode, "tag": tag},
                       out / "lat" / f"{sid}__{cond}.pt")
            rows.write(json.dumps(rec) + "\n")
            rows.flush()
            print(json.dumps(rec), flush=True)
            if n_vid < args.videos and cond == "both":
                import imageio.v2 as iio
                top = np.concatenate([g_[:, ::4, ::4] for g_ in gen], 2)
                bot = np.concatenate([g_[:, ::4, ::4] for g_ in gt], 2)
                iio.mimwrite(out / "videos" / f"{sid}.mp4", np.concatenate([top, bot], 1), fps=16, quality=7,
                             macro_block_size=4)
                n_vid += 1


if __name__ == "__main__":
    main()
