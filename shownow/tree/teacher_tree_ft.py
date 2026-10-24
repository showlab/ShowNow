"""Teacher (bidirectional) fine-tune in the branch-tree layout: one sequence = shared 2 s history + W/A/S/D futures.

Per fork: slots [H(9) | W(12) | A(12) | S(12) | D(12)] (57 latents instead of 4 x 21 = 84); history clean (t=0),
the 4 futures noised with the SAME noise and t; history attends only history, each branch attends history + itself.
Loss = per-branch flow MSE on the futures + lambda * counterfactual difference loss over the 6 branch pairs:
|| (pred_a - pred_b) - (target_a - target_b) ||^2  (with shared noise, target_a - target_b = clean_b - clean_a), which
directly rewards getting the action-dependent part right.
Conditioning as teacher v2: action embedding (time path) + nominal camera trajectory x0.5 per branch.
Init: teacher v2. Launch: sbatch -N 1 -w <node> slurm/train_3node.sbatch shownow.tree.teacher_tree_ft --out ...
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SolarWM/src"))
from shownow.train.student_common import tokens  # noqa: E402
from shownow.train.teacher_action_ft import (ForkDataset, build_teacher, camera_tokens, is_action_param, next_action,  # noqa: E402
                                             next_branch, split_files)
from shownow.tree.tree import FRAME_TOKENS, bidir_layout, enable_tree, tree_call  # noqa: E402

H, F, K = 9, 12, 4


def tree_inputs(fork, device, cam_scale, actions=None, cam_rel=None):
    """Pack a fork (K = number of branches, 4 for Doom forks, 1 for single trajectories) into the bidir tree layout.
    Returns clean slots [1, H + K*F, ...], prompt, camera, action tokens. A fork may carry its own camera scale
    (SolarWM-Data subset: translation already in SolarWM units)."""
    lat = fork["latents"].to(device)
    K = lat.shape[0]
    acts = (fork["actions"] if actions is None else actions).to(device)
    cam_scale = float(fork.get("cam_scale", cam_scale))
    cam = camera_tokens(fork["cam_rel"] if cam_rel is None else cam_rel, K, device, cam_scale)  # [K, 21*405]
    x = torch.cat([lat[0:1, :H]] + [lat[k:k + 1, H:] for k in range(K)], 1)
    a = torch.cat([acts[0:1, :H]] + [acts[k:k + 1, H:] for k in range(K)], 1)
    c = {n: torch.cat([cam[n][0:1, :H * FRAME_TOKENS]] + [cam[n][k:k + 1, H * FRAME_TOKENS:] for k in range(K)], 1)
         for n in ("viewmats", "K")}
    return x, fork["prompt"].to(device)[None], c, a


def tree_loss(diffusion, root, fork, device, cam_scale, lam, t_index=None, actions=None, cam_rel=None, gen=None,
              action_drop=0.0):
    from solarwm.backends.wan22.objectives import weighted_masked_mse  # noqa: F401

    clean, prompt, cam, acts = tree_inputs(fork, device, cam_scale, actions, cam_rel)
    K = fork["latents"].shape[0]
    layout = bidir_layout(K, H, F)
    if action_drop > 0:
        for k in range(K):
            if torch.rand(1, generator=gen).item() < action_drop:
                acts[:, H + F * k:H + F * (k + 1)] = 0
    sched = diffusion.scheduler
    if t_index is None:
        t_index = int(torch.randint(0, 1000, (1,), generator=gen).item())
    tval = sched.timesteps.to(device=device, dtype=torch.float32)[t_index]
    n = torch.randn((F, *clean.shape[2:]), generator=gen).to(device, clean.dtype)
    noise = torch.cat([torch.zeros_like(clean[:, :H])] + [n[None]] * K, 1)
    t = torch.zeros(1, clean.shape[1], device=device)
    t[:, H:] = tval
    noisy = clean.clone()
    fut = sched.add_noise(clean[0, H:].float(), noise[0, H:].float(), t[0, H:]).to(clean.dtype)
    noisy[0, H:] = fut
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = tree_call(root, noisy, tokens(t), prompt, cam, layout, action_tokens=tokens(acts.float()).long())
    target = (noise - clean).float()
    w = sched.training_weight(t[0, H:]).float()[0]
    p = pred[0, H:].float().unflatten(0, (K, F))  # [K,F,C,h,w]
    g = target[0, H:].unflatten(0, (K, F))
    mse = w * ((p - g) ** 2).mean()
    pairs = list(itertools.combinations(range(K), 2))
    if not pairs:  # single trajectory: no counterfactual pairs
        return mse, mse.detach(), torch.zeros((), device=device)
    diff = w * torch.stack([(((p[a] - p[b]) - (g[a] - g[b])) ** 2).mean() for a, b in pairs]).mean()
    return mse + lam * diff, mse.detach(), diff.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--cam-dir", default=str(REPO / "data/cf_vizdoom_10k_cam"))
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--init", default=str(REPO / "runs/teacher_ft/wasd_v2/teacher_action_step000500.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-action", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="counterfactual difference loss weight")
    ap.add_argument("--accum", type=int, default=1, help="forks per rank per optimizer step (1 node x --accum 2 = the "
                    "2-node batch)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--action-drop", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-forks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-v3", action="store_true",
                    help="roaming/switch Doom forks + SolarWM subset 1:1, true/nominal/static camera (tree/cond_v3.py)")
    ap.add_argument("--p-true", type=float, default=0.4)
    ap.add_argument("--p-static", type=float, default=0.2)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    from solarwm.backends.wan22.runtime.modeling.causal_model import CausalWanAttentionBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    diffusion, _ = build_teacher(args.init)
    inner = diffusion.module
    enable_tree(inner)
    fsdp = FSDP(inner, auto_wrap_policy=ModuleWrapPolicy({CausalWanAttentionBlock}),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=torch.bfloat16),
                device_id=device, use_orig_params=True, sync_module_states=False, limit_all_gathers=True)
    apply_activation_checkpointing(fsdp, checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, CausalWanAttentionBlock))
    diffusion.module = fsdp
    fsdp.train()
    act = [p for n, p in fsdp.named_parameters() if is_action_param(n)]
    base = [p for n, p in fsdp.named_parameters() if not is_action_param(n)]
    opt = torch.optim.AdamW([{"params": base, "lr": args.lr}, {"params": act, "lr": args.lr_action}],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    base_lrs = [args.lr, args.lr_action]

    if args.data_v3:
        from shownow.tree.cond_v3 import MixSampler, V3Dataset, pick_condition, v3_split
        td, ts, vd, _ = v3_split(seed=args.seed)
        if rank == 0:
            print(f"v3 data: doom train {len(td)} swm train {len(ts)} doom val {len(vd)}", flush=True)
        ds = V3Dataset(td + ts)
        sampler = MixSampler(len(td), len(ts), rank, world, args.seed)
        val_ood = vd
        val_ds = lambda files: V3Dataset(files)  # noqa: E731
    else:
        train_f, val_ood, _ = split_files(Path(args.data), args.seed)
        ds = ForkDataset(train_f, args.cam_dir, "nominal")
        sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        val_ds = lambda files: ForkDataset(files, args.cam_dir, "nominal")  # noqa: E731
    cgen = torch.Generator().manual_seed(args.seed * 7 + rank)
    modes = {}
    loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=4, prefetch_factor=4,
                                         persistent_workers=True)
    gen = torch.Generator().manual_seed(args.seed * 1000 + rank)

    def evaluate(step):
        fsdp.eval()
        mine = val_ood[rank::world][: max(1, args.val_forks // world)] or val_ood[rank % len(val_ood):][:1]
        vds = val_ds(mine)  # every rank runs the same number of forwards (FSDP collectives)
        rows = []
        with torch.no_grad():
            for i in range(len(vds)):
                fork = vds[i]
                if args.data_v3:
                    fork["cam_rel"] = fork["cam_nom"]  # validate with what inference provides
                wrong = fork["actions"].clone()
                wrong[:, H:] = next_action(wrong[:, H:], wrong.shape[0])
                cam_w = fork["cam_rel"][next_branch(fork["cam_rel"].shape[0])]
                r = []
                for ti in (250, 500, 750):
                    for kw in ({}, {"actions": wrong, "cam_rel": cam_w}):
                        g = torch.Generator().manual_seed(ti)
                        _, mse, diff = tree_loss(diffusion, fsdp, fork, device, args.cam_scale, args.lam, t_index=ti,
                                                 gen=g, **kw)
                        r.append([mse.item(), diff.item()])
                r = torch.tensor(r).view(3, 2, 2).mean(0)  # [cond(ok,swapped), (mse, diff)]
                rows.append(r.flatten())
        t = torch.stack(rows).to(device).mean(0) if rows else torch.zeros(4, device=device)
        dist.all_reduce(t)
        t /= world
        fsdp.train()
        if rank == 0:
            rec = {"step": step, "val_mse": t[0].item(), "val_diff": t[1].item(), "val_mse_swapped": t[2].item(),
                   "val_diff_swapped": t[3].item(), "condition_gap": (t[2] - t[0]).item()}
            print("VAL", json.dumps(rec), flush=True)
            with open(out / "val.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def save(step):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True,
                                                                                           rank0_only=True)):
            sd = fsdp.state_dict()
        if rank == 0:
            sd = {k.replace("_checkpoint_wrapped_module.", ""): v.to(torch.bfloat16) for k, v in sd.items()}
            torch.save({"generator": sd, "step": step, "args": vars(args), "layout": "bidir_tree"},
                       out / f"teacher_tree_step{step:06d}.pt")
            print(f"saved step {step}", flush=True)
        dist.barrier()

    evaluate(0)
    step, t0, epoch, micro = 0, time.time(), 0, 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for fork in loader:
            for g_, b in zip(opt.param_groups, base_lrs):
                g_["lr"] = b * min(1.0, (step + 1) / args.warmup)
            if args.data_v3:
                m_ = pick_condition(fork, cgen, args.p_true, args.p_static, args.p_uncond)
                modes[m_] = modes.get(m_, 0) + 1
            loss, mse, diff = tree_loss(diffusion, fsdp, fork, device, args.cam_scale, args.lam, gen=gen,
                                        action_drop=args.action_drop)
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            gn = fsdp.clip_grad_norm_(1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                v = torch.stack([loss.detach(), mse, diff])
                dist.all_reduce(v)
                v /= world
                if rank == 0:
                    rec = {"step": step, "loss": v[0].item(), "mse": v[1].item(), "diff": v[2].item(),
                           "grad_norm": float(gn), "sec_per_step": (time.time() - t0) / step, "modes_rank0": modes}
                    print("TRAIN", json.dumps(rec), flush=True)
                    with open(out / "train.jsonl", "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
            if step % args.val_every == 0:
                evaluate(step)
            if step % args.save_every == 0 or step == args.steps:
                save(step)
            if step >= args.steps:
                break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
