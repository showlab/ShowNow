"""Stage1 AR student (causal teacher forcing, AnyFlow-v1.5 flow map) on ViZDoom counterfactual forks (W/A/S/D).

Same conditioning as teacher v2 (shownow/train/teacher_action_ft.py): per-latent action embedding added to the time
embedding + camera = action-derived nominal trajectory (x cam_scale). Objective, sample-type mix, fused guidance and
gated (t,r) embedding follow SolarWM `train_stage1_tf_anyflow_v1_5_81f.yaml`; the loop is ours because SolarWM's engine is
tied to its online WDS pipeline. Init: teacher v2 weights (Stage0.5 layout) -> causal AnyFlow model, delta embedding
copied from the time embedding (as SolarWM does from Stage0.5).

Launch: sbatch slurm/train_3node.sbatch shownow.train.student_stage1 --out runs/student_stage1/<name> ...
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SolarWM/src"))
from shownow.train.teacher_action_ft import (FRAME_TOKENS, LATENTS, ForkDataset, add_action_conditioning,  # noqa: E402
                                             camera_tokens, split_files, is_action_param, next_action, next_branch)

BLOCK = 3
STAGE1_CFG = REPO / "SolarWM/configs/examples/wan22_ti2v_5b/train_stage1_tf_anyflow_v1_5_81f.yaml"


def load_state(path: str, field: str = "generator") -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sd = sd.get(field, sd)
    return {k: v for k, v in sd.items() if not k.startswith("_shownow_inner.")}


def build_student(init: str, from_teacher: bool, field: str = "generator"):
    """Causal AnyFlow Wan + action embedding. from_teacher: init from a Stage0.5-layout checkpoint (delta <- time)."""
    from solarwm.backends.wan22.runtime.components import build_diffusion_architecture

    cfg = yaml.safe_load(STAGE1_CFG.read_text())
    cfg["model"]["base_path"] = str(REPO / "models/SolarWM-5B-base")
    diffusion = build_diffusion_architecture(cfg)
    add_action_conditioning(diffusion.module)
    res = diffusion.module.load_state_dict(load_state(init, field), strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("delta_embedding.")]
    assert not missing and not res.unexpected_keys, (missing[:5], res.unexpected_keys[:5])
    if from_teacher:
        assert res.missing_keys, "expected delta_embedding to be missing in a teacher checkpoint"
        assert diffusion.module.initialize_anyflow_delta_from_time()
    else:
        assert not res.missing_keys
    return diffusion, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--cam-dir", default=str(REPO / "data/cf_vizdoom_10k_cam"))
    ap.add_argument("--camera", default="nominal", choices=["rot", "nominal", "true"])
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--init", default=str(REPO / "runs/teacher_ft/wasd_v2/teacher_action_step000500.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-action", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--branches", type=int, default=4, help="branches of a fork per micro-batch (memory knob)")
    ap.add_argument("--action-drop", type=float, default=0.1)
    ap.add_argument("--guidance", type=float, default=3.0, help="AnyFlow fused guidance (uncond = neg. prompt, no action)")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=BLOCK, help="latents per AR step (SolarWM 3; 1 = per-latent AR)")
    ap.add_argument("--history", type=int, default=15, help="max clean history latents a block attends to")
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
    from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens
    from solarwm.backends.wan22.runtime.stage1_anyflow import anyflow_v15_objective, load_negative_prompt_embedding
    from solarwm.training.ema import ShardedEMA
    from solarwm.training.schedule import make_warmup_cosine
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    diffusion, cfg = build_student(args.init, from_teacher=True)
    # block-causal geometry: history window measured in blocks (SolarWM: 5 blocks x 3 latents = 15 latents)
    assert args.history % args.block == 0 and LATENTS % args.block == 0
    diffusion.module.max_prior_clean_chunks = args.history // args.block
    cfg["train"]["anyflow_negative_embedding"] = "conditioning/wan_negemb_cn.pth"
    negative = load_negative_prompt_embedding(cfg, device=device)
    inner = diffusion.module
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
    ema = ShardedEMA(fsdp, decay=args.ema_decay, device=device, dtype=torch.float32)

    act_params = [p for n, p in fsdp.named_parameters() if is_action_param(n)]
    base_params = [p for n, p in fsdp.named_parameters() if not is_action_param(n)]
    opt = torch.optim.AdamW([{"params": base_params, "lr": args.lr}, {"params": act_params, "lr": args.lr_action}],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    sched = make_warmup_cosine(opt, warmup_steps=args.warmup, total_steps=args.steps, min_lr_ratio=0.1)

    train_f, val_ood, _ = split_files(Path(args.data), args.seed)
    if rank == 0:
        print(f"forks: train {len(train_f)}  val_ood {len(val_ood)}  world {world}", flush=True)
    ds = ForkDataset(train_f, args.cam_dir, args.camera)
    sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=4, prefetch_factor=4,
                                         persistent_workers=True)
    gen = torch.Generator().manual_seed(args.seed * 1000 + rank)
    mc = cfg["model"]
    shift, ntt = float(mc["timestep_shift"]), 1000

    def objective(fork, logical_rank, logical_world, action_override=None, cam_override=None, gen_=None, drop=0.0):
        B = args.branches
        idx = torch.arange(4) if B == 4 else torch.randperm(4, generator=gen_)[:B].sort().values
        clean = fork["latents"][idx].to(device)
        cam = camera_tokens((fork["cam_rel"] if cam_override is None else cam_override)[idx]
                            if "cam_rel" in fork else fork["c2w_rot"], B, device, args.cam_scale)
        acts = (fork["actions"] if action_override is None else action_override)[idx].clone()
        if drop > 0:
            acts[torch.rand(B, generator=gen_) < drop] = 0
        act_tok = expand_timesteps_to_tokens(acts.float().to(device), FRAME_TOKENS).long()
        cond = {"prompt_embeds": fork["prompt"].to(device)[None].expand(B, -1, -1), "action_tokens": act_tok}
        aug = expand_timesteps_to_tokens(torch.zeros(B, LATENTS, device=device), FRAME_TOKENS)
        none_tok = torch.zeros_like(act_tok)

        def model_u(sample, t_value, r_value, c):
            inner._action_tokens = c.get("action_tokens", none_tok)  # guidance branch: negative prompt, no action
            try:
                return diffusion.forward_train_tf(
                    sample, clean, c, cam, expand_timesteps_to_tokens(t_value, FRAME_TOKENS), aug,
                    num_frame_per_block=args.block, sequence_length=LATENTS * FRAME_TOKENS,
                    r_timestep_tokens=expand_timesteps_to_tokens(r_value, FRAME_TOKENS),
                    augmentation_r_timestep_tokens=aug)
            finally:
                inner._action_tokens = None

        mask = torch.ones(B, LATENTS, dtype=torch.bool, device=device)
        mask[:, 0] = False  # TI2V first-latent anchor
        noise = None
        if gen_ is not None:  # deterministic validation noise
            noise = torch.randn(clean.shape[1:], generator=gen_).to(device, clean.dtype)[None].expand_as(clean)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return anyflow_v15_objective(
                clean=clean, condition=cond, model_u=model_u, scheduler=diffusion.scheduler, loss_mask=mask,
                logical_dp_rank=logical_rank, logical_dp_world_size=logical_world, shift=shift,
                num_train_timesteps=ntt, epsilon=float(cfg["train"]["anyflow_epsilon"]),
                diffusion_ratio=float(cfg["train"]["anyflow_diffusion_ratio"]),
                consistency_ratio=float(cfg["train"]["anyflow_consistency_ratio"]),
                guidance=args.guidance, negative_prompt_embeds=negative, noise=noise,
                generator=None if gen_ is None else torch.Generator(device=device).manual_seed(int(gen_.initial_seed())),
                gather_fn=(lambda x: x) if gen_ is not None else None)

    def evaluate(step):
        """Map-disjoint val objective (fixed noise / time pairs) with correct vs swapped action+trajectory."""
        fsdp.eval()
        files = val_ood[rank::world][:2]
        vds = ForkDataset(files, args.cam_dir, args.camera)
        rows = []
        with torch.no_grad():
            for i in range(len(vds)):
                fork = vds[i]
                wrong = fork["actions"].clone()
                wrong[:, 9:] = next_action(wrong[:, 9:], wrong.shape[0])
                cam_w = fork["cam_rel"][next_branch(fork["cam_rel"].shape[0])] if "cam_rel" in fork else None
                r = []
                for seed in (11, 22):
                    for ao, co in ((None, None), (wrong, cam_w)):
                        g = torch.Generator().manual_seed(seed)
                        r.append(objective(fork, 0, 1, ao, co, gen_=g).loss.item())
                rows.append([(r[0] + r[2]) / 2, (r[1] + r[3]) / 2])
        t = torch.tensor(rows, device=device).mean(0) if rows else torch.zeros(2, device=device)
        dist.all_reduce(t)
        t /= world
        fsdp.train()
        if rank == 0:
            rec = {"step": step, "val_loss": t[0].item(), "val_loss_swapped_condition": t[1].item(),
                   "condition_gap": (t[1] - t[0]).item()}
            print("VAL", json.dumps(rec), flush=True)
            with open(out / "val.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def save(step):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg_sd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        payload = {}
        for name in ("generator", "generator_ema"):
            with ema.swapped_into(fsdp) if name == "generator_ema" else contextlib.nullcontext():
                with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg_sd):
                    sd = fsdp.state_dict()
            if rank == 0:
                payload[name] = {k.replace("_checkpoint_wrapped_module.", ""): v.to(torch.bfloat16)
                                 for k, v in sd.items() if not k.startswith("_shownow_inner.")}
            del sd
        if rank == 0:
            torch.save({**payload, "step": step, "args": vars(args), "stage": "stage1_anyflow_v1_5"},
                       out / f"student_stage1_step{step:06d}.pt")
            print(f"saved step {step}", flush=True)
        dist.barrier()

    evaluate(0)
    step, t0, epoch = 0, time.time(), 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for fork in loader:
            res = objective(fork, rank, world, gen_=None, drop=args.action_drop)
            res.loss.backward()
            gn = fsdp.clip_grad_norm_(1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ema.update(fsdp, decay=0.0 if step < 50 else args.ema_decay)
            step += 1
            if step % 10 == 0:
                l = res.loss.detach().clone()
                dist.all_reduce(l)
                if rank == 0:
                    rec = {"step": step, "loss": (l / world).item(), "grad_norm": float(gn),
                           "lr": opt.param_groups[0]["lr"], "sec_per_step": (time.time() - t0) / step}
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
