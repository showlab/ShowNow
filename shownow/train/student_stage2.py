"""Stage2 SGF/DMD few-step AR student on ViZDoom counterfactual forks (W/A/S/D), history-prefixed.

Follows SolarWM `train_stage2_sgf_81f.yaml`: critic update every outer step, student every 5th; student rollout with KV
cache and 4-step schedule [1000,750,500,250], random exit step, gradient through a teacher-forcing replay; DMD gradient
from real score (teacher, CFG 3) vs fake score (critic); critic trained with flow matching on the student's samples.
Differences (ours): the 2 s real history (latents 0-8) is a clean prefix committed to the KV cache and excluded from all
losses; every role gets the action embedding (uncond branch = negative prompt + no action, camera kept) and the nominal
camera trajectory; teacher/critic = our action teacher v2 (Stage0.5 layout); student = Stage1 EMA with the AnyFlow delta
embedding dropped (as SolarWM does).

Launch: sbatch slurm/train_3node.sbatch shownow.train.student_stage2 --student <stage1 ckpt> --out runs/student_stage2/<name>
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
from shownow.train.student_common import HIST_LATENTS, history_rollout, tokens  # noqa: E402
from shownow.train.teacher_action_ft import (ACTION_IN_MISSING_OK, LATENTS, ForkDataset,  # noqa: E402
                                             add_action_conditioning, build_teacher, camera_tokens, is_action_param,
                                             split_files)

STAGE2_CFG = REPO / "SolarWM/configs/examples/wan22_ti2v_5b/train_stage2_sgf_81f.yaml"


def build_stage2_student(path: str, field: str = "generator_ema", block: int = 3, history: int = 15):
    """block = latents per AR step, history = clean history latents a block attends to (SolarWM: 3 / 15)."""
    from solarwm.backends.wan22.runtime.components import build_diffusion_architecture

    cfg = yaml.safe_load(STAGE2_CFG.read_text())
    cfg["model"]["base_path"] = str(REPO / "models/SolarWM-5B-base")
    assert history % block == 0
    cfg["model"].update(num_frame_per_block=block, max_prior_clean_chunks=history // block,
                        local_attn_size=history + block)
    diffusion = build_diffusion_architecture(cfg)
    add_action_conditioning(diffusion.module)
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        sd = payload[field] if field in payload else payload["generator"]
    dropped = [k for k in sd if k.startswith("delta_embedding.")]
    sd = {k: v for k, v in sd.items() if not k.startswith("delta_embedding.") and not k.startswith("_shownow_inner.")}
    res = diffusion.module.load_state_dict(sd, strict=False)
    assert not [k for k in res.missing_keys if not k.startswith(ACTION_IN_MISSING_OK)] and not res.unexpected_keys, \
        (res.missing_keys[:5], res.unexpected_keys[:5])
    return diffusion, cfg, dropped


def load_full(fsdp_module, sd: dict) -> None:
    """Write a full (unsharded, wrapper-free key names) state dict into an FSDP(use_orig_params) module."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    seen = set()
    with FSDP.summon_full_params(fsdp_module, writeback=True, rank0_only=False), torch.no_grad():
        for name, p in fsdp_module.named_parameters():
            key = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
            if key not in sd and key.startswith(ACTION_IN_MISSING_OK):
                continue  # older checkpoint without the token-path action embedding: keep the zero init
            p.copy_(sd[key].to(device=p.device, dtype=p.dtype))
            seen.add(key)
    missing = [k for k in sd if k not in seen and not k.startswith("delta_embedding.")]
    assert not missing, missing[:5]


def fsdp_wrap(module, device, train: bool):
    from solarwm.backends.wan22.runtime.modeling.causal_model import CausalWanAttentionBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    f = FSDP(module, auto_wrap_policy=ModuleWrapPolicy({CausalWanAttentionBlock}),
             sharding_strategy=ShardingStrategy.FULL_SHARD,
             mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                            buffer_dtype=torch.bfloat16,
                                            # the root cast would copy the fp32 KV-cache camera metadata to bf16,
                                            # so cache commits land in throwaway copies (K stays 0 -> NaN)
                                            cast_root_forward_inputs=False),
             device_id=device, use_orig_params=True, sync_module_states=False, limit_all_gathers=True)
    if train:
        apply_activation_checkpointing(f, checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, CausalWanAttentionBlock))
    return f


def denoising_list(args, cfg) -> list:
    """student denoising timesteps (raw, before warping): --denoising-steps or the SolarWM config's 4-step list"""
    if args.denoising_steps:
        return [int(x) for x in args.denoising_steps.split(",")]
    return cfg["train"]["denoising_step_list"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--cam-dir", default=str(REPO / "data/cf_vizdoom_10k_cam"))
    ap.add_argument("--camera", default="nominal")
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--student", required=True, help="Stage1 checkpoint (EMA weights used)")
    ap.add_argument("--teacher", default=str(REPO / "runs/teacher_ft/wasd_v2/teacher_action_step000500.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000, help="outer steps (critic every step, student every 5th)")
    ap.add_argument("--ratio", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lr-action", type=float, default=1e-5)
    ap.add_argument("--critic-lr", type=float, default=4e-7)
    ap.add_argument("--denoising-steps", default="",
                    help="comma list overriding the config's denoising_step_list [1000,750,500,250], e.g. 1000,667,333")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--branches", type=int, default=2)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--ema-start", type=int, default=39, help="student steps before EMA starts")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--block", type=int, default=3, help="latents per AR step (must match the Stage1 student)")
    ap.add_argument("--history", type=int, default=15, help="clean history latents a block attends to")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="Stage2 checkpoint to continue from: live student <- 'generator', EMA <- 'generator_ema', "
                         "critic <- 'critic' if saved (else re-initialised from the teacher), step counters continue")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="outer steps after (re)start with critic updates only (use when the critic is fresh)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed * 7919 + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    from solarwm.backends.wan22.objectives import apply_timestep_shift  # noqa: F401
    from solarwm.backends.wan22.sgf import (compute_kl_gradient, reference_cfg, sample_shifted_score_timesteps,
                                            sgf_critic_flow_loss, sgf_student_loss, should_update_student,
                                            warp_denoising_steps)
    from solarwm.training.ema import ShardedEMA
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    resume = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True) if args.resume else None
    student, cfg, dropped = build_stage2_student(args.resume or args.student, block=args.block, history=args.history)
    assert resume is not None or len(dropped) == 4, dropped
    s_inner = student.module
    student.module = fsdp_wrap(s_inner, device, train=True)
    ema, start_step, student_step = None, 0, 0
    if resume is not None:  # module holds the EMA weights now -> EMA shadow; then live weights <- 'generator'
        ema = ShardedEMA(student.module, decay=args.ema_decay, device=device, dtype=torch.float32)
        load_full(student.module, resume["generator"])
        start_step, student_step = int(resume["step"]), int(resume["student_step"])
    teacher, _ = build_teacher(args.teacher)
    t_inner = teacher.module
    t_inner.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    critic, _ = build_teacher(args.teacher)
    c_inner = critic.module
    critic.module = fsdp_wrap(c_inner, device, train=True)
    if resume is not None and "critic" in resume:
        load_full(critic.module, resume["critic"])
    if rank == 0 and resume is not None:
        print(f"RESUME from {args.resume}: step {start_step}, student_step {student_step}, critic "
              f"{'restored' if 'critic' in resume else 'fresh from teacher'}", flush=True)
    del resume
    neg = torch.load(REPO / "models/SolarWM-5B-base/conditioning/wan_negemb_cn.pth", weights_only=True)
    neg = neg["negative_prompt_embeds"].reshape(1, -1, 4096).to(device, torch.bfloat16)

    def adam(params_groups, betas=(0.0, 0.999)):
        return torch.optim.AdamW(params_groups, betas=betas, eps=1e-8, weight_decay=0.0)
    s_act = [p for n, p in student.module.named_parameters() if is_action_param(n)]
    s_base = [p for n, p in student.module.named_parameters() if not is_action_param(n)]
    s_opt = adam([{"params": s_base, "lr": args.lr}, {"params": s_act, "lr": args.lr_action}])
    c_opt = adam([{"params": list(critic.module.parameters()), "lr": args.critic_lr}])
    steps_raw = warp_denoising_steps((denoising_list(args, cfg)), student.scheduler.timesteps.to(device),
                                     num_train_timesteps=1000)
    steps_raw = [float(s) for s in steps_raw]
    cache_frames = int(cfg["model"]["local_attn_size"])
    root = s_inner

    train_f, val_ood, _ = split_files(Path(args.data), args.seed)
    ds = ForkDataset(train_f, args.cam_dir, args.camera)
    sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=4, prefetch_factor=4,
                                         persistent_workers=True)

    def batches():
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            epoch += 1
            yield from loader
    it = batches()

    def prepare(fork):
        B = args.branches
        idx = torch.randperm(4)[:B].sort().values
        clean = fork["latents"][idx].to(device)
        cam = camera_tokens(fork["cam_rel"][idx] if "cam_rel" in fork else fork["c2w_rot"], B, device, args.cam_scale)
        acts = fork["actions"][idx].to(device)
        cond = {"prompt_embeds": fork["prompt"].to(device)[None].expand(B, -1, -1)}
        uncond = {"prompt_embeds": neg.expand(B, -1, -1)}
        return clean, cond, uncond, cam, acts

    def score(model, inner, noisy, cond, cam, t, acts):
        inner._action_tokens = tokens(acts.float()).long()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                flow = model(noisy, cond, cam, tokens(t), sequence_length=LATENTS * 405)
        finally:
            inner._action_tokens = None
        return flow, model.flow_to_x0(noisy, flow, t)

    def noised(x, clean):
        B = x.shape[0]
        t = sample_shifted_score_timesteps(batch_size=B, num_frames=LATENTS, device=device, shift=5.0,
                                           num_train_timesteps=1000, min_timestep=20.0, max_timestep=980.0)
        t[:, :HIST_LATENTS] = 0.0
        noise = torch.randn_like(x)
        xn = student.scheduler.add_noise(x.flatten(0, 1).float(), noise.flatten(0, 1).float(),
                                         t.flatten(0, 1)).unflatten(0, x.shape[:2]).to(x.dtype)
        xn[:, :HIST_LATENTS] = clean[:, :HIST_LATENTS]
        return xn, t, noise

    def save(step):
        cfg_sd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        payload = {}
        for name in ("generator", "generator_ema"):
            if name == "generator_ema" and ema is None:
                continue
            with ema.swapped_into(student.module) if name == "generator_ema" else contextlib.nullcontext():
                with FSDP.state_dict_type(student.module, StateDictType.FULL_STATE_DICT, cfg_sd):
                    sd = student.module.state_dict()
            if rank == 0:
                payload[name] = {k.replace("_checkpoint_wrapped_module.", ""): v.to(torch.bfloat16)
                                 for k, v in sd.items()}
            del sd
        with FSDP.state_dict_type(critic.module, StateDictType.FULL_STATE_DICT, cfg_sd):
            sd = critic.module.state_dict()  # saved so the next continuation keeps the critic
        if rank == 0:
            payload["critic"] = {k.replace("_checkpoint_wrapped_module.", ""): v.to(torch.bfloat16)
                                 for k, v in sd.items()}
        del sd
        if rank == 0:
            torch.save({**payload, "step": step, "student_step": student_step, "args": vars(args),
                        "stage": "stage2_sgf"}, out / f"student_stage2_step{step:06d}.pt")
            print(f"saved step {step}", flush=True)
        dist.barrier()

    t0 = time.time()
    student.module.eval()  # SolarWM runs all roles in eval mode (no dropout); grads still flow
    critic.module.eval()
    for step in range(start_step + 1, args.steps + 1):
        rec = {"step": step}
        if should_update_student(step - 1, args.ratio) and step > start_step + args.critic_warmup:
            s_opt.zero_grad(set_to_none=True)
            clean, cond, uncond, cam, acts = prepare(next(it))
            ro = history_rollout(student, s_inner, root, clean, cond, cam, acts, steps_raw, cache_frames,
                                 require_grad=True, block=args.block)
            gen = ro.output
            xn, t, _ = noised(gen.detach(), clean)
            none = torch.zeros_like(acts)
            with torch.no_grad():
                _, fake = score(critic, c_inner, xn, cond, cam, t, acts)
                _, real_c = score(teacher, t_inner, xn, cond, cam, t, acts)
                _, real_u = score(teacher, t_inner, xn, uncond, cam, t, none)
                real = reference_cfg(real_c, real_u, args.guidance)
                grad = compute_kl_gradient(fake_x0=fake, real_x0=real, student_output=gen, mask=ro.loss_mask,
                                           normalize=True)
            loss = sgf_student_loss(gen, grad, mask=ro.loss_mask)
            loss.backward()
            rec["student_grad_norm"] = float(student.module.clip_grad_norm_(10.0))
            s_opt.step()
            student_step += 1
            rec.update(loss_student=loss.item(), dmd_grad=float(grad.abs().mean()), exit=ro.exit_index)
            if student_step >= args.ema_start:
                if ema is None:
                    ema = ShardedEMA(student.module, decay=args.ema_decay, device=device, dtype=torch.float32)
                else:
                    ema.update(student.module)
        c_opt.zero_grad(set_to_none=True)
        clean, cond, uncond, cam, acts = prepare(next(it))
        with torch.no_grad():
            ro = history_rollout(student, s_inner, root, clean, cond, cam, acts, steps_raw, cache_frames,
                                 require_grad=False, block=args.block)
        gen = ro.output.detach()
        xn, t, noise = noised(gen, clean)
        flow, _ = score(critic, c_inner, xn, cond, cam, t, acts)
        closs = sgf_critic_flow_loss(flow, noise=noise, clean=gen, mask=ro.loss_mask)
        if step <= 3 or not torch.isfinite(closs):
            print(f"FINITE rank {rank} step {step}: clean {bool(torch.isfinite(clean).all())} gen "
                  f"{bool(torch.isfinite(gen).all())} xn {bool(torch.isfinite(xn).all())} flow "
                  f"{bool(torch.isfinite(flow).all())} gen_std {gen[:, HIST_LATENTS:].float().std().item():.3f}",
                  flush=True)
        closs.backward()
        rec["critic_grad_norm"] = float(critic.module.clip_grad_norm_(10.0))
        c_opt.step()
        rec.update(loss_critic=closs.item(), student_step=student_step,
                   sec_per_step=(time.time() - t0) / (step - start_step))
        if rank == 0 and (step % 5 == 0 or step <= 3):
            print("TRAIN", json.dumps(rec), flush=True)
            with open(out / "train.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            save(step)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
