"""BRANCH-TREE Stage2 SGF/DMD student (all 4 branches of a fork per sample, shared history computed once).

Same recipe as shownow/train/student_stage2.py (critic every outer step, student every 5th, 4-step schedule, random
exit, DMD gradient teacher-CFG vs critic, critic flow matching), but:
  pass 1  = tree window rollout of the 4 futures together (sgf_tree_rollout, cache-free),
  pass 2  = gradient-carrying teacher-forcing replay in the tf_layout tree,
  scores  = teacher / critic in the bidirectional tree layout (57 slots instead of 4 x 21).
Launch: sbatch -N 1 -w <node> slurm/train_3node.sbatch shownow.tree.student_stage2_tree --student <tree s1 ckpt> ...
"""

from __future__ import annotations

import argparse
import contextlib
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
from shownow.train.student_stage2 import build_stage2_student, fsdp_wrap, load_full  # noqa: E402
from shownow.train.teacher_action_ft import ForkDataset, build_teacher, is_action_param, split_files  # noqa: E402
from shownow.tree.student_stage1_tree import subsample_branches  # noqa: E402
from shownow.tree.teacher_tree_ft import tree_inputs  # noqa: E402
from shownow.tree.tree import bidir_layout, enable_tree, tf_layout, tree_call  # noqa: E402
from shownow.tree.tree_samplers import pack_tree, sgf_tree_rollout  # noqa: E402

H, K = 9, 4


def denoising_list(args, cfg) -> list:
    """student denoising timesteps (raw, before warping): --denoising-steps or the SolarWM config's 4-step list"""
    if args.denoising_steps:
        return [int(x) for x in args.denoising_steps.split(",")]
    return cfg["train"]["denoising_step_list"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--cam-dir", default=str(REPO / "data/cf_vizdoom_10k_cam"))
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True, help="tree teacher checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--ratio", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lr-action", type=float, default=1e-5)
    ap.add_argument("--critic-lr", type=float, default=4e-7)
    ap.add_argument("--denoising-steps", default="",
                    help="comma list overriding the config's denoising_step_list [1000,750,500,250], e.g. 1000,667,333")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--ema-start", type=int, default=39)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--save-steps", default="", help="extra comma-separated outer steps to save (e.g. 1000,2000)")
    ap.add_argument("--resume", default="", help="continue a run from one of its checkpoints (same args, same --out): "
                    "generator, generator_ema (EMA state) and critic weights and the step counters are restored; the "
                    "optimizer moments are not saved and restart (AdamW betas (0, 0.999)); data order is reshuffled")
    ap.add_argument("--block", type=int, default=3, help="latents per AR chunk (must match the tree Stage1 student)")
    ap.add_argument("--history", type=int, default=15, help="clean prior latents a chunk attends to")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-v3", action="store_true",
                    help="roaming/switch Doom forks + SolarWM subset 1:1, true/nominal/static camera; teacher CFG "
                         "unconditional = negative prompt + no action + static camera (tree/cond_v3.py)")
    ap.add_argument("--p-true", type=float, default=0.3)
    ap.add_argument("--p-static", type=float, default=0.1)
    ap.add_argument("--branches", type=int, default=0,
                    help="random subset of a fork's branches per sample (memory knob, e.g. 4 for 8-action forks); 0 = all")
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per student / critic update (1 node x 8 GPUs with --accum 3 = 24-GPU batch)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    resume = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True) if args.resume else None
    start = resume["step"] + 1 if resume else 1
    rseed = args.seed + start - 1  # shuffling / noise seed (the data split keeps args.seed)
    torch.manual_seed(rseed * 7919 + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    from solarwm.backends.wan22.sgf import (compute_kl_gradient, reference_cfg, sample_shifted_score_timesteps,
                                            sgf_critic_flow_loss, sgf_student_loss, should_update_student,
                                            warp_denoising_steps)
    from solarwm.training.ema import ShardedEMA
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    student, cfg, dropped = build_stage2_student(args.resume or args.student, block=args.block, history=args.history)
    max_prior = args.history // args.block
    assert len(dropped) == (0 if resume else 4), dropped
    enable_tree(student.module)
    student.module = fsdp_wrap(student.module, device, train=True)
    teacher, _ = build_teacher(args.teacher)
    teacher.module.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    enable_tree(teacher.module)
    critic, _ = build_teacher(args.teacher)
    enable_tree(critic.module)
    critic.module = fsdp_wrap(critic.module, device, train=True)
    neg = torch.load(REPO / "models/SolarWM-5B-base/conditioning/wan_negemb_cn.pth", weights_only=True)
    neg = neg["negative_prompt_embeds"].reshape(1, -1, 4096).to(device, torch.bfloat16)

    s_act = [p for n, p in student.module.named_parameters() if is_action_param(n)]
    s_base = [p for n, p in student.module.named_parameters() if not is_action_param(n)]
    s_opt = torch.optim.AdamW([{"params": s_base, "lr": args.lr}, {"params": s_act, "lr": args.lr_action}],
                              betas=(0.0, 0.999), eps=1e-8, weight_decay=0.0)
    c_opt = torch.optim.AdamW(critic.module.parameters(), lr=args.critic_lr, betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0)
    ema = None
    if resume:  # the module holds generator_ema now: seed the EMA from it, then load the raw generator and the critic
        if "generator_ema" in resume:
            ema = ShardedEMA(student.module, decay=args.ema_decay, device=device, dtype=torch.float32)
        load_full(student.module, {k: v for k, v in resume["generator"].items() if not k.startswith("_shownow_inner.")})
        load_full(critic.module, {k: v for k, v in resume["critic"].items() if not k.startswith("_shownow_inner.")})
        if rank == 0:
            print(f"resumed {args.resume}: step {resume['step']}, student_step {resume['student_step']}, "
                  f"ema {'restored' if ema is not None else 'not started'}", flush=True)
    fewstep = [float(s) for s in warp_denoising_steps((denoising_list(args, cfg)),
                                                      student.scheduler.timesteps.to(device), num_train_timesteps=1000)]
    layouts = {}
    cur = {}  # per-sample layouts / mask / unconditional camera (K = 4 forks or K = 1 trajectories)

    if args.data_v3:
        from shownow.tree.cond_v3 import MixSampler, V3Dataset, pick_condition, static_future, v3_split
        td, ts, _, _ = v3_split(seed=args.seed)
        ds = V3Dataset(td + ts)
        sampler = MixSampler(len(td), len(ts), rank, world, rseed)
    else:
        train_f, _, _ = split_files(Path(args.data), args.seed)
        ds = ForkDataset(train_f, args.cam_dir, "nominal")
        sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=rseed)
    cgen = torch.Generator().manual_seed(rseed * 7 + rank)
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
        fork = subsample_branches(fork, args.branches, cgen)  # memory knob (8-action forks); 0 = all branches
        from shownow.train.teacher_action_ft import camera_tokens
        if args.data_v3:  # Stage2 rollouts: no unconditional samples (the teacher's CFG handles that side)
            pick_condition(fork, cgen, args.p_true, args.p_static, 0.0)
        K = fork["latents"].shape[0]
        if K not in layouts:
            m_ = torch.zeros(1, H + K * 12, 1, 1, 1, dtype=torch.bool, device=device)
            m_[:, H:] = True
            layouts[K] = (tf_layout(K, max_prior=max_prior, block=args.block), bidir_layout(K), m_)
        cur["L_tf"], cur["L_bi"], cur["mask"] = layouts[K]
        clean_t, prompt, cam_t, acts_t = tree_inputs(fork, device, args.cam_scale)  # tree-packed
        cur["cam_u"] = tree_inputs({**fork, "cam_rel": static_future(fork["cam_rel"])}, device, args.cam_scale)[2] \
            if args.data_v3 else cam_t
        cam_b = camera_tokens(fork["cam_rel"], K, device, float(fork.get("cam_scale", args.cam_scale)))  # per branch
        return fork["latents"].to(device), clean_t, prompt, cam_t, cam_b, fork["actions"].to(device), acts_t

    def rollout(lat, prompt, cam_b, acts, cam_t, acts_t, clean_t, require_grad):
        exit_index = int(torch.randint(0, len(fewstep), (1,), device=device).item())
        committed, noisy_exit = sgf_tree_rollout(student, student.module, lat, prompt, cam_b, acts, fewstep, exit_index,
                                                 max_prior_chunks=max_prior, block=args.block)
        c_t, n_t = pack_tree(committed), pack_tree(noisy_exit)
        if not require_grad:
            return c_t, exit_index
        t_exit = fewstep[exit_index]
        n = c_t.shape[1]
        tt = torch.cat([torch.zeros(1, n, device=device), torch.full((1, n), t_exit, device=device)], 1)
        tt[:, n:n + H] = 0.0  # noisy history slots = clean history, t = 0
        n_t[:, :H] = clean_t[:, :H]
        cam2 = {k: torch.cat([cam_t[k], cam_t[k]], 1) for k in ("viewmats", "K")}
        a2 = tokens(torch.cat([acts_t, acts_t], 1).float()).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flow = tree_call(student.module, torch.cat([c_t, n_t], 1), tokens(tt), prompt, cam2, cur["L_tf"],
                             action_tokens=a2)
        out = student.flow_to_x0(n_t, flow, tt[:, n:])
        out = torch.cat([clean_t[:, :H], out[:, H:]], 1)
        return out, exit_index

    def noised(x, clean_t):
        t = sample_shifted_score_timesteps(batch_size=1, num_frames=x.shape[1], device=device, shift=5.0,
                                           num_train_timesteps=1000, min_timestep=20.0, max_timestep=980.0)
        t[:, :H] = 0.0
        noise = torch.randn_like(x)
        xn = student.scheduler.add_noise(x[0].float(), noise[0].float(), t[0])[None].to(x.dtype)
        xn[:, :H] = clean_t[:, :H]
        return xn, t, noise

    def score(model, xn, prompt, cam_t, t, acts_t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flow = tree_call(model.module, xn, tokens(t), prompt, cam_t, cur["L_bi"],
                             action_tokens=tokens(acts_t.float()).long())
        return flow, model.flow_to_x0(xn, flow, t)

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
            sd = critic.module.state_dict()  # kept so a later continuation does not restart the critic
        if rank == 0:
            payload["critic"] = {k.replace("_checkpoint_wrapped_module.", ""): v.to(torch.bfloat16)
                                 for k, v in sd.items()}
        del sd
        if rank == 0:
            torch.save({**payload, "step": step, "student_step": student_step, "args": vars(args),
                        "stage": "stage2_sgf", "layout": "tree"}, out / f"student_stage2_tree_step{step:06d}.pt")
            print(f"saved step {step}", flush=True)
        dist.barrier()

    extra_saves = {int(v) for v in args.save_steps.split(",") if v}
    student_step, t0 = (resume["student_step"] if resume else 0), time.time()
    student.module.eval()
    critic.module.eval()
    for step in range(start, args.steps + 1):
        rec = {"step": step}
        if should_update_student(step - 1, args.ratio):
            s_opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                lat, clean_t, prompt, cam_t, cam_b, acts, acts_t = prepare(next(it))
                gen, ex = rollout(lat, prompt, cam_b, acts, cam_t, acts_t, clean_t, require_grad=True)
                xn, t, _ = noised(gen.detach(), clean_t)
                with torch.no_grad():
                    _, fake = score(critic, xn, prompt, cam_t, t, acts_t)
                    _, real_c = score(teacher, xn, prompt, cam_t, t, acts_t)
                    _, real_u = score(teacher, xn, neg, cur["cam_u"], t, torch.zeros_like(acts_t))
                    real = reference_cfg(real_c, real_u, args.guidance)
                    grad = compute_kl_gradient(fake_x0=fake, real_x0=real, student_output=gen, mask=cur["mask"],
                                               normalize=True)
                loss = sgf_student_loss(gen, grad, mask=cur["mask"])
                (loss / args.accum).backward()
            rec["student_grad_norm"] = float(student.module.clip_grad_norm_(10.0))
            s_opt.step()
            student_step += 1
            rec.update(loss_student=loss.item(), dmd_grad=float(grad.abs().mean()), exit=ex)
            if student_step >= args.ema_start:
                if ema is None:
                    ema = ShardedEMA(student.module, decay=args.ema_decay, device=device, dtype=torch.float32)
                else:
                    ema.update(student.module)
        c_opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            lat, clean_t, prompt, cam_t, cam_b, acts, acts_t = prepare(next(it))
            with torch.no_grad():
                gen, _ = rollout(lat, prompt, cam_b, acts, cam_t, acts_t, clean_t, require_grad=False)
            xn, t, noise = noised(gen, clean_t)
            flow, _ = score(critic, xn, prompt, cam_t, t, acts_t)
            closs = sgf_critic_flow_loss(flow, noise=noise, clean=gen, mask=cur["mask"])
            (closs / args.accum).backward()
        rec["critic_grad_norm"] = float(critic.module.clip_grad_norm_(10.0))
        c_opt.step()
        rec.update(loss_critic=closs.item(), student_step=student_step, sec_per_step=(time.time() - t0) / step)
        if rank == 0 and (step % 5 == 0 or step <= 3):
            print("TRAIN", json.dumps(rec), flush=True)
            with open(out / "train.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if step % args.save_every == 0 or step == args.steps or step in extra_saves:
            save(step)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
