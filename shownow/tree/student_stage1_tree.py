"""BRANCH-TREE Stage1 AR student: like shownow/train/student_stage1.py, but one sample = the whole fork as a branch tree
(teacher-forcing layout [clean H+4 branches | noisy H+4 branches], tree mask, shared RoPE positions), so the shared
history is computed once per fork. AnyFlow-v1.5 objective over the 57 noisy slots (slot 0 = TI2V anchor).

Baseline docstring follows.
Stage1 AR student (causal teacher forcing, AnyFlow-v1.5 flow map) on ViZDoom counterfactual forks (W/A/S/D).

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
import itertools
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
from shownow.train.teacher_action_ft import (ACTION_IN_MISSING_OK, FRAME_TOKENS, LATENTS, ForkDataset,  # noqa: E402
                                             add_action_conditioning, camera_tokens, is_action_param, next_action,
                                             next_branch, split_files)
from shownow.tree.teacher_tree_ft import tree_inputs  # noqa: E402
from shownow.tree.tree import enable_tree, tf_layout, tree_call  # noqa: E402

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
    missing = [k for k in res.missing_keys if not k.startswith("delta_embedding.") and not k.startswith(ACTION_IN_MISSING_OK)]
    assert not missing and not res.unexpected_keys, (missing[:5], res.unexpected_keys[:5])
    if from_teacher:
        assert any(k.startswith("delta_embedding.") for k in res.missing_keys), \
            "expected delta_embedding to be missing in a teacher checkpoint"
        assert diffusion.module.initialize_anyflow_delta_from_time()
    else:
        assert not [k for k in res.missing_keys if not k.startswith(ACTION_IN_MISSING_OK)]
    return diffusion, cfg


def subsample_branches(fork: dict, n: int, gen) -> dict:
    """Random subset of n branches of a fork (memory knob; all branches if the fork has <= n). Every per-branch
    tensor (leading dim = number of branches: latents, actions, camera trajectories) is sliced with the same indices;
    future action ids keep their meaning (the branch's own key), so the subset is still a valid counterfactual tree."""
    K = fork["latents"].shape[0]
    if n <= 0 or K <= n:
        return fork
    idx = torch.randperm(K, generator=gen)[:n].sort().values
    out = dict(fork)
    for k, v in fork.items():
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == K and k != "prompt":
            out[k] = v[idx]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--cam-dir", default=str(REPO / "data/cf_vizdoom_10k_cam"))
    ap.add_argument("--camera", default="nominal", choices=["rot", "nominal", "true"])
    ap.add_argument("--cam-scale", type=float, default=0.5)
    ap.add_argument("--init", required=True, help="tree teacher checkpoint (Stage0.5 layout)")
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
    ap.add_argument("--resume", default=None, help="tree Stage1 checkpoint to continue from (live weights; optimizer "
                    "moments and EMA restart)")
    ap.add_argument("--resume-warmup", type=int, default=20)
    ap.add_argument("--block", type=int, default=BLOCK, help="latents per AR chunk (3 = SolarWM B3, 1 = per-latent B1)")
    ap.add_argument("--history", type=int, default=15, help="clean prior latents a noisy chunk attends to")
    ap.add_argument("--data-v3", action="store_true",
                    help="roaming/switch Doom forks + SolarWM subset 1:1, true/nominal/static camera (tree/cond_v3.py)")
    ap.add_argument("--p-true", type=float, default=0.4)
    ap.add_argument("--p-static", type=float, default=0.2)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--lam-cf", type=float, default=0.0, help="counterfactual difference loss weight (K > 1 forks)")
    ap.add_argument("--cf-every", type=int, default=2, help="apply the counterfactual loss every n steps")
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per optimizer step (1 node x 8 GPUs with --accum 3 = the 24-GPU effective "
                         "batch; micro-batch m uses AnyFlow logical rank rank + world * m of world * accum)")
    args = ap.parse_args()
    assert args.history % args.block == 0

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

    start_step = 0
    if args.resume:
        diffusion, cfg = build_student(args.resume, from_teacher=False, field="generator")
        start_step = int(torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)["step"])
    else:
        diffusion, cfg = build_student(args.init, from_teacher=True)
    cfg["train"]["anyflow_negative_embedding"] = "conditioning/wan_negemb_cn.pth"
    negative = load_negative_prompt_embedding(cfg, device=device)
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
    ema = ShardedEMA(fsdp, decay=args.ema_decay, device=device, dtype=torch.float32)

    act_params = [p for n, p in fsdp.named_parameters() if is_action_param(n)]
    base_params = [p for n, p in fsdp.named_parameters() if not is_action_param(n)]
    opt = torch.optim.AdamW([{"params": base_params, "lr": args.lr}, {"params": act_params, "lr": args.lr_action}],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    sched = make_warmup_cosine(opt, warmup_steps=args.warmup, total_steps=args.steps, min_lr_ratio=0.1)
    for _ in range(start_step):  # continue the cosine schedule where the resumed run stopped
        sched.step()
    resume_lrs = [g["lr"] for g in opt.param_groups]

    if args.data_v3:
        from shownow.tree.cond_v3 import MixSampler, V3Dataset, pick_condition, static_future, v3_split
        td, ts, val_ood, _ = v3_split(seed=args.seed)
        if rank == 0:
            print(f"v3 data: doom train {len(td)} swm train {len(ts)} doom val {len(val_ood)} world {world}", flush=True)
        ds = V3Dataset(td + ts)
        sampler = MixSampler(len(td), len(ts), rank, world, args.seed)
        val_ds = lambda files: V3Dataset(files)  # noqa: E731
    else:
        train_f, val_ood, _ = split_files(Path(args.data), args.seed)
        if rank == 0:
            print(f"forks: train {len(train_f)}  val_ood {len(val_ood)}  world {world}", flush=True)
        ds = ForkDataset(train_f, args.cam_dir, args.camera)
        sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        val_ds = lambda files: ForkDataset(files, args.cam_dir, args.camera)  # noqa: E731
    cgen = torch.Generator().manual_seed(args.seed * 7 + rank)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=4, prefetch_factor=4,
                                         persistent_workers=True)
    gen = torch.Generator().manual_seed(args.seed * 1000 + rank)
    mc = cfg["model"]
    shift, ntt = float(mc["timestep_shift"]), 1000

    layouts = {}

    def layout_for(K):
        if K not in layouts:
            layouts[K] = tf_layout(K, max_prior=args.history // args.block, block=args.block)
        return layouts[K]

    def make_model_u(fork, clean, prompt, cam, act_tok, layout):
        """AnyFlow u(x, t, r | c). The guidance (unconditional) branch gets the negative prompt, no action and - with
        --data-v3 - a STATIC future camera, so guidance amplifies the motion that action + camera ask for."""
        cam2 = {n: torch.cat([cam[n], cam[n]], 1) for n in ("viewmats", "K")}
        if args.data_v3:
            _, _, cam_s, _ = tree_inputs({**fork, "cam_rel": static_future(fork["cam_rel"])}, device, args.cam_scale)
            cam2_u = {n: torch.cat([cam_s[n], cam_s[n]], 1) for n in ("viewmats", "K")}
        else:
            cam2_u = cam2
        none_tok = torch.zeros_like(act_tok)
        zero = torch.zeros(1, clean.shape[1], device=device)

        def model_u(sample, t_value, r_value, c):
            a = c.get("action_tokens", none_tok)  # guidance branch: negative prompt, no action
            cc = cam2 if "action_tokens" in c else cam2_u
            tt = torch.cat([zero, t_value], 1)
            rr = torch.cat([zero, r_value], 1)
            return tree_call(fsdp, torch.cat([clean, sample], 1), expand_timesteps_to_tokens(tt, FRAME_TOKENS),
                             c["prompt_embeds"], cc, layout, r_tok=expand_timesteps_to_tokens(rr, FRAME_TOKENS),
                             action_tokens=torch.cat([a, a], 1))
        return model_u

    def cf_loss(fork, gen_):
        """Counterfactual difference loss on the instantaneous velocity (r = t): branches share the noise, so the
        target difference between branches a, b is clean_b - clean_a; rewards the action-dependent part directly."""
        clean, prompt, cam, acts = tree_inputs(fork, device, args.cam_scale)
        K = fork["latents"].shape[0]
        act_tok = expand_timesteps_to_tokens(acts.float(), FRAME_TOKENS).long()
        model_u = make_model_u(fork, clean, prompt, cam, act_tok, layout_for(K))
        F_ = (clean.shape[1] - 9) // K
        ti = int(torch.randint(20, 980, (1,), generator=gen_).item())
        tval = diffusion.scheduler.timesteps.to(device=device, dtype=torch.float32)[ti]
        n_h = torch.randn((9, *clean.shape[2:]), generator=gen_).to(device, clean.dtype)
        n_f = torch.randn((F_, *clean.shape[2:]), generator=gen_).to(device, clean.dtype)
        noise = torch.cat([n_h[None]] + [n_f[None]] * K, 1)
        t = torch.full((1, clean.shape[1]), float(tval), device=device)
        noisy = diffusion.scheduler.add_noise(clean[0].float(), noise[0].float(), t[0]).to(clean.dtype)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model_u(noisy, t, t, {"prompt_embeds": prompt, "action_tokens": act_tok})
        p = pred[0, 9:].float().unflatten(0, (K, F_))
        g = (noise - clean)[0, 9:].float().unflatten(0, (K, F_))
        w = diffusion.scheduler.training_weight(t[0, 9:]).float().mean()
        pairs = list(itertools.combinations(range(K), 2))
        if not pairs:  # single trajectory: same forward/backward (FSDP collectives must match on every rank), zero loss
            return 0.0 * pred.float().mean()
        return w * torch.stack([(((p[a] - p[b]) - (g[a] - g[b])) ** 2).mean() for a, b in pairs]).mean()

    def objective(fork, logical_rank, logical_world, action_override=None, cam_override=None, gen_=None, drop=0.0):
        clean, prompt, cam, acts = tree_inputs(fork, device, args.cam_scale, action_override, cam_override)
        K = fork["latents"].shape[0]
        layout = layout_for(K)
        if drop > 0:
            for k in range(K):
                if torch.rand(1, generator=gen_).item() < drop:
                    acts[:, 9 + 12 * k:21 + 12 * k] = 0
        act_tok = expand_timesteps_to_tokens(acts.float(), FRAME_TOKENS).long()
        cond = {"prompt_embeds": prompt, "action_tokens": act_tok}
        n_slots = clean.shape[1]
        model_u = make_model_u(fork if cam_override is None else {**fork, "cam_rel": cam_override}, clean, prompt,
                               cam, act_tok, layout)

        mask = torch.ones(1, n_slots, dtype=torch.bool, device=device)
        mask[:, 0] = False  # TI2V first-latent anchor
        noise = None
        if gen_ is not None:
            noise = torch.randn(clean.shape, generator=gen_).to(device, clean.dtype)
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
        files = val_ood[rank::world][:2] or val_ood[rank % len(val_ood):][:1]
        vds = val_ds(files)
        rows = []
        with torch.no_grad():
            for i in range(len(vds)):
                fork = vds[i]
                if args.data_v3:
                    fork["cam_rel"] = fork["cam_nom"]  # validate with what inference provides
                wrong = fork["actions"].clone()
                wrong[:, 9:] = next_action(wrong[:, 9:], wrong.shape[0])
                cam_w = fork["cam_rel"][next_branch(fork["cam_rel"].shape[0])] if "cam_rel" in fork else None
                r = []
                # diffusion-type samples only (logical ranks 0-1 of 4): with one tree per call the consistency /
                # flow-map losses have no diffusion reference for AnyFlow's adaptive rescale -> not comparable
                for lr_ in (0, 1):
                    for ao, co in ((None, None), (wrong, cam_w)):
                        g = torch.Generator().manual_seed(11 + lr_)
                        r.append(objective(fork, lr_, 4, ao, co, gen_=g).loss.item())
                rows.append([sum(r[0::2]) / 2, sum(r[1::2]) / 2])
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
            torch.save({**payload, "step": step, "args": vars(args), "stage": "stage1_anyflow_v1_5", "layout": "tf_tree"},
                       out / f"student_stage1_step{step:06d}.pt")
            print(f"saved step {step}", flush=True)
        dist.barrier()

    if start_step == 0:
        evaluate(0)
    step, t0, epoch = start_step, time.time(), 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        micro = []
        for fork in loader:
            micro.append(fork)
            if len(micro) < args.accum:
                continue
            if args.resume and step - start_step < args.resume_warmup:
                for g_, lr0 in zip(opt.param_groups, resume_lrs):
                    g_["lr"] = lr0 * (step - start_step + 1) / args.resume_warmup
            cf = None
            for m_i, fork in enumerate(micro):
                fork = subsample_branches(fork, args.branches, cgen)
                if args.data_v3:
                    pick_condition(fork, cgen, args.p_true, args.p_static, args.p_uncond)
                res = objective(fork, rank + world * m_i, world * args.accum, gen_=None, drop=args.action_drop)
                loss = res.loss
                if args.lam_cf > 0 and step % args.cf_every == 0:  # every rank (zero for single trajectories)
                    cf = cf_loss(fork, cgen)
                    loss = loss + args.lam_cf * cf
                (loss / args.accum).backward()
            micro = []
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
                           "cf_rank0": None if cf is None else cf.item(),
                           "lr": opt.param_groups[0]["lr"], "sec_per_step": (time.time() - t0) / max(1, step - start_step)}
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
