"""Action-conditioned fine-tuning of the SolarWM-5B Stage0.5 teacher on ViZDoom counterfactual forks (W/A/S/D).

Design: docs/phase2_teacher_finetune_design.md.  Launch (1 node, 8 GPUs):
  torchrun --standalone --nproc-per-node=8 -m shownow.train.teacher_action_ft --out runs/teacher_ft/<name>
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SolarWM/src"))

FRAME_TOKENS = 405
LATENTS = 21
FUTURE0 = 9  # first future latent
# action-id vocabulary: 0 none, 1 W, 2 A, 3 S, 4 D (4-action data). 8-action data (W A S D TURN_L TURN_R ATTACK JUMP,
# shownow/data/vizdoom_preencode_8a.py) uses SHOWNOW_N_ACTIONS=9; every script building the model reads it here.
N_ACTIONS = int(os.environ.get("SHOWNOW_N_ACTIONS", "5"))


def next_action(ids: torch.Tensor, K: int) -> torch.Tensor:
    """future action ids 1..K -> the next branch's id (K=4: W->A, A->S, S->D, D->W), for wrong-action probes"""
    return ids % K + 1


def next_branch(K: int) -> list:
    """branch index permutation matching next_action (K=4: [1, 2, 3, 0])"""
    return [(i + 1) % K for i in range(K)]
VAL_MAPS = {"MAP29", "MAP30", "MAP31", "MAP32"}
FX_NORM = 969.6969696969696 / (960.0 * 2)
FY_NORM = 969.6969696969696 / (540.0 * 2)


# ----------------------------------------------------------------------------- model
class ActionEmbed(torch.nn.Module):
    """Per-token action id -> additive time-embedding offset (DiT-style class conditioning).
    Last layer zero-initialised: at step 0 the model is exactly the pretrained teacher."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.table = torch.nn.Embedding(N_ACTIONS, hidden)
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden, dim))
        torch.nn.init.normal_(self.table.weight, std=0.02)
        torch.nn.init.zeros_(self.proj[1].weight)
        torch.nn.init.zeros_(self.proj[1].bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.table(ids))


def is_action_param(name: str) -> bool:
    """Parameters of the action conditioning (own learning rate): time-path embed + token-path embed."""
    return "action_embed" in name or "action_in" in name


ACTION_IN_MISSING_OK = ("action_in.",)  # token-path action embedding: absent in older checkpoints


def action_input(module, tok: torch.Tensor, ids) -> torch.Tensor:
    """Token-path action conditioning: tok [B, L, dim] + action_in(ids [B, L]) (zero-initialised: no-op at start)."""
    if ids is None or not hasattr(module, "action_in"):
        return tok
    return tok + module.action_in(ids.reshape(-1)).view(*ids.shape, -1).to(tok.dtype)


def add_action_conditioning(module: torch.nn.Module) -> None:
    """Two paths: action_embed adds to the time embedding (adaLN modulation of every block, as before) and action_in
    adds a per-token vector to the patch-embedded tokens (direct residual-stream signal)."""
    module.action_embed = ActionEmbed(int(module.dim))
    module.action_in = ActionEmbed(int(module.dim))
    module._action_tokens = None
    orig = module._time_condition

    def _time_condition(self, timestep, reference, r_timestep=None):
        out = orig(timestep, reference, r_timestep)
        ids = self._action_tokens
        if ids is not None:
            n, m = timestep.shape[-1], ids.shape[-1]
            if n == 2 * m:  # causal teacher forcing: [clean | noisy] tokens of the same latents
                ids = torch.cat([ids, ids], dim=-1)
            elif n < m:  # AR window: ids cover [clean history | current block]; head modulation uses the block only
                ids = ids[..., m - n:]
            if ids.shape != timestep.shape:
                raise RuntimeError(f"action tokens {tuple(ids.shape)} != timestep {tuple(timestep.shape)}")
            out = out + self.action_embed(ids.reshape(-1)).to(out.dtype)
        return out

    module._time_condition = types.MethodType(_time_condition, module)


def build_teacher(weights: str):
    from solarwm.backends.wan22.runtime.checkpoint import normalize_model_state
    from solarwm.backends.wan22.runtime.components import build_diffusion_architecture

    cfg = yaml.safe_load((REPO / "SolarWM/configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml").read_text())
    cfg["model"]["base_path"] = str(REPO / "models/SolarWM-5B-base")
    diffusion = build_diffusion_architecture(cfg)
    payload = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    field = "generator"  # release default for SolarWM-5B-bid-stage0p5-81f is "live"
    state = {k: v for k, v in payload[field].items() if not k.startswith("_shownow_inner.")}
    has_action = any(k.startswith("action_embed.") for k in state)
    if not has_action:  # released Stage0.5 weights
        state = normalize_model_state(state, field=field)
    add_action_conditioning(diffusion.module)
    own = diffusion.module.state_dict()
    for k in ("action_embed.table.weight", "action_in.table.weight"):  # smaller vocabulary in the checkpoint (e.g. a
        if k in state and state[k].shape[0] < own[k].shape[0]:     # 4-action model loaded with SHOWNOW_N_ACTIONS=9)
            grown = own[k].clone().to(state[k].dtype)
            grown[: state[k].shape[0]] = state[k]
            print(f"build_teacher: grew {k} {tuple(state[k].shape)} -> {tuple(grown.shape)}", flush=True)
            state[k] = grown
    res = diffusion.module.load_state_dict(state, strict=False)
    missing = [k for k in res.missing_keys if not (k.startswith("action_embed.") and not has_action)
               and not k.startswith(ACTION_IN_MISSING_OK)]
    assert not missing and not res.unexpected_keys, (missing[:5], res.unexpected_keys[:5])
    del payload, state
    return diffusion, cfg


# ----------------------------------------------------------------------------- data
class ForkDataset(torch.utils.data.Dataset):
    def __init__(self, files, cam_dir=None, camera="rot"):
        self.files, self.cam_dir, self.camera = files, cam_dir, camera

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fork = torch.load(self.files[i], map_location="cpu", weights_only=False)
        if self.camera != "rot":  # per-branch relative c2w [4,21,4,4] (shownow/data/vizdoom_camera_traj.py)
            fork["cam_rel"] = torch.load(Path(self.cam_dir) / self.files[i].name, weights_only=True)[self.camera]
        return fork


def split_files(root: Path, seed: int = 0):
    """Map-disjoint validation (MAP29-32) + 2% in-distribution validation; map per fork from the source manifest."""
    maps = {}
    shards = sorted(root.glob("index_shard*.json"))  # roaming/switch data (vizdoom_preencode_sw.py)
    if shards:
        for f in shards:
            maps.update(json.loads(f.read_text()))
    else:
        for line in open(REPO / "data/cf_vizdoom_10k/train_manifest.jsonl"):
            r = json.loads(line)
            maps[r["sample_id"]] = r["map"]
    keep = set(json.loads((root / "keep.json").read_text())) if (root / "keep.json").exists() else None  # checked forks
    rng = random.Random(seed)
    train, val_ood, val_id = [], [], []
    for f in sorted(root.glob("*.pt")):
        mp = maps.get(f.stem)
        if mp is None or (keep is not None and f.stem not in keep):
            continue
        if mp in VAL_MAPS:
            val_ood.append(f)
        elif rng.random() < 0.02:
            val_id.append(f)
        else:
            train.append(f)
    return train, val_ood, val_id


def camera_tokens(c2w: torch.Tensor, n: int, device, scale: float = 1.0) -> dict:
    """Relative c2w, [21,4,4] shared by the n clips (rotation-only) or [n,21,4,4] per clip (with translation)
    -> PRoPE viewmats (w2c) and K per token."""
    c2w = c2w.float().clone()
    if c2w.ndim == 3:
        c2w = c2w[None].expand(n, -1, -1, -1).clone()
    c2w[..., :3, 3] *= scale
    w2c = torch.linalg.inv(c2w)
    view = w2c[:, :, None].expand(n, LATENTS, FRAME_TOKENS, 4, 4).reshape(n, -1, 4, 4)
    K = torch.zeros(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2] = FX_NORM, FY_NORM, 0.5, 0.5, 1.0
    K = K.expand(n, view.shape[1], 3, 3)
    return {"viewmats": view.contiguous().to(device), "K": K.contiguous().to(device)}


# ----------------------------------------------------------------------------- loss
def fork_loss(diffusion, fsdp_model, fork, device, t_index=None, action_override=None, action_drop=0.0, gen=None,
              cam_scale=1.0):
    """Counterfactual micro-batch: the 4 branches (W/A/S/D) of one fork, same noise and same t."""
    from solarwm.backends.wan22.objectives import weighted_masked_mse
    from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens

    clean = fork["latents"].to(device)  # [4,21,48,30,54] bf16
    B = clean.shape[0]
    prompt = {"prompt_embeds": fork["prompt"].to(device)[None].expand(B, -1, -1)}
    cam = camera_tokens(fork.get("cam_rel", fork["c2w_rot"]), B, device, cam_scale)
    actions = fork["actions"].to(device).clone() if action_override is None else action_override.to(device)
    if action_drop > 0:
        drop = torch.rand(B, generator=gen, device="cpu") < action_drop
        actions[drop.to(device), FUTURE0:] = 0
    sched = diffusion.scheduler
    if t_index is None:
        t_index = int(torch.randint(0, 1000, (1,), generator=gen).item())
    tval = sched.timesteps.to(device=device, dtype=torch.float32)[t_index]
    timestep = torch.zeros(B, LATENTS, device=device)
    timestep[:, FUTURE0:] = tval
    noise = torch.randn(clean.shape[1:], generator=gen, dtype=torch.float32).to(device, torch.bfloat16)
    noise = noise[None].expand(B, *noise.shape).contiguous()
    noisy_f = sched.add_noise(clean[:, FUTURE0:].flatten(0, 1).float(), noise[:, FUTURE0:].flatten(0, 1).float(),
                              timestep[:, FUTURE0:].flatten(0, 1)).unflatten(0, (B, LATENTS - FUTURE0))
    noisy = clean.clone()
    noisy[:, FUTURE0:] = noisy_f.to(clean.dtype)
    ttok = expand_timesteps_to_tokens(timestep, FRAME_TOKENS)
    fsdp_model._shownow_inner._action_tokens = expand_timesteps_to_tokens(actions.float(), FRAME_TOKENS).long()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = diffusion(noisy, prompt, cam, ttok, sequence_length=LATENTS * FRAME_TOKENS)
    fsdp_model._shownow_inner._action_tokens = None
    target = sched.training_target(clean, noise, timestep)
    weight = sched.training_weight(timestep).view(B, LATENTS, 1, 1, 1)
    mask = torch.zeros(B, LATENTS, dtype=torch.bool, device=device)
    mask[:, FUTURE0:] = True
    per_clip = torch.stack([weighted_masked_mse(pred[i:i + 1], target[i:i + 1], mask[i:i + 1], weight[i:i + 1])
                            for i in range(B)])
    return per_clip.mean(), per_clip.detach()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/cf_vizdoom_10k_lat"))
    ap.add_argument("--teacher", default=str(REPO / "models/SolarWM-5B-bid-stage0p5-81f/model.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-action", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--action-drop", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-forks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="rot", choices=["rot", "nominal", "true"],
                    help="future camera: rotation only (v1), action-derived nominal trajectory, or real trajectory")
    ap.add_argument("--cam-dir", default=str(REPO / "data/cf_vizdoom_10k_cam"))
    ap.add_argument("--cam-scale", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="optional checkpoint to start from (e.g. teacher v1)")
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

    diffusion, cfg = build_teacher(args.teacher)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=True, mmap=True)["generator"]
        sd = {k: v for k, v in sd.items() if not k.startswith("_shownow_inner.")}
        for k in ("action_embed.table.weight", "action_in.table.weight"):  # e.g. 4-action teacher -> 8 actions
            cur = diffusion.module.state_dict()[k]
            if k in sd and sd[k].shape[0] < cur.shape[0]:
                grown = cur.clone().to(sd[k].dtype)
                grown[: sd[k].shape[0]] = sd[k]
                sd[k] = grown
                if rank == 0:
                    print(f"init: grew {k} {tuple(sd[k].shape)} (rows >= old count keep their fresh init)", flush=True)
        res = diffusion.module.load_state_dict(sd, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        del sd
    inner = diffusion.module
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from solarwm.backends.wan22.runtime.modeling.causal_model import CausalWanAttentionBlock

    fsdp = FSDP(inner, auto_wrap_policy=ModuleWrapPolicy({CausalWanAttentionBlock}),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=torch.bfloat16),
                device_id=device, use_orig_params=True, sync_module_states=False, limit_all_gathers=True)
    apply_activation_checkpointing(fsdp, checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, CausalWanAttentionBlock))
    fsdp._shownow_inner = inner
    diffusion.module = fsdp
    fsdp.train()

    act_params = [p for n, p in fsdp.named_parameters() if is_action_param(n)]
    base_params = [p for n, p in fsdp.named_parameters() if not is_action_param(n)]
    opt = torch.optim.AdamW([{"params": base_params, "lr": args.lr}, {"params": act_params, "lr": args.lr_action}],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    base_lrs = [args.lr, args.lr_action]

    train_f, val_ood, val_id = split_files(Path(args.data), args.seed)
    if rank == 0:
        print(f"forks: train {len(train_f)}  val_ood(map-disjoint) {len(val_ood)}  val_id {len(val_id)}", flush=True)
        (out / "split.json").write_text(json.dumps({"train": [f.stem for f in train_f], "val_ood": [f.stem for f in val_ood],
                                                    "val_id": [f.stem for f in val_id]}))
    sampler = torch.utils.data.DistributedSampler(ForkDataset(train_f, args.cam_dir, args.camera), num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(ForkDataset(train_f, args.cam_dir, args.camera), batch_size=None, sampler=sampler, num_workers=4,
                                         prefetch_factor=4, persistent_workers=True)
    gen = torch.Generator().manual_seed(args.seed * 1000 + rank)

    def evaluate(step):
        """Map-disjoint val loss at fixed t, with correct vs swapped actions (action sensitivity); with a camera
        trajectory also the loss with action AND trajectory swapped between branches (full-condition sensitivity)."""
        fsdp.eval()
        files = val_ood[rank::world][: max(1, args.val_forks // world)]
        rows = []
        val_ds = ForkDataset(files, args.cam_dir, args.camera)
        with torch.no_grad():
            for fi in range(len(val_ds)):
                fork = val_ds[fi]
                for ti in (250, 500, 750):
                    g = torch.Generator().manual_seed(ti)
                    l_ok, _ = fork_loss(diffusion, fsdp, fork, device, t_index=ti, gen=g, cam_scale=args.cam_scale)
                    wrong = fork["actions"].clone()
                    wrong[:, FUTURE0:] = next_action(wrong[:, FUTURE0:], wrong.shape[0])
                    g = torch.Generator().manual_seed(ti)
                    l_bad, _ = fork_loss(diffusion, fsdp, fork, device, t_index=ti, action_override=wrong, gen=g,
                                         cam_scale=args.cam_scale)
                    none = fork["actions"].clone(); none[:, FUTURE0:] = 0
                    g = torch.Generator().manual_seed(ti)
                    l_none, _ = fork_loss(diffusion, fsdp, fork, device, t_index=ti, action_override=none, gen=g,
                                          cam_scale=args.cam_scale)
                    l_both = l_bad
                    if "cam_rel" in fork:  # W's trajectory for A, A's for S, ... (same permutation as next_action)
                        g = torch.Generator().manual_seed(ti)
                        l_both, _ = fork_loss(diffusion, fsdp, {**fork, "cam_rel": fork["cam_rel"][next_branch(fork["cam_rel"].shape[0])]}, device,
                                              t_index=ti, action_override=wrong, gen=g, cam_scale=args.cam_scale)
                    rows.append([l_ok.item(), l_bad.item(), l_none.item(), l_both.item()])
        t = torch.tensor(rows, device=device).mean(0) if rows else torch.zeros(4, device=device)
        dist.all_reduce(t)
        t /= world
        fsdp.train()
        if rank == 0:
            rec = {"step": step, "val_loss_correct_action": t[0].item(), "val_loss_swapped_action": t[1].item(),
                   "val_loss_no_action": t[2].item(), "action_gap": (t[1] - t[0]).item(),
                   "val_loss_swapped_action_and_camera": t[3].item(), "condition_gap": (t[3] - t[0]).item()}
            print("VAL", json.dumps(rec), flush=True)
            with open(out / "val.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def save(step):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg_sd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg_sd):
            sd = fsdp.state_dict()
        if rank == 0:
            sd = {k.replace("_checkpoint_wrapped_module.", ""): v.to(torch.bfloat16) for k, v in sd.items()
                  if not k.startswith("_shownow_inner.")}
            torch.save({"generator": sd, "step": step, "args": vars(args)}, out / f"teacher_action_step{step:06d}.pt")
            print(f"saved step {step}", flush=True)
        dist.barrier()

    evaluate(0)
    step, t0, epoch = 0, time.time(), 0
    while step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for fork in loader:
            lr_scale = min(1.0, (step + 1) / args.warmup)
            for g_, b in zip(opt.param_groups, base_lrs):
                g_["lr"] = b * lr_scale
            loss, per_clip = fork_loss(diffusion, fsdp, fork, device, action_drop=args.action_drop, gen=gen,
                                       cam_scale=args.cam_scale)
            loss.backward()
            gn = fsdp.clip_grad_norm_(1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                l = loss.detach().clone()
                dist.all_reduce(l)
                if rank == 0:
                    rec = {"step": step, "loss": (l / world).item(), "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"],
                           "sec_per_step": (time.time() - t0) / step}
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
