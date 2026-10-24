"""Branch-tree forward equivalence checks (run on 1 GPU):
  python tests/tree/test_tree_equivalence.py
1) full layout (K=1, all visible) == native bidirectional teacher forward        (plumbing: RoPE, camera, time, action)
2) bidir tree K=4 branch outputs == bidir tree K=1 per branch                      (branch isolation, shared positions)
3) tf tree K=1 == native forward_train_tf; tf tree K=4 == K=1 per branch           (causal student training layout)
4) window tree K=4 == native forward_inference_window per branch                   (cache-free AR inference layout)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SOLARWM_COMPILE_FLEX", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
import torch  # noqa: E402

from shownow.train.student_common import tokens  # noqa: E402
from shownow.train.teacher_action_ft import build_teacher, camera_tokens  # noqa: E402
from shownow.tree.tree import (FRAME_TOKENS, bidir_layout, enable_tree, full_layout, tf_layout, tree_call,  # noqa: E402
                               window_layout)

dev = torch.device("cuda")
DT = torch.float32 if os.environ.get("TREE_FP32") else torch.bfloat16
BLK = int(os.environ.get("TREE_BLOCK", "3"))  # latents per chunk (3 = B3, 1 = B1); history window 15 latents
MAXP = 15 // BLK
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
fork = torch.load(REPO / "data/cf_vizdoom_10k_lat/cf_00009.pt", weights_only=False)
cam_rel = torch.load(REPO / "data/cf_vizdoom_10k_cam/cf_00009.pt", weights_only=True)["nominal"]
clean = fork["latents"].to(dev, DT)  # [4,21,C,H,W]
acts = fork["actions"].to(dev)  # [4,21]
prompt = fork["prompt"].to(dev, DT)[None]
cam4 = camera_tokens(cam_rel, 4, dev, 0.5)  # per branch [4, 21*405, ...]


def rel(a, b):
    """relative L2 error"""
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()


def tree_cam(cams_per_slot):
    return {k: torch.cat([c[k] for c in cams_per_slot], 1) for k in ("viewmats", "K")}


def branch_cam(k, lo, hi):
    return {n: cam4[n][k:k + 1, lo * FRAME_TOKENS:hi * FRAME_TOKENS] for n in ("viewmats", "K")}


def act_tok(k, lo, hi):
    return tokens(acts[k:k + 1, lo:hi].float()).long()


results = {}
# ---------------------------------------------------------------- teacher (bidirectional)
teacher, _ = build_teacher(str(REPO / "runs/teacher_ft/wasd_v2/teacher_action_step000500.pt"))
inner = teacher.module.to(dev, DT).eval()
enable_tree(inner)
x = clean.clone()
x[:, 9:] = torch.randn_like(x[:, 9:])
t = torch.zeros(4, 21, device=dev)
t[:, 9:] = 600.0
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16):
    inner._action_tokens = tokens(acts[:1].float()).long()
    native = teacher(x[:1], {"prompt_embeds": prompt}, branch_cam(0, 0, 21), tokens(t[:1]), sequence_length=21 * 405)
    inner._action_tokens = None
    full = tree_call(inner, x[:1], tokens(t[:1]), prompt, branch_cam(0, 0, 21), full_layout(21),
                     action_tokens=tokens(acts[:1].float()).long())
    results["1 full==native bidir"] = rel(full, native)

    # K=4 tree: [H | W | A | S | D]
    L4 = bidir_layout(4)
    xs = torch.cat([x[0:1, :9]] + [x[k:k + 1, 9:] for k in range(4)], 1)
    ts = torch.cat([t[0:1, :9]] + [t[k:k + 1, 9:] for k in range(4)], 1)
    cams = tree_cam([branch_cam(0, 0, 9)] + [branch_cam(k, 9, 21) for k in range(4)])
    at = torch.cat([act_tok(0, 0, 9)] + [act_tok(k, 9, 21) for k in range(4)], 1)
    out4 = tree_call(inner, xs, tokens(ts), prompt, cams, L4, action_tokens=at)
    L1 = bidir_layout(1)
    errs = []
    for k in range(4):
        o1 = tree_call(inner, torch.cat([x[0:1, :9], x[k:k + 1, 9:]], 1), tokens(torch.cat([t[0:1, :9], t[k:k + 1, 9:]], 1)),
                       prompt, tree_cam([branch_cam(0, 0, 9), branch_cam(k, 9, 21)]), L1,
                       action_tokens=torch.cat([act_tok(0, 0, 9), act_tok(k, 9, 21)], 1))
        errs.append(rel(out4[:, 9 + 12 * k:21 + 12 * k], o1[:, 9:]))
        if k == 0:
            errs.append(rel(out4[:, :9], o1[:, :9]))
    results["2 bidir tree K4 == K1 per branch (max)"] = max(errs)
    # how much does removing history->future attention change the teacher? (not an equivalence, informative)
    results["info: bidir tree vs native bidir (future, branch W)"] = rel(out4[:, 9:21], native[:, 9:])
del teacher, inner
torch.cuda.empty_cache()

# ---------------------------------------------------------------- causal student (Stage1 AnyFlow)
from shownow.train.student_stage1 import build_student  # noqa: E402
stu, _ = build_student(str(REPO / "runs/student_stage1/smoke/student_stage1_step000020.pt"), from_teacher=False,
                       field="generator_ema")
inner = stu.module.to(dev, DT).eval()
inner.max_prior_clean_chunks = MAXP
enable_tree(inner)
noisy = clean.clone()
noisy[:, 1:] = torch.randn_like(noisy[:, 1:])
tn = torch.full((4, 21), 700.0, device=dev)
tn[:, 0] = 0
rn = torch.full((4, 21), 300.0, device=dev)
rn[:, 0] = 0
aug = torch.zeros(4, 21, device=dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16):
    nat = []
    for k in range(4):
        inner._action_tokens = tokens(acts[k:k + 1].float()).long()
        nat.append(stu.forward_train_tf(noisy[k:k + 1], clean[k:k + 1], {"prompt_embeds": prompt}, branch_cam(k, 0, 21),
                                        tokens(tn[:1]), tokens(aug[:1]), num_frame_per_block=BLK, sequence_length=21 * 405,
                                        r_timestep_tokens=tokens(rn[:1]), augmentation_r_timestep_tokens=tokens(aug[:1])))
        inner._action_tokens = None
    inner._action_tokens = tokens(acts.float()).long()
    nat_b4 = stu.forward_train_tf(noisy, clean, {"prompt_embeds": prompt.expand(4, -1, -1)}, cam4, tokens(tn), tokens(aug),
                                  num_frame_per_block=BLK, sequence_length=21 * 405, r_timestep_tokens=tokens(rn),
                                  augmentation_r_timestep_tokens=tokens(aug))
    inner._action_tokens = None
    results["floor: native TF batch4 vs batch1 (max over branches)"] = max(rel(nat_b4[k:k + 1], nat[k]) for k in range(4))
    # tree K=4: slots [clean H | clean W..D futures | noisy H | noisy W..D futures]
    Lt = tf_layout(4, max_prior=MAXP, block=BLK)
    cl = [clean[0:1, :9]] + [clean[k:k + 1, 9:] for k in range(4)]
    ny = [noisy[0:1, :9]] + [noisy[k:k + 1, 9:] for k in range(4)]
    xs = torch.cat(cl + ny, 1)
    tt = torch.cat([aug[0:1, :9]] + [aug[k:k + 1, 9:] for k in range(4)] + [tn[0:1, :9]] + [tn[k:k + 1, 9:] for k in range(4)], 1)
    rr = torch.cat([aug[0:1, :9]] + [aug[k:k + 1, 9:] for k in range(4)] + [rn[0:1, :9]] + [rn[k:k + 1, 9:] for k in range(4)], 1)
    cams1 = [branch_cam(0, 0, 9)] + [branch_cam(k, 9, 21) for k in range(4)]
    at1 = [act_tok(0, 0, 9)] + [act_tok(k, 9, 21) for k in range(4)]
    out = tree_call(inner, xs, tokens(tt), prompt, tree_cam(cams1 + cams1), Lt, r_tok=tokens(rr),
                    action_tokens=torch.cat(at1 + at1, 1))
    # out = noisy slots: [H(9) | W | A | S | D]
    errs = [rel(out[:, :9], nat[0][:, :9])] + [rel(out[:, 9 + 12 * k:21 + 12 * k], nat[k][:, 9:]) for k in range(4)]
    results["3 tf tree K4 == native TF per branch (max)"] = max(errs)
    results["3 (history slots vs native, branch W)"] = errs[0]

    # ---------------------------------------------------------------- window (block 3: history 0-8 -> noisy 9-11)
    nb = torch.randn(4, BLK, *clean.shape[2:], device=dev, dtype=clean.dtype)
    tw = torch.full((4, BLK), 500.0, device=dev)
    rw = torch.full((4, BLK), 250.0, device=dev)
    natw = []
    for k in range(4):
        inner._action_tokens = torch.cat([act_tok(k, 0, 9), act_tok(k, 9, 9 + BLK)], 1)
        natw.append(stu.forward_inference_window(nb[k:k + 1], clean[k:k + 1, :9], {"prompt_embeds": prompt},
                                                 branch_cam(k, 0, 9 + BLK), tokens(tw[:1]), num_frame_per_block=BLK,
                                                 sequence_length=BLK * 405, clean_history_timestep=0.0,
                                                 r_timestep_tokens=tokens(rw[:1])))
        inner._action_tokens = None
    Lw = window_layout(4, hist_frames=9, gen_frames=0, block=BLK)
    xs = torch.cat([clean[0:1, :9]] + [nb[k:k + 1] for k in range(4)], 1)
    tt = torch.cat([torch.zeros(1, 9, device=dev)] + [tw[k:k + 1] for k in range(4)], 1)
    rr = torch.cat([torch.zeros(1, 9, device=dev)] + [rw[k:k + 1] for k in range(4)], 1)
    outw = tree_call(inner, xs, tokens(tt), prompt,
                     tree_cam([branch_cam(0, 0, 9)] + [branch_cam(k, 9, 9 + BLK) for k in range(4)]), Lw,
                     r_tok=tokens(rr), action_tokens=torch.cat([act_tok(0, 0, 9)] + [act_tok(k, 9, 9 + BLK) for k in range(4)], 1))
    results["4 window tree K4 == native window per branch (max)"] = max(
        rel(outw[:, BLK * k:BLK * k + BLK], natw[k]) for k in range(4))

for k, v in results.items():
    print(f"{k:60s} {v:.3e}")
