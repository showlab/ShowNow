"""Zero-training check: can W/A/S/D be expressed as camera trajectories that SolarWM's camera prior follows?

For held-out forks (map-disjoint MAP29-32) generate the 4 futures (W/A/S/D) from the same clean history and the same
noise, under different camera conditions, and measure whether the generated futures match their own ground-truth
branch (4-way counterfactual retrieval, by latent distance and by optical-flow similarity).

Camera conditions (future latents 9-20; history always uses its real rotation, and real translation when the
condition has translation):
  rot        rotation only (what teacher v1 was trained with)            -> no action information at all
  rot_act    rotation only + v1 action embedding (v1 only)               -> v1's current capability
  true       real per-branch trajectory incl. collisions (upper bound, leaks the future)
  true_K     'true' with the real 130 deg Doom intrinsics instead of the SolarWM default K
  nom_s{x}[_act]  nominal trajectory from action + free-motion Doom kinematics (no collisions), translation scaled by x

Launch: 8 independent ranks, 1 GPU each.
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
from shownow.train.teacher_action_ft import FRAME_TOKENS, FUTURE0, LATENTS, FX_NORM, FY_NORM, build_teacher  # noqa: E402

BRANCHES = ["W", "A", "S", "D"]
AXIS = {"W": (2, 1.0), "S": (2, -1.0), "A": (0, -1.0), "D": (0, 1.0)}  # OpenCV camera: x right, z forward
LAT_PIX = np.array([0, *(1 + 4 * k for k in range(20))])
N_HIST = 33
SRC = REPO / "data/cf_vizdoom_10k"
LAT = REPO / "data/cf_vizdoom_10k_lat"
DOOM_K = (298.4369012191991 / 1280.0, 298.4369012191991 * 0.675 / 480.0)  # fx/W, fy/H after resize+crop


# ----------------------------------------------------------------------------- trajectories
def nominal_curves(train_ids: list, n: int = 600) -> dict:
    """Median free-motion displacement (metres, along the action axis, relative to the fork pose) per future frame,
    from training-map forks whose branch was not blocked (>= 12 m travelled in 3 s)."""
    rng = np.random.default_rng(0)
    ids = rng.choice(train_ids, size=min(n, len(train_ids)), replace=False)
    curves = {k: [] for k in BRANCHES}
    for sid in ids:
        m = json.load(open(SRC / sid / "meta.json"))
        fork = np.array(m["history"]["c2w"][-1])
        for k in BRANCHES:
            c = np.array(m["futures"][k]["c2w"])
            d = np.einsum("ij,nj->ni", fork[:3, :3].T, c[:, :3, 3] - fork[:3, 3])
            ax, sgn = AXIS[k]
            along = sgn * d[:, ax]
            if along[-1] >= 12.0:
                curves[k].append(along)
    return {k: np.median(np.stack(v), 0) for k, v in curves.items()}, {k: len(v) for k, v in curves.items()}


def branch_c2w_abs(m: dict, fork_pt: dict, k: str) -> np.ndarray:
    h_idx, f_idx = fork_pt["meta"]["history_src_idx"], fork_pt["meta"]["future_src_idx"]
    frames = [m["history"]["c2w"][i] for i in h_idx] + [m["futures"][k]["c2w"][i] for i in f_idx]
    return np.array(frames, dtype=np.float64)[LAT_PIX]  # [21,4,4]


def nominal_c2w_abs(m: dict, fork_pt: dict, k: str, curve: np.ndarray) -> np.ndarray:
    out = branch_c2w_abs(m, fork_pt, k).copy()
    fork = np.array(m["history"]["c2w"][-1], dtype=np.float64)
    f_idx = fork_pt["meta"]["future_src_idx"]
    ax, sgn = AXIS[k]
    for li in range(FUTURE0, LATENTS):
        src = f_idx[LAT_PIX[li] - N_HIST]
        d = np.zeros(3)
        d[ax] = sgn * curve[src]
        out[li] = fork.copy()
        out[li][:3, 3] = fork[:3, 3] + fork[:3, :3] @ d  # view direction fixed during futures
    return out


def relative(c2w: np.ndarray, scale: float = 1.0, rot_only: bool = False) -> np.ndarray:
    rel = np.einsum("ij,njk->nik", np.linalg.inv(c2w[0]), c2w)
    if rot_only:
        rel[:, :3, 3] = 0.0
    else:
        rel[:, :3, 3] *= scale
    return rel.astype(np.float32)


def camera_batch(rels: list, K: tuple, device) -> dict:
    """list of 4 [21,4,4] relative c2w -> PRoPE viewmats (w2c) and normalised K per token."""
    views = []
    for r in rels:
        w2c = torch.linalg.inv(torch.from_numpy(r).float())
        views.append(w2c[:, None].expand(LATENTS, FRAME_TOKENS, 4, 4).reshape(-1, 4, 4))
    view = torch.stack(views)
    Km = torch.zeros(3, 3)
    Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2], Km[2, 2] = K[0], K[1], 0.5, 0.5, 1.0
    Km = Km.expand(view.shape[0], view.shape[1], 3, 3)
    return {"viewmats": view.contiguous().to(device), "K": Km.contiguous().to(device)}


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(diffusion, inner, clean, prompt, cam, actions, steps, seed, device):
    """Future latents 9-20 from noise with clean history latents 0-8 (training layout); 4 branches, shared noise."""
    from solarwm.backends.wan22.runtime.scheduler import build_wan_flow_unipc_scheduler
    from solarwm.backends.wan22.runtime.stage0p5 import expand_timesteps_to_tokens

    B = clean.shape[0]
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(clean.shape[1:], generator=g).to(device, torch.bfloat16)
    x = clean.clone()
    x[:, FUTURE0:] = noise[None, FUTURE0:].expand(B, -1, -1, -1, -1)
    sched = build_wan_flow_unipc_scheduler(num_train_timesteps=1000, shift=5.0, num_inference_steps=steps, device=device)
    inner._action_tokens = expand_timesteps_to_tokens(actions.float(), FRAME_TOKENS).long()
    for t in sched.timesteps:
        ts = torch.zeros(B, LATENTS, device=device)
        ts[:, FUTURE0:] = float(t.item())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flow = diffusion(x, prompt, cam, expand_timesteps_to_tokens(ts, FRAME_TOKENS),
                             sequence_length=LATENTS * FRAME_TOKENS)
        flow[:, :FUTURE0] = 0
        x = sched.step(flow.to(x.dtype), t, x, return_dict=False)[0].to(torch.bfloat16)
        x[:, :FUTURE0] = clean[:, :FUTURE0]
    inner._action_tokens = None
    return x


# ----------------------------------------------------------------------------- metrics
def flow_features(frames: np.ndarray) -> np.ndarray:
    """uint8 [48,H,W,3] future frames -> concatenated coarse Farneback flow (t -> t+4), [n, 8, 14, 2] flattened."""
    import cv2

    g = [cv2.cvtColor(cv2.resize(f, (216, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in frames]
    feats = []
    for i in range(0, len(g) - 4, 4):
        fl = cv2.calcOpticalFlowFarneback(g[i], g[i + 4], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        feats.append(cv2.resize(fl, (14, 8), interpolation=cv2.INTER_AREA))
    return np.stack(feats).reshape(-1)


WINDOWS = {"0-0.5s": (33, 41), "0.5-1s": (41, 49), "1-3s": (49, 81)}  # pixel frames (16 fps, future starts at 33)
LAT_WINDOWS = {"0-0.5s": (9, 11), "0.5-1s": (11, 13), "1-3s": (13, 21)}


@torch.no_grad()
def mse_matrices(gen_pix: torch.Tensor, gt_pix: torch.Tensor, gen_lat: torch.Tensor, gt_lat: torch.Tensor) -> dict:
    """4x4 MSE (generated branch i vs GT branch j) per time window, in pixel space ([-1,1], 2x average-pooled) and
    latent space. Also the static baseline (last history frame repeated) vs each GT branch, and GT-vs-GT."""
    def pool(x):
        return torch.nn.functional.avg_pool2d(x.float().flatten(0, 1), 2).unflatten(0, x.shape[:2])
    out = {}
    for w, (a, b) in WINDOWS.items():
        g, t = pool(gen_pix[:, a:b]), pool(gt_pix[:, a:b])
        static = pool(gt_pix[:1, N_HIST - 1:N_HIST]).expand_as(t[:1])
        out[f"pix_{w}"] = ((g[:, None] - t[None]) ** 2).flatten(2).mean(-1).tolist()
        out[f"pix_gtgt_{w}"] = ((t[:, None] - t[None]) ** 2).flatten(2).mean(-1).tolist()
        out[f"pix_static_{w}"] = ((static - t) ** 2).flatten(1).mean(-1).tolist()
    for w, (a, b) in LAT_WINDOWS.items():
        g, t = gen_lat[:, a:b].float(), gt_lat[:, a:b].float()
        out[f"lat_{w}"] = ((g[:, None] - t[None]) ** 2).flatten(2).mean(-1).tolist()
    return out


def to_uint8(pix: torch.Tensor) -> np.ndarray:
    """[T,3,H,W] in [-1,1] -> uint8 [T,H,W,3]."""
    return ((pix.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def retrieval(gen: np.ndarray, gt: np.ndarray, similarity: bool) -> list:
    """gen/gt [4, D]; returns per-branch correctness of nearest GT branch."""
    if similarity:
        a = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
        b = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)
        pick = (a @ b.T).argmax(1)
    else:
        pick = ((gen[:, None] - gt[None]) ** 2).mean(-1).argmin(1)
    return [bool(p == i) for i, p in enumerate(pick)]


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs/camera_prior_eval/v0"))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--forks", type=int, default=48)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--models", default="v1,base")
    ap.add_argument("--conds", default="rot,rot_act,true,true_K,nom_s0.5,nom_s1,nom_s2")
    ap.add_argument("--videos-per-rank", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    split = json.load(open(REPO / "runs/teacher_ft/wasd_v1/split.json"))
    curves, counts = nominal_curves(split["train"])
    if args.rank == 0:
        json.dump({"counts": counts, "curves_m": {k: v[[25, 51, 77, 103]].round(2).tolist() for k, v in curves.items()},
                   "args": vars(args)}, open(out / "nominal.json", "w"), indent=1)
    val = sorted(split["val_ood"])[: args.forks][args.rank::args.world]

    from solarwm.backends.wan22.runtime.components import Wan5BVAE
    vae = Wan5BVAE(REPO / "models/SolarWM-5B-base/vae/Wan2.2_VAE.pth").to(device, dtype=torch.bfloat16)
    weights = {"base": None, "v1": REPO / "runs/teacher_ft/wasd_v1/teacher_action_step002000.pt",
               "v2p": REPO / "runs/teacher_ft/wasd_v2_pilot/teacher_action_step000300.pt",
               "v2": REPO / "runs/teacher_ft/wasd_v2/teacher_action_step000500.pt"}
    conds = args.conds.split(",")
    rows_path = out / f"rows_rank{args.rank}.jsonl"
    done = {(r["model"], r["sample_id"], r["cond"]) for r in map(json.loads, open(rows_path))} if rows_path.exists() else set()

    for model_name in args.models.split(","):
        diffusion, _ = build_teacher(str(REPO / "models/SolarWM-5B-bid-stage0p5-81f/model.pt"))
        inner = diffusion.module
        if weights[model_name] is not None:
            sd = torch.load(weights[model_name], map_location="cpu", weights_only=True, mmap=True)["generator"]
            # the v1 checkpoint also holds a duplicate copy under "_shownow_inner." (FSDP attribute alias)
            sd = {k: v for k, v in sd.items() if not k.startswith("_shownow_inner.")}
            res = inner.load_state_dict(sd, strict=True)
            assert not res.missing_keys and not res.unexpected_keys
            del sd
        inner.to(device, dtype=torch.bfloat16).eval()
        for fi, sid in enumerate(val):
            fork = torch.load(LAT / f"{sid}.pt", map_location="cpu", weights_only=False)
            m = json.load(open(SRC / sid / "meta.json"))
            clean = fork["latents"].to(device)
            prompt = {"prompt_embeds": fork["prompt"].to(device)[None].expand(4, -1, -1)}
            gt_lat = clean[:, FUTURE0:].float().flatten(1).cpu().numpy()
            gt_pix = vae.decode(clean)  # [4,81,3,H,W]
            gt_flow = np.stack([flow_features(to_uint8(gt_pix[i, N_HIST:])) for i in range(4)])
            from shownow.eval.ego_motion import ego_motion
            gt_ego = [ego_motion(to_uint8(gt_pix[i, N_HIST - 1:N_HIST + 16]))["pred"] for i in range(4)]
            free = [m["futures"][k]["summary"]["displacement_m"] >= 12.0 for k in BRANCHES]
            for cond in conds:
                if (model_name, sid, cond) in done or (cond.endswith("act") and model_name == "base"):
                    continue
                t0 = time.time()
                acts = fork["actions"].clone()
                if not cond.endswith("_act"):
                    acts[:, FUTURE0:] = 0  # camera-only: no action token for the future
                K = (FX_NORM, FY_NORM)
                if cond in ("rot", "rot_act"):
                    rels = [relative(branch_c2w_abs(m, fork, k), rot_only=True) for k in BRANCHES]
                elif cond.startswith("true"):
                    rels = [relative(branch_c2w_abs(m, fork, k)) for k in BRANCHES]
                    if cond == "true_K":
                        K = DOOM_K
                else:
                    s = float(cond.split("_s")[1].replace("_act", ""))
                    rels = [relative(nominal_c2w_abs(m, fork, k, curves[k]), scale=s) for k in BRANCHES]
                cam = camera_batch(rels, K, device)
                x = sample(diffusion, inner, clean, prompt, cam, acts.to(device), args.steps, seed=fi, device=device)
                pix = vae.decode(x)
                gen_lat = x[:, FUTURE0:].float().flatten(1).cpu().numpy()
                frames = [to_uint8(pix[i, N_HIST:]) for i in range(4)]
                gen_flow = np.stack([flow_features(f) for f in frames])
                from shownow.eval.ego_motion import ego_motion
                ego = [ego_motion(to_uint8(pix[i, N_HIST - 1:N_HIST + 16]))["pred"] for i in range(4)]
                row = {"model": model_name, "sample_id": sid, "cond": cond, "map": m["environment"]["map"],
                       "free": free, "acc_latent": retrieval(gen_lat, gt_lat, False),
                       "acc_flow": retrieval(gen_flow, gt_flow, True),
                       "ego_pred": ego, "gt_ego_pred": gt_ego, "mse": mse_matrices(pix, gt_pix, x, clean),
                       "flow_cos_own": [float(np.dot(gen_flow[i], gt_flow[i]) /
                                              (np.linalg.norm(gen_flow[i]) * np.linalg.norm(gt_flow[i]) + 1e-8))
                                        for i in range(4)],
                       "sec": round(time.time() - t0, 1)}
                with open(rows_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if fi < args.videos_per_rank:
                    import imageio.v2 as iio2
                    gt_frames = [to_uint8(gt_pix[i, N_HIST:]) for i in range(4)]
                    # rows: generated / ground truth; columns: W A S D (each 432x240)
                    import cv2
                    def small(f):
                        return np.stack([cv2.resize(z, (432, 240), interpolation=cv2.INTER_AREA) for z in f])
                    top = np.concatenate([small(f) for f in frames], axis=2)
                    bot = np.concatenate([small(f) for f in gt_frames], axis=2)
                    iio2.mimwrite(out / "videos" / f"{sid}_{model_name}_{cond}.mp4", np.concatenate([top, bot], 1),
                                  fps=16, quality=7, macro_block_size=8)
        del diffusion, inner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
