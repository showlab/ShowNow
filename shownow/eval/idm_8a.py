"""Latent inverse-dynamics model (IDM) for the 8-action Doom forks (W A S D TURN_L TURN_R ATTACK JUMP;
data/cf_vizdoom_8a_lat + data/cf_vizdoom_8a_cam). 8-action version of idm.py (that file is unchanged).

Same CNN as idm.py: three consecutive Wan2.2 latents (t-1, t, t+1; 48x30x54 each) -> the action held during latent t.
Every branch holds its action for the whole future, so future latents t = 10..19 of branch b are labelled with the
branch's action where the real (GT) motion during latent t shows it unambiguously:
  W/A/S/D  : the real camera moves the way the key says (cos(local displacement, key axis) > 0.5), as in idm.py
  TURN_L/R : the real yaw changes by > MIN_YAW_DEG during latent t in the key's direction (60 deg/s -> ~15 deg/latent)
  JUMP     : the real camera height changes by > MIN_DY_M during latent t (airborne)
  ATTACK   : always (continuous fire: weapon / muzzle animation)
Latent 9 (first future latent) is never labelled (Doom momentum). On generated latents the same mask, computed from the
REAL branch with the same action, is applied, so generated and real accuracies use the same latents.

  python -m shownow.eval.idm_8a train --out runs/idm_8a/v1
  python -m shownow.eval.idm_8a eval-real --ckpt runs/idm_8a/v1/idm.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from shownow.eval.idm import IDM, collate, windows  # noqa: E402

LAT = REPO / "data/cf_vizdoom_8a_lat"
CAM = REPO / "data/cf_vizdoom_8a_cam"
BRANCHES = ["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
NA = len(BRANCHES)
H = 9
AXIS = {1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0), 4: (1.0, 0.0)}  # key -> local (x right, z forward)
MIN_YAW_DEG = 5.0
MIN_DY_M = 0.03


def label_mask(actions: torch.Tensor, cam_true: torch.Tensor) -> torch.Tensor:
    """actions [K,21] (future ids 1..8), cam_true [K,21,4,4] relative c2w (OpenCV, metres) -> bool [K,21]."""
    K, T = actions.shape
    m = torch.zeros(K, T, dtype=torch.bool)
    for b in range(K):
        for t in range(H + 1, T - 1):
            k = int(actions[b, t])
            c0, c1 = cam_true[b, t].double(), cam_true[b, t + 1].double()
            d = c0[:3, :3].T @ (c1[:3, 3] - c0[:3, 3])  # motion during latent t in the camera frame
            R = c0[:3, :3].T @ c1[:3, :3]
            yaw = math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2])))  # + = turned right (x right, z forward)
            if k in AXIS:
                v = torch.stack([d[0], d[2]])
                n = float(v.norm())
                m[b, t] = n > 1e-4 and float(v @ torch.tensor(AXIS[k], dtype=v.dtype)) / n > 0.5
            elif k == 5:
                m[b, t] = yaw < -MIN_YAW_DEG
            elif k == 6:
                m[b, t] = yaw > MIN_YAW_DEG
            elif k == 7:
                m[b, t] = True
            elif k == 8:
                m[b, t] = abs(float(d[1])) > MIN_DY_M
    return m


class ForkSamples(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        fork = torch.load(f, map_location="cpu", weights_only=False)
        cam = torch.load(CAM / f.name, weights_only=True)["true"]
        m = label_mask(fork["actions"], cam)
        xs, ys = [], []
        for b in range(m.shape[0]):
            ts = [t for t in range(H + 1, 20) if m[b, t]]
            if ts:
                xs.append(windows(fork["latents"][b], ts))
                ys.append(fork["actions"][b, ts] - 1)
        if not xs:
            return torch.zeros(0, 144, 30, 54, dtype=torch.bfloat16), torch.zeros(0, dtype=torch.long)
        return torch.cat(xs), torch.cat(ys)


def splits():
    from shownow.train.teacher_action_ft import split_files
    tr, vo, vi = split_files(LAT)
    return tr, sorted(vo, key=lambda f: f.name)


@torch.no_grad()
def evaluate(model, files, dev, workers=16):
    dl = torch.utils.data.DataLoader(ForkSamples(files), batch_size=4, num_workers=workers, collate_fn=collate)
    conf = np.zeros((NA, NA), dtype=np.int64)
    model.eval()
    for x, y in dl:
        if not len(y):
            continue
        p = model(x.to(dev)).argmax(1).cpu()
        np.add.at(conf, (y.numpy(), p.numpy()), 1)
    model.train()
    return conf


def conf_summary(conf):
    return {"acc": float(np.trace(conf) / max(conf.sum(), 1)),
            "per_action": {b: round(float(conf[i, i] / max(conf[i].sum(), 1)), 3) for i, b in enumerate(BRANCHES)},
            "n": int(conf.sum())}


def make_idm():
    model = IDM()
    model.net[-1] = torch.nn.Linear(model.net[-1].in_features, NA)  # 8 actions
    return model


def load_idm(ckpt, dev):
    model = make_idm().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return model.eval()


def train(args):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    tr, vo = splits()
    rng = np.random.default_rng(0)
    tr = [tr[i] for i in rng.permutation(len(tr))[: args.train_forks]]
    vo = vo[: args.val_forks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = make_idm().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(ForkSamples(tr), batch_size=args.forks_per_batch, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=collate, persistent_workers=True,
                                     prefetch_factor=4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.05)
    step, t0, log = 0, time.time(), open(out / "log.jsonl", "a")
    print(json.dumps({"train_forks": len(tr), "val_forks": len(vo), "steps": total}), flush=True)
    for ep in range(args.epochs):
        for x, y in dl:
            if not len(y):
                continue
            x, y = x.to(dev, non_blocking=True).float(), y.to(dev)
            if args.noise > 0:  # robustness to generated latents: random per-sample latent noise
                x = x + torch.randn_like(x) * (torch.rand(len(x), 1, 1, 1, device=dev) * args.noise)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "epoch": ep, "loss": float(loss), "n": len(y), "sec": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        conf = evaluate(model, vo, dev)
        rec = {"epoch": ep, "val": conf_summary(conf), "conf": conf.tolist()}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "args": vars(args), "val": rec}, out / "idm.pt")


def eval_real(args):
    dev = torch.device("cuda")
    model = load_idm(args.ckpt, dev)
    _, vo = splits()
    for name, files in (("val48 (tree_eval_8a forks)", vo[:48]), ("val all", vo)):
        print(name, json.dumps(conf_summary(evaluate(model, files, dev))), flush=True)


@torch.no_grad()
def score_generated(model, gen_lat, real_lat, actions, cam_true, dev):
    """gen/real [K,21,...] future latents of one fork -> per-branch IDM accuracy on the labelled latents (same mask
    for generated and real) and the prediction lists."""
    m = label_mask(actions, cam_true)
    res = {"key": [], "pred_gen": [], "pred_real": []}
    for b in range(gen_lat.shape[0]):
        ts = [t for t in range(H + 1, 20) if m[b, t]]
        if not ts:
            continue
        pg = model(windows(gen_lat[b].to(dev), ts)).argmax(1).cpu() + 1
        pr = model(windows(real_lat[b].to(dev), ts)).argmax(1).cpu() + 1
        res["key"] += [int(actions[b, t]) for t in ts]
        res["pred_gen"] += pg.tolist()
        res["pred_real"] += pr.tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval-real"])
    ap.add_argument("--out", default="runs/idm_8a/v1")
    ap.add_argument("--ckpt", default="runs/idm_8a/v1/idm.pt")
    ap.add_argument("--train-forks", type=int, default=9433)
    ap.add_argument("--val-forks", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--forks-per-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.3)
    a = ap.parse_args()
    {"train": train, "eval-real": eval_real}[a.cmd](a)


if __name__ == "__main__":
    main()
