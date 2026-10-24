"""Latent inverse-dynamics model (IDM) for Doom action-following evaluation.

A small CNN reads three consecutive Wan2.2 latents (t-1, t, t+1; 48x30x54 each) and predicts the W/A/S/D key held
during latent t. Trained on the real v3 key-switch Doom training forks; validated on the map-disjoint validation forks;
then applied to generated latents (same latent space, no decoder in the loop).

Labels are only used where the real motion during latent t is unambiguous:
  * not the first latent after a key switch (Doom momentum: the view reverses ~170 ms after the press);
  * the real camera moves the way the key says (cos(local displacement, key axis) > 0.5) -> walls / blocked excluded.
On generated videos the same mask (from the REAL branch with the same key schedule) is applied, so generated and real
accuracies are measured on the same latents.

  python -m shownow.eval.idm train --out runs/idm/v1
  python -m shownow.eval.idm eval-real --ckpt runs/idm/v1/idm.pt
  python -m shownow.eval.idm eval-gen --ckpt runs/idm/v1/idm.pt --gen runs/action_eval/<run>/lat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from shownow.tree.cond_v3 import cam_dir_for, v3_split  # noqa: E402

H = 9  # history latents
AXIS = {1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0), 4: (1.0, 0.0)}  # key -> local (x right, z forward)


def label_mask(actions: torch.Tensor, cam_true: torch.Tensor) -> torch.Tensor:
    """actions [K,21] (1..4), cam_true [K,21,4,4] relative c2w (pose = first frame of each latent)
    -> bool [K,21]: latent t has an unambiguous real motion label (t in 1..19)."""
    K, T = actions.shape
    m = torch.zeros(K, T, dtype=torch.bool)
    for b in range(K):
        for t in range(1, T - 1):
            k = int(actions[b, t])
            if k not in AXIS or int(actions[b, t - 1]) != k:
                continue
            c0, c1 = cam_true[b, t].double(), cam_true[b, t + 1].double()
            d = c0[:3, :3].T @ (c1[:3, 3] - c0[:3, 3])  # motion during latent t, in the camera frame
            v = torch.stack([d[0], d[2]])
            n = float(v.norm())
            if n < 1e-4:
                continue
            m[b, t] = float(v @ torch.tensor(AXIS[k], dtype=v.dtype)) / n > 0.5
    return m


def windows(lat: torch.Tensor, ts: list) -> torch.Tensor:
    """lat [21,48,30,54] -> [n,144,30,54] (t-1, t, t+1)."""
    return torch.stack([lat[t - 1:t + 2].flatten(0, 1) for t in ts])


class IDM(nn.Module):
    def __init__(self, cin=144, w=128):
        super().__init__()

        def blk(a, b, s):
            return nn.Sequential(nn.Conv2d(a, b, 3, s, 1), nn.GroupNorm(8, b), nn.SiLU())

        self.net = nn.Sequential(blk(cin, w, 1), blk(w, w, 2), blk(w, 2 * w, 1), blk(2 * w, 2 * w, 2),
                                 blk(2 * w, 4 * w, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 4))

    def forward(self, x):  # x [n,144,30,54] -> logits [n,4] for W A S D
        return self.net(x.float())


class ForkSamples(torch.utils.data.Dataset):
    """One item = all labelled (window, key) samples of one fork (real latents)."""

    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        fork = torch.load(f, map_location="cpu", weights_only=False)
        cam = torch.load(cam_dir_for(f.parent) / f.name, weights_only=True)["true"]
        m = label_mask(fork["actions"], cam)
        xs, ys = [], []
        for b in range(m.shape[0]):
            ts = [t for t in range(H, 20) if m[b, t]]  # future latents only (balanced key schedules)
            if ts:
                xs.append(windows(fork["latents"][b], ts))
                ys.append(fork["actions"][b, ts] - 1)
        if not xs:
            return torch.zeros(0, 144, 30, 54, dtype=torch.bfloat16), torch.zeros(0, dtype=torch.long)
        return torch.cat(xs), torch.cat(ys)


def collate(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])


@torch.no_grad()
def evaluate(model, files, dev, workers=16):
    dl = torch.utils.data.DataLoader(ForkSamples(files), batch_size=4, num_workers=workers, collate_fn=collate)
    conf = np.zeros((4, 4), dtype=np.int64)
    model.eval()
    for x, y in dl:
        if not len(y):
            continue
        p = model(x.to(dev)).argmax(1).cpu()
        np.add.at(conf, (y.numpy(), p.numpy()), 1)
    model.train()
    return conf


def conf_summary(conf):
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    per = [float(conf[i, i] / max(conf[i].sum(), 1)) for i in range(4)]
    return {"acc": acc, "W/A/S/D": [round(v, 3) for v in per], "n": int(conf.sum())}


def train(args):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    td, _, vd, _ = v3_split()
    rng = np.random.default_rng(0)
    td = [td[i] for i in rng.permutation(len(td))[: args.train_forks]]
    vd = sorted(vd, key=lambda f: f.name)[: args.val_forks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = IDM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(ForkSamples(td), batch_size=args.forks_per_batch, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=collate, persistent_workers=True,
                                     prefetch_factor=4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.05)
    step, t0, log = 0, time.time(), open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        for x, y in dl:
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
                rec = {"step": step, "epoch": ep, "loss": float(loss), "n": len(y), "sec": time.time() - t0}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        conf = evaluate(model, vd, dev)
        rec = {"epoch": ep, "val": conf_summary(conf), "conf": conf.tolist()}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "args": vars(args), "val": rec}, out / "idm.pt")


def load_idm(ckpt, dev):
    model = IDM().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return model.eval()


def eval_real(args):
    dev = torch.device("cuda")
    model = load_idm(args.ckpt, dev)
    _, _, vd, _ = v3_split()
    vd = sorted(vd, key=lambda f: f.name)
    for name, files in (("val48 (action-eval forks)", vd[:48]), ("val all", vd)):
        print(name, json.dumps(conf_summary(evaluate(model, files, dev))), flush=True)


@torch.no_grad()
def score_generated(model, gen_lat, real_lat, actions, cam_true, dev):
    """gen/real [K,21,...]; -> dict of per-latent IDM predictions on generated and real future latents."""
    m = label_mask(actions, cam_true)
    out = {"key": [], "pred_gen": [], "pred_real": [], "labelled": [], "branch": [], "t": [], "after_switch": []}
    for b in range(gen_lat.shape[0]):
        ts = list(range(H, 20))  # future latents with a t+1 neighbour
        pg = model(windows(gen_lat[b], ts).to(dev)).argmax(1).cpu() + 1
        pr = model(windows(real_lat[b], ts).to(dev)).argmax(1).cpu() + 1
        for i, t in enumerate(ts):
            out["key"].append(int(actions[b, t]))
            out["pred_gen"].append(int(pg[i]))
            out["pred_real"].append(int(pr[i]))
            out["labelled"].append(bool(m[b, t]))
            out["branch"].append(b)
            out["t"].append(t)
            # latents since the last key change (0 = first latent of a new key)
            s = 0
            while t - s - 1 >= H and int(actions[b, t - s - 1]) == int(actions[b, t]):
                s += 1
            out["after_switch"].append(s if t - s - 1 >= H - 1 and int(actions[b, t - s - 1]) != int(actions[b, t]) else -1)
    return out


def eval_gen(args):
    """Score saved generated latents (<run>/lat/*.pt from action_follow_eval --save-lat) with the IDM.
    Reports, on the same latents for generated and real video:
      idm_acc      labelled latents (settled key, real camera moves as the key says)
      idm_acc_all  every future latent except the first latent of a new key
      response     per key switch: latents until the IDM first reads the new key on the generated video
                   (0 = already in the first latent of the new key); also on the real video (Doom momentum)."""
    dev = torch.device("cuda")
    model = load_idm(args.ckpt, dev)
    runs = {}
    for f in sorted(Path(args.gen).glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fp = Path(d["fork"])
        fork = torch.load(fp, map_location="cpu", weights_only=False)
        cam = torch.load(cam_dir_for(fp.parent) / fp.name, weights_only=True)["true"]
        r = score_generated(model, d["gen"], fork["latents"], fork["actions"], cam, dev)
        resp = {"gen": [], "real": []}
        for b in range(4):
            idx = [i for i in range(len(r["t"])) if r["branch"][i] == b]
            for i in idx:
                if r["after_switch"][i] != 0:
                    continue
                key = r["key"][i]
                seg = [j for j in idx if r["t"][j] >= r["t"][i] and r["key"][j] == key]
                seg = [j for n_, j in enumerate(seg) if r["t"][j] == r["t"][i] + n_]  # contiguous segment
                for tag, pk in (("gen", "pred_gen"), ("real", "pred_real")):
                    hit = [n_ for n_, j in enumerate(seg) if r[pk][j] == key]
                    resp[tag].append(hit[0] if hit else None)
        runs.setdefault((d["cond"], d["mode"]), []).append((r, resp))
    res = {}
    for (cond, mode), items in runs.items():
        key = np.concatenate([np.array(r["key"]) for r, _ in items])
        pg = np.concatenate([np.array(r["pred_gen"]) for r, _ in items])
        pr = np.concatenate([np.array(r["pred_real"]) for r, _ in items])
        lab = np.concatenate([np.array(r["labelled"]) for r, _ in items])
        settled = np.concatenate([np.array(r["after_switch"]) != 0 for r, _ in items])

        def resp_stats(tag):
            v = [x for _, rs in items for x in rs[tag]]
            hits = [x for x in v if x is not None]
            return {"within_1_latent": float(np.mean([x is not None and x <= 1 for x in v])),
                    "mean_latents": float(np.mean(hits)) if hits else None, "never": float(np.mean([x is None for x in v])),
                    "n_switches": len(v)}

        # paired with the real video of the same branch and switch: generated minus real response (latents), so the
        # game's own momentum (~170 ms to reverse) cancels; None (never responded within the segment) counts as the
        # segment length + 1 is NOT assumed -- such switches are reported separately.
        pairs = [(g, r_) for _, rs in items for g, r_ in zip(rs["gen"], rs["real"]) if g is not None and r_ is not None]
        extra = np.array([g - r_ for g, r_ in pairs]) if pairs else np.zeros(0)
        res[f"{cond}/{mode}"] = {
            "response_extra_latents": {"mean": float(extra.mean()) if len(extra) else None,
                                       "frac_same_or_faster": float((extra <= 0).mean()) if len(extra) else None,
                                       "n_pairs": len(extra)},
            "forks": len(items),
            "idm_acc_gen": float((pg == key)[lab].mean()), "idm_acc_real": float((pr == key)[lab].mean()),
            "idm_acc_all_gen": float((pg == key)[settled].mean()), "idm_acc_all_real": float((pr == key)[settled].mean()),
            "per_key_gen": [round(float((pg == key)[lab & (key == k)].mean()), 3) for k in (1, 2, 3, 4)],
            "n_labelled": int(lab.sum()), "response_gen": resp_stats("gen"), "response_real": resp_stats("real")}
    print(json.dumps(res, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- motion IDM (inertia-aware action following)
# Target = the REAL motion of the same branch under the same key schedule (it already contains Doom's momentum:
# acceleration, braking and reversal after a switch, and walls), not the key itself. The motion IDM regresses the
# camera-frame displacement (x right, z forward; metres) during latent t from latents (t-1, t, t+1), trained on ALL
# real future latents (switch transitions and blocked ones included).


def local_motion(cam_true: torch.Tensor) -> torch.Tensor:
    """[K,21,4,4] -> [K,21,2] displacement during latent t in the camera frame of latent t (t = 20: zeros)."""
    c = cam_true.double()
    d = c[:, 1:, :3, 3] - c[:, :-1, :3, 3]
    loc = torch.einsum("ktji,ktj->kti", c[:, :-1, :3, :3], d)  # R^T d
    out = torch.zeros(c.shape[0], c.shape[1], 2, dtype=torch.float32)
    out[:, :-1] = loc[..., [0, 2]].float()
    return out


class ForkMotion(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        fork = torch.load(f, map_location="cpu", weights_only=False)
        mv = local_motion(torch.load(cam_dir_for(f.parent) / f.name, weights_only=True)["true"])
        ts = list(range(H, 20))
        return (torch.cat([windows(fork["latents"][b], ts) for b in range(4)]),
                torch.cat([mv[b, ts] for b in range(4)]))


MOTION_SCALE = 0.5  # metres per latent ~ walking speed; targets are divided by it


class MotionIDM(IDM):
    def __init__(self):
        super().__init__()
        self.net[-1] = nn.Linear(self.net[-1].in_features, 2)


def train_motion(args):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    td, _, vd, _ = v3_split()
    rng = np.random.default_rng(0)
    td = [td[i] for i in rng.permutation(len(td))[: args.train_forks]]
    vd = sorted(vd, key=lambda f: f.name)[: args.val_forks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = MotionIDM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(ForkMotion(td), batch_size=args.forks_per_batch, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=collate, persistent_workers=True,
                                     prefetch_factor=4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.05)
    step, t0, log = 0, time.time(), open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True).float(), y.to(dev) / MOTION_SCALE
            if args.noise > 0:
                x = x + torch.randn_like(x) * (torch.rand(len(x), 1, 1, 1, device=dev) * args.noise)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.smooth_l1_loss(model(x).float(), y, beta=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(json.dumps({"step": step, "loss": float(loss), "sec": time.time() - t0}), flush=True)
        rec = {"epoch": ep, "val": motion_val(model, vd, dev)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "args": vars(args), "val": rec}, out / "motion_idm.pt")


def direction_agree(p, q, min_norm=0.05):
    """p, q [n,2] displacements -> bool [n]: same direction (cos > 0.5), or both (nearly) still."""
    np_, nq = p.norm(dim=1), q.norm(dim=1)
    still_p, still_q = np_ < min_norm, nq < min_norm
    cos = (p * q).sum(1) / (np_ * nq).clamp_min(1e-6)
    return torch.where(still_p | still_q, still_p & still_q, cos > 0.5)


@torch.no_grad()
def motion_val(model, files, dev):
    model.eval()
    ps, ys = [], []
    for i in range(len(files)):
        x, y = ForkMotion(files)[i]
        ps.append(model(x.to(dev)).float().cpu() * MOTION_SCALE)
        ys.append(y)
    model.train()
    p, y = torch.cat(ps), torch.cat(ys)
    return {"dir_agree": float(direction_agree(p, y).float().mean()), "mae_m": float((p - y).abs().mean()),
            "n": len(y)}


@torch.no_grad()
def eval_motion(args):
    """Inertia-aware action following of saved generated latents: on EVERY future latent (switch transitions and
    walls included), does the generated video move like the real game does under the same key schedule?
      dir_vs_real_cam   generated motion (motion IDM) vs the real camera motion       (ground truth)
      dir_vs_real_idm   generated motion vs the motion IDM on the real video          (same estimator on both)
      ref_real_idm      motion IDM on the real video vs the real camera (estimator ceiling)
    also split into transition latents (first 2 latents of a new key) and settled ones."""
    dev = torch.device("cuda")
    model = MotionIDM().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    acc = {}
    for f in sorted(Path(args.gen).glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fp = Path(d["fork"])
        fork = torch.load(fp, map_location="cpu", weights_only=False)
        mv = local_motion(torch.load(cam_dir_for(fp.parent) / fp.name, weights_only=True)["true"])
        a = fork["actions"]
        ts = list(range(H, 20))
        for b in range(4):
            pg = model(windows(d["gen"][b], ts).to(dev)).float().cpu() * MOTION_SCALE
            pr = model(windows(fork["latents"][b], ts).to(dev)).float().cpu() * MOTION_SCALE
            y = mv[b, ts]
            trans = torch.tensor([int(a[b, t]) != int(a[b, t - 1]) or int(a[b, t - 1]) != int(a[b, t - 2]) for t in ts])
            # final action-following score: latents where the real game is blocked by geometry are
            # excluded (real still = wall; real moving off the key axis while the key is settled = sliding along a
            # wall); in a momentum transition (real still drifting off the new key) the generated motion counts as
            # correct if it matches the real motion OR already moves along the new key.
            ax = torch.tensor([AXIS[int(a[b, t])] for t in ts])
            ny, ng = y.norm(dim=1), pg.norm(dim=1)
            real_on = ((y * ax).sum(1) / ny.clamp_min(1e-6) > 0.5) & (ny >= 0.05)
            real_off = ~real_on & (ny >= 0.05)
            gen_on = ((pg * ax).sum(1) / ng.clamp_min(1e-6) > 0.5) & (ng >= 0.05)
            agree = direction_agree(pg, y)
            keep = real_on | (real_off & trans)
            okf = torch.where(real_off & trans, agree | gen_on, agree)
            e = acc.setdefault(f'{d["cond"]}/{d["mode"]}', {"cam": [], "idm": [], "ref": [], "trans": [], "err": [],
                                                            "af_ok": [], "af_keep": []})
            nr = pr.norm(dim=1)
            ref_on = ((pr * ax).sum(1) / nr.clamp_min(1e-6) > 0.5) & (nr >= 0.05)
            e.setdefault("af_ref", []).append(torch.where(real_off & trans, direction_agree(pr, y) | ref_on,
                                                          direction_agree(pr, y)))
            e["af_ok"].append(okf)
            e["af_keep"].append(keep)
            e["cam"].append(direction_agree(pg, y))
            e["idm"].append(direction_agree(pg, pr))
            e["ref"].append(direction_agree(pr, y))
            e["trans"].append(trans)
            e["err"].append((pg - y).norm(dim=1))
    res = {}
    for k, e in acc.items():
        cam, idm, ref, tr, err = (torch.cat(e[n]) for n in ("cam", "idm", "ref", "trans", "err"))
        f_ = lambda v, m: round(float(v[m].float().mean()), 3)  # noqa: E731
        allm = torch.ones_like(tr)
        ok_, keep_ = torch.cat(e["af_ok"]), torch.cat(e["af_keep"])
        res.setdefault(k, {})["action_follow"] = {"acc": f_(ok_, keep_), "real_video_ceiling": f_(torch.cat(e["af_ref"]), keep_), "acc_transition": f_(ok_, keep_ & tr),
                                                  "acc_settled": f_(ok_, keep_ & ~tr), "n": int(keep_.sum()),
                                                  "excluded_frac": round(1 - float(keep_.float().mean()), 3)}
        res[k] |= {sub: {"dir_vs_real_cam": f_(cam, m), "dir_vs_real_idm": f_(idm, m), "ref_real_idm": f_(ref, m),
                        "err_m": f_(err, m), "n": int(m.sum())}
                  for sub, m in (("all", allm), ("transition", tr), ("settled", ~tr))}
    print(json.dumps(res, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval-real", "eval-gen", "train-motion", "eval-motion"])
    ap.add_argument("--out", default="runs/idm/v1")
    ap.add_argument("--ckpt", default="runs/idm/v1/idm.pt")
    ap.add_argument("--train-forks", type=int, default=8000)
    ap.add_argument("--val-forks", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--forks-per-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--gen", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    {"train": train, "eval-real": eval_real, "eval-gen": eval_gen, "train-motion": train_motion,
     "eval-motion": eval_motion}[args.cmd](args)


if __name__ == "__main__":
    main()
