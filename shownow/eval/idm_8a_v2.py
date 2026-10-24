"""8-action latent IDM v2 (scorer only): labels from the engine control log checked against the engine state, trained
on hold + key-switch data (manifest: shownow/eval/idm_8a_data.py). v1 (idm_8a.py, runs/idm_8a/v1) is unchanged.

Labels. Key per future latent = the latent's key in the pre-encoded forks (switch schedule / held branch key); verified
against the engine control log (button rows of the 32 fps frames 8j+1..8j+7 of future latent j, which show only that
key). A latent is IDENTIFIABLE (scored / trained) when the engine state shows the key's effect during the latent:
  W/A/S/D   the real camera moves along the key axis (cos > 0.5) during the latent      (as v1; walls / momentum fail)
  TURN_L/R  the real yaw changes > 5 deg the right way during the latent                 (as v1)
  ATTACK    the weapon entered its attack state within this ATTACK segment (attack_ready 1 -> 0, or an ammo drop) and is
            in it during the latent (some frame not ready, or an ammo drop)            (v1: always)
  JUMP      a take-off (on_ground 1 -> 0) happened within this JUMP segment and the player is airborne in the latent
                                                                                          (v1: |dy| > 3 cm)
Reasons for non-identifiable latents are kept per latent (coverage report). Latent 20 has no t+1 and is never used.
Nothing is removed because a scorer misreads it.

Label schemes (window = latents t-1, t, t+1 as in v1):
  center   one label: the key of latent t (v1 scheme), trained on identifiable centres
  seq      three heads: keys of t-1, t, t+1, each trained where identifiable; scoring uses the centre head

  python -m shownow.eval.idm_8a_v2 labels --manifest runs/idm_8a/data_v2/split.json --out runs/idm_8a/labels_v2.pt
  python -m shownow.eval.idm_8a_v2 train --scheme center --data runs/idm_8a/data_v2 --out runs/idm_8a/v2_center
  python -m shownow.eval.idm_8a_v2 eval --ckpt runs/idm_8a/v2_center/idm.pt --sets val_switch,val_hold,real_sw8a,sw8i_e2_val
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from shownow.eval.idm import IDM, collate, windows  # noqa: E402

BRANCHES = ["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
NA = 8
H = 9
TS = list(range(H + 1, 20))  # scored / trained window centres (as v1 and e2_8a_score)
AXIS = {1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0), 4: (1.0, 0.0)}  # key -> local (x right, z forward)
MIN_YAW_DEG = 5.0
REASONS = ["ok", "move_blocked", "move_off_axis", "turn_small", "attack_no_evidence", "jump_no_takeoff", "no_next"]
BUTTON = {1: 0, 3: 1, 2: 2, 4: 3, 8: 4, 7: 6}  # key -> index in the control rows [MF MB ML MR JUMP CROUCH ATTACK ...]


def raw_dir(lat_file: Path) -> Path:
    return lat_file.parent.parent / lat_file.parent.name.replace("_lat", "_32fps") / lat_file.stem


def row_key(r) -> int:
    """control row -> key id (0 = none / several)."""
    on = [k for k, i in BUTTON.items() if r[i] > 0]
    if abs(r[9]) > 1e-6:
        on.append(6 if r[9] > 0 else 5)
    return on[0] if len(on) == 1 else 0


def fork_labels(lat_file: str) -> dict:
    """identifiable mask / reason codes [8,21] and the control-log check of one fork."""
    from shownow.tree.cond_v3 import cam_dir_for
    f = Path(lat_file)
    fork = torch.load(f, map_location="cpu", weights_only=False)
    acts = fork["actions"].long()
    cam = torch.load(cam_dir_for(f.parent) / f.name, weights_only=True)["true"].double()
    meta = json.load(open(raw_dir(f) / "meta.json"))
    names = fork.get("meta", {}).get("branches", BRANCHES)
    reason = torch.full((8, 21), -1, dtype=torch.int8)
    log_mismatch = 0
    hist_last = meta["history"]["state"][-1]
    for b, br in enumerate(names):
        st, rows = meta["futures"][br]["state"], meta["futures"][br]["actions"]
        keys = [int(k) for k in acts[b]]
        for t in range(H, 21):
            j, k = t - H, keys[t]
            log = Counter(row_key(rows[n]) for n in range(8 * j + 1, 8 * j + 8))
            log_mismatch += int(log.most_common(1)[0][0] != k)
            if t == 20:
                reason[b, t] = 6
                continue
            s0 = t
            while s0 > H and keys[s0 - 1] == k:
                s0 -= 1
            c0, c1 = cam[b, t], cam[b, t + 1]
            d = c0[:3, :3].T @ (c1[:3, 3] - c0[:3, 3])
            R = c0[:3, :3].T @ c1[:3, :3]
            yaw = math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2])))
            seg = [hist_last if s0 == H else st[8 * (s0 - H) - 1]] + st[8 * (s0 - H): 8 * j + 8]  # segment start..latent end
            lat = st[8 * j: 8 * j + 8]
            if k in AXIS:
                v = torch.stack([d[0], d[2]])
                n = float(v.norm())
                ok = n > 1e-4 and float(v @ torch.tensor(AXIS[k], dtype=v.dtype)) / n > 0.5
                r = 0 if ok else (1 if n < 0.02 else 2)
            elif k in (5, 6):
                r = 0 if (yaw < -MIN_YAW_DEG if k == 5 else yaw > MIN_YAW_DEG) else 3
            elif k == 7:
                rd = [s["attack_ready"] for s in seg]
                am = [s["weapon_ammo"] for s in seg]
                started = any(a > 0 and b_ == 0 for a, b_ in zip(rd, rd[1:])) or any(b_ < a for a, b_ in zip(am, am[1:]))
                prev_ammo = seg[-9]["weapon_ammo"] if len(seg) >= 9 else seg[0]["weapon_ammo"]
                busy = any(s["attack_ready"] == 0 for s in lat) or min(s["weapon_ammo"] for s in lat) < prev_ammo
                r = 0 if started and busy else 4
            else:
                g = [int(s["on_ground"]) for s in seg]
                took = any(a and not b_ for a, b_ in zip(g, g[1:]))
                air = any(not s["on_ground"] for s in lat)
                r = 0 if took and air else 5
            reason[b, t] = r
    return {"sid": f.stem, "reason": reason, "log_mismatch": log_mismatch, "weapon": meta["weapon"]["name"]}


def build_labels(args):
    sp = json.load(open(args.manifest))
    files = sorted({f for k, v in sp.items() if k.startswith(("train", "val", "real")) for f in v} |
                   {f for v in sp["report_sw8i"].values() for f in v})
    if args.limit:
        files = files[:args.limit]
    t0 = time.time()
    with Pool(args.workers) as p:
        res = p.map(fork_labels, files, chunksize=8)
    out = {r["sid"]: {"reason": r["reason"], "weapon": r["weapon"]} for r in res}
    torch.save(out, args.out)
    allr = torch.cat([r["reason"][:, H + 1:20].flatten() for r in res])
    print(json.dumps({"forks": len(res), "sec": round(time.time() - t0),
                      "control_log_mismatch_latents": int(sum(r["log_mismatch"] for r in res)),
                      "latents_checked": len(res) * 8 * 12,
                      "reasons": {REASONS[i]: int((allr == i).sum()) for i in range(len(REASONS))}}, indent=1))


# ----------------------------------------------------------------------------- model / data
def make_idm(scheme: str):
    m = IDM()
    m.net[-1] = torch.nn.Linear(m.net[-1].in_features, NA * (3 if scheme == "seq" else 1))
    return m


def load_idm(ckpt, dev):
    s = torch.load(ckpt, map_location=dev, weights_only=False)
    m = make_idm(s["args"]["scheme"]).to(dev)
    m.load_state_dict(s["model"])
    m.scheme = s["args"]["scheme"]
    return m.eval()


def center_logits(model, x):
    o = model(x)
    return o[:, NA:2 * NA] if getattr(model, "scheme", "center") == "seq" else o


def scorer(ckpt, dev):
    """IDM v1 or v2 checkpoint -> fn(windows [n,144,30,54]) -> centre logits [n,8] (v2 seq: the centre head)."""
    if Path(ckpt).parent.name == "v1":
        from shownow.eval.idm_8a import load_idm as load_v1
        return load_v1(ckpt, dev)
    m = load_idm(ckpt, dev)
    return lambda x: center_logits(m, x)


def reasons(sid: str, labels_path=REPO / "runs/idm_8a/labels_v2.pt") -> torch.Tensor:
    """v2 reason codes [8,21] of a real fork (0 = identifiable; REASONS), from the engine state."""
    if not hasattr(reasons, "cache"):
        reasons.cache = torch.load(labels_path, weights_only=False)
    return reasons.cache[sid]["reason"]


def identifiable(sid: str) -> torch.Tensor:
    """v2 scoring mask [8,21] of a real fork (the frozen label rules)."""
    return reasons(sid) == 0


class Windows(torch.utils.data.Dataset):
    """one item = every window (t in TS) of every branch of one fork, with the labels [n,3] (-1 = not identifiable)
    of latents t-1, t, t+1 and the stratum info (after switch, segment held for the whole future)."""

    def __init__(self, files, labels):
        self.files, self.labels = list(files), labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = Path(self.files[i])
        fork = torch.load(f, map_location="cpu", weights_only=False)
        acts = fork["actions"].long()
        reason = self.labels[f.stem]["reason"]
        xs, ys, info = [], [], []
        for b in range(acts.shape[0]):
            keys = acts[b].tolist()
            xs.append(windows(fork["latents"][b], TS))
            for t in TS:
                y = [keys[u] - 1 if u >= H and int(reason[b, u]) == 0 else -1 for u in (t - 1, t, t + 1)]
                s0 = t
                while s0 > H and keys[s0 - 1] == keys[t]:
                    s0 -= 1
                ys.append(y)
                info.append([t - s0, int(len(set(keys[H:])) == 1), int(reason[b, t])])
        return torch.cat(xs), torch.tensor(ys), torch.tensor(info)


def collate3(batch):
    return tuple(torch.cat([b[i] for b in batch]) for i in range(3))


@torch.no_grad()
def predict(model, files, labels, dev, workers=16):
    dl = torch.utils.data.DataLoader(Windows(files, labels), batch_size=4, num_workers=workers, collate_fn=collate3)
    P, Y, I = [], [], []
    model.eval()
    for x, y, info in dl:
        P.append(center_logits(model, x.to(dev)).argmax(1).cpu())
        Y.append(y[:, 1])
        I.append(info)
    model.train()
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()


def metrics(p, y, info) -> dict:
    m = y >= 0
    conf = np.zeros((NA, NA), dtype=np.int64)
    np.add.at(conf, (y[m], p[m]), 1)
    rec = conf.diagonal() / np.maximum(conf.sum(1), 1)
    acc = lambda mm: (float((p[mm] == y[mm]).mean()), int(mm.sum())) if mm.any() else (None, 0)  # noqa: E731
    return {"acc": acc(m), "macro": float(rec[conf.sum(1) > 0].mean()), "recall": dict(zip(BRANCHES, rec.round(3).tolist())),
            "hold_branch": acc(m & (info[:, 1] == 1)), "transition": acc(m & (info[:, 1] == 0) & (info[:, 0] < 2)),
            "settled": acc(m & (info[:, 1] == 0) & (info[:, 0] >= 2)),
            "coverage": float(m.mean()), "conf": conf.tolist()}


def set_files(sp, name):
    if name.startswith("sw8i_"):
        return sp["report_sw8i"][name[5:]]
    return sp[name]


def train(args):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    sp = json.load(open(Path(args.data) / "split.json"))
    labels = torch.load(args.labels, weights_only=False)
    tr = sp["train"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    model = make_idm(args.scheme).to(dev)
    model.scheme = args.scheme
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(Windows(tr, labels), batch_size=args.forks_per_batch, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=collate3, persistent_workers=True, prefetch_factor=4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.05)
    step, t0, log, best = 0, time.time(), open(out / "log.jsonl", "a"), -1.0
    print(json.dumps({"train_forks": len(tr), "steps": total}), flush=True)
    for ep in range(args.epochs):
        for x, y, _ in dl:
            x, y = x.to(dev, non_blocking=True).float(), y.to(dev)
            if args.noise > 0:
                x = x + torch.randn_like(x) * (torch.rand(len(x), 1, 1, 1, device=dev) * args.noise)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x).float()
            if args.scheme == "seq":
                loss = sum(F.cross_entropy(o[:, i * NA:(i + 1) * NA], y[:, i], ignore_index=-1) * (1.0 if i == 1 else args.side_w)
                           for i in range(3) if (y[:, i] >= 0).any())
            else:
                loss = F.cross_entropy(o, y[:, 1], ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "epoch": ep, "loss": float(loss), "sec": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        rec = {"epoch": ep}
        for name in ("val_switch", "val_hold"):
            rec[name] = metrics(*predict(model, sp[name], labels, dev))
        print(json.dumps({k: (v if k == "epoch" else {kk: vv for kk, vv in v.items() if kk != "conf"}) for k, v in rec.items()}),
              flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        state = {"model": model.state_dict(), "args": vars(args), "val": rec}
        torch.save(state, out / f"idm_ep{ep}.pt")
        if rec["val_switch"]["macro"] > best:  # selection: macro accuracy on the switch validation forks only
            best = rec["val_switch"]["macro"]
            torch.save(state, out / "idm.pt")


def evaluate(args):
    dev = torch.device("cuda")
    sp = json.load(open(args.manifest))
    labels = torch.load(args.labels, weights_only=False)
    res = {}
    for ck in args.ckpt.split(","):
        if ck.endswith("v1/idm.pt"):
            from shownow.eval.idm_8a import load_idm as load_v1
            model = load_v1(ck, dev)
            model.scheme = "center"
        else:
            model = load_idm(ck, dev)
        res[ck] = {name: metrics(*predict(model, set_files(sp, name), labels, dev)) for name in args.sets.split(",")}
        for name, r in res[ck].items():
            print(ck, name, json.dumps({k: v for k, v in r.items() if k != "conf"}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("labels")
    lb.add_argument("--manifest", required=True)
    lb.add_argument("--out", required=True)
    lb.add_argument("--workers", type=int, default=48)
    lb.add_argument("--limit", type=int, default=0)
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--labels", default="runs/idm_8a/labels_v2.pt")
    tr.add_argument("--out", required=True)
    tr.add_argument("--scheme", choices=["center", "seq"], default="center")
    tr.add_argument("--side-w", type=float, default=0.5, help="seq: loss weight of the t-1 / t+1 heads")
    tr.add_argument("--epochs", type=int, default=6)
    tr.add_argument("--forks-per-batch", type=int, default=8)
    tr.add_argument("--workers", type=int, default=24)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--noise", type=float, default=0.3)
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True, help="comma-separated; runs/idm_8a/v1/idm.pt = v1")
    ev.add_argument("--manifest", default="runs/idm_8a/data_v2/split.json")
    ev.add_argument("--labels", default="runs/idm_8a/labels_v2.pt")
    ev.add_argument("--sets", default="val_switch,val_hold,real_sw8a,sw8i_e2_val")
    ev.add_argument("--out", default="")
    a = ap.parse_args()
    {"labels": build_labels, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
