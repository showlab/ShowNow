"""E3 for the 8-action models (W A S D TURN_L TURN_R ATTACK JUMP): 60 s rollouts on the fixed key-switch eval scenes
(shownow/eval/sw8a_split.py: training maps by default; SHOWNOW_SPLIT=test -> test scenes) and their long-horizon metrics.
8-action version of the E3 protocol (shownow/eval/long_eval.py, that file is unchanged).

  gen --mode realtime_opt  the deployed ShowNow path (tree_opt.RTStepper, --opts): at every latent all 8 key
                           candidates in one shared-history tree pass, the scripted key's candidate is committed.
                           Saves latents, keys and the camera poses used.
  gen --mode reference     normal (non-speculative) inference: cache-free sliding window (15 prior latents + 1 noisy,
                           window-relative RoPE), one future; --from <realtime run> replays its keys and poses.
                           --kind s1 (flow map, --nfe) or s2 (SGF, --denoising-steps).
  gen --block 3            SolarWM-FT (B3) content: reference inference one AR block of 3 latents per step, the key
                           read once per block (as deployed); the camera follows the applied keys, the metrics score
                           the scripted keys (saved as keys / keys_applied).
  Both start from the scene's 15 real latents (9 history + the first 6 of its W branch); then 60 s (240 latents) with a
  new key every 2 s (never the same key twice in a row, W twice as likely, seed 1000 + video). Camera: KeyStream8
  (shownow/data/nominal_8a.py) from the fork pose, the W branch's keys replayed over the intro (as play_server).
  score  TAEHV decode and metrics (as long_eval.py): key IDM accuracy with the 8-action IDM (runs/idm_8a/v1) on
         settled latents (>= 2 latents after a switch; walls may legitimately stop movement, so this is a lower bound
         for W/A/S/D), per key and per 5 s window; latent FD per 5 s window vs all real latents (8 branches x 21) of
         the scenes; sharpness / contrast vs real; washed-out windows; latent-boundary ratio; red flashes / min.

  SHOWNOW_N_ACTIONS=9 python -m shownow.eval.long_eval_8a gen --kind s2 --mode realtime_opt --student <ckpt> \
      --out runs/long_eval_8a/<name>/rt
  SHOWNOW_N_ACTIONS=9 python -m shownow.eval.long_eval_8a gen --kind s2 --student <ckpt> \
      --from runs/long_eval_8a/<name>/rt --out runs/long_eval_8a/<name>/ref
  python -m shownow.eval.long_eval_8a score runs/long_eval_8a/<name>/rt [more run dirs] [--mp4]
  python -m shownow.eval.long_eval_8a table runs/long_eval_8a/*/*
  IDM v2: score --idm runs/idm_8a/<v2>/idm.pt --metrics-name long_metrics_v2.json (v1 results are kept); the E3 key
  accuracy has no paired real future, so every settled latent (>= 2 after a switch) of every method is scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
sys.path.insert(0, str(REPO / "third_party/taehv"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

BRANCHES = ["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
P = 15  # real latents before the scripted minute
WIN = 20  # latents per 5 s window
SETTLE = 2


def scenes(n):
    """E3 scenes: the fixed key-switch eval scenes of shownow/eval/sw8a_split.py (held-out test split with
    SHOWNOW_SPLIT=test)."""
    from shownow.eval.sw8a_split import split
    from shownow.tree.cond_v3 import split_name
    return [Path(f) for f in split()["test_e3" if split_name() == "test" else "scenes"]][:n]


def key_schedule(n: int, per: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys, prev = [], 0
    while len(keys) < n:
        prev = int(rng.choice([k for k in (1, 1, 2, 3, 4, 5, 6, 7, 8) if k != prev]))
        keys += [prev] * per
    return np.array(keys[:n])


def intro(fork, camf):
    """the P real latents' poses / actions and the key stream after them (as play_server.load_scene: nominal camera of
    branch 0, its scheduled keys replayed over the intro's future latents 9..P-1)."""
    from shownow.data.nominal_8a import KeyStream8
    stream = KeyStream8(camf["fork"].double().numpy())
    for k in fork["actions"][0, 9:P]:
        stream.commit(int(k))
    return [camf["nominal"][0, i].float() for i in range(P)], [int(a) for a in fork["actions"][0, :P]], stream


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def gen(args):
    from shownow.eval.long_drift_eval import camera_tokens_long
    from shownow.train.student_common import flowmap_schedule, tokens
    from shownow.tree.tree import tree_call, window_layout
    from shownow.tree.cond_v3 import cam_dir_for
    from shownow.tree.tree_eval import load
    from solarwm.backends.wan22.sgf import warp_denoising_steps
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    (out / "lat").mkdir(parents=True, exist_ok=True)
    (out / f"args_rank{args.rank}.json").write_text(json.dumps(vars(args), indent=1))
    d = load(args.kind, args.student, args.field, dev)
    root = d.module
    if args.kind == "s1":
        ts, rs = flowmap_schedule(args.nfe, device=dev)
        pairs = list(zip(ts.tolist(), rs.tolist()))
    else:
        steps = warp_denoising_steps([int(v) for v in args.denoising_steps.split(",")], d.scheduler.timesteps.to(dev),
                                     num_train_timesteps=1000)
        pairs = [(float(t), None) for t in steps]
    stepper = None
    if args.mode == "realtime_opt":
        from shownow.latency.tree_opt import RTStepper, parse_opts
        tab_t, tab_s = d.scheduler.timesteps.float(), d.scheduler.sigmas.float()
        sig = [float(tab_s[(tab_t - t).abs().argmin()]) for t, _ in pairs]
        stepper = RTStepper(root, len(BRANCHES), [t for t, _ in pairs], sig, P=args.history,
                            opts=parse_opts(args.opts), device=dev)
    n_lat, per = args.seconds * 4, int(round(args.switch_sec * 4))
    BL = args.block
    assert n_lat % BL == 0 and (BL == 1 or (args.mode == "reference" and not args.src))
    for vi, scene in list(enumerate(scenes(args.videos)))[args.rank::args.world]:
        dst = out / "lat" / f"video{vi}.pt"
        if dst.exists():
            continue
        fork = torch.load(scene, map_location="cpu", weights_only=False)
        camf = torch.load(cam_dir_for(scene.parent) / scene.name, weights_only=True)
        if args.src:  # reference: replay the keys / poses of a realtime run
            src = torch.load(Path(args.src) / "lat" / f"video{vi}.pt", map_location="cpu", weights_only=False)
            keys, poses = src["keys"].numpy(), src["poses"]
            acts_intro = [int(a) for a in fork["actions"][0, :P]]
        else:
            keys = key_schedule(n_lat, per, 1000 + vi)
            poses, acts_intro, stream = intro(fork, camf)
        prompt = fork["prompt"].to(dev)[None]
        seq = [fork["latents"][0, i].to(dev, torch.bfloat16) for i in range(P)]
        g = torch.Generator(device=dev).manual_seed(vi + 10007 * args.seed)
        if stepper is not None:  # deployed path: all 8 candidates per latent, commit the scripted key
            def cams(ps):
                return camera_tokens_long(torch.stack(ps)[None], dev, args.cam_scale)
            K, ns = len(BRANCHES), len(pairs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                stepper.reset()
                stepper.set_text(root.text_embedding(prompt))
                acts_seq = list(acts_intro)
                if not args.no_ring:
                    for c0 in range(P):
                        stepper.commit_frame(seq[c0][None, None], cams([poses[c0]]), acts_intro[c0])
                for key in keys:
                    key = int(key)
                    if args.no_ring:  # no K/V ring: re-encode the last P latents from scratch (as normal inference)
                        stepper.reset()
                        for c0 in range(len(seq) - args.history, len(seq)):
                            stepper.commit_frame(seq[c0][None, None], cams([poses[c0]]), acts_seq[c0])
                    cand = [torch.from_numpy(stream.peek(k)).float() for k in range(1, K + 1)]
                    stepper.set_inputs(cams(cand), list(range(1, K + 1)))
                    shape = (1, K, *seq[0].shape)
                    noise = [torch.randn(shape, generator=g, device=dev, dtype=torch.bfloat16)]
                    noise += [torch.randn(shape, generator=g, device=dev) for _ in range(ns - 1)]
                    x = stepper.gen(noise)
                    seq.append(x[0, key - 1].clone())
                    acts_seq.append(key)
                    stepper.choose(key - 1, run_commit=not args.no_ring)
                    poses.append(torch.from_numpy(stream.commit(key)).float())
        else:
            applied = np.asarray(keys)
            if BL > 1:  # SolarWM-FT: the key held at the block start drives the whole block
                applied = np.array([keys[(j // BL) * BL] for j in range(len(keys))])
            if not args.src:
                poses = poses + [torch.from_numpy(stream.commit(int(k))).float() for k in applied]
            acts = torch.cat([torch.tensor(acts_intro), torch.from_numpy(applied).long()])[None].to(dev)
            cam = camera_tokens_long(torch.stack(list(poses))[None], dev, args.cam_scale)
            n_tok = cam["viewmats"].shape[1] // acts.shape[1]
            for j in range(P, acts.shape[1], BL):  # one AR block: window = 15 prior latents + the block
                lo = max(0, j - args.history)
                n_h = j - lo
                layout = window_layout(1, n_h, 0, t0=lo, block=BL)
                c = {k_: v_[:, lo * n_tok:(j + BL) * n_tok] for k_, v_ in cam.items()}
                a = tokens(acts[:, lo:j + BL].float()).long()
                ctx = torch.stack(seq[lo:j])[None]
                x = torch.randn((1, BL, *ctx.shape[2:]), generator=g, device=dev, dtype=torch.bfloat16)
                for i, (t, r) in enumerate(pairs):
                    tt = torch.zeros(1, n_h + BL, device=dev)
                    tt[:, n_h:] = t
                    rr = None
                    if r is not None:
                        rr = torch.zeros_like(tt)
                        rr[:, n_h:] = r
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = tree_call(root, torch.cat([ctx, x], 1), tokens(tt), prompt, c, layout,
                                      r_tok=None if rr is None else tokens(rr), action_tokens=a)
                    if r is not None:  # flow map
                        x = (x.float() - (t - r) / 1000.0 * o.float()).to(torch.bfloat16)
                        continue
                    x0 = d.flow_to_x0(x, o, torch.full((1, BL), t, device=dev))
                    if i < len(pairs) - 1:
                        tn = torch.full((BL,), pairs[i + 1][0], device=dev)
                        x = d.scheduler.add_noise(x0[0].float(), torch.randn(x0[0].shape, generator=g, device=dev),
                                                  tn)[None].to(torch.bfloat16)
                    else:
                        x = x0.to(torch.bfloat16)
                seq += [x[0, i] for i in range(BL)]
        extra = {"keys_applied": torch.from_numpy(applied)} if stepper is None and BL > 1 else {}
        torch.save({**extra, "lat": torch.stack(seq).cpu(), "keys": torch.from_numpy(np.asarray(keys)),
                    "poses": torch.stack(list(poses)), "scene": str(scene), "sid": scene.stem, "kind": args.kind,
                    "mode": (args.mode if not args.src else "reference_replay") + (f"_b{BL}" if BL > 1 else ""),
                    "student": args.student}, dst)
        print(json.dumps({"video": vi, "sid": scene.stem,
                          "keys": " ".join(BRANCHES[int(k) - 1] for k in keys[::per])}), flush=True)


# ----------------------------------------------------------------------------- scoring
def score(args):
    import imageio.v2 as iio
    from taehv import TAEHV, StreamingTAEHV
    from shownow.eval.idm import windows
    from shownow.eval.idm_8a_v2 import scorer
    from shownow.eval.long_drift_summary import fd
    from shownow.eval.long_eval import boundary_ratio, frame_stats, lat_feat, red_jumps
    dev = torch.device("cuda")
    tae = StreamingTAEHV(TAEHV(str(REPO / "third_party/taehv/taew2_2.pth")).to(dev, torch.float16).eval())

    def decode(lat):
        tae.reset()
        outf = []
        for i in range(lat.shape[0]):
            fr = tae.decode(lat[None, i:i + 1].to(dev).half())
            while fr is not None:
                outf.append((fr[0].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu())
                fr = tae.decode()
        return torch.cat(outf).numpy()

    idm = scorer(args.idm, dev)
    real_cache = {}

    def real_ref(scene):  # the 8 real branches of the fork (21 latents each)
        if scene not in real_cache:
            rl = torch.load(scene, map_location="cpu", weights_only=False)["latents"].float()
            rframes = [decode(rl[b]) for b in range(rl.shape[0])]
            st = [frame_stats(fr) for fr in rframes]
            real_cache[scene] = dict(all=lat_feat(rl.flatten(0, 1)), nb=rl.shape[0],
                                     sharp=float(np.mean([a.mean() for a, _ in st])),
                                     con=float(np.mean([b.mean() for _, b in st])),
                                     bound=float(np.mean([boundary_ratio(fr, 20) for fr in rframes])),
                                     red=float(np.mean([red_jumps(fr[33:]) for fr in rframes])))
        return real_cache[scene]

    for run in map(Path, args.runs):
        files = sorted(run.glob("lat/video*.pt"), key=lambda f: int(f.stem[5:]))
        refs = [real_ref(s) for s in sorted({torch.load(f, map_location="cpu", weights_only=False)["scene"]
                                            for f in files})]
        pool_sharp, pool_con = np.mean([r["sharp"] for r in refs]), np.mean([r["con"] for r in refs])
        per_video, gen_feats = [], []
        (run / "videos").mkdir(exist_ok=True)
        conf = np.zeros((8, 8), dtype=np.int64)
        for f in files:
            src = torch.load(f, map_location="cpu", weights_only=False)
            lat, keys = src["lat"].float(), src["keys"].long()
            N = lat.shape[0]
            frames = decode(lat)
            fut = frames[-4 * (N - P):]
            sh, co = frame_stats(fut)
            n_win = (N - P) // WIN
            gen_feats.append(lat_feat(lat[P:]).numpy())
            sharp = [float(sh[w * 4 * WIN:(w + 1) * 4 * WIN].mean() / pool_sharp) for w in range(n_win)]
            con = [float(co[w * 4 * WIN:(w + 1) * 4 * WIN].mean() / pool_con) for w in range(n_win)]
            ts = list(range(P, N - 1))
            with torch.no_grad():
                wx = windows(lat, ts).to(dev)
                pk = torch.cat([idm(wx[i:i + 64]) for i in range(0, len(ts), 64)]).argmax(1).cpu() + 1
            ok, ok_win = [], [[] for _ in range(n_win)]
            for i, t in enumerate(ts):
                j = t - P
                key = int(keys[j])
                s = 0
                while j - s - 1 >= 0 and int(keys[j - s - 1]) == key:
                    s += 1
                if s < SETTLE:
                    continue
                ok.append(int(pk[i]) == key)
                conf[key - 1, int(pk[i]) - 1] += 1
                if j // WIN < n_win:
                    ok_win[j // WIN].append(ok[-1])
            rec = {"video": f.stem, "sid": src["sid"], "key_idm_acc": float(np.mean(ok)),
                   "key_idm_acc_win": [float(np.mean(v)) for v in ok_win], "sharp_win": sharp, "contrast_win": con,
                   "washed_windows": int(sum(1 for a, b in zip(sharp, con) if a < 0.5 or b < 0.5)),
                   "boundary": boundary_ratio(frames, N - P), "red_jumps": red_jumps(fut)}
            per_video.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_win")}), flush=True)
            if args.mp4:
                iio.mimwrite(run / "videos" / f"{f.stem}_{src['sid']}.mp4", frames, fps=16, quality=7,
                             macro_block_size=8)
        agg = {"run": str(run), "n_videos": len(per_video)}
        for k in ("key_idm_acc", "boundary", "red_jumps", "washed_windows"):
            agg[k] = float(np.mean([r[k] for r in per_video]))
        for k in ("key_idm_acc_win", "sharp_win", "contrast_win"):
            agg[k] = [float(v) for v in np.mean([r[k] for r in per_video], 0)]
        agg["key_idm_acc_last15"] = float(np.mean(agg["key_idm_acc_win"][-3:]))
        agg["per_key"] = {b: round(float(conf[i, i] / max(conf[i].sum(), 1)), 3) for i, b in enumerate(BRANCHES)}
        agg["conf"] = conf.tolist()
        agg["boundary_real"] = float(np.mean([r["bound"] for r in refs]))
        agg["red_real_per_3s"] = float(np.mean([r["red"] for r in refs]))
        real = torch.cat([r["all"] for r in refs]).numpy()
        half = torch.cat([r["all"].unflatten(0, (r["nb"], 21)) for r in refs], 1)
        agg["fd_floor"] = fd(half[:4].flatten(0, 1).numpy(), half[4:].flatten(0, 1).numpy())
        g_all = np.stack(gen_feats)
        agg["fd_win"] = [fd(g_all[:, w * WIN:(w + 1) * WIN].reshape(-1, 96), real) for w in range(g_all.shape[1] // WIN)]
        agg["sharp_last"] = agg["sharp_win"][-1]
        agg["idm"] = str(args.idm)
        json.dump({"agg": agg, "videos": per_video}, open(run / args.metrics_name, "w"), indent=1)
        print("AGG", json.dumps({k: v for k, v in agg.items() if k != "conf"}), flush=True)


def table(args):
    rows = ["| run | videos | key IDM acc (all / last 15 s) | " + " | ".join(BRANCHES) +
            " | FD first/mean/last 5 s (floor) | sharp last | washed/video | boundary (real) | red/min |",
            "|---|---|---|" + "---|" * len(BRANCHES) + "---|---|---|---|---|"]
    for run in map(Path, args.runs):
        if not (run / args.metrics_name).exists():
            continue
        a = json.load(open(run / args.metrics_name))["agg"]
        fw = a["fd_win"]
        rows.append(f"| {run.name} | {a['n_videos']} | {a['key_idm_acc']:.3f} / {a['key_idm_acc_last15']:.3f} | " +
                    " | ".join(f"{a['per_key'][b]:.2f}" for b in BRANCHES) +
                    f" | {fw[0]:.1f}/{np.mean(fw):.1f}/{fw[-1]:.1f} ({a['fd_floor']:.1f}) | {a['sharp_last']:.2f} | "
                    f"{a['washed_windows']:.1f} | {a['boundary']:.2f} ({a['boundary_real']:.2f}) | "
                    f"{a['red_jumps']:.1f} |")
    print("\n".join(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--kind", required=True, choices=["s1", "s2"])
    g.add_argument("--student", required=True)
    g.add_argument("--field", default="generator_ema")
    g.add_argument("--nfe", type=int, default=3)
    g.add_argument("--denoising-steps", default="1000,667,333")
    g.add_argument("--mode", default="reference", choices=["reference", "realtime_opt"])
    g.add_argument("--opts", default="exact", help="realtime_opt: tree_opt option set")
    g.add_argument("--from", dest="src", default="", help="reference: realtime run whose keys / poses are replayed")
    g.add_argument("--out", required=True)
    g.add_argument("--videos", type=int, default=10)
    g.add_argument("--seconds", type=int, default=60)
    g.add_argument("--switch-sec", type=float, default=2.0)
    g.add_argument("--block", type=int, default=1, help="latents per AR block (3 = SolarWM-FT, reference only)")
    g.add_argument("--seed", type=int, default=0, help="noise seed offset (video vi uses vi + 10007 * seed)")
    g.add_argument("--no-ring", action="store_true",
                   help="realtime_opt ablation: no K/V ring, the last 15 latents are re-encoded every step (opts "
                        "without fused_commit)")
    g.add_argument("--history", type=int, default=15)
    g.add_argument("--cam-scale", type=float, default=0.5)
    g.add_argument("--rank", type=int, default=0)
    g.add_argument("--world", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--mp4", action="store_true")
    s.add_argument("--idm", default=str(REPO / "runs/idm_8a/v1/idm.pt"), help="IDM v1 or v2 checkpoint")
    s.add_argument("--metrics-name", default="long_metrics.json", help="v2 scoring: e.g. long_metrics_v2.json")
    t = sub.add_parser("table")
    t.add_argument("runs", nargs="+")
    t.add_argument("--metrics-name", default="long_metrics.json")
    args = ap.parse_args()
    {"gen": gen, "score": score, "table": table}[args.cmd](args)


if __name__ == "__main__":
    main()
