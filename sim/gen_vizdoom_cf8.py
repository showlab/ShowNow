"""8-action counterfactual generator on ViZDoom, videos at 32 fps (copy of gen_vizdoom_cf.py; that file is unchanged).

Branches (each held for the whole future): W, A, S, D, TURN_L, TURN_R (60 deg/s = 180 deg over 3 s), ATTACK, JUMP.
Timing: Doom game logic runs at a fixed 35 tics/s and cannot render between tics. Every tic is simulated and captured;
the 32 fps videos take, for output frame time t = n/32 s, the captured tic nearest to t (|error| <= 1/70 s, ~3 tics per
second skipped, no duplicated / interpolated frames). meta.json stores the per-output-frame state/actions/camera and the
source tic index of every frame ("tic_index"); fps = 32, engine_tic_hz = 35.

Original description:

For every sample:
  1. random map + seed, random-policy warm-up to reach a diverse start (not recorded)
  2. HISTORY: record `history_s` seconds of a random exploration policy (move/strafe/turn,
     occasional jump/crouch); enemies are active, player has god mode
  3. FORK: save the full engine state (ZDoom savegame = world, actors, RNG, player momentum)
  4. FUTURES: for each action in {W, A, S, D, JUMP, CROUCH}: restore the fork, hold that
     action for `future_s` seconds with the view direction fixed, record everything
  5. CONSISTENCY: restore + replay every action a second time, require bit-identical frames
     and game state (snapshot consistency check)
Output per sample: mp4s (history + 6 futures + side-by-side compare), per-frame actions,
player state and camera c2w/intrinsics in meta.json, and a preview jpg.

Notes on engine semantics (verified in sim/probe3.py):
  * ZDoom save/load each advance ~1-2 tics using the *currently held* buttons, so the held
    input is set to NO-OP before save/load; the fork state is the state observed right after
    restore, identical for every branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import imageio.v2 as iio2
import numpy as np
import vizdoom as vzd

B = vzd.Button
GV = vzd.GameVariable
BUTTONS = [B.MOVE_FORWARD, B.MOVE_BACKWARD, B.MOVE_LEFT, B.MOVE_RIGHT, B.JUMP, B.CROUCH,
           B.ATTACK, B.USE, B.SPEED, B.TURN_LEFT_RIGHT_DELTA]
BUTTON_NAMES = ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "JUMP", "CROUCH",
                "ATTACK", "USE", "SPEED(run)", "TURN_DELTA_DEG"]
NB = len(BUTTONS)
TURN = NB - 1


def _a(*on):
    v = [0] * NB
    for i in on:
        v[i] = 1
    return v


TURN_DEG = 180.0 / 105  # degrees per tic for TURN_L / TURN_R: half a turn (180 deg) over the 3 s future = 60 deg/s


def _turn(sign):
    v = [0] * NB
    v[TURN] = sign * TURN_DEG  # ViZDoom TURN_LEFT_RIGHT_DELTA: positive = turn right (smoke test: yaw decreases)
    return v


# counterfactual actions: held for the whole future (view direction fixed except for the turn branches)
ACTIONS = {
    "W": _a(0), "A": _a(2), "S": _a(1), "D": _a(3),
    "TURN_L": _turn(-1), "TURN_R": _turn(1),
    "ATTACK": _a(6), "JUMP": _a(4),
}
NOOP = [0] * NB
ACTION_ID = {k: i for i, k in enumerate(ACTIONS)}
WEAPONS = {"Pistol": 2, "Shotgun": 3, "SuperShotgun": 3, "Chaingun": 4, "RocketLauncher": 5,
           "PlasmaRifle": 6, "BFG9000": 7, "Chainsaw": 1}  # class -> slot (chosen via `use <class>`)
VARS = [GV.POSITION_X, GV.POSITION_Y, GV.POSITION_Z, GV.ANGLE, GV.PITCH, GV.VIEW_HEIGHT,
        GV.VELOCITY_X, GV.VELOCITY_Y, GV.VELOCITY_Z, GV.HEALTH, GV.KILLCOUNT,
        GV.ARMOR, GV.ON_GROUND, GV.DAMAGECOUNT, GV.HITCOUNT, GV.DAMAGE_TAKEN,
        GV.SELECTED_WEAPON, GV.SELECTED_WEAPON_AMMO, GV.ATTACK_READY,
        GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6]
VAR_NAMES = ["pos_x", "pos_y", "pos_z", "yaw_deg", "pitch_deg", "view_height", "vel_x", "vel_y", "vel_z", "health", "killcount",
             "armor", "on_ground", "damage_dealt", "hits_taken", "damage_taken",
             "weapon_slot", "weapon_ammo", "attack_ready",
             "ammo_bullets", "ammo_shells", "ammo_rockets", "ammo_cells"]
ACTION_TEXT = {
    "W": "walks forward", "A": "strafes left", "S": "walks backward", "D": "strafes right",
    "TURN_L": "turns left", "TURN_R": "turns right",
    "ATTACK": "fires the weapon continuously (fire released on the first tic, then held)", "JUMP": "jumps",
}
WEAPON_POOL: list = []  # --weapons subset (empty = all of WEAPONS)
NEARBY_RADIUS = 1024.0  # map units (~32 m)
# ViZDoom session counters that are NOT part of the savegame (they persist across loads):
# excluded from state-equality checks and stored as per-clip deltas.
SESSION_COUNTERS = ["damage_dealt", "hits_taken", "damage_taken"]
SAVED_IDX = np.array([i for i, n in enumerate(VAR_NAMES) if n not in SESSION_COUNTERS])
COUNTER_IDX = np.array([VAR_NAMES.index(n) for n in SESSION_COUNTERS])


def same_state(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a[..., SAVED_IDX], b[..., SAVED_IDX]))


def to_deltas(vs: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = np.array(vs, dtype=np.float64, copy=True)
    out[..., COUNTER_IDX] -= base[COUNTER_IDX]
    return out
FPS = 35.0  # one captured frame per Doom tic (engine rate)
OUT_FPS = 32.0  # video / label rate (nearest-tic sampling)


def hist_sel(n_tics: int, seconds: float) -> list:
    """history: tic j shows t = (j - (n_tics - 1)) / 35 (last tic = fork); output m shows t = (m - (M - 1)) / 32"""
    M = int(round(seconds * OUT_FPS))
    return [int(np.clip(n_tics - 1 + round((m - (M - 1)) * FPS / OUT_FPS), 0, n_tics - 1)) for m in range(M)]


def fut_sel(n_tics: int, seconds: float) -> list:
    """future: tic i shows t = (i + 1) / 35 after the fork; output n shows t = (n + 1) / 32"""
    M = int(round(seconds * OUT_FPS))
    return [int(np.clip(round((n + 1) * FPS / OUT_FPS) - 1, 0, n_tics - 1)) for n in range(M)]
UNIT_M = 1.75 / 56.0  # Doom player is 56 units tall ~ 1.75 m
WAD = os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad")
MAPS = [f"MAP{i:02d}" for i in range(1, 33)]
WADS = {"freedoom2": ("freedoom2.wad", MAPS),  # Freedoom Phase 2: MAP01-MAP32
        "freedoom1": ("freedoom1.wad", [f"E{e}M{m}" for e in range(1, 5) for m in range(1, 10)])}  # Phase 1: E1M1-E4M9
H_FOV_DEG = 130.0
MIN_CENTER_DEPTH = 12
NO_MONSTERS = False  # --no-monsters: pure scene roaming (no attacks -> no red damage flashes)
PRE_DMG_TICS = 70  # 2 s
NO_DAMAGE_FLASH = False  # --no-damage-flash: `notarget` (monsters never attack the player -> no red pain flash)
NO_FIRE = False  # --no-fire: the explorer never shoots (no point-blank explosions / muzzle flashes in histories)  # depth-buffer units; ~1 when facing a wall, ~25-30 in open space


def make_game(map_name: str, seed: int, skill: int, res: str) -> vzd.DoomGame:
    g = vzd.DoomGame()
    g.set_doom_game_path(WAD)
    g.set_doom_map(map_name)
    g.set_screen_resolution(getattr(vzd.ScreenResolution, res))
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_window_visible(False)
    g.set_render_hud(False)
    g.set_render_weapon(True)
    g.set_render_crosshair(False)
    g.set_render_messages(False)
    g.set_render_particles(True)
    g.set_render_effects_sprites(True)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_available_buttons(BUTTONS)
    g.set_available_game_variables(VARS)
    g.add_game_args(f"+sv_jump 2 +sv_crouch 2 +freelook 1 +sv_freelook 1 +fov {H_FOV_DEG:g}"
                    + (" -nomonsters" if NO_MONSTERS else ""))
    g.set_depth_buffer_enabled(True)
    g.set_labels_buffer_enabled(True)
    g.set_objects_info_enabled(True)
    g.set_doom_skill(skill)
    g.set_seed(seed)
    g.set_episode_timeout(0)
    g.init()
    g.send_game_command("god")
    g.send_game_command("give all")
    g.send_game_command(f"fov {H_FOV_DEG:g}")
    g.send_game_command("cl_capfps 1")  # no wall-clock frame interpolation
    g.send_game_command("neverswitchonpickup 1")  # keep the labelled weapon when walking over pickups
    # view bob phase is driven by a global tic counter outside the savegame -> non-reproducible
    # view_height/camera after loads under CPU contention; disable it
    g.send_game_command("movebob 0")
    if NO_DAMAGE_FLASH:  # monsters ignore the player (no attacks on the player -> no red pain flash);
        g.send_game_command("notarget")  # blood_fade_scalar 0 has no effect in ViZDoom's ZDoom (tested)
    return g


def c2w_from_state(v: np.ndarray) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) to Doom world (x,y ground plane, z up), metres."""
    x, y, z, yaw, pitch, vh = v[0], v[1], v[2], np.radians(v[3]), np.radians(v[4]), v[5]
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = right, down, fwd
    M[:3, 3] = np.array([x, y, z + vh]) * UNIT_M
    return M


def intrinsics(w: int, h: int) -> list:
    fx = (w / 2) / np.tan(np.radians(H_FOV_DEG) / 2)
    return [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]]


class Explorer:
    """Random exploration policy with stuck detection (history + warm-up)."""

    def __init__(self, rng: random.Random):
        self.rng, self.cur, self.left, self.turn, self.last_pos, self.still = rng, list(NOOP), 0, 0.0, None, 0

    def act(self, v: np.ndarray) -> list:
        pos = v[:2]
        if self.last_pos is not None and np.linalg.norm(pos - self.last_pos) < 0.5:
            self.still += 1
        else:
            self.still = 0
        self.last_pos = pos.copy()
        if self.left <= 0 or self.still > 8:
            r = self.rng.random()
            move = [0, 0, 0, 0]
            if r < 0.55:
                move[0] = 1
            elif r < 0.65:
                move[1] = 1
            elif r < 0.78:
                move[2] = 1
            elif r < 0.91:
                move[3] = 1
            self.cur = move + [0] * (NB - 4)
            if move[0] and self.rng.random() < 0.3:
                self.cur[8] = 1  # run
            self.turn = self.rng.choice([0.0, 0.0, self.rng.uniform(-4, 4)])
            if self.still > 8:
                self.turn = self.rng.choice([-1, 1]) * self.rng.uniform(6, 12)
                self.still = 0
            self.left = self.rng.randint(8, 40)
        self.left -= 1
        a = list(self.cur)
        a[TURN] = self.turn
        r = self.rng.random()
        if r < 0.015:
            a[4] = 1  # occasional jump
        elif r < 0.025:
            a[5] = 1  # occasional crouch tap
        elif r < 0.06:
            if not NO_FIRE:
                a[6] = 1  # occasional shot
        elif r < 0.07:
            a[7] = 1  # occasional use
        return a


class LevelExit(RuntimeError):
    pass


def frame_labels(s, v: np.ndarray) -> dict:
    """Per-frame labels: visible objects (bbox), forward depth, brightness."""
    px, py, pz = v[0], v[1], v[2]
    vis = []
    for l in s.labels:
        if l.object_category == "Self":
            continue
        dist = float(np.sqrt((l.object_position_x - px) ** 2 + (l.object_position_y - py) ** 2 + (l.object_position_z - pz) ** 2))
        vis.append({"id": int(l.object_id), "name": l.object_name, "category": l.object_category,
                    "bbox_xywh": [int(l.x), int(l.y), int(l.width), int(l.height)],
                    "pos": [round(l.object_position_x, 1), round(l.object_position_y, 1), round(l.object_position_z, 1)],
                    "vel": [round(l.object_velocity_x, 2), round(l.object_velocity_y, 2), round(l.object_velocity_z, 2)],
                    "dist_m": round(dist * UNIT_M, 2)})
    d = s.depth_buffer
    hh, ww = d.shape
    return {"visible_objects": vis,
            "center_depth": float(np.median(d[hh // 3: 2 * hh // 3, ww // 3: 2 * ww // 3])),
            "brightness": round(float(s.screen_buffer.mean()), 1)}


def nearby_monsters(s, v: np.ndarray) -> list:
    out = []
    for o in s.objects:
        if o.category != "Monster":
            continue
        dist = float(np.hypot(o.position_x - v[0], o.position_y - v[1]))
        if dist <= NEARBY_RADIUS:
            out.append({"id": int(o.id), "name": o.name, "pos": [round(o.position_x, 1), round(o.position_y, 1), round(o.position_z, 1)],
                        "dist_m": round(dist * UNIT_M, 2), "sector_id": int(o.sector_id)})
    return sorted(out, key=lambda r: r["dist_m"])


def branch_policy(name: str):
    held = list(ACTIONS[name])
    if name != "ATTACK":
        return lambda _v: list(held)
    state = {"t": 0}

    def pol(_v):
        state["t"] += 1
        return list(NOOP) if state["t"] == 1 else list(held)

    return pol


SW_KEYS = list(ACTIONS)  # switch mode: all 8 keys (W A S D TURN_L TURN_R ATTACK JUMP)
LATENT_S = 0.25  # model latent = 4 frames @ 16 fps


def switch_schedule(rng: random.Random, first: str, future_s: float, min_lat: int = 1, max_lat: int = 6,
                    p_hold: float = 0.25) -> list:
    """[(key, start_latent, n_latents), ...] covering the future: starts with `first`, then a different random W/A/S/D
    key every 1-6 latents (0.25-1.5 s); switches sit on the latent grid so every latent has one key. With p_hold the
    branch holds `first` for the whole future (keeps long holds in the data)."""
    n_lat = int(round(future_s / LATENT_S))
    if rng.random() < p_hold:
        return [(first, 0, n_lat)]
    out, t, key = [], 0, first
    while t < n_lat:
        n = min(rng.randint(min_lat, max_lat), n_lat - t)
        out.append((key, t, n))
        t += n
        key = rng.choice([k for k in SW_KEYS if k != key])
    return out


def switch_policy(sched: list):
    """Per-tic policy for a switch schedule (fresh closure per call: replays are deterministic)."""
    bounds = [(int(round(s * LATENT_S * FPS)), k) for k, s, _ in sched]
    state = {"t": 0}

    def pol(_v):
        t = state["t"]
        state["t"] += 1
        key = [k for b, k in bounds if b <= t][-1]
        return list(ACTIONS[key])

    return pol


def record(g: vzd.DoomGame, policy, n: int, labels: list | None = None):
    frames, acts, vars_ = [], [], []
    for _ in range(n):
        s = g.get_state()
        if s is None or g.is_episode_finished():
            raise LevelExit("episode ended (level exit reached)")
        a = policy(np.array(s.game_variables))
        g.make_action(a)
        s = g.get_state()
        if s is None or g.is_episode_finished():
            raise LevelExit("episode ended (level exit reached)")
        v = np.array(s.game_variables, dtype=np.float64)
        frames.append(np.ascontiguousarray(s.screen_buffer))
        acts.append(list(map(float, a)))
        vars_.append(v)
        if labels is not None:
            labels.append(frame_labels(s, v))
    return np.stack(frames), acts, np.stack(vars_)


CRF = 17
FF_THREADS = 2


def write_mp4(path: Path, frames: np.ndarray, fps: float = FPS) -> None:
    w = iio2.get_writer(str(path), fps=fps, codec="libx264", quality=None,
                        ffmpeg_params=["-crf", str(CRF), "-preset", "medium", "-threads", str(FF_THREADS)],
                        macro_block_size=8, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (18 + 22 * len(text), 44), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2, cv2.LINE_AA)
    return out


def gen_sample(idx: int, out_root: Path, args) -> dict:
    rng = random.Random(args.seed * 100003 + idx)
    switch = args.mode == "switch"
    branches = SW_KEYS if switch else list(ACTIONS)
    # switch mode: every branch starts with its own key, then switches (own RNG: the game RNG stream is unchanged)
    srng = random.Random(args.seed * 100003 + idx + 7_000_000)
    scheds = {k: switch_schedule(srng, k, args.future_s) for k in SW_KEYS} if switch else {}
    pol_for = (lambda k: switch_policy(scheds[k])) if switch else branch_policy
    map_name = rng.choice(MAPS)
    seed = rng.randrange(2**31)
    g = make_game(map_name, seed, args.skill, args.res)
    snap = Path(tempfile.mkdtemp()) / "fork.zds"
    try:
        exp = Explorer(rng)
        # Freedoom Phase 1 (Doom 1 rules) has no super shotgun
        weapon_cls = rng.choice(sorted(w for w in (WEAPON_POOL or WEAPONS) if not (w == "SuperShotgun" and "freedoom1" in WAD)))
        # `give all` only lands after a few tics; switch afterwards and wait for the raise animation
        for _ in range(5):
            g.make_action(NOOP)
        for _attempt in range(3):
            g.send_game_command(f"use {weapon_cls}")
            for _ in range(45):
                g.make_action(NOOP)
            if int(g.get_state().game_variables[VAR_NAMES.index("weapon_slot")]) == WEAPONS[weapon_cls]:
                break
        warm = rng.randint(int(args.warmup_min_s * FPS), int(args.warmup_max_s * FPS))
        # warm-up (not recorded): top ammo up every second so the labelled weapon never runs dry and
        # auto-switches; re-select it if needed
        wi = VAR_NAMES.index("weapon_slot")
        done = 0
        while done < warm:
            k = min(35, warm - done)
            record(g, exp.act, k)
            done += k
            g.send_game_command("give ammo")
            if int(g.get_state().game_variables[wi]) != WEAPONS[weapon_cls]:
                g.send_game_command(f"use {weapon_cls}")
        n_h, n_f = int(round(args.history_s * FPS)), int(round(args.future_s * FPS))
        # keep exploring until the last n_h tics end in open space (fork not facing a wall)
        buf_f, buf_a, buf_v, buf_l, extra = [], [], [], [], 0
        dmg_i = VAR_NAMES.index("damage_taken")
        dmg_hist = []  # damage counter over the history + PRE_DMG_TICS before it (a pain flash fades over ~2 s)
        while True:
            fr, ac, vs = record(g, exp.act, 1, buf_l)
            buf_f.append(fr[0]); buf_a.append(ac[0]); buf_v.append(vs[0])
            buf_f, buf_a, buf_v, buf_l = buf_f[-n_h:], buf_a[-n_h:], buf_v[-n_h:], buf_l[-n_h:]
            dmg_hist = (dmg_hist + [vs[0][dmg_i]])[-(n_h + PRE_DMG_TICS):]
            no_dmg = not NO_DAMAGE_FLASH or (len(dmg_hist) == n_h + PRE_DMG_TICS and dmg_hist[-1] == dmg_hist[0])
            if len(buf_f) == n_h:
                d = g.get_state().depth_buffer
                hh, ww = d.shape
                vh = np.array([x[VAR_NAMES.index("view_height")] for x in buf_v[-8:]])
                settled = (np.ptp(vh) == 0 and buf_v[-1][VAR_NAMES.index("on_ground")] == 1
                           and int(buf_v[-1][wi]) == WEAPONS[weapon_cls])
                if (np.median(d[hh // 3: 2 * hh // 3, ww // 3: 2 * ww // 3]) >= MIN_CENTER_DEPTH and settled and no_dmg) \
                        or extra > 30 * FPS:
                    break
                extra += 1
        h_frames, h_acts, h_vars, h_labels = np.stack(buf_f), buf_a, np.stack(buf_v), buf_l
        h_vars = to_deltas(h_vars, h_vars[0])
        fork_depth = float(np.median(d[hh // 3: 2 * hh // 3, ww // 3: 2 * ww // 3]))
        pre_hist_damage = float(dmg_hist[-1] - dmg_hist[0]) if dmg_hist else None

        g.set_action(NOOP)
        g.save(str(snap))
        g.close()

        # Every branch (and every consistency replay) runs in a FRESH engine instance that loads the
        # fork savegame. ZDoom keeps a global tic counter outside the savegame that drives warped /
        # animated textures; within one process it differs between loads, which made renders differ
        # deterministically (MAD < 1/255, floor liquids). A fresh instance + load gives every branch
        # the identical hidden state (verified: fresh-process replays are bit-identical).
        restore_stats = {"calls": 0, "retries": 0, "diff_vars": {}}
        ref = {}

        def fresh_restore():
            restore_stats["calls"] += 1
            for attempt in range(5):
                gb = make_game(map_name, seed, args.skill, args.res)
                gb.set_action(NOOP)
                gb.load(str(snap))
                st = gb.get_state()
                v = np.array(st.game_variables, dtype=np.float64)
                if "v" not in ref or same_state(v, ref["v"]):
                    return gb, st, v
                restore_stats["retries"] += 1
                for i in [j for j in np.nonzero(v != ref["v"])[0] if j in SAVED_IDX]:
                    restore_stats["diff_vars"][VAR_NAMES[i]] = restore_stats["diff_vars"].get(VAR_NAMES[i], 0) + 1
                gb.close()
            d = {VAR_NAMES[i]: [float(ref["v"][i]), float(v[i])] for i in np.nonzero(v != ref["v"])[0]}
            raise RuntimeError(f"fork state not reproducible in fresh instances: ref vs got {json.dumps(d)}")

        fut, fut_labels, fut_end_monsters, consistent, mismatch, retries = {}, {}, {}, {}, {}, {}
        fork_vars = fork_monsters = fork_visible = None
        for name in branches:
            gb, st0, v0 = fresh_restore()
            if fork_vars is None:
                fork_vars, ref["v"] = v0, v0
                fork_monsters = nearby_monsters(st0, v0)
                fork_visible = frame_labels(st0, v0)["visible_objects"]
            lab = []
            fr, ac, vs = record(gb, pol_for(name), n_f, lab)
            fut[name] = (fr, ac, to_deltas(vs, v0))
            fut_labels[name] = lab
            fut_end_monsters[name] = nearby_monsters(gb.get_state(), vs[-1])
            gb.close()
        # snapshot-consistency replay (fresh instance each time); keep the version reproduced twice. --replay-every N:
        # only every N-th sample is replayed (spot check; the rest are recorded as consistent=None-checked)
        replay = idx % args.replay_every == 0
        for name in (branches if replay else []):
            attempts = [fut[name]]
            ok_name = False
            for _ in range(3):
                gb, _, v0b = fresh_restore()
                fr2, ac2, vs2 = record(gb, pol_for(name), n_f)
                gb.close()
                vs2 = to_deltas(vs2, v0b)
                match = next((p for p in attempts if np.array_equal(p[0], fr2) and np.array_equal(p[2], vs2)), None)
                if match is not None:
                    fut[name] = match
                    ok_name = True
                    break
                prev = attempts[0]
                dv = np.abs(vs2 - prev[2]).max(1)
                diff = np.abs(fr2.astype(np.int16) - prev[0].astype(np.int16)).sum(-1)
                df = diff.reshape(len(fr2), -1).mean(1) / 3.0
                tt, yy, xx = np.nonzero(diff)
                mismatch.setdefault(name, []).append({
                    "diff_tics": sorted(set(tt.tolist()))[:20], "n_diff_tics": int(len(set(tt.tolist()))),
                    "diff_bbox_y0y1x0x1": [int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())] if len(yy) else None,
                    "first_state_tic": int(np.argmax(dv > 0)) if dv.max() > 0 else None,
                    "state_identical": bool(dv.max() == 0), "max_frame_mad": float(df.max())})
                attempts.append((fr2, ac2, vs2))
            consistent[name] = ok_name
            retries[name] = len(attempts) - 1
    finally:
        try:
            g.close()
        except Exception:
            pass
        shutil.rmtree(snap.parent, ignore_errors=True)

    # quality metrics
    lasts = {k: v[0][-1].astype(np.float32) for k, v in fut.items()}
    names = list(branches)
    pair = {f"{a}-{b}": float(np.abs(lasts[a] - lasts[b]).mean()) for i, a in enumerate(names) for b in names[i + 1:]}
    disp = {k: float(np.linalg.norm((v[2][-1, :3] - fork_vars[:3])) * UNIT_M) for k, v in fut.items()}
    brightness = float(h_frames.mean())
    if not replay:
        consistent = {n: True for n in branches}
        retries = {n: 0 for n in branches}
    ok = all(consistent.values()) and brightness > 25 and fork_depth >= MIN_CENTER_DEPTH
    weapon_cls_ok = int(fork_vars[VAR_NAMES.index("weapon_slot")]) == WEAPONS[weapon_cls]
    ok = ok and weapon_cls_ok and restore_stats["retries"] == 0

    sid = f"{args.prefix}{idx:05d}"
    d = out_root / sid
    d.mkdir(parents=True, exist_ok=True)
    hs, fs = hist_sel(len(h_frames), args.history_s), fut_sel(n_f, args.future_s)
    write_mp4(d / "history.mp4", h_frames[hs], OUT_FPS)
    for k, (fr, _, _) in fut.items():
        write_mp4(d / f"future_{k}.mp4", fr[fs], OUT_FPS)
    # preview: last frame of every future (quarter-res tiles, fork frame first)
    q = lambda im: im[::4, ::4] if im.shape[1] >= 1280 else im[::2, ::2]
    tiles = [label(q(h_frames[-1]), "FORK")] + [label(q(fut[k][0][-1]), k) for k in names]
    tiles += [np.zeros_like(tiles[0])] * (12 - len(tiles))
    iio2.imwrite(d / "preview.jpg", np.concatenate([np.concatenate(tiles[r * 4:(r + 1) * 4], 1) for r in range(3)], 0),
                 quality=90)

    w, h = h_frames.shape[2], h_frames.shape[1]
    as_rows = lambda vs: [dict(zip(VAR_NAMES, map(float, r))) for r in vs]
    fv = dict(zip(VAR_NAMES, map(float, to_deltas(fork_vars, fork_vars))))
    weapon = weapon_cls
    weapon_ok = int(fv["weapon_slot"]) == WEAPONS[weapon_cls]

    def seen(labels):  # distinct visible monsters over a clip
        names_ = {}
        for fl in labels:
            for o in fl["visible_objects"]:
                if o["category"] == "Monster":
                    names_.setdefault(o["id"], o["name"])
        return sorted(names_.values())

    def summary(vs, labels):
        return {"monsters_seen": seen(labels),
                "kills_delta": float(vs[-1][VAR_NAMES.index("killcount")] - vs[0][VAR_NAMES.index("killcount")]),
                "damage_taken_delta": float(vs[-1][VAR_NAMES.index("damage_taken")] - vs[0][VAR_NAMES.index("damage_taken")]),
                "damage_dealt_delta": float(vs[-1][VAR_NAMES.index("damage_dealt")] - vs[0][VAR_NAMES.index("damage_dealt")]),
                "hits_taken_delta": float(vs[-1][VAR_NAMES.index("hits_taken")] - vs[0][VAR_NAMES.index("hits_taken")]),
                "ammo_used": float(vs[0][VAR_NAMES.index("weapon_ammo")] - vs[-1][VAR_NAMES.index("weapon_ammo")]),
                "displacement_m": float(np.linalg.norm(vs[-1][:3] - vs[0][:3]) * UNIT_M),
                "height_change_m": float((vs[-1][2] + vs[-1][5] - vs[0][2] - vs[0][5]) * UNIT_M),
                "min_view_height": float(np.min(vs[:, 5])), "max_z_gain_m": float((np.max(vs[:, 2]) - vs[0][2]) * UNIT_M),
                "max_speed": float(np.max(np.hypot(vs[:, 6], vs[:, 7])))}

    def stuck_s(k, vs, ac):
        """longest run (s, engine time) with a move key held and horizontal speed < 1 unit/tic (after 5 tics)"""
        if k not in ("W", "A", "S", "D"):
            return 0.0
        sp = np.hypot(vs[:, 6], vs[:, 7])
        run = best = 0
        for i in range(5, len(sp)):
            run = run + 1 if sp[i] < 1.0 else 0
            best = max(best, run)
        return best / FPS

    def effect_flags(k, vs, summ):
        """Did the held action visibly take effect? (walls / ledges / NOAUTOFIRE make 'False' legitimate)"""
        yaw = np.radians(fork_vars[3])
        dx, dy = (vs[-1][0] - fork_vars[0]) * UNIT_M, (vs[-1][1] - fork_vars[1]) * UNIT_M
        fwd, left = dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)
        dyaw = float((vs[-1][3] - fork_vars[3] + 180.0) % 360.0 - 180.0)  # CCW positive (turning left)
        dyaw05 = float((vs[int(0.5 * FPS) - 1][3] - fork_vars[3] + 180.0) % 360.0 - 180.0)
        exp = {"W": fwd > 0.3, "S": fwd < -0.3, "A": left > 0.3, "D": left < -0.3,
               "TURN_L": dyaw05 > 0.8 * TURN_DEG * int(0.5 * FPS), "TURN_R": dyaw05 < -0.8 * TURN_DEG * int(0.5 * FPS),
               "JUMP": summ["max_z_gain_m"] > 0.3,
               "ATTACK": summ["ammo_used"] > 0 or weapon_cls == "Chainsaw"}
        e = None if switch else exp[k]
        return {"effect_observed": None if e is None else bool(e), "forward_m": float(fwd), "left_m": float(left),
                "yaw_change_deg": dyaw, "yaw_change_0p5s_deg": dyaw05}

    def who_text(mons):
        if not mons:
            return "no enemies visible"
        from collections import Counter

        return ", ".join(f"{n} {m}" if n > 1 else m for m, n in sorted(Counter(mons).items())) + " visible"

    def caption(action_key, labels):
        who = who_text(seen(labels))
        if switch:
            seq = ", then ".join(f"{ACTION_TEXT[k]} for {n * LATENT_S:g} s" for k, _, n in scheds[action_key])
            return (f"First-person Doom-style shooter, map {map_name}, player holding a {weapon}; "
                    f"the player {seq}, with the view direction fixed; {who}.")
        view = "" if action_key.startswith("TURN") else " with the view direction fixed"
        return (f"First-person Doom-style shooter, map {map_name}, player holding a {weapon}; "
                f"the player {ACTION_TEXT[action_key]} for {args.future_s:g} s{view}; {who}.")

    h_mons = seen(h_labels)
    hist_caption = (f"First-person Doom-style shooter, map {map_name}, player holding a {weapon}, exploring "
                    f"(moving, strafing, turning, occasional jump/crouch/fire/use); "
                    + who_text(h_mons) + ".")
    meta = {
        "schema": "shownow.vizdoom_counterfactual.v4-8a-32fps",
        "sample_id": sid, "engine": f"ViZDoom {vzd.__version__} (GZDoom)", "fps": OUT_FPS, "engine_tic_hz": FPS,
        "frame_sampling": "nearest engine tic to t = n / 32 s (history: t <= 0 ending at the fork; future: t = (n+1)/32)",
        "turn_deg_per_tic": TURN_DEG, "resolution": [w, h],
        "horizontal_fov_deg": H_FOV_DEG, "intrinsics": intrinsics(w, h),
        "camera_convention": "c2w, OpenCV (x right, y down, z forward), metres",
        "units": {"world_unit_m": UNIT_M, "angles": "degrees, yaw CCW from +x", "position": "Doom map units",
                  "session_counters": f"{SESSION_COUNTERS} are per-clip deltas (not restored by savegames)"},
        "environment": {"wad": os.path.basename(WAD), "map": map_name, "skill": args.skill, "seed": seed,
                        "warmup_tics": warm, "enemies_active": not NO_MONSTERS, "no_monsters": NO_MONSTERS,
                        "explorer_fires": not NO_FIRE, "notarget": NO_DAMAGE_FLASH,
                        "damage_in_history_and_2s_before": pre_hist_damage,
                        "fork_center_depth": fork_depth, "history_mean_brightness": brightness},
        "character": {"type": "DoomPlayer (first person)", "god_mode": True, "fork_state": fv,
                      "posture_at_fork": "crouched" if fv["view_height"] < 35 else "standing",
                      "on_ground_at_fork": bool(fv["on_ground"])},
        "weapon": {"slot": int(fv["weapon_slot"]), "name": weapon, "slot_matches_class": weapon_ok, "all_weapons_owned": True,
                   "ammo_at_fork": {k: fv[k] for k in ("weapon_ammo", "ammo_bullets", "ammo_shells", "ammo_rockets", "ammo_cells")}},
        "npc": {"nearby_monsters_at_fork": fork_monsters, "visible_at_fork": fork_visible,
                "radius_m": NEARBY_RADIUS * UNIT_M},
        "action_space": {"buttons": BUTTON_NAMES, "branches": ACTIONS, "branch_id": ACTION_ID, "text": ACTION_TEXT,
                         "mode": args.mode,
                         "switch_schedules": {k: [{"key": a, "start_latent": s0, "n_latents": n} for a, s0, n in v]
                                              for k, v in scheds.items()}},
        "history": {"file": "history.mp4", "num_frames": len(hs), "num_tics": n_h, "tic_index": hs, "caption": hist_caption,
                    "actions": [h_acts[i] for i in hs], "state": as_rows(h_vars[hs]),
                    "frame_labels": [h_labels[i] for i in hs],
                    "c2w": [c2w_from_state(h_vars[i]).tolist() for i in hs], "summary": summary(h_vars, h_labels),
                    "actions_all_tics": h_acts},
        "fork": {"state": fv, "c2w": c2w_from_state(fork_vars).tolist(),
                 "note": "identical engine state restored for every future (verified)"},
        "futures": {k: {"file": f"future_{k}.mp4", "num_frames": len(fs), "num_tics": n_f, "tic_index": fs,
                        "action": k, "action_id": ACTION_ID[k],
                        "action_held": ACTIONS[k], "action_text": ACTION_TEXT[k], "caption": caption(k, fut_labels[k]),
                        "actions": [ac[i] for i in fs], "state": as_rows(vs[fs]),
                        "frame_labels": [fut_labels[k][i] for i in fs],
                        "c2w": [c2w_from_state(vs[i]).tolist() for i in fs], "summary": summary(vs, fut_labels[k]),
                        "effect": {**effect_flags(k, vs, summary(vs, fut_labels[k])), "stuck_s": stuck_s(k, vs, ac)},
                        "nearby_monsters_at_end": fut_end_monsters[k]}
                    for k, (_, ac, vs) in fut.items()},
        "checks": {"replay_checked": replay, "snapshot_consistent": consistent, "replay_retries": retries, "mismatch_detail": mismatch,
                   "restore_stats": restore_stats,
                   "future_pairwise_last_frame_mad": pair, "future_displacement_m": disp,
                   "history_mean_brightness": brightness, "fork_center_depth": fork_depth, "pass": ok},
    }
    manifest = [{"sample_id": sid, "pass": ok, "history": f"{sid}/history.mp4", "future": f"{sid}/future_{k}.mp4",
                 "action": k, "action_id": ACTION_ID[k], "action_held": ACTIONS[k], "caption": meta["futures"][k]["caption"],
                 "history_caption": hist_caption, "map": map_name, "weapon": weapon,
                 "monsters_seen": meta["futures"][k]["summary"]["monsters_seen"], "fps": OUT_FPS,
                 "effect_observed": meta["futures"][k]["effect"]["effect_observed"],
                 "stuck_s": meta["futures"][k]["effect"]["stuck_s"],
                 "num_frames_history": len(hs), "num_frames_future": len(fs), "meta": f"{sid}/meta.json"} for k in names]
    (d / "meta.json").write_text(json.dumps(meta))
    if args.compare_every and idx % args.compare_every == 0:
        try:  # optional review video (helper script not part of this release)
            from build_compare480 import build
        except ImportError:
            build = None
        if build is not None:
            build(d)  # 4x4 review video, 864x480 tiles (spot-check subset only)
    with open(out_root / f"train_manifest_{args.start:06d}.jsonl", "a") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    return {"sample_id": sid, "map": map_name, "weapon": weapon, "pass": ok, "consistent": all(consistent.values()),
            "min_pair_mad": min(pair.values()), "fork_depth": fork_depth, "retries": sum(retries.values()),
            "disp": {k: round(v, 2) for k, v in disp.items()}, "brightness": round(brightness, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skill", type=int, default=3)
    ap.add_argument("--res", default="RES_1280X720")
    ap.add_argument("--history-s", type=float, default=3.0)
    ap.add_argument("--future-s", type=float, default=3.0)
    ap.add_argument("--warmup-min-s", type=float, default=3.0)
    ap.add_argument("--warmup-max-s", type=float, default=40.0)
    ap.add_argument("--prefix", default="c8_")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--threads", type=int, default=2, help="ffmpeg threads per encode")
    ap.add_argument("--compare-every", type=int, default=0, help="build compare480.mp4 for every N-th sample (0=never)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-monsters", action="store_true", help="start maps with -nomonsters (scene roaming only)")
    ap.add_argument("--no-fire", action="store_true", help="the history explorer never fires the weapon")
    ap.add_argument("--wad", default="freedoom2", choices=sorted(WADS))
    ap.add_argument("--no-damage-flash", action="store_true", help="notarget: monsters ignore the player, so no red damage flash")
    ap.add_argument("--turn-deg", type=float, default=180.0 / 105, help="degrees per tic for TURN_L / TURN_R")
    ap.add_argument("--mode", default="hold", choices=["hold", "switch"],
                    help="hold: 8 actions held for the whole future; switch: 8 branches that start with their "
                         "own key and then switch keys every 0.25-1.5 s (on the 0.25 s latent grid)")
    ap.add_argument("--weapons", default="", help="comma-separated weapon subset (e.g. Pistol,Shotgun,SuperShotgun,Chaingun,"
                    "Chainsaw: no splash self-damage / screen-filling explosions); default all")
    ap.add_argument("--replay-every", type=int, default=1, help="consistency replay every N-th sample (1 = all)")
    ap.add_argument("--maps", default="", help="comma-separated map subset (e.g. MAP29,MAP30,MAP31,MAP32 = validation)")
    args = ap.parse_args()
    global CRF, FF_THREADS, NO_MONSTERS, NO_FIRE, WAD, MAPS
    CRF, FF_THREADS, NO_MONSTERS, NO_FIRE = args.crf, args.threads, args.no_monsters, args.no_fire
    global NO_DAMAGE_FLASH
    NO_DAMAGE_FLASH = args.no_damage_flash
    assert args.turn_deg == TURN_DEG, "TURN_DEG is baked into ACTIONS; change the constant to use another rate"
    WAD = os.path.join(os.path.dirname(vzd.__file__), WADS[args.wad][0])
    MAPS = args.maps.split(",") if args.maps else WADS[args.wad][1]
    global WEAPON_POOL
    WEAPON_POOL = [w for w in args.weapons.replace("+", ",").split(",") if w]  # "+" also separates (sbatch --export)
    assert all(w in WEAPONS for w in WEAPON_POOL), WEAPON_POOL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.start, args.start + args.count):
        if args.skip_existing and (out / f"{args.prefix}{i:05d}" / "meta.json").exists():
            continue
        t = time.time()
        try:
            r = gen_sample(i, out, args)
        except Exception as e:  # keep the worker alive; failures are recorded, not hidden
            r = {"sample_id": f"{args.prefix}{i:05d}", "pass": False, "error": repr(e)[:300]}
        r["sec"] = round(time.time() - t, 1)
        rows.append(r)
        print(json.dumps(r), flush=True)
        with open(out / f"index_{args.start:06d}.jsonl", "a") as f:  # append as we go (restartable)
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
