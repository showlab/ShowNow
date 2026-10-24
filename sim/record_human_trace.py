"""Record human key timing in ViZDoom for latency replay (an alternative to synthetic Poisson presses).

Run on a machine with a display (laptop / desktop), not on the cluster:
    pip install vizdoom
    python record_human_trace.py --name alice --sessions 10

Each session is one 60 s game on a random Freedoom2 map. Play naturally (explore, fight, dodge). Keys:
    W / S / A / D       forward / back / strafe left / strafe right
    Left / Right arrow  turn left / right
    Ctrl or F           fire
    Space               jump
Only these eight actions are recorded. Mouse look is disabled (the model has eight discrete actions).

Output: one JSON per session, human_traces/<name>_<k>.json, with
    tics   per game tic (35 per second), the list of held action ids (1 W, 2 A, 3 S, 4 D, 5 TURN_L, 6 TURN_R, 7 ATTACK, 8 JUMP)
    events the single-key trace replayed by the latency harness: [[t_seconds, key or null], ...]
The model takes one action at a time. If several keys are held, the most recently pressed one is active; releasing
it falls back to the most recently pressed key that is still held, or to idle (null).
"""

import argparse
import json
import random
import time
from pathlib import Path

import vizdoom as vzd

B = vzd.Button
BUTTONS = [B.MOVE_FORWARD, B.MOVE_LEFT, B.MOVE_BACKWARD, B.MOVE_RIGHT, B.TURN_LEFT, B.TURN_RIGHT, B.ATTACK, B.JUMP]
IDS = [1, 2, 3, 4, 5, 6, 7, 8]  # action ids in BUTTONS order (W, A, S, D, TURN_L, TURN_R, ATTACK, JUMP)
TIC = 1.0 / 35
BINDS = ["unbindall", "bind w +forward", "bind s +back", "bind a +moveleft", "bind d +moveright",
         "bind leftarrow +left", "bind rightarrow +right", "bind ctrl +attack", "bind f +attack",
         "bind space +jump", "bind escape menu_main"]


def single_key_events(tics):
    """Per-tic held sets -> [(t, key or None)] change events, most recent press wins."""
    order, prev, ev, cur = [], set(), [], None
    for i, held in enumerate(tics):
        held = set(held)
        for k in held - prev:
            order.append(k)
        order = [k for k in order if k in held]
        new = order[-1] if order else None
        if new != cur:
            ev.append([round(i * TIC, 4), new])
            cur = new
        prev = held
    return ev


def play(seconds, seed):
    g = vzd.DoomGame()
    g.set_doom_game_path(str(Path(vzd.__file__).parent / "freedoom2.wad"))
    g.set_doom_map(f"MAP{random.Random(seed).randint(1, 32):02d}")
    g.set_screen_resolution(vzd.ScreenResolution.RES_1024X576)
    g.set_window_visible(True)
    g.set_mode(vzd.Mode.ASYNC_SPECTATOR)  # real time, the human drives
    g.set_available_buttons(BUTTONS)
    g.add_game_args("+freelook 0 +sv_jump 2 +fov 130")
    g.set_episode_timeout(int(seconds * 35))
    g.set_doom_skill(3)
    g.set_seed(seed)
    g.init()
    for b in BINDS:
        g.send_game_command(b)
    g.send_game_command("god")
    tics = []
    while not g.is_episode_finished():
        g.advance_action()
        a = g.get_last_action()
        tics.append([IDS[i] for i, v in enumerate(a) if v])
    g.close()
    return tics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="human_traces")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for k in range(a.sessions):
        f = out / f"{a.name}_{k:02d}.json"
        if f.exists():
            continue
        input(f"Session {k + 1}/{a.sessions} ({a.seconds:.0f} s). Press Enter to start...")
        seed = int(time.time()) % 100000
        tics = play(a.seconds, seed)
        ev = single_key_events(tics)
        f.write_text(json.dumps({"player": a.name, "session": k, "seed": seed, "tic_s": TIC, "seconds": a.seconds,
                                 "tics": tics, "events": ev}))
        n_press = sum(1 for _, x in ev if x is not None)
        print(f"saved {f}: {len(tics)} tics, {n_press} key changes")


if __name__ == "__main__":
    main()
