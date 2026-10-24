"""The E1 protocol without any model code: Poisson key traces, the trace clock, the display clock and per-press scoring.
Shared by latency_harness (ShowNow / baselines on our model) and ext_latency (external world models in their own envs).
FPS / FRAME_S are module globals (set them before creating a Display to change the display clock)."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import zlib

FPS = 16
FRAME_S = 1.0 / FPS
# key ids of the traces: W A S D (1-4); SHOWNOW_N_ACTIONS=9 -> 8 keys (+ TURN_L TURN_R ATTACK JUMP)
# keys pressed by the traces: every action of the model, or the first SHOWNOW_TRACE_KEYS of them (4 = W A S D, the
# key set of the external-model E1 runs, also for the 8-action model)
KEYS = tuple(range(1, int(os.environ.get("SHOWNOW_TRACE_KEYS") or int(os.environ.get("SHOWNOW_N_ACTIONS", "5")) - 1) + 1))


# ----------------------------------------------------------------------------- traces
def make_trace_keyset(kind: str, seed: int, seconds: float, keyset, p_release: float = 0.2, min_gap: float = 0.15) -> list:
    """Action-count scaling traces: event times AND the release / press decisions come from one RNG stream that does
    not depend on the key set, the key identities from a second stream, so every key set with >= 2 keys gets the
    identical press / release timeline. A press while a key is held switches to a DIFFERENT key of the set (least used,
    ties random); a press from idle takes the least-used key. With a single key a press while held is impossible, so
    that event releases the key instead (K = 1 measures idle -> action only)."""
    assert kind.startswith("poisson"), kind
    rate = float(kind[len("poisson"):])
    trng = random.Random(1000 * seed + zlib.crc32(kind.encode()) % 997)
    krng = random.Random(7919 * seed + zlib.crc32(kind.encode()) % 991 + 17)
    keyset = list(keyset)
    used = {k: 0 for k in keyset}
    ev, t, cur = [], 0.5, None
    while t < seconds:
        rel = trng.random() < p_release
        if cur is not None and (rel or len(keyset) == 1):
            cur = None
        else:
            cand = [k for k in keyset if k != cur]
            m = min(used[k] for k in cand)
            cur = krng.choice([c for c in cand if used[c] == m])
            used[cur] += 1
        ev.append((round(t, 4), cur))
        t += min_gap + trng.expovariate(1.0 / (1.0 / rate - min_gap))
    return ev


def make_trace_diag(kind: str, seed: int, seconds: float, keyset) -> list:
    """Diagnostic traces (no releases; every event switches to a DIFFERENT key of the set; key identity balanced):
      iso<h>    isolated switches: a key is held h s, then the next switch comes at h s + U(0, 0.25 s) (one latent
                period), so switch times fall uniformly on the decision / commit grid (the phase is measured post hoc
                from the run's timeline) and no switch is overridden
      hold<h>   long holds: as iso<h> with a large h (predictable, few switches)
      burst<n>  rapid bursts: n switches 150-300 ms apart (human refractory floor 150 ms), then a 2-3 s hold
    The press at 0.5 s is from idle; all later events are switches. Times use one RNG stream, keys another, so all key
    sets (K >= 2) get the identical timeline."""
    trng = random.Random(1000 * seed + zlib.crc32(kind.encode()) % 997)
    krng = random.Random(7919 * seed + zlib.crc32(kind.encode()) % 991 + 17)
    keyset = list(keyset)
    assert len(keyset) >= 2, "diagnostic traces need at least two keys"
    used = {k: 0 for k in keyset}

    def nxt(cur):
        cand = [k for k in keyset if k != cur]
        m = min(used[k] for k in cand)
        k = krng.choice([c for c in cand if used[c] == m])
        used[k] += 1
        return k

    ev, t, cur = [], 0.5, None
    if kind.startswith(("iso", "hold")):
        h = float(kind[3:] if kind.startswith("iso") else kind[4:])
        while t < seconds:
            cur = nxt(cur)
            ev.append((round(t, 4), cur))
            t += h + trng.uniform(0.0, 0.25)
    elif kind.startswith("burst"):
        n = int(kind[len("burst"):])
        cur = nxt(cur)
        ev.append((round(t, 4), cur))
        t += 2.0
        while t < seconds:
            for i in range(n):
                if t >= seconds:
                    break
                cur = nxt(cur)
                ev.append((round(t, 4), cur))
                t += trng.uniform(0.15, 0.30) if i < n - 1 else trng.uniform(2.0, 3.0)
    else:
        raise ValueError(kind)
    return ev


def make_trace(kind: str, seed: int, seconds: float, p_release: float = 0.2, min_gap: float = 0.15) -> list:
    """[(t_s, key or None)].
    "poisson<rate>": Poisson-like arrivals (user presses are unpredictable, so they fall uniformly on the system's
      decision cycle) with a human refractory period: gap = min_gap + Exp(mean 1/rate - min_gap), i.e. the mean rate is
      kept and no two presses are closer than min_gap (150 ms). An event while a key is held releases it with
      p_release (-> idle), otherwise switches to another key; an event while idle presses a key (press from idle).
      Keys 1..4 (W A S D) are balanced (least-used next). The first press is at 0.5 s.
    "file:<path>": a recorded human session (jsonl of {"t": s, "key": 1-4 or null}, from play_server --record-dir)."""
    if kind.startswith("file:"):
        ev = [(float(r["t"]), r["key"]) for r in map(json.loads, open(kind[5:]))]
        return [(round(t, 4), k) for t, k in ev if t < seconds]
    assert kind.startswith("poisson"), kind
    rate = float(kind[len("poisson"):])
    assert 1.0 / rate > min_gap, (rate, min_gap)
    rng = random.Random(1000 * seed + zlib.crc32(kind.encode()) % 997)  # reproducible across processes
    used = {k: 0 for k in KEYS}

    def next_key(prev):
        cand = [k for k in KEYS if k != prev]
        m = min(used[k] for k in cand)
        k = rng.choice([c for c in cand if used[c] == m])
        used[k] += 1
        return k

    ev, t, cur = [], 0.5, None
    while t < seconds:
        if cur is not None and rng.random() < p_release:
            cur = None
        else:
            cur = next_key(cur)
        ev.append((round(t, 4), cur))
        t += min_gap + rng.expovariate(1.0 / (1.0 / rate - min_gap))
    return ev


class TraceKeys:
    def __init__(self, trace, t0):
        self.trace, self.t0 = trace, t0

    def current(self, now=None):
        now = (time.time() if now is None else now) - self.t0
        k = None
        for t, key in self.trace:
            if t > now:
                break
            k = key
        return k

    def done(self, seconds):
        return time.time() - self.t0 >= seconds


class Display:
    """FPS playback clock (16 by default, --fps) over a queue of frame tags (step, key); records play times and stalls."""

    def __init__(self, keys: TraceKeys, seconds: float):
        self.q, self.lock, self.keys, self.seconds = [], threading.Lock(), keys, seconds
        self.played, self.stall_ticks, self.held_ticks, self.stop = [], 0, 0, False
        self.stall_times = []
        self.dropped = 0  # queued frames discarded by replace() (preemption)
        self.last_tick = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def push(self, tags):
        with self.lock:
            self.q.extend(tags)

    def replace(self, tags, max_drop: int) -> int:
        """preemption: discard up to max_drop of the still-queued frames (the newest ones, so playback stays continuous
        until the cut) and queue tags after the rest. Returns the number of discarded frames."""
        with self.lock:
            d = min(len(self.q), max_drop)
            if d:
                del self.q[len(self.q) - d:]
            self.q.extend(tags)
            self.dropped += d
        return d

    def pending(self):
        return len(self.q)

    def _run(self):
        t = time.time()
        while not self.stop:
            now = time.time()
            with self.lock:
                self.last_tick = now
                if self.q:
                    tag = self.q.pop(0)
                    self.played.append((now, tag))
                    if self.keys.current(now) is not None and now - self.keys.t0 < self.seconds:
                        self.held_ticks += 1
                elif self.keys.current(now) is not None and self.played and now - self.keys.t0 < self.seconds:
                    self.stall_ticks += 1
                    self.stall_times.append(round(now - self.keys.t0, 3))
                    self.held_ticks += 1
            t += FRAME_S
            time.sleep(max(0.0, t - time.time()))
            if time.time() - t > 0.5:
                t = time.time()


def score(trace, steps, seconds):
    """per press: latency without / with VAE, overridden / unserved."""
    presses = []
    held = None
    for i, (t, k) in enumerate(trace):
        if k is not None and k != held:
            t_next = next((t2 for t2, k2 in trace[i + 1:] if k2 != k), seconds)
            presses.append((t, k, t_next))
        held = k
    out = []
    prev_of = {}
    held = None
    for t, k in trace:
        if k is not None and k != held:
            prev_of[t] = held
        held = k
    for t, k, t_next in presses:
        s = next((s for s in steps if s["key"] == k and s["t_decision"] >= t - 1e-4), None)
        if s is None or s["t_decision"] > t_next:
            out.append({"t": t, "key": k, "status": "overridden" if t_next < seconds else "unserved",
                        "from_idle": prev_of.get(t) is None})
            continue
        out.append({"t": t, "key": k, "status": "served", "lat_no_vae": s["t_commit"] - t,
                    "lat_vae": None if s["t_play"] is None else s["t_play"] - t,
                    "wait_decision": s["t_decision"] - t, "after_decision": s["t_commit"] - s["t_decision"],
                    # breakdown (seconds): candidate ready after the press (0 if before), decision, commit, decode, queue
                    "ready_after": max(0.0, s.get("t_ready", t) - t), "t_decision_abs": s["t_decision"],
                    "decode_s": s["t_decoded"] - s["t_commit"],
                    "queue_s": None if s["t_play"] is None else s["t_play"] - s["t_decoded"], "hit": s.get("hit"),
                    "from_idle": prev_of.get(t) is None})
    return out
