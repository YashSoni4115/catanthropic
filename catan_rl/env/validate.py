"""Ground-truth validation of per-viewer belief trackers (debug channel only).

    python -m catan_rl.env.validate --random-games 500 --ab-games 100 --workers 8
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from catanatron.players.minimax import AlphaBetaPlayer

from catan_rl.env import COLORS, CatanEnv, random_agent
from catan_rl.env import event_adapter as ea

PHASES = ("early", "mid", "late")


def tracker_hand(h):  # catanatron (WOOD,BRICK,SHEEP,WHEAT,ORE) -> tracker (BRICK,WOOL,ORE,GRAIN,LUMBER)
    return (h[1], h[2], h[4], h[3], h[0])


class ValidatingEnv(CatanEnv):
    """Checks after every action that each viewer's belief gives every opponent's true hand p>0,
    and records log p(true), |E[count]-true|, and tracker update time per event."""

    def reset(self, seed, seat_agents):
        self.track_viewers = frozenset(c.value for c in COLORS)
        self.rows: List[tuple] = []  # (action_idx, logp, mae, n_events, seconds)
        self._t, self._n = 0.0, 0
        return super().reset(seed, seat_agents)

    def _apply(self, action):
        trackers = self.trackers
        for t in trackers.values():
            if not hasattr(t, "_orig_apply"):
                t._orig_apply = t.apply
                t.apply = self._timed(t._orig_apply)
        self._t, self._n = 0.0, 0
        super()._apply(action)
        truth = {c: tracker_hand(h) for c, h in ea.hands(self.game.state).items()}
        logps, maes = [], []
        for v, t in trackers.items():
            for c, h in truth.items():
                if c == v:
                    continue
                pmf = t.players[c].pmf
                p = pmf.get(h, 0.0)
                if p <= 0.0:
                    raise AssertionError(f"seed={self.seed} viewer={v} opp={c} true={h} has p=0")
                logps.append(math.log(p))
                exp = [sum(q * hh[i] for hh, q in pmf.items()) for i in range(5)]
                maes.append(sum(abs(e - x) for e, x in zip(exp, h)) / 5)
        self.rows.append((len(self.rows), sum(logps) / len(logps), sum(maes) / len(maes), self._n, self._t))

    def _timed(self, fn):
        def wrapped(e):
            s = time.perf_counter()
            fn(e)
            self._t += time.perf_counter() - s
            self._n += 1
        return wrapped


def run_game(args):
    seed, ab = args
    agents = [random_agent] * 4
    if ab:
        agents[1] = AlphaBetaPlayer(COLORS[1])
    env = ValidatingEnv().reset(seed, agents)
    env.play()
    n = len(env.rows)
    out = {ph: [0.0, 0.0, 0, 0, 0.0] for ph in PHASES}  # sum logp, sum mae, actions, events, secs
    for i, lp, mae, ne, sec in env.rows:
        o = out[PHASES[min(2, 3 * i // n)]]
        o[0] += lp; o[1] += mae; o[2] += 1; o[3] += ne; o[4] += sec
    return out, env.forced_decisions, env.agent_decisions


def summarize(results) -> Dict[str, Dict[str, float]]:
    agg = {ph: [0.0, 0.0, 0, 0, 0.0] for ph in PHASES}
    for out, *_ in results:
        for ph in PHASES:
            agg[ph] = [a + b for a, b in zip(agg[ph], out[ph])]
    return {ph: {"mean_logp_true": a[0] / a[2], "mean_abs_err": a[1] / a[2],
                 "us_per_event": 1e6 * a[4] / max(a[3], 1), "actions": a[2], "events": a[3]}
            for ph, a in agg.items()}


def run(n, ab, workers, start=0):
    jobs = [(start + s, ab) for s in range(n)]
    t = time.perf_counter()
    if workers == 1:
        res = [run_game(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(run_game, jobs, chunksize=4))
    return res, time.perf_counter() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random-games", type=int, default=500)
    p.add_argument("--ab-games", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    for name, n, ab in (("random", a.random_games, False), ("alphabeta_seat1", a.ab_games, True)):
        res, dt = run(n, ab, a.workers)
        f = sum(r[1] for r in res); d = sum(r[2] for r in res)
        print(f"\n== {name}: {n} games, 0 contradictions, true hand p>0 after every action ({dt:.1f}s)")
        print(f"   decisions: forced={f} agent={d} forced_frac={f / (f + d):.3f}")
        print(f"   {'phase':6} {'mean log p(true)':>17} {'MAE/resource':>13} {'us/event':>9} {'events':>9}")
        for ph, m in summarize(res).items():
            print(f"   {ph:6} {m['mean_logp_true']:17.4f} {m['mean_abs_err']:13.4f} {m['us_per_event']:9.1f} {m['events']:9d}")


if __name__ == "__main__":
    main()
