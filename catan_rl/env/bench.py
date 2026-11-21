"""Throughput of CatanEnv with random agents.

    python -m catan_rl.env.bench --games 200 --workers 1 8
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

from catan_rl.env import CatanEnv, random_agent


def _game(args):
    seed, track = args
    env = CatanEnv(track_beliefs=track, record_events=track).reset(seed, [random_agent] * 4)
    env.play()
    return env.forced_decisions, env.agent_decisions


def bench(n, workers, track):
    jobs = [(s, track) for s in range(n)]
    t = time.perf_counter()
    if workers == 1:
        res = [_game(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_game, jobs, chunksize=8))
    dt = time.perf_counter() - t
    forced, agent = sum(r[0] for r in res), sum(r[1] for r in res)
    return n / dt, (forced + agent) / dt, agent / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    a = p.parse_args()
    print(f"{'trackers':9} {'workers':>7} {'games/s':>9} {'decisions/s':>12} {'agent dec/s':>12}")
    for track in (False, True):
        for w in a.workers:
            n = a.games * (4 if w > 1 else 1)
            g, d, ad = bench(n, w, track)
            print(f"{'on' if track else 'off':9} {w:7d} {g:9.1f} {d:12.0f} {ad:12.0f}")


if __name__ == "__main__":
    main()
