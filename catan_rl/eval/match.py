"""Seat-rotated, seed-paired matches between named agents.

For each seed, 4 games are played; in rotation r the agent in lineup slot s sits at turn-order
position (s + r) % 4, so every slot occupies every position exactly once per seed. The same
seeds across lineups give identical board, turn order, dev deck and dice (fork RNG streams).
The 4 games of a seed are NOT independent, so all CIs resample seeds (clusters).
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence

import catanatron
from catanatron import Game, RandomPlayer

from catan_rl.env import COLORS, CatanEnv
from catan_rl.eval import agents as A
from catan_rl.eval.stats import bootstrap_mean_ci, cluster_bootstrap, wilson

FORK_BASE = "ecf9311"  # upstream catanatron commit the fork branches from


def turn_order(seed: int) -> List:
    """Colors in turn order for ``seed`` when players are passed in COLORS order (seats stream)."""
    return list(Game([RandomPlayer(c) for c in COLORS], seed=seed).state.colors)


def play_one(job) -> dict:
    lineup, seed, rot = job
    order = turn_order(seed)
    slot_of = {order[(s + rot) % 4]: s for s in range(4)}  # color -> lineup slot
    random.seed(seed * 4 + rot)  # in case a bot draws from the global `random` module
    seat_agents = {c: A.get(lineup[slot_of[c]]).factory(c) for c in COLORS}
    t = time.perf_counter()
    env = CatanEnv(record_events=False).reset(seed, seat_agents)
    env.play()
    st = env.game.state
    assert list(st.colors) == order, "turn order drifted from seats stream"
    w = env.winner
    return {
        "seed": seed, "rotation": rot,
        "agents": {c.value: lineup[slot_of[c]] for c in order},
        "slots": {c.value: slot_of[c] for c in order},
        "turn_order": [c.value for c in order],
        "winner": w.value if w else None,
        "winner_slot": slot_of[w] if w else None,
        "vp": {c.value: st.player_state[f"P{st.color_to_index[c]}_ACTUAL_VICTORY_POINTS"] for c in order},
        "num_actions": len(st.action_records),
        "num_turns": st.num_turns,
        "truncated": w is None,
        "forced_decisions": env.forced_decisions,
        "agent_decisions": env.agent_decisions,
        "wall_time": time.perf_counter() - t,
    }


def run_match(agent_names: Sequence[str], n_seeds: int, seed_base: int = 0, workers: int = 1) -> List[dict]:
    lineup = tuple(agent_names)
    assert len(lineup) == 4
    for n in lineup:
        A.get(n)
    jobs = [(lineup, seed_base + s, r) for s in range(n_seeds) for r in range(4)]
    if workers == 1:
        return [play_one(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(play_one, jobs, chunksize=2))


# ---------------------------------------------------------------------------- statistics
def _color_of(r, slot):
    return next(c for c, s in r["slots"].items() if s == slot)


def wins_by_seed(results: List[dict], slot: int) -> Dict[int, List[int]]:
    """seed -> [wins of ``slot``, games]."""
    acc: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        acc[r["seed"]][0] += r["winner_slot"] == slot
        acc[r["seed"]][1] += 1
    return dict(sorted(acc.items()))


def clustered_win_rate(results: List[dict], slots: Sequence[int]) -> dict:
    """Pooled win rate of ``slots`` (e.g. all slots of one agent), clustered by seed, Wilson as reference."""
    per = [wins_by_seed(results, s) for s in slots]
    seeds = sorted(per[0])
    k = [sum(p[s][0] for p in per) for s in seeds]
    g = [sum(p[s][1] for p in per) for s in seeds]
    out = cluster_bootstrap(k, g)
    out["wilson_ci95_reference"] = list(wilson(sum(k), sum(g)))
    return out


def metrics(results: List[dict], lineup: Sequence[str]) -> dict:
    n = len(results)
    slots = []
    for s, name in enumerate(lineup):
        vps = [r["vp"][_color_of(r, s)] for r in results]
        by_pos = []
        for p in range(4):
            games = [r for r in results if r["turn_order"].index(_color_of(r, s)) == p]
            k = sum(r["winner_slot"] == s for r in games)
            by_pos.append({"position": p, "games": len(games), "wins": k, "win_rate": k / len(games) if games else None})
        slots.append({"slot": s, "agent": A.label(name), "win_rate": clustered_win_rate(results, [s]),
                      "mean_final_vp": sum(vps) / n, "win_rate_by_position": by_pos})
    names = list(dict.fromkeys(lineup))
    forced = sum(r["forced_decisions"] for r in results)
    agent = sum(r["agent_decisions"] for r in results)
    return {
        "games": n,
        "per_slot": slots,
        "per_agent": {A.label(x): clustered_win_rate(results, [s for s, y in enumerate(lineup) if y == x]) for x in names},
        "no_winner_games": sum(r["truncated"] for r in results),
        "mean_game_length_turns": sum(r["num_turns"] for r in results) / n,
        "mean_game_length_actions": sum(r["num_actions"] for r in results) / n,
        "decisions": {"forced": forced, "agent": agent},
    }


def paired_comparison(results_a: List[dict], results_b: List[dict], slot: int = 0,
                      n_boot: int = 10000, seed: int = 0) -> dict:
    """Win-rate difference (A - B) of ``slot`` on shared seeds; CI resamples seeds (clusters) jointly."""
    a, b = wins_by_seed(results_a, slot), wins_by_seed(results_b, slot)
    shared = sorted(set(a) & set(b))
    diffs = [a[s][0] / a[s][1] - b[s][0] / b[s][1] for s in shared]
    mean, lo, hi = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
    return {"slot": slot, "n_seeds": len(shared), "mean_diff": mean, "ci95": [lo, hi]}


# ---------------------------------------------------------------------------- provenance / io
def _git(path: str, *args: str):
    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def provenance(lineup: Sequence[str], seeds: Sequence[int]) -> dict:
    fork = os.path.dirname(os.path.dirname(catanatron.__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "python": sys.version,
        "platform": platform.platform(),
        "fork_path": fork,
        "fork_commit": _git(fork, "rev-parse", "HEAD"),
        "fork_dirty": bool(_git(fork, "status", "--porcelain", "--untracked-files=no")),
        "fork_base_commit": _git(fork, "rev-parse", FORK_BASE) or FORK_BASE,
        "repo_commit": _git(repo, "rev-parse", "HEAD"),
        "repo_dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        "lineup": list(lineup),
        "peeking": [A.get(x).peeking for x in lineup],
        "seeds": list(seeds),
    }


def save(out_dir: str, results: List[dict], m: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)


def load(out_dir: str) -> List[dict]:
    with open(os.path.join(out_dir, "results.jsonl")) as f:
        return [json.loads(line) for line in f]
