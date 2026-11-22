"""BC dataset: value_function teacher decisions.

Lineups (by seed): half 4x value_function; half 1-2 value_function vs weighted_random (random seats).
Seeds 200000.. (disjoint from eval 0..124, winprob 0..2999, diagnostics 100000..). Split by game 80/10/10 (sha256).
Per non-forced teacher decision: public_features(game, seat) [state], action_features for every legal action
(env.legal_actions order), teacher's chosen position, action group.

python -m catan_rl.policy.dataset --games 3000 --workers 8 --out runs/policy/bc_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_rl.env import COLORS, CatanEnv
from catan_rl.env.actions import DOMESTIC, to_index
from catan_rl.policy.action_features import action_features
from catan_rl.policy.teacher import GROUPS, group, vf_choice, vf_values
from catan_rl.winprob.features import public_features

SEED_BASE = 200_000


def lineup(seed: int):
    rng = random.Random(f"bc_lineup:{seed}")
    if seed % 2 == 0:
        return ("value_function",) * 4
    k = rng.choice((1, 2))
    names = ["value_function"] * k + ["weighted_random"] * (4 - k)
    rng.shuffle(names)
    return tuple(names)


def split_of(seed: int) -> str:
    h = int(hashlib.sha256(f"bc:{seed}".encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else "val" if h == 8 else "test"


def legal_actions(game):
    """Legal actions sorted like env.legal_actions() (by action index)."""
    acts = [a for a in game.playable_actions if a.action_type not in DOMESTIC]
    return sorted(acts, key=to_index)


def play_game(seed: int) -> dict:
    names = lineup(seed)
    xs, acts, n_act, choice, grp, aidx = [], [], [], [], [], []

    def teacher(env, color):
        game = env.game
        raw = [a for a in game.playable_actions if a.action_type not in DOMESTIC]
        t = raw[vf_choice(vf_values(game, color, raw))]
        legal = legal_actions(game)
        xs.append(public_features(game, color))
        acts.append(action_features(game, color, legal))
        n_act.append(len(legal))
        choice.append(legal.index(t))
        aidx.append([to_index(a) for a in legal])
        grp.append(GROUPS.index(group(t.action_type, game.state.is_initial_build_phase)))
        return to_index(t)

    seats = {c: teacher if n == "value_function" else WeightedRandomPlayer(c) for c, n in zip(COLORS, names)}
    random.seed(seed)
    env = CatanEnv(record_events=False).reset(seed, seats)
    env.play()
    return {"seed": seed, "x": np.array(xs, np.float32), "a": np.concatenate(acts).astype(np.float16),
            "aidx": np.concatenate(aidx).astype(np.int16), "n_act": np.array(n_act, np.int16),
            "choice": np.array(choice, np.int16), "group": np.array(grp, np.int8), "lineup": ",".join(names)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/policy/bc_v1")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    with Pool(a.workers) as p:
        games = list(p.imap(play_game, range(SEED_BASE, SEED_BASE + a.games), chunksize=4))
    nd = [len(g["choice"]) for g in games]
    out = {k: np.concatenate([g[k] for g in games]) for k in ("x", "a", "aidx", "n_act", "choice", "group")}
    out["seed"] = np.repeat([g["seed"] for g in games], nd)
    out["split"] = np.array([split_of(s) for s in out["seed"]])
    np.savez(f"{a.out}/dataset.npz", **out)
    meta = {"games": a.games, "seed_base": SEED_BASE, "decisions": int(sum(nd)), "actions": int(len(out["a"])),
            "seconds": time.time() - t, "groups": list(GROUPS),
            "lineups": {k: sum(g["lineup"] == k for g in games) for k in sorted({g["lineup"] for g in games})}}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "lineups"}, indent=1))


if __name__ == "__main__":
    main()
