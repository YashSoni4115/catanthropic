"""Why does Φ-greedy lose to value_function?  (report only)

  agree:   4 value_function seats; at every non-forced decision record vf's action and the action
           phi_greedy would take from the same viewer information (its own belief determinization).
  ranking: on sampled states, the top-K actions by vf score; per action, win rate of the acting seat
           from N belief-determinized rollouts (all seats value_function, paired: the same determinization
           and RNG streams for every action). Spearman(Φ-greedy score, win rate), Spearman(vf score, win rate).

python -m catan_rl.policy.diagnose --games 200 --states 300 --n 64 --workers 8 --out runs/policy/diagnose
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from catanatron.players.value import ValueFunctionPlayer
from scipy.stats import spearmanr

from catan_rl.agents.phi_greedy import PhiGreedyAgent
from catan_rl.env import COLORS, CatanEnv
from catan_rl.env.actions import DOMESTIC, to_index
from catan_rl.env.determinize import DeterminizeError
from catan_rl.policy.teacher import group, vf_choice, vf_values
from catan_rl.winprob.phi import load_default

SEED_BASE = 100_000  # disjoint from eval (0..124) and winprob (0..2999)


def _game(seed, targets=None, n=64, k=4):
    """targets None: agreement records. Else {decision: id}: ranking rows at those decisions."""
    phi = load_default()
    pg = {c: PhiGreedyAgent(c, phi) for c in COLORS}
    recs, rows = [], []

    def agent(env, color):
        acts = [a for a in env.game.playable_actions if a.action_type not in DOMESTIC]
        vals = vf_values(env.game, color, acts)
        t = to_index(acts[vf_choice(vals)])
        d, legal = env.agent_decisions, env.legal_actions()
        if targets is None:
            p = pg[color].choose(env, legal)
            ty = acts[vf_choice(vals)].action_type
            recs.append({"seed": seed, "decision": d, "n_legal": len(legal),
                         "group": group(ty, env.game.state.is_initial_build_phase), "agree": int(p == t)})
        elif d in targets:
            rows.append(_rank(env, color, acts, vals, pg[color], legal, targets[d], n, k))
        return t

    random.seed(seed)
    env = CatanEnv(track_viewers=COLORS, record_events=False).reset(seed, {c: agent for c in COLORS})
    env.play()
    return recs if targets is None else [r for r in rows if r]


def _rank(env, color, acts, vals, pg, legal, sid, n, k):
    top = sorted(range(len(acts)), key=lambda i: -vals[i])[:k]
    pg.choose(env, legal)
    keys = [(acts[i].action_type, acts[i].value) for i in top]
    wins = np.zeros(len(top))
    for r in range(n):
        try:
            g = env.determinize(color, f"rank:{sid}:{r}")
        except DeterminizeError:
            return None
        by_key = {(a.action_type, a.value): a for a in g.playable_actions}
        for j, key in enumerate(keys):
            g2 = g.copy()
            g2.execute(by_key[key])
            g2.state.players = [ValueFunctionPlayer(c) for c in g2.state.colors]
            g2.play()
            wins[j] += g2.winning_color() == color
    return {"id": sid, "seed": int(sid.split(":")[0]), "vf": [vals[i] for i in top],
            "phi": [float(pg.last_scores[to_index(acts[i])]) for i in top], "win": (wins / n).tolist(),
            "group": group(acts[top[0]].action_type, env.game.state.is_initial_build_phase)}


def _job(job):
    return _game(*job)


def summarize_agree(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["group"]].append(r["agree"])
    return {"overall": float(np.mean([r["agree"] for r in recs])), "decisions": len(recs),
            "by_group": {g: {"agree": float(np.mean(v)), "n": len(v)} for g, v in sorted(by.items())}}


def summarize_rank(rows):
    from catan_rl.winprob.eval import cluster_mean_ci
    out = {"states": len(rows)}
    for key in ("phi", "vf"):
        rho, g = [], []
        for r in rows:
            if len(r["win"]) >= 3 and np.ptp(r["win"]) > 0 and np.ptp(r[key]) > 0:
                rho.append(spearmanr(r[key], r["win"])[0]), g.append(r["seed"])
        out[f"spearman_{key}_vs_rollout"] = {"mean_per_state": list(map(float, cluster_mean_ci(np.array(rho), np.array(g)))),
                                             "states_used": len(rho)}
        out[f"top1_is_best_{key}"] = float(np.mean([np.argmax(r[key]) in np.flatnonzero(np.array(r["win"]) == max(r["win"]))
                                                   for r in rows]))
    out["best_minus_vf_choice_winrate"] = float(np.mean([max(r["win"]) - r["win"][0] for r in rows]))
    out["phi_choice_minus_vf_choice_winrate"] = float(np.mean([r["win"][int(np.argmax(r["phi"]))] - r["win"][0] for r in rows]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--states", type=int, default=300)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/policy/diagnose")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    seeds = range(SEED_BASE, SEED_BASE + a.games)
    t = time.time()
    with Pool(a.workers) as p:
        recs = [r for g in p.imap(_job, [(s,) for s in seeds]) for r in g]
    agree = summarize_agree(recs)
    agree["seconds"] = time.time() - t
    json.dump(agree, open(f"{a.out}/agree.json", "w"), indent=1)
    print(json.dumps(agree, indent=1), flush=True)
    pool = [r for r in recs if r["n_legal"] >= 2]
    pick = random.Random("rank").sample(pool, a.states)
    targets = defaultdict(dict)
    for r in pick:
        targets[r["seed"]][r["decision"]] = f"{r['seed']}:{r['decision']}"
    t = time.time()
    with Pool(a.workers) as p:
        rows = [r for g in p.imap_unordered(_job, [(s, tg, a.n, a.k) for s, tg in sorted(targets.items())]) for r in g]
    json.dump(rows, open(f"{a.out}/rank_rows.json", "w"))
    rank = summarize_rank(rows)
    rank["seconds"], rank["n_rollouts"], rank["rollout_policy"] = time.time() - t, a.n, "value_function x4"
    json.dump(rank, open(f"{a.out}/rank.json", "w"), indent=1)
    print(json.dumps(rank, indent=1))


if __name__ == "__main__":
    main()
