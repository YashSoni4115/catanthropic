"""Checkpoint diagnostics (report only).

  profile: per learner seat, N games vs 3 value_function (argmax): final VP, length, turn-limit games,
           VP sources in wins, action-type counts per game.
  kl:      on the checkpoint's own decisions vs 3 value_function: how often its argmax differs from bc's,
           overall and by action type, and the mean BC log-prob of the chosen action.
  windows: learner win rate per league slice in windows of W updates from a run's log.jsonl.

python -m catan_rl.rl.profile profile --ckpts bc,runs/rl/B2_shape/final.pt --games 100 --seed-base 7000
python -m catan_rl.rl.profile kl --ckpt runs/rl/B2_shape/final.pt --games 50 --seed-base 7100
python -m catan_rl.rl.profile windows --log runs/rl/B2_shape/log.jsonl --window 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from multiprocessing import get_context

import numpy as np
from catanatron.models.enums import CITY, SETTLEMENT, ActionType
from catanatron.players.value import ValueFunctionPlayer

from catan_rl.env import COLORS, CatanEnv
from catan_rl.eval.match import turn_order
from catan_rl.policy.teacher import GROUPS, group

ACTION_KINDS = {"settlements": {ActionType.BUILD_SETTLEMENT}, "cities": {ActionType.BUILD_CITY},
                "roads": {ActionType.BUILD_ROAD}, "dev_buys": {ActionType.BUY_DEVELOPMENT_CARD},
                "dev_plays": {ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
                              ActionType.PLAY_ROAD_BUILDING},
                "trades": {ActionType.MARITIME_TRADE}, "robber_moves": {ActionType.MOVE_ROBBER}}
SOURCES = ("settlements", "cities", "longest_road", "largest_army", "vp_cards")


def _agent(ckpt, color):
    import torch
    torch.set_num_threads(1)
    if ckpt == "bc":
        from catan_rl.policy.agent import PolicyAgent
        return PolicyAgent(color)
    from catan_rl.rl.agent import PPOAgent
    return PPOAgent(color, ckpt)


def _setup(seed):
    order = turn_order(seed)
    return order[seed % 4]  # learner's turn position rotates with the seed


def vp_sources(st, c) -> dict:
    i = st.color_to_index[c]
    b = st.buildings_by_color[c]
    ps = st.player_state
    return {"settlements": len(b[SETTLEMENT]), "cities": 2 * len(b[CITY]), "longest_road": 2 * bool(ps[f"P{i}_HAS_ROAD"]),
            "largest_army": 2 * bool(ps[f"P{i}_HAS_ARMY"]),
            "vp_cards": ps[f"P{i}_ACTUAL_VICTORY_POINTS"] - ps[f"P{i}_VICTORY_POINTS"]}


def _profile_game(job):
    ckpt, seed = job
    me = _setup(seed)
    agents = {c: _agent(ckpt, c) if c == me else ValueFunctionPlayer(c) for c in COLORS}
    random.seed(seed)
    env = CatanEnv(record_events=False).reset(seed, agents)
    env.play()
    st = env.game.state
    acts = Counter()
    for r in st.action_records:
        if r.action.color == me:
            for k, ts in ACTION_KINDS.items():
                acts[k] += r.action.action_type in ts
    src = vp_sources(st, me)
    return {"seed": seed, "win": env.winner == me, "truncated": env.winner is None, "turns": st.num_turns,
            "vp": st.player_state[f"P{st.color_to_index[me]}_ACTUAL_VICTORY_POINTS"], "src": src, "acts": dict(acts)}


def profile(ckpts, games, seed_base, workers=8) -> dict:
    out = {}
    with get_context("spawn").Pool(workers) as p:
        for ck in ckpts:
            rows = p.map(_profile_game, [(ck, seed_base + s) for s in range(games)])
            wins = [r for r in rows if r["win"]]
            tot = {k: sum(r["src"][k] for r in wins) for k in SOURCES}
            all_vp = sum(tot.values()) or 1
            out[ck] = {"games": len(rows), "win_rate": len(wins) / len(rows),
                       "mean_final_vp": float(np.mean([r["vp"] for r in rows])),
                       "mean_turns": float(np.mean([r["turns"] for r in rows])),
                       "turn_limit_frac": float(np.mean([r["truncated"] for r in rows])),
                       "win_vp_share": {k: tot[k] / all_vp for k in SOURCES},
                       "win_vp_mean": {k: tot[k] / max(len(wins), 1) for k in SOURCES},
                       "all_vp_mean": {k: float(np.mean([r["src"][k] for r in rows])) for k in SOURCES},
                       "actions_per_game": {k: float(np.mean([r["acts"].get(k, 0) for r in rows])) for k in ACTION_KINDS}}
    return out


def _kl_game(job):
    import torch
    from catan_rl.env.actions import to_index
    from catan_rl.policy.action_features import action_features
    from catan_rl.policy.dataset import legal_actions
    from catan_rl.rl.agent import PPOAgent
    from catan_rl.rl.net import ActorCritic, prep
    from catan_rl.winprob.features import public_features
    ckpt, seed = job
    torch.set_num_threads(1)
    me = _setup(seed)
    pol = PPOAgent(me, ckpt)
    bc = ActorCritic.from_bc()
    rows = []

    def seat(env, color):
        g = env.game
        acts = legal_actions(g)
        X, A, M = prep(pol.net.norm, [public_features(g, color)], [action_features(g, color, acts)])
        with torch.no_grad():
            z = pol.net(X, A, M)[0][0]
            zb = bc(X, A, M)[0][0]
        i, j = int(z.argmax()), int(zb.argmax())
        rows.append({"differs": i != j, "bc_logp": float(torch.log_softmax(zb, -1)[i]),
                     "group": group(acts[i].action_type, g.state.is_initial_build_phase)})
        return to_index(acts[i])  # argmax, identical to PPOAgent

    agents = {c: seat if c == me else ValueFunctionPlayer(c) for c in COLORS}
    random.seed(seed)
    CatanEnv(record_events=False).reset(seed, agents).play()
    return rows


def kl(ckpt, games, seed_base, workers=8) -> dict:
    with get_context("spawn").Pool(workers) as p:
        rows = [r for g in p.map(_kl_game, [(ckpt, seed_base + s) for s in range(games)]) for r in g]
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    summ = lambda rs: {"n": len(rs), "differs": float(np.mean([r["differs"] for r in rs])),
                       "mean_bc_logp": float(np.mean([r["bc_logp"] for r in rs])),
                       "mean_bc_prob": float(np.mean([np.exp(r["bc_logp"]) for r in rs]))}
    return {"games": games, "decisions": len(rows), "overall": summ(rows),
            "by_group": {g: summ(by[g]) for g in GROUPS if by[g]}}


def windows(log, w=50) -> list:
    recs = [json.loads(l) for l in open(log)]
    out = []
    for s in range(0, len(recs), w):
        ch = recs[s:s + w]
        row = {"updates": f"{ch[0]['update']}-{ch[-1]['update']}"}
        for cat in ("self", "bc", "vf", "past"):
            n = sum(r["learner_seats"][cat] for r in ch)
            k = sum((r["win_rate"][cat] or 0) * r["learner_seats"][cat] for r in ch)
            row[cat] = (k / n if n else None, n)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "kl", "windows"])
    ap.add_argument("--ckpts", default="bc")
    ap.add_argument("--ckpt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "profile":
        r = profile(a.ckpts.split(","), a.games, a.seed_base, a.workers)
    elif a.cmd == "kl":
        r = kl(a.ckpt, a.games, a.seed_base, a.workers)
    else:
        r = windows(a.log, a.window)
        for row in r:
            print(row["updates"].rjust(9), "  ".join(f"{c}={v[0]:.3f}(n={v[1]})" if v[0] is not None else f"{c}=-"
                                                     for c, v in row.items() if c != "updates"))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    if a.cmd != "windows":
        print(json.dumps(r, indent=1))
    return r


if __name__ == "__main__":
    main()
