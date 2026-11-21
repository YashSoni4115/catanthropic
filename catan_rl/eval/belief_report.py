"""Belief/determinizer report on seeded games (ground truth used for scoring only).

    python -m catan_rl.eval.belief_report --games 200 --agent value_function --every 5 --workers 8 --out DIR

Per measured decision and viewer (all 4 viewers tracked):
  * log P(true opponents' hands) under independent trackers vs the conservation-conditioned joint
    (log P_joint = log P_indep - log Z, Z = independent probability of the per-resource sums),
  * |E[hidden VP cards] - true| per opponent,
  * one determinization: time and rejection tries (acceptance = samples / tries).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from catan_rl.env import COLORS, CatanEnv
from catan_rl.env.actions import DOMESTIC, to_index
from catan_rl.env.determinize import DeterminizeError, conservation_sums, joint_support, own_hand, phase
from catan_rl.eval import agents as A
from catan_rl.tests.belief_helpers import true_dev, true_hand

PHASES = ("setup", "early", "mid", "late")


def run_game(job):
    seed, agent, every = job
    rows, n = [], [0]
    bots = {c: A.get(agent).factory(c) for c in COLORS}

    def measure(env):
        st = env.game.state
        ph = phase(st)
        for v in (c.value for c in COLORS):
            trk, db = env.belief(v), env.dev_belief(v)
            opp = [c for c in st.colors if c.value != v]
            pmfs = {c.value: trk.players[c.value].pmf for c in opp}
            lp = sum(math.log(pmfs[c.value][true_hand(st, c)]) for c in opp)
            S = conservation_sums(st.resource_freqdeck, own_hand(trk, v))
            try:
                _, Z = joint_support(pmfs, S)
                lj = lp - math.log(Z)
            except DeterminizeError:
                lj = None
            vp_err = [abs(db.expected_count(c.value, "VICTORY_POINT") - true_dev(st, c)["VICTORY_POINT"]) for c in opp]
            t = time.perf_counter()
            g = env.determinize(v, (seed, n[0], v))
            rows.append({"phase": ph, "logp_indep": lp, "logp_joint": lj, "vp_abs_err": vp_err,
                         "det_s": time.perf_counter() - t, "tries": g.determinization["tries"],
                         "fallback": g.determinization["method"] != "rejection"})

    def agent_fn(env, color):
        n[0] += 1
        if n[0] % every == 0:
            measure(env)
        acts = [a for a in env.game.playable_actions if a.action_type not in DOMESTIC]
        return to_index(bots[color].decide(env.game, acts))

    env = CatanEnv(track_viewers=[c.value for c in COLORS], record_events=False)
    env.reset(seed, {c: agent_fn for c in COLORS}).play()
    return rows


def summarize(rows):
    out = {}
    for ph in PHASES + ("all",):
        r = [x for x in rows if ph == "all" or x["phase"] == ph]
        if not r:
            continue
        j = [x for x in r if x["logp_joint"] is not None]
        errs = [e for x in r for e in x["vp_abs_err"]]
        out[ph] = {
            "n": len(r),
            "mean_logp_indep": sum(x["logp_indep"] for x in j) / len(j),
            "mean_logp_joint": sum(x["logp_joint"] for x in j) / len(j),
            "joint_skipped": len(r) - len(j),
            "mean_hidden_vp_abs_err": sum(errs) / len(errs),
            "acceptance_rate": len(r) / sum(x["tries"] for x in r),
            "fallbacks": sum(x["fallback"] for x in r),
            "determinizations_per_s": len(r) / sum(x["det_s"] for x in r),
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--agent", default="value_function")
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="runs/belief_report")
    a = p.parse_args(argv)
    t = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for g in ex.map(run_game, [(s, a.agent, a.every) for s in range(a.games)]) for r in g]
    s = summarize(rows)
    s["meta"] = {"games": a.games, "agent": a.agent, "every": a.every, "elapsed_s": time.perf_counter() - t}
    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"{'phase':6} {'n':>6} {'logP indep':>11} {'logP joint':>11} {'VP |err|':>9} {'accept':>7} {'fallbk':>6} {'det/s':>7}")
    for ph in PHASES + ("all",):
        if ph in s:
            x = s[ph]
            print(f"{ph:6} {x['n']:6d} {x['mean_logp_indep']:11.3f} {x['mean_logp_joint']:11.3f} "
                  f"{x['mean_hidden_vp_abs_err']:9.3f} {x['acceptance_rate']:7.3f} {x['fallbacks']:6d} {x['determinizations_per_s']:7.0f}")
    print(s["meta"])


if __name__ == "__main__":
    main()
