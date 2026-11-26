"""Rollout workers: play one league game, return per-decision trajectories of the learner seats only.

League (per game, from an RNG keyed by (run seed, game seed)):
  50% self-play: 4 learner seats (parameter sharing, all learn)
  25% learner x k + frozen bc_policy x (4-k), k in {1,2,3}
  15% learner x k + value_function x (4-k)
  10% learner x k + one past checkpoint x (4-k)
Reward per learner seat: outcome (win 1, else 0; truncated games 0) at its last decision; with shaping,
r_t += coef * (gamma * Phi(s_{t+1}) - Phi(s_t)), Phi = PhiModel win prob of that seat at its own consecutive
decision points and Phi(terminal) = outcome, so the shaping terms telescope to coef * (outcome - Phi(s_0)).
"""
from __future__ import annotations

import random
import time
from typing import Dict, List

import numpy as np
import torch
from catanatron.players.value import ValueFunctionPlayer

from catan_rl.env import COLORS, CatanEnv
from catan_rl.rl.agent import NetSeat
from catan_rl.rl.net import ActorCritic

CATEGORIES = ("self", "bc", "vf", "past")
_W: Dict[str, object] = {}


def init_worker():
    torch.set_num_threads(1)


def sample_lineup(rng: random.Random, n_past: int):
    u = rng.random()
    cat = "self" if u < 0.5 else "bc" if u < 0.75 else "vf" if u < 0.9 else "past"
    if cat == "self":
        return cat, ["learner"] * 4
    k = rng.choice((1, 2, 3))
    opp = {"bc": "bc", "vf": "vf", "past": f"past:{rng.randrange(n_past)}"}[cat]
    seats = ["learner"] * k + [opp] * (4 - k)
    rng.shuffle(seats)
    return cat, seats


def rewards(phis: np.ndarray, outcome: float, shaping: bool, coef: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    r = np.zeros(len(phis))
    r[-1] = outcome
    if shaping:
        nxt = np.append(phis[1:], outcome)
        r += coef * (gamma * nxt - phis)
    return r


def _net(key, params, ds, da, norm, db=0):
    net = _W.get(key)
    if net is None or net.db != db:
        net = _W[key] = ActorCritic(ds, da, norm, db)
    net.load_params(params)
    return net


def play_game(job: dict) -> dict:
    """job: seed, seats, category, params, past {j: params}, dims (ds, da, norm[, db]), shaping, coef, gamma, phi_version.
    db > 0: learner and past nets use belief features; their seats get the viewer's own BeliefTracker."""
    t0 = time.perf_counter()
    ds, da, norm, *rest = job["dims"]
    db = int(rest[0]) if rest else 0
    phi = None
    if job["shaping"]:
        from catan_rl.winprob.phi import load_default
        phi = load_default()
        assert phi.version == job["phi_version"], f"Φ changed mid-run: {phi.version} != {job['phi_version']}"
    learner = _net("learner", job["params"], ds, da, norm, db)
    recs, agents, track = {}, {}, []
    for c, s in zip(COLORS, job["seats"]):
        if s == "learner":
            recs[c] = []
            agents[c] = NetSeat(learner, c, "sample", record=recs[c], phi=phi)
            track += [c.value] if db else []
        elif s == "bc":
            from catan_rl.policy.agent import PolicyAgent
            agents[c] = PolicyAgent(c)
        elif s == "vf":
            agents[c] = ValueFunctionPlayer(c)
        else:
            j = int(s.split(":")[1])
            agents[c] = NetSeat(_net(f"past{j}", job["past"][j], ds, da, norm, db), c, "sample", salt=f"past{j}")
            track += [c.value] if db else []
    random.seed(job["seed"])
    env = CatanEnv(track_viewers=track, record_events=False).reset(job["seed"], agents)
    env.play()
    w = env.winner
    trajs = []
    for c, rec in recs.items():
        if not rec:
            continue
        out = float(w == c)
        phis = np.array([r["phi"] for r in rec])
        trajs.append({"viewer": c.value, "seed": job["seed"], "category": job["category"], "outcome": out,
                      "x": np.stack([r["x"] for r in rec]), "A": [r["A"] for r in rec],
                      "choice": np.array([r["choice"] for r in rec]), "logp": np.array([r["logp"] for r in rec]),
                      "value": np.array([r["value"] for r in rec]), "phi": phis,
                      "reward": rewards(phis, out, job["shaping"], job["coef"], job["gamma"])})
    return {"seed": job["seed"], "category": job["category"], "winner": w.value if w else None,
            "learner_seats": [c.value for c in recs], "trajs": trajs, "decisions": env.agent_decisions,
            "seconds": time.perf_counter() - t0}
