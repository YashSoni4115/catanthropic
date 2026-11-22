"""Φ-greedy: one-ply expectimax on the calibrated win-probability model, over one determinization.

At each non-forced decision of seat ``viewer``:
  1. g = env.determinize(viewer, decision_seed)  (hidden info re-sampled from viewer's beliefs only)
  2. for every legal action a: outcomes of a on g — one copy for deterministic actions; the chance
     spectrum for dice rolls (2..12), dev-card buys (unseen-card composition on g) and robber steals
     (victim's sampled hand composition) — scored by E[Φ(outcome, viewer)[viewer]]
  3. argmax, ties broken by an RNG seeded from the decision seed.
The decision seed depends only on (agent seed, game seed, number of actions so far), so a decision is a
pure function of the viewer's information: the real game is never read except through the determinizer.
Use as a CatanEnv seat callback; the env must track the viewer (track_viewers).
"""
from __future__ import annotations

import random
import time
from typing import List, Tuple

import numpy as np
from catanatron.models.enums import RESOURCES, ActionRecord, ActionType
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.state_functions import get_player_freqdeck

from catan_rl.env.actions import DOMESTIC, to_index
from catan_rl.winprob.phi import DEFAULT_PATH, PhiModel, load_default


def _steal_spectrum(game, action) -> List[Tuple[object, float]]:
    """Robber move with a steal: outcome per card weighted by the (determinized) victim hand."""
    _, victim = action.value
    hand = get_player_freqdeck(game.state, victim) if victim is not None else [0] * 5
    tot = sum(hand)
    if tot == 0:
        return execute_spectrum(game, action)
    out = []
    for card, n in zip(RESOURCES, hand):
        if n:
            g = game.copy()
            g.execute(action, validate_action=False, action_record=ActionRecord(action=action, result=card))
            out.append((g, n / tot))
    return out


def outcomes(game, action) -> List[Tuple[object, float]]:
    if action.action_type == ActionType.MOVE_ROBBER:
        return _steal_spectrum(game, action)
    return execute_spectrum(game, action)


class PhiGreedyAgent:
    def __init__(self, viewer, phi: PhiModel = None, seed: int = 0, phi_path: str = DEFAULT_PATH):
        self.viewer = getattr(viewer, "value", viewer)
        self.phi = phi or load_default(phi_path)
        self.seed = seed
        self.decisions, self.time = 0, 0.0
        self.last_scores = None

    def decision_seed(self, env) -> str:
        return f"phi_greedy:{self.seed}:{env.seed}:{len(env.game.state.action_records)}"

    def choose(self, env, legal: List[int]) -> int:
        s = self.decision_seed(env)
        g = env.determinize(self.viewer, s)
        acts = {to_index(a): a for a in g.playable_actions if a.action_type not in DOMESTIC}
        missing = set(legal) - set(acts)
        assert not missing, f"determinized game lost legal actions {missing}"
        games, owner, probs = [], [], []
        for j, i in enumerate(legal):
            for og, p in outcomes(g, acts[i]):
                games.append(og), owner.append(j), probs.append(p)
        P = self.phi.phi_batch(games, self.viewer)[:, 0]
        scores = np.bincount(owner, weights=P * np.asarray(probs), minlength=len(legal))
        scores /= np.bincount(owner, weights=probs, minlength=len(legal))
        self.last_scores = dict(zip(legal, scores))
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        return legal[best[random.Random(s).randrange(len(best))]]

    def __call__(self, env, color) -> int:
        assert getattr(color, "value", color) == self.viewer
        t = time.perf_counter()
        legal = env.legal_actions()
        a = legal[0] if len(legal) == 1 else self.choose(env, legal)
        self.decisions += 1
        self.time += time.perf_counter() - t
        return a


def factory(color):
    return PhiGreedyAgent(color)
