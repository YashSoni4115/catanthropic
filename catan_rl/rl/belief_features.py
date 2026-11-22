"""Optional belief-derived action features (--belief-features; extraction only, not used in training yet).

Inputs: the viewer's BeliefTracker (public events only) + public game state + own hand. Never true opponent hands.
Per action (zeros where not applicable):
  rob_p_{r}      P(stolen card = r) for MOVE_ROBBER with a victim: E_h[h_r / |h|] under the victim's belief pmf
  rob_value      sum_r rob_p_r * need_r, need_r = 1 if r is missing for a settlement/city/dev card from own hand
  mono_gain      MONOPOLY r: expected cards gained = sum over opponents E[h_r]
  yop_need       YEAR_OF_PLENTY: number of chosen cards that fill an own-hand deficit
  yop_opp_hold   YEAR_OF_PLENTY: expected opponent holdings of the chosen resources (scarcity proxy)
"""
from __future__ import annotations

from typing import List

import numpy as np
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.state_functions import player_key

from catan_rl.winprob.features import RECIPES, _to_cat

NAMES = [f"rob_p_{r}" for r in RESOURCES] + ["rob_value", "mono_gain", "yop_need", "yop_opp_hold"]
DIM = len(NAMES)


def _expect(pmf):
    """(E[h] per catanatron resource, E[h_r/|h|] per resource) from a tracker pmf."""
    E, S = np.zeros(5), np.zeros(5)
    for h, w in pmf.items():
        c = np.array(_to_cat(h), float)
        E += w * c
        if c.sum() > 0:
            S += w * c / c.sum()
    return E, S


def belief_action_features(game, color, tracker, actions: List[Action]) -> np.ndarray:
    st = game.state
    v = getattr(color, "value", color)
    me = player_key(st, color)
    own = np.array([st.player_state[f"{me}_{r}_IN_HAND"] for r in RESOURCES])
    need = np.zeros(5)
    for rec in RECIPES.values():  # catanatron resource order
        need = np.maximum(need, (np.array(rec) > own).astype(float))
    opps = [c.value for c in st.colors if c.value != v]
    stats = {o: _expect(tracker.players[o].pmf) for o in opps}
    E_opp = sum(stats[o][0] for o in opps)
    X = np.zeros((len(actions), DIM), np.float32)
    for i, a in enumerate(actions):
        t = a.action_type
        if t == ActionType.MOVE_ROBBER and a.value[1] is not None:
            p = stats[a.value[1].value][1]
            X[i, :5], X[i, 5] = p, float(p @ need)
        elif t == ActionType.PLAY_MONOPOLY:
            X[i, 6] = E_opp[RESOURCES.index(a.value)]
        elif t == ActionType.PLAY_YEAR_OF_PLENTY:
            idx = [RESOURCES.index(r) for r in a.value]
            X[i, 7] = sum(need[j] for j in set(idx))
            X[i, 8] = sum(E_opp[j] for j in idx)
    return X
