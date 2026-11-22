"""value_function teacher, re-expressed so its per-action scores are visible.

``vf_values`` mirrors ``ValueFunctionPlayer.decide`` (fork: catanatron/players/value.py, ``decide`` and
``base_fn``): copy the REAL game, execute the action, score with base_fn(DEFAULT_WEIGHTS); the choice is
the first strict maximum in playable_actions order. A test asserts it equals ValueFunctionPlayer.decide.
"""
from __future__ import annotations

from typing import List

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

VALUE_FN = base_fn(DEFAULT_WEIGHTS)

GROUPS = ("initial_settlement", "initial_road", "settlement", "city", "road", "dev_buy", "robber",
          "trade", "end_turn", "roll", "play_dev", "discard")
_G = {ActionType.BUILD_SETTLEMENT: "settlement", ActionType.BUILD_CITY: "city", ActionType.BUILD_ROAD: "road",
      ActionType.BUY_DEVELOPMENT_CARD: "dev_buy", ActionType.MOVE_ROBBER: "robber", ActionType.MARITIME_TRADE: "trade",
      ActionType.END_TURN: "end_turn", ActionType.ROLL: "roll", ActionType.DISCARD_RESOURCE: "discard",
      ActionType.PLAY_KNIGHT_CARD: "play_dev", ActionType.PLAY_YEAR_OF_PLENTY: "play_dev",
      ActionType.PLAY_MONOPOLY: "play_dev", ActionType.PLAY_ROAD_BUILDING: "play_dev"}


def group(action_type: ActionType, initial: bool) -> str:
    g = _G[action_type]
    return f"initial_{g}" if initial and g in ("settlement", "road") else g


def vf_values(game, color, actions) -> List[float]:
    out = []
    for a in actions:
        g = game.copy()
        g.execute(a)
        out.append(VALUE_FN(g, color))
    return out


def vf_choice(values) -> int:
    best, bi = float("-inf"), None
    for i, v in enumerate(values):
        if v > best:
            best, bi = v, i
    return bi
