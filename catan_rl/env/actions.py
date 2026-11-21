"""Canonical 4-player action index space.

Same construction and ordering as catanatron's ``gym.envs.action_space.get_action_array``
(re-implemented here so we never import ``catanatron.gym``; a test asserts equality).
Domestic trades are not part of the space and are never offered by the engine.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.map import build_map
from catanatron.models.player import Color

COLORS: Tuple[Color, ...] = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
DOMESTIC = {ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE,
            ActionType.CONFIRM_TRADE, ActionType.CANCEL_TRADE}


@lru_cache(maxsize=None)
def action_array(colors: Tuple[Color, ...] = COLORS, map_type: str = "BASE") -> List[Tuple]:
    m = build_map(map_type)
    n = len(m.land_nodes)
    trades = [(ActionType.MARITIME_TRADE, tuple(k * [i] + [None] * (4 - k) + [j]))
              for k in (4, 3, 2) for i in RESOURCES for j in RESOURCES if i != j]
    return sorted(
        [
            (ActionType.ROLL, None),
            *[(ActionType.DISCARD_RESOURCE, r) for r in RESOURCES],
            *[(ActionType.BUILD_ROAD, tuple(sorted(e))) for e in get_edges(m.land_nodes)],
            *[(ActionType.BUILD_SETTLEMENT, i) for i in range(n)],
            *[(ActionType.BUILD_CITY, i) for i in range(n)],
            (ActionType.BUY_DEVELOPMENT_CARD, None),
            (ActionType.PLAY_KNIGHT_CARD, None),
            *[(ActionType.PLAY_YEAR_OF_PLENTY, (a, RESOURCES[j]))
              for i, a in enumerate(RESOURCES) for j in range(i, len(RESOURCES))],
            *[(ActionType.PLAY_YEAR_OF_PLENTY, (a,)) for a in RESOURCES],
            (ActionType.PLAY_ROAD_BUILDING, None),
            *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
            *[(ActionType.MOVE_ROBBER, (c, v)) for c in m.land_tiles for v in [None, *colors]],
            *trades,
            (ActionType.END_TURN, None),
        ],
        key=str,
    )


@lru_cache(maxsize=None)
def action_index() -> Dict[Tuple, int]:
    return {key: i for i, key in enumerate(action_array())}


NUM_ACTIONS = len(action_array())


def to_index(action: Action) -> int:
    if action.action_type in DOMESTIC:
        raise ValueError(f"domestic trade not in action space: {action}")
    return action_index()[(action.action_type, action.value)]
