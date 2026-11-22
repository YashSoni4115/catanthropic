"""Per-action feature vectors for the BC policy. Inputs: the acting seat's public + own information only.

For each legal action: type one-hot, resource args, node/robber-target descriptors, and the change in
value_function's components (``vf_components``: public VP, production, enemy production, reachability,
buildable nodes, hand synergy, ...; fork catanatron/players/value.py base_fn) caused by the action. Those
components read only public state and the actor's own hand, and are computed on a copy of the game:
  * deterministic actions: executed on the copy;
  * ROLL: expectation over the 11 dice outcomes (execute_spectrum); never the copy's actual next roll;
  * MOVE_ROBBER: executed as (tile, no victim): the stolen card is unknown; victim hand size is a feature;
  * BUY_DEVELOPMENT_CARD: executed on the copy; the drawn card's type does not enter any feature
    (dev cards count only in total, public VP excludes VP cards).
A twin-game test (hidden opponent cards / deck order changed) asserts identical features.
"""
from __future__ import annotations

from typing import List

import numpy as np
from catanatron.features import build_production_features, reachability_features, resource_hand_features
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, Action, ActionType
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.players.value import value_production
from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards, player_key,
                                        player_num_dev_cards, player_num_resource_cards)

TYPES = [ActionType.ROLL, ActionType.DISCARD_RESOURCE, ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT,
         ActionType.BUILD_CITY, ActionType.BUY_DEVELOPMENT_CARD, ActionType.PLAY_KNIGHT_CARD,
         ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_ROAD_BUILDING, ActionType.PLAY_MONOPOLY,
         ActionType.MOVE_ROBBER, ActionType.MARITIME_TRADE, ActionType.END_TURN]
VF_NAMES = ["public_vp", "production", "enemy_production", "num_tiles", "reach0", "reach1", "buildable_nodes",
            "longest_road", "no_buildable", "hand_synergy", "hand_size", "over7", "devs", "knights"]
VF_SCALE = np.array([1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 1, 1, 0.1, 1, 0.1, 0.1], dtype=np.float32)
PORTS = [None, *RESOURCES]
NAMES = ([f"type_{t.name}" for t in TYPES] + [f"res_a_{r}" for r in RESOURCES] + [f"res_b_{r}" for r in RESOURCES]
         + ["trade_ratio", "monopoly_gain"] + [f"node_prod_{r}" for r in RESOURCES] + ["node_prod_total", "node_variety"]
         + [f"node_port_{p}" for p in PORTS] + [f"rob_res_{r}" for r in RESOURCES]
         + ["rob_pips", "rob_own_weight", "rob_opp_weight", "rob_has_victim", "rob_victim_hand", "rob_victim_vp"]
         + [f"rob_victim_seat{i}" for i in (1, 2, 3)] + [f"d_{n}" for n in VF_NAMES] + [f"after_{n}" for n in VF_NAMES])
DIM = len(NAMES)
_PROD = build_production_features(True)


def vf_components(game, color) -> np.ndarray:
    """value_function's base_fn terms (DEFAULT_WEIGHTS structure), unweighted. Public + own hand only."""
    st = game.state
    s = _PROD(game, color)
    reach = reachability_features(game, color, 2)
    h = resource_hand_features(game, color)
    city = (max(2 - h["P0_WHEAT_IN_HAND"], 0) + max(3 - h["P0_ORE_IN_HAND"], 0)) / 5.0
    sett = sum(max(1 - h[f"P0_{r}_IN_HAND"], 0) for r in ("WHEAT", "SHEEP", "BRICK", "WOOD")) / 4.0
    b = st.buildings_by_color[color]
    tiles = set()
    for n in b[SETTLEMENT] + b[CITY]:
        tiles.update(st.board.map.adjacent_tiles[n])
    nb = len(st.board.buildable_node_ids(color))
    hand = player_num_resource_cards(st, color)
    return np.array([
        st.player_state[f"{player_key(st, color)}_VICTORY_POINTS"], value_production(s, "P0"),
        value_production(s, "P1", False), len(tiles), sum(reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES),
        sum(reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES), nb, get_longest_road_length(st, color),
        float(nb == 0), (2 - city - sett) / 2, hand, float(hand > 7), player_num_dev_cards(st, color),
        get_played_dev_cards(st, color, "KNIGHT")], dtype=np.float64)


def _after(game, action) -> np.ndarray:
    c = action.color
    if action.action_type == ActionType.ROLL:
        return sum(p * vf_components(g, c) for g, p in execute_spectrum(game, action))
    if action.action_type == ActionType.MOVE_ROBBER:
        action = Action(c, action.action_type, (action.value[0], None))
    g = game.copy()
    g.execute(action, validate_action=False)
    return vf_components(g, c)


def action_features(game, color, actions: List[Action]) -> np.ndarray:
    st = game.state
    m = st.board.map
    order = list(st.colors)
    k = order.index(color)
    me = player_key(st, color)
    port_of = {n: p for p, ns in m.port_nodes.items() for n in ns}
    base = vf_components(game, color)
    X = np.zeros((len(actions), DIM), dtype=np.float32)
    o = {n: i for i, n in enumerate(NAMES)}
    for i, a in enumerate(actions):
        x, t, v = X[i], a.action_type, a.value
        x[o[f"type_{t.name}"]] = 1
        if t in (ActionType.DISCARD_RESOURCE, ActionType.PLAY_MONOPOLY):
            x[o[f"res_a_{v}"]] = 1
        if t == ActionType.PLAY_MONOPOLY:  # opponents' total holding of v is public: 19 - bank - own
            x[o["monopoly_gain"]] = (19 - st.resource_freqdeck[RESOURCES.index(v)] - st.player_state[f"{me}_{v}_IN_HAND"]) / 5
        if t == ActionType.PLAY_YEAR_OF_PLENTY:
            x[o[f"res_a_{v[0]}"]] += 1
            if len(v) > 1:
                x[o[f"res_b_{v[1]}"]] += 1
        if t == ActionType.MARITIME_TRADE:
            give = [r for r in v[:-1] if r is not None]
            x[o[f"res_a_{give[0]}"]], x[o[f"res_b_{v[-1]}"]], x[o["trade_ratio"]] = 1, 1, len(give) / 4
        if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            prod = m.node_production[v]
            for r in RESOURCES:
                x[o[f"node_prod_{r}"]] = prod.get(r, 0) * 36 / 5
            x[o["node_prod_total"]], x[o["node_variety"]] = sum(prod.values()) * 36 / 15, len(prod) / 3
            if v in port_of:
                x[o[f"node_port_{port_of[v]}"]] = 1
        if t == ActionType.MOVE_ROBBER:
            coord, victim = v
            tile = m.land_tiles[coord]
            if tile.resource:
                x[o[f"rob_res_{tile.resource}"]] = 1
            x[o["rob_pips"]] = (6 - abs(7 - tile.number)) / 5 if tile.number else 0
            for n in tile.nodes.values():
                bld = st.board.buildings.get(n)
                if bld:
                    x[o["rob_own_weight" if bld[0] == color else "rob_opp_weight"]] += 2 if bld[1] == CITY else 1
            if victim is not None:
                vk = player_key(st, victim)
                x[o["rob_has_victim"]] = 1
                x[o["rob_victim_hand"]] = player_num_resource_cards(st, victim) / 10
                x[o["rob_victim_vp"]] = st.player_state[f"{vk}_VICTORY_POINTS"] / 10
                x[o[f"rob_victim_seat{(order.index(victim) - k) % 4}"]] = 1
        after = _after(game, a)
        n = len(VF_NAMES)
        x[o["d_public_vp"]:o["d_public_vp"] + n] = (after - base) * VF_SCALE
        x[o["after_public_vp"]:o["after_public_vp"] + n] = after * VF_SCALE
    return X
