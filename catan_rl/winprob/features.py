"""Viewer features for win-probability states. Inputs: observed_state(viewer), the viewer's trackers,
the static board map (public) and, for block B, the fork's create_sample (leak-tested). Never true hands.

Block A (engineered) layout, all floats, names in ``names_a()``:
  per player (viewer, then opponents in turn order): PLAYER_FIELDS
  viewer: own hand (5) + own dev cards in hand (5)
  per opponent: belief fields (indep tracker + constrained joint from 64 hand samples + dev belief)
  global: bank (5), dev deck size, turn, viewer must roll, pips blocked by robber (all / viewer)
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from catanatron.features import create_sample_vector
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES

from catan_rl.env.determinize import CAT_OF_TRK, conservation_sums, own_hand, sample_hands
from catan_rl.env.observation import observed_state

PLAYED = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING")
RECIPES = {"settlement": (1, 1, 1, 1, 0), "city": (0, 0, 0, 2, 3), "dev": (0, 0, 1, 1, 1)}  # catanatron order
PLAYER_FIELDS = (["vp", "settlements", "cities", "roads", "longest_road", "has_road", "knights", "has_army", "ports"]
                 + [f"income_{r}" for r in RESOURCES] + ["income_total", "hand_total"]
                 + [f"played_{d}" for d in PLAYED] + ["unplayed_dev"])
BELIEF_FIELDS = ([f"{m}_{k}_{r}" for m in ("indep", "joint") for k in ("E", "P1") for r in RESOURCES]
                 + [f"{m}_afford_{b}" for m in ("indep", "joint") for b in RECIPES]
                 + ["dev_p_vp", "dev_p_playable_knight", "dev_e_hidden_vp"])
N_JOINT = 64


def pips(n) -> int:
    return 0 if n is None else 6 - abs(7 - n)


def names_a() -> List[str]:
    out = [f"p{i}_{f}" for i in range(4) for f in PLAYER_FIELDS]
    out += [f"own_{r}" for r in RESOURCES] + [f"own_dev_{d}" for d in DEVELOPMENT_CARDS]
    out += [f"o{i}_{f}" for i in range(1, 4) for f in BELIEF_FIELDS]
    return out + [f"bank_{r}" for r in RESOURCES] + ["dev_deck", "turn", "must_roll", "robber_pips_all", "robber_pips_viewer"]


def _node_tiles(cmap) -> Dict[int, List[Tuple]]:
    out: Dict[int, List[Tuple]] = {}
    for coord, t in cmap.land_tiles.items():
        for n in t.nodes.values():
            out.setdefault(n, []).append(coord)
    return out


def _to_cat(h) -> Tuple[int, ...]:
    """tracker-order hand -> catanatron-order hand."""
    v = [0] * 5
    for i, x in enumerate(h):
        v[CAT_OF_TRK[i]] = x
    return tuple(v)


def _belief_block(pmf_cat: Dict[Tuple, float], samples: List[Tuple]) -> List[float]:
    def stats(dist):  # list of (hand, weight)
        E = [sum(w * h[i] for h, w in dist) for i in range(5)]
        P1 = [sum(w for h, w in dist if h[i] > 0) for i in range(5)]
        aff = [sum(w for h, w in dist if all(a >= b for a, b in zip(h, rec))) for rec in RECIPES.values()]
        return E, P1, aff
    Ei, Pi, Ai = stats(list(pmf_cat.items()))
    Ej, Pj, Aj = stats([(h, 1 / len(samples)) for h in samples])
    return Ei + Pi + Ej + Pj + Ai + Aj


def _public(game, viewer):
    """(head, tail, obs, players, v): every block-A feature except opponent beliefs, from observed_state only."""
    v = getattr(viewer, "value", viewer)
    st = game.state
    obs = observed_state(st, next(c for c in st.colors if c.value == v))
    order = obs["seat_order"]
    k = order.index(v)
    players = order[k:] + order[:k]
    tiles = {c: (r, n) for c, r, n in obs["tiles"]}
    node_tiles = _node_tiles(st.board.map)
    port_nodes = {n: r for r, ns in obs["ports"] for n in ns}
    income = {p: [0.0] * 5 for p in players}
    blocked = {p: 0.0 for p in players}
    ports = {p: set() for p in players}
    for node, color, btype in obs["buildings"]:
        mult = 2 if btype == "CITY" else 1
        if node in port_nodes:
            ports[color].add(port_nodes[node])
        for coord in node_tiles.get(node, []):
            res, num = tiles[coord]
            if res is None:
                continue
            w = pips(num) * mult / 36
            if coord == obs["robber"]:
                blocked[color] += pips(num) * mult
            else:
                income[color][RESOURCES.index(res)] += w
    x: List[float] = []
    for p in players:
        d = obs["players"][p]
        x += [d["victory_points"], 5 - d["settlements_available"], 4 - d["cities_available"], 15 - d["roads_available"],
              d["longest_road_length"], d["has_road"], d["played_dev"]["KNIGHT"], d["has_army"], len(ports[p])]
        x += income[p] + [sum(income[p]), d["num_resources"]] + [d["played_dev"][c] for c in PLAYED] + [d["num_dev_cards"]]
    me = obs["players"][v]
    x += [me["hand"][r] for r in RESOURCES] + [me["dev_cards"][c] for c in DEVELOPMENT_CARDS]
    tail = list(obs["bank"]) + [obs["dev_deck_size"], obs["turn"],
                                float(not obs["is_initial_build_phase"] and not me["has_rolled"] and obs["current_player"] == v),
                                sum(blocked.values()), blocked[v]]
    return x, tail, obs, players, v


def names_public() -> List[str]:
    return [n for n in names_a() if not (n[0] == "o" and n[1].isdigit())]


def public_features(game, viewer) -> np.ndarray:
    """Block A without opponent beliefs (columns of names_public()); no tracker needed."""
    head, tail, *_ = _public(game, viewer)
    return np.asarray(head + tail, dtype=np.float32)


def engineered(game, viewer, tracker, dev_belief, seed) -> np.ndarray:
    x, tail, obs, players, v = _public(game, viewer)
    # beliefs: independent tracker marginals + constrained joint (N_JOINT samples of the determinizer's hand sampler)
    opps = players[1:]
    pmfs = {o: tracker.players[o].pmf for o in opps}
    S = conservation_sums(obs["bank"], own_hand(tracker, v))
    rng = random.Random(f"{seed}:features:{v}")
    draws = [sample_hands(pmfs, S, rng)[0] for _ in range(N_JOINT)]
    for o in opps:
        x += _belief_block({_to_cat(h): w for h, w in pmfs[o].items()}, [_to_cat(dr[o]) for dr in draws])
        x += [dev_belief.p_at_least_one(o, "VICTORY_POINT"), dev_belief.p_at_least_one(o, "KNIGHT", old_only=True),
              dev_belief.expected_count(o, "VICTORY_POINT")]
    x += tail
    return np.asarray(x, dtype=np.float32)


def sample_vector(game, viewer) -> np.ndarray:
    """Block B: the fork's create_sample for the viewer (P0). Kept only because it passes the leak test."""
    color = next(c for c in game.state.colors if c.value == getattr(viewer, "value", viewer))
    return np.asarray(create_sample_vector(game, color), dtype=np.float32)
