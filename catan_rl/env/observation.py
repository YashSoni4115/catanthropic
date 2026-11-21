"""Per-viewer projection of the live catanatron State (modeled on serialization.client_view).

Visible: board, robber, bank counts, dev deck size, every player's public fields, and the
viewer's own hand / dev cards / actual VP. Opponents expose only totals and public fields.
Never included: opponents' hands, dev identities, ACTUAL_VICTORY_POINTS, *_OWNED_AT_START,
deck order, any RNG state, the seed.
"""
from __future__ import annotations

from typing import Any, Dict

from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.state import State

PUBLIC_FIELDS = ("VICTORY_POINTS", "ROADS_AVAILABLE", "SETTLEMENTS_AVAILABLE", "CITIES_AVAILABLE",
                 "HAS_ROAD", "HAS_ARMY", "HAS_ROLLED", "LONGEST_ROAD_LENGTH")


def _player(state: State, color, own: bool) -> Dict[str, Any]:
    ps, k = state.player_state, f"P{state.color_to_index[color]}_"
    out: Dict[str, Any] = {f.lower(): ps[k + f] for f in PUBLIC_FIELDS}
    out["played_dev"] = {d: ps[f"{k}PLAYED_{d}"] for d in DEVELOPMENT_CARDS}
    out["num_resources"] = sum(ps[f"{k}{r}_IN_HAND"] for r in RESOURCES)
    out["num_dev_cards"] = sum(ps[f"{k}{d}_IN_HAND"] for d in DEVELOPMENT_CARDS)
    if own:
        out["hand"] = {r: ps[f"{k}{r}_IN_HAND"] for r in RESOURCES}
        out["dev_cards"] = {d: ps[f"{k}{d}_IN_HAND"] for d in DEVELOPMENT_CARDS}
        out["actual_victory_points"] = ps[k + "ACTUAL_VICTORY_POINTS"]
        out["has_played_dev_this_turn"] = ps[k + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
    return out


def observed_state(state: State, viewer) -> Dict[str, Any]:
    b, m = state.board, state.board.map
    return {
        "viewer": viewer.value,
        "seat_order": [c.value for c in state.colors],
        "current_player": state.current_color().value,
        "prompt": state.current_prompt.value,
        "turn": state.num_turns,
        "is_initial_build_phase": state.is_initial_build_phase,
        "discard_counts": list(state.discard_counts),
        "tiles": sorted((c, t.resource, t.number) for c, t in m.land_tiles.items()),
        "ports": sorted((str(r), sorted(n)) for r, n in m.port_nodes.items()),
        "robber": b.robber_coordinate,
        "buildings": sorted((n, c.value, t) for n, (c, t) in b.buildings.items()),
        "roads": sorted((e, c.value) for e, c in b.roads.items() if e[0] < e[1]),
        "bank": list(state.resource_freqdeck),
        "dev_deck_size": len(state.development_listdeck),
        "players": {c.value: _player(state, c, c == viewer) for c in state.colors},
    }
