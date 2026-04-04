"""Catan game-state extraction and projection utilities.

This module supports two sources:
- live Java `game.Game` / `board.Board` objects (through a JVM bridge)
- JSON snapshots

It returns a model-ready payload with:
- `omniscient_state`: full information (when available)
- `observed_state`: perspective-limited information

Backward-compatible top-level fields (`robber`, `tiles`, `players`, etc.) default to
**mirroring `observed_state`** so agents do not accidentally receive omniscient
hidden information. Use `top_level_mirror=\"omniscient\"` only for offline ML
pipelines that want a convenience view of full state.

Design goals:
- stable ML-friendly naming
- forward-compatible schema and encoding metadata
- explicit hidden-information handling
- compatibility with existing sparse board extraction
- ML-stable player ordering: current (viewer) player first, then opponents sorted
  by `(player_id, player_name)` lexicographically
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SCHEMA_VERSION = "2.1.0"

BOARD_SIZE = 7
STRUCTURE_ORIENTATION_COUNT = 2
ROAD_ORIENTATION_COUNT = 3

# Hex tile coordinates that receive terrain in `board.Board` (Java col=x, row=y).
ENGINE_BOARD_LAYOUT_ID = "catanthropic_java_board_7x7_19_hex"
VALID_SLOT_MASKS_VERSION = "1.0.0"

VALID_TILE_COORDINATES: Set[Tuple[int, int]] = {
    (1, 1),
    (2, 1),
    (3, 1),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (2, 4),
    (3, 4),
    (4, 4),
    (5, 4),
    (3, 5),
    (4, 5),
    (5, 5),
}

RESOURCE_TYPES = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER", "DESERT")
TRACKED_RESOURCES = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER")

STRUCTURE_TYPES = ("settlement", "city")

PORT_INDEX_MAP = {
    0: "three_to_one",
    1: "brick_two_to_one",
    2: "wool_two_to_one",
    3: "ore_two_to_one",
    4: "grain_two_to_one",
    5: "lumber_two_to_one",
}

DEV_CARD_SCHEMA_VERSION = "1.0"
DEV_CARD_LABELS = {
    "Knight": "knight",
    "Progress": "progress",
    "Victory Point": "victory_point",
    "Road building": "road_building",
    "Year of plenty": "year_of_plenty",
    "Monopoly": "monopoly",
}

# -----------------------------------------------------------------------------
# DERIVED_FEATURES_EXTENSION — placeholder for future engineered features
# (expected production, robber-blocked production, settlement quality, city upgrade
# value, reachable expansion spots, longest-road threat). The extractor stays focused
# on normalized state; call `build_derived_features_placeholder()` from assembly.
# -----------------------------------------------------------------------------


def build_derived_features_placeholder() -> Dict[str, Any]:
    """Reserved schema hook for future derived features; returns empty placeholder."""
    return {
        "schema_status": "placeholder",
        "planned_metrics": [
            "expected_production",
            "robber_blocked_production",
            "settlement_quality",
            "city_upgrade_value",
            "reachable_expansion_spots",
            "longest_road_threat",
        ],
    }


def _coerce_iterable_to_list(value: Any) -> List[Any]:
    """Convert Java collections, arrays, iterables, or Python sequences into a Python list.

    Live JVM bridges may return types that are not `list`/`tuple`; `None` yields `[]`.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        pass

    to_array = getattr(value, "toArray", None)
    if callable(to_array):
        try:
            return _coerce_iterable_to_list(to_array())
        except Exception:
            pass

    size_attr = getattr(value, "size", None)
    if callable(size_attr):
        try:
            n = int(size_attr())
            get = getattr(value, "get", None)
            if callable(get):
                return [get(i) for i in range(n)]
        except Exception:
            pass

    return []


def _coerce_2d_grid(grid: Any) -> List[List[Any]]:
    rows = _coerce_iterable_to_list(grid)
    return [_coerce_iterable_to_list(r) for r in rows]


def _coerce_3d_grid(grid: Any) -> List[List[List[Any]]]:
    layers = _coerce_iterable_to_list(grid)
    return [_coerce_2d_grid(layer) for layer in layers]


def _call(obj: Any, name: str, default: Any = None) -> Any:
    attr = getattr(obj, name, None)
    if callable(attr):
        try:
            return attr()
        except TypeError:
            return default
    return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _in_bounds(x_coord: int, y_coord: int) -> bool:
    return 0 <= x_coord < BOARD_SIZE and 0 <= y_coord < BOARD_SIZE


def _normalize_resource_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in RESOURCE_TYPES:
        return normalized
    return "UNKNOWN"


def _normalize_structure_type(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in STRUCTURE_TYPES:
            return cleaned
    as_int = _safe_int(value)
    if as_int == 1:
        return "city"
    return "settlement"


def _normalize_ports(value: Any) -> Dict[str, bool]:
    ports = {name: False for name in PORT_INDEX_MAP.values()}
    if value is None:
        return ports

    if isinstance(value, dict):
        for key, flag in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in ports:
                ports[normalized_key] = bool(flag)
        return ports

    iterable = _coerce_iterable_to_list(value)
    if not iterable:
        return ports

    for index, flag in enumerate(iterable):
        mapped = PORT_INDEX_MAP.get(index)
        if mapped is not None:
            ports[mapped] = bool(flag)
    return ports


def _location_to_dict(location: Any) -> Optional[Dict[str, int]]:
    if location is None:
        return None
    x_coord = _safe_int(_call(location, "getXCoord"))
    y_coord = _safe_int(_call(location, "getYCoord"))
    if x_coord is None or y_coord is None:
        return None
    return {"x": x_coord, "y": y_coord}


def _player_identity(player: Any) -> Dict[str, Optional[str]]:
    if player is None:
        return {"player_id": None, "player_name": None}

    player_name = _safe_str(_call(player, "getName")) or _safe_str(player)

    candidate_id_methods = ("getId", "getPlayerId", "getUUID", "getUid")
    player_id: Optional[str] = None
    for method_name in candidate_id_methods:
        candidate = _call(player, method_name)
        if candidate is not None:
            player_id = _safe_str(candidate)
            break

    if player_id is None and player_name is not None:
        player_id = f"name:{player_name}"

    return {"player_id": player_id, "player_name": player_name}


def _owner_ref_from_snapshot(value: Any) -> Dict[str, Optional[str]]:
    if value is None:
        return {"owner_id": None, "owner_name": None}

    if isinstance(value, dict):
        owner_id = _safe_str(value.get("owner_id") or value.get("player_id") or value.get("id"))
        owner_name = _safe_str(value.get("owner_name") or value.get("player_name") or value.get("name"))
        if owner_id is None and owner_name is not None:
            owner_id = f"name:{owner_name}"
        return {"owner_id": owner_id, "owner_name": owner_name}

    owner_name = _safe_str(value)
    owner_id = f"name:{owner_name}" if owner_name is not None else None
    return {"owner_id": owner_id, "owner_name": owner_name}


# Coordinate adjacency helpers below are derived from the engine's grid indexing — they are
# NOT verified by calling the Java engine. They match `board.Board`'s slanted hex layout;
# validate against known snapshots or tests before treating as engine-guaranteed.


def _vertex_adjacent_tiles(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    candidates: List[Tuple[int, int]]
    if orientation == 0:
        candidates = [
            (x_coord, y_coord),
            (x_coord, y_coord + 1),
            (x_coord + 1, y_coord + 1),
        ]
    else:
        candidates = [
            (x_coord, y_coord),
            (x_coord, y_coord - 1),
            (x_coord - 1, y_coord - 1),
        ]

    output: List[Dict[str, int]] = []
    for tx, ty in candidates:
        if _in_bounds(tx, ty):
            output.append({"x": tx, "y": ty})
    return output


def _vertex_incident_roads(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    candidates: List[Tuple[int, int, int]]
    if orientation == 0:
        candidates = [
            (x_coord, y_coord, 0),
            (x_coord, y_coord, 1),
            (x_coord, y_coord + 1, 2),
        ]
    else:
        candidates = [
            (x_coord, y_coord - 1, 0),
            (x_coord - 1, y_coord - 1, 1),
            (x_coord - 1, y_coord - 1, 2),
        ]

    output: List[Dict[str, int]] = []
    for rx, ry, ro in candidates:
        if _in_bounds(rx, ry) and 0 <= ro < ROAD_ORIENTATION_COUNT:
            output.append({"x": rx, "y": ry, "orientation": ro})
    return output


def _road_endpoint_vertices(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    if orientation == 0:
        candidates = [(x_coord, y_coord, 0), (x_coord, y_coord + 1, 1)]
    elif orientation == 1:
        candidates = [(x_coord, y_coord, 0), (x_coord + 1, y_coord + 1, 1)]
    else:
        candidates = [(x_coord, y_coord - 1, 0), (x_coord + 1, y_coord + 1, 1)]

    output: List[Dict[str, int]] = []
    for vx, vy, vo in candidates:
        if _in_bounds(vx, vy) and 0 <= vo < STRUCTURE_ORIENTATION_COUNT:
            output.append({"x": vx, "y": vy, "orientation": vo})
    return output


def _tile_adjacent_vertices(x_coord: int, y_coord: int) -> List[Dict[str, int]]:
    candidates = [
        (x_coord, y_coord, 0),
        (x_coord, y_coord, 1),
        (x_coord + 1, y_coord + 1, 1),
        (x_coord - 1, y_coord - 1, 0),
        (x_coord, y_coord + 1, 1),
        (x_coord, y_coord - 1, 0),
    ]

    output: List[Dict[str, int]] = []
    for vx, vy, vo in candidates:
        if _in_bounds(vx, vy) and 0 <= vo < STRUCTURE_ORIENTATION_COUNT:
            output.append({"x": vx, "y": vy, "orientation": vo})
    return output


def _engine_structure_vertices_for_hex_tile(lx: int, ly: int) -> List[Tuple[int, int, int]]:
    """Vertices adjacent to a hex tile, matching `Board.java` rollStructures pattern."""
    candidates = [
        (lx, ly, 0),
        (lx, ly, 1),
        (lx + 1, ly + 1, 1),
        (lx - 1, ly - 1, 0),
        (lx, ly + 1, 1),
        (lx, ly - 1, 0),
    ]
    out: List[Tuple[int, int, int]] = []
    for x, y, o in candidates:
        if _in_bounds(x, y) and 0 <= o < STRUCTURE_ORIENTATION_COUNT:
            out.append((x, y, o))
    return out


def _build_valid_structure_slots() -> frozenset:
    acc: Set[Tuple[int, int, int]] = set()
    for lx, ly in VALID_TILE_COORDINATES:
        acc.update(_engine_structure_vertices_for_hex_tile(lx, ly))
    return frozenset(acc)


def _build_valid_road_slots(structure_slots: frozenset) -> frozenset:
    """Road slot is valid if both endpoint vertices are engine-playable structure slots."""
    valid: Set[Tuple[int, int, int]] = set()
    for x_coord in range(BOARD_SIZE):
        for y_coord in range(BOARD_SIZE):
            for orientation in range(ROAD_ORIENTATION_COUNT):
                endpoints = _road_endpoint_vertices(x_coord, y_coord, orientation)
                if len(endpoints) != 2:
                    continue
                t0 = (endpoints[0]["x"], endpoints[0]["y"], endpoints[0]["orientation"])
                t1 = (endpoints[1]["x"], endpoints[1]["y"], endpoints[1]["orientation"])
                if t0 in structure_slots and t1 in structure_slots:
                    valid.add((x_coord, y_coord, orientation))
    return frozenset(valid)


_VALID_STRUCTURE_SLOTS: frozenset = _build_valid_structure_slots()
_VALID_ROAD_SLOTS: frozenset = _build_valid_road_slots(_VALID_STRUCTURE_SLOTS)


def _validate_adjacency_against_board(board: Any) -> Optional[List[str]]:
    """Optional hook: compare coordinate adjacency to live `board.Board` (not implemented)."""
    del board
    return None


def _build_slot_ordering_metadata() -> Dict[str, Any]:
    tile_slots: List[Dict[str, int]] = []
    structure_slots: List[Dict[str, int]] = []
    road_slots: List[Dict[str, int]] = []

    for x_coord in range(BOARD_SIZE):
        for y_coord in range(BOARD_SIZE):
            tile_slots.append({"x": x_coord, "y": y_coord})
            for orientation in range(STRUCTURE_ORIENTATION_COUNT):
                structure_slots.append({
                    "x": x_coord,
                    "y": y_coord,
                    "orientation": orientation,
                })
            for orientation in range(ROAD_ORIENTATION_COUNT):
                road_slots.append({
                    "x": x_coord,
                    "y": y_coord,
                    "orientation": orientation,
                })

    # Invalid slots are padding in the 7×7 envelope, not "empty but legal" placements.
    tile_slot_valid = [(s["x"], s["y"]) in VALID_TILE_COORDINATES for s in tile_slots]
    structure_slot_valid = [
        (s["x"], s["y"], s["orientation"]) in _VALID_STRUCTURE_SLOTS for s in structure_slots
    ]
    road_slot_valid = [
        (s["x"], s["y"], s["orientation"]) in _VALID_ROAD_SLOTS for s in road_slots
    ]

    return {
        "tile_slots": tile_slots,
        "structure_slots": structure_slots,
        "road_slots": road_slots,
        "ordering_rule": "x_then_y_then_orientation",
        "tile_slot_valid": tile_slot_valid,
        "structure_slot_valid": structure_slot_valid,
        "road_slot_valid": road_slot_valid,
        "valid_slot_masks_version": VALID_SLOT_MASKS_VERSION,
        "engine_board_layout": ENGINE_BOARD_LAYOUT_ID,
        "slot_mask_note": (
            "False marks non-playable coordinate padding; True marks slots that can correspond "
            "to real terrain (tiles) or build locations (structures/roads) on this engine layout."
        ),
    }


def _normalize_roll_history_item(item: Any) -> Any:
    if isinstance(item, dict):
        return {str(k): _normalize_roll_history_item(v) for k, v in sorted(item.items())}
    return _safe_int(item, item)


def _normalize_action_item(item: Any) -> Any:
    if isinstance(item, dict):
        return {str(k): _normalize_action_item(v) for k, v in sorted(item.items())}
    if item is None:
        return None
    return _safe_str(item) if not isinstance(item, (int, float, bool, str)) else item


def _extract_turn_metadata(game: Any = None, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snapshot = snapshot or {}

    turn_index = snapshot.get("turn_index")
    if turn_index is None and game is not None:
        turn_index = _call(game, "getTurnIndex")

    current_phase = snapshot.get("current_phase")
    if current_phase is None and game is not None:
        current_phase = _call(game, "getCurrentPhase")

    last_action = snapshot.get("last_action")
    if last_action is None and game is not None:
        last_action = _call(game, "getLastAction")

    current_player_id = snapshot.get("current_player_id")
    current_player_name = snapshot.get("current_player_name")
    if game is not None and (current_player_id is None or current_player_name is None):
        live_current_player = _call(game, "getCurrentPlayer")
        live_identity = _player_identity(live_current_player)
        if current_player_id is None:
            current_player_id = live_identity.get("player_id")
        if current_player_name is None:
            current_player_name = live_identity.get("player_name")

    return {
        "turn_index": _safe_int(turn_index),
        "current_player_id": current_player_id,
        "current_player_name": current_player_name,
        "current_phase": _safe_str(current_phase),
        "last_action": _safe_str(last_action),
    }


def _extract_transition_context(game: Any = None, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snapshot = snapshot or {}
    # Prefer snapshot fields when present; otherwise fill from live game.
    latest_dice_roll = snapshot.get("latest_dice_roll")
    if latest_dice_roll is None and game is not None:
        latest_dice_roll = _call(game, "getLatestDiceRoll")

    roll_history = snapshot.get("roll_history")
    if roll_history is None and game is not None:
        roll_history = _call(game, "getRollHistory")
    roll_history = _coerce_iterable_to_list(roll_history)

    recent_actions = snapshot.get("recent_actions")
    if recent_actions is None and game is not None:
        recent_actions = _call(game, "getRecentActions")
    recent_actions = _coerce_iterable_to_list(recent_actions)

    roll_history_norm = [_normalize_roll_history_item(x) for x in roll_history]
    recent_actions_norm = [_normalize_action_item(x) for x in recent_actions]

    prev_sid = snapshot.get("previous_state_id")
    if prev_sid is None:
        tc = snapshot.get("transition_context")
        if isinstance(tc, dict):
            dt = tc.get("delta_tracking")
            if isinstance(dt, dict):
                prev_sid = dt.get("previous_state_id")
    if prev_sid is None and game is not None:
        getter = getattr(game, "getPreviousStateId", None)
        if callable(getter):
            try:
                prev_sid = getter()
            except Exception:
                prev_sid = None

    return {
        "latest_dice_roll": _safe_int(latest_dice_roll),
        "roll_history": roll_history_norm,
        "recent_actions": recent_actions_norm,
        "delta_tracking": {
            "previous_state_id": prev_sid,
            "delta_supported": False,
            "delta_placeholder": None,
        },
    }


def _extract_dev_card_summary_from_player(player: Any) -> Dict[str, Any]:
    get_dev_cards_type = getattr(player, "getDevCardsType", None)
    grouped = {
        "knight": 0,
        "progress": 0,
        "victory_point": 0,
    }
    progress_subtypes = {
        "road_building": 0,
        "year_of_plenty": 0,
        "monopoly": 0,
    }

    if callable(get_dev_cards_type):
        grouped["knight"] = _safe_int(get_dev_cards_type("Knight"), 0) or 0
        grouped["progress"] = _safe_int(get_dev_cards_type("Progress"), 0) or 0
        grouped["victory_point"] = _safe_int(get_dev_cards_type("Victory Point"), 0) or 0

        progress_subtypes["road_building"] = _safe_int(get_dev_cards_type("Road building"), 0) or 0
        progress_subtypes["year_of_plenty"] = _safe_int(get_dev_cards_type("Year of plenty"), 0) or 0
        progress_subtypes["monopoly"] = _safe_int(get_dev_cards_type("Monopoly"), 0) or 0

    # Assumption:
    # `grouped.progress` already includes all progress subtypes.
    # Therefore `total_cards` is computed only from grouped buckets to avoid double counting.
    total_cards = grouped["knight"] + grouped["progress"] + grouped["victory_point"]

    return {
        "schema_version": DEV_CARD_SCHEMA_VERSION,
        "grouped_counts": grouped,
        "progress_subtype_counts": progress_subtypes,
        "total_cards": total_cards,
        "counting_assumption": "progress_subtypes_are_subset_of_grouped_progress",
    }


def _normalize_dev_card_summary_from_snapshot(raw: Any) -> Dict[str, Any]:
    grouped = {
        "knight": 0,
        "progress": 0,
        "victory_point": 0,
    }
    progress_subtypes = {
        "road_building": 0,
        "year_of_plenty": 0,
        "monopoly": 0,
    }

    if isinstance(raw, dict):
        if "grouped_counts" in raw:
            grouped_raw = raw.get("grouped_counts", {})
            if isinstance(grouped_raw, dict):
                for key in grouped:
                    grouped[key] = _safe_int(grouped_raw.get(key), 0) or 0

            subtype_raw = raw.get("progress_subtype_counts", {})
            if isinstance(subtype_raw, dict):
                for key in progress_subtypes:
                    progress_subtypes[key] = _safe_int(subtype_raw.get(key), 0) or 0
        else:
            grouped["knight"] = _safe_int(raw.get("Knight"), 0) or 0
            grouped["progress"] = _safe_int(raw.get("Progress"), 0) or 0
            grouped["victory_point"] = _safe_int(raw.get("Victory Point"), 0) or 0
            progress_subtypes["road_building"] = _safe_int(raw.get("Road building"), 0) or 0
            progress_subtypes["year_of_plenty"] = _safe_int(raw.get("Year of plenty"), 0) or 0
            progress_subtypes["monopoly"] = _safe_int(raw.get("Monopoly"), 0) or 0

    total_cards = grouped["knight"] + grouped["progress"] + grouped["victory_point"]

    return {
        "schema_version": DEV_CARD_SCHEMA_VERSION,
        "grouped_counts": grouped,
        "progress_subtype_counts": progress_subtypes,
        "total_cards": total_cards,
        "counting_assumption": "progress_subtypes_are_subset_of_grouped_progress",
    }


def _extract_resources_from_player(player: Any) -> Dict[str, int]:
    getter = getattr(player, "getNumberResourcesType", None)
    resources = {resource: 0 for resource in TRACKED_RESOURCES}
    if not callable(getter):
        return resources

    for resource in TRACKED_RESOURCES:
        resources[resource] = _safe_int(getter(resource), 0) or 0
    return resources


def _extract_board_ownership(board_state: Dict[str, Any], player_id: Optional[str]) -> Dict[str, Any]:
    if player_id is None:
        return {"settlements": [], "cities": [], "roads": []}

    owned_structures = [
        s for s in board_state.get("structures", []) if s.get("owner_id") == player_id
    ]
    settlements = [s for s in owned_structures if s.get("type") == "settlement"]
    cities = [s for s in owned_structures if s.get("type") == "city"]
    roads = [r for r in board_state.get("roads", []) if r.get("owner_id") == player_id]

    return {
        "settlements": settlements,
        "cities": cities,
        "roads": roads,
    }


def _extract_remaining_pieces(player: Any) -> Dict[str, Optional[int]]:
    total_limits = {"roads": 15, "settlements": 5, "cities": 4}

    roads_placed = _safe_int(_call(player, "getNumbRoads"))
    settlements_placed = _safe_int(_call(player, "getNumbSettlements"))
    cities_placed = _safe_int(_call(player, "getNumbCities"))

    def _remaining(limit: int, placed: Optional[int]) -> Optional[int]:
        if placed is None:
            return None
        if 0 <= placed <= limit:
            return limit - placed
        return None

    return {
        "roads_remaining": _remaining(total_limits["roads"], roads_placed),
        "settlements_remaining": _remaining(total_limits["settlements"], settlements_placed),
        "cities_remaining": _remaining(total_limits["cities"], cities_placed),
        "roads_placed_reported": roads_placed,
        "settlements_placed_reported": settlements_placed,
        "cities_placed_reported": cities_placed,
    }


def _extract_player_summary_live(player: Any, board_state: Dict[str, Any]) -> Dict[str, Any]:
    identity = _player_identity(player)
    ownership = _extract_board_ownership(board_state, identity["player_id"])
    resources = _extract_resources_from_player(player)
    dev_cards = _extract_dev_card_summary_from_player(player)
    remaining_pieces = _extract_remaining_pieces(player)

    return {
        "player_id": identity["player_id"],
        "player_name": identity["player_name"],
        "victory_points_visible": _safe_int(_call(player, "getVictoryPoints")),
        "victory_point_breakdown_visible": None,
        "knights_played_visible": _safe_int(_call(player, "getNumbKnights")),
        "largest_army_flag": bool(_call(player, "hasLargestArmy", False)),
        "resource_cards": resources,
        "total_resource_cards": sum(resources.values()),
        "development_cards": dev_cards,
        "ports": _normalize_ports(_call(player, "getPorts")),
        "remaining_pieces": remaining_pieces,
        "board_presence": {
            "settlement_count": len(ownership["settlements"]),
            "city_count": len(ownership["cities"]),
            "road_count": len(ownership["roads"]),
            "settlements": ownership["settlements"],
            "cities": ownership["cities"],
            "roads": ownership["roads"],
        },
    }


def _extract_players_from_game(
    game: Any,
    board_state: Dict[str, Any],
    players: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    if players is None:
        players = _call(game, "getPlayers")
    players = _coerce_iterable_to_list(players)

    summaries: List[Dict[str, Any]] = []
    for player in players:
        summaries.append(_extract_player_summary_live(player, board_state))
    return summaries


def _extract_players_from_snapshot(
    snapshot_players: Any,
    board_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    snapshot_players = _coerce_iterable_to_list(snapshot_players)

    summaries: List[Dict[str, Any]] = []
    for raw in snapshot_players:
        if not isinstance(raw, dict):
            continue

        player_name = _safe_str(raw.get("player_name") or raw.get("name"))
        player_id = _safe_str(raw.get("player_id") or raw.get("id"))
        if player_id is None and player_name is not None:
            player_id = f"name:{player_name}"

        resource_cards_raw = raw.get("resource_cards")
        resource_cards = {resource: 0 for resource in TRACKED_RESOURCES}
        if isinstance(resource_cards_raw, dict):
            for resource in TRACKED_RESOURCES:
                resource_cards[resource] = _safe_int(resource_cards_raw.get(resource), 0) or 0

        ownership = _extract_board_ownership(board_state, player_id)

        summaries.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "victory_points_visible": _safe_int(raw.get("victory_points_visible") or raw.get("victory_points")),
                "victory_point_breakdown_visible": raw.get("victory_point_breakdown_visible"),
                "knights_played_visible": _safe_int(raw.get("knights_played_visible") or raw.get("numb_knights")),
                "largest_army_flag": bool(raw.get("largest_army_flag", False)),
                "resource_cards": resource_cards,
                "total_resource_cards": _safe_int(raw.get("total_resource_cards"), sum(resource_cards.values())) or 0,
                "development_cards": _normalize_dev_card_summary_from_snapshot(raw.get("development_cards")),
                "ports": _normalize_ports(raw.get("ports")),
                "remaining_pieces": raw.get("remaining_pieces") or {
                    "roads_remaining": None,
                    "settlements_remaining": None,
                    "cities_remaining": None,
                    "roads_placed_reported": None,
                    "settlements_placed_reported": None,
                    "cities_placed_reported": None,
                },
                "board_presence": {
                    "settlement_count": len(ownership["settlements"]),
                    "city_count": len(ownership["cities"]),
                    "road_count": len(ownership["roads"]),
                    "settlements": ownership["settlements"],
                    "cities": ownership["cities"],
                    "roads": ownership["roads"],
                },
            }
        )

    return summaries


def _extract_public_game_state(
    game: Any,
    board_state: Dict[str, Any],
    players: List[Dict[str, Any]],
) -> Dict[str, Any]:
    del board_state
    largest_army_owner_id: Optional[str] = None
    largest_army_owner_name: Optional[str] = None
    for summary in players:
        if summary.get("largest_army_flag"):
            largest_army_owner_id = summary.get("player_id")
            largest_army_owner_name = summary.get("player_name")
            break

    longest_road_owner_id: Optional[str] = None
    longest_road_owner_name: Optional[str] = None
    longest_road_length: Optional[int] = None

    board_obj = _call(game, "getBoard") if game is not None else None
    if board_obj is not None and players:
        live_players = _coerce_iterable_to_list(_call(game, "getPlayers"))
        live_by_id: Dict[str, Any] = {}
        for candidate in live_players:
            identity = _player_identity(candidate)
            pid = identity.get("player_id")
            if pid is not None and pid not in live_by_id:
                live_by_id[pid] = candidate

        best_length = -1
        best_player: Optional[Dict[str, Any]] = None
        tie = False
        for summary in players:
            pid = summary.get("player_id")
            match_player = live_by_id.get(pid) if pid is not None else None
            if match_player is None:
                continue

            length: Optional[int] = None
            try:
                length = _safe_int(board_obj.findLongestRoad(match_player), None)
            except Exception:
                length = None

            if length is None:
                continue

            if length > best_length:
                best_length = length
                best_player = summary
                tie = False
            elif length == best_length:
                tie = True

        if best_player is not None:
            longest_road_length = best_length
            if not tie:
                longest_road_owner_id = best_player.get("player_id")
                longest_road_owner_name = best_player.get("player_name")

    return {
        "largest_army_owner": {
            "player_id": largest_army_owner_id,
            "player_name": largest_army_owner_name,
        },
        "longest_road_owner": {
            "player_id": longest_road_owner_id,
            "player_name": longest_road_owner_name,
            "length": longest_road_length,
        },
        "visible_knight_counts": {
            summary.get("player_id") or summary.get("player_name"): summary.get("knights_played_visible")
            for summary in players
        },
    }


def _extract_supply_state(game: Any, players: List[Dict[str, Any]]) -> Dict[str, Any]:
    deck = _call(game, "getDeck") if game is not None else None

    remaining_dev_cards = None
    if deck is not None:
        # Current Java API exposes `isEmpty` but not exact count.
        is_empty = _call(deck, "isEmpty")
        if is_empty is True:
            remaining_dev_cards = 0

    remaining_piece_supply: Dict[str, Dict[str, Optional[int]]] = {}
    for summary in players:
        pid = summary.get("player_id") or summary.get("player_name")
        remaining_piece_supply[str(pid)] = summary.get("remaining_pieces", {})

    return {
        "bank_resource_cards_remaining": None,
        "development_cards_remaining": remaining_dev_cards,
        "player_piece_supply": remaining_piece_supply,
    }


def _reconcile_robber_flags(
    robber_norm: Optional[Dict[str, int]],
    tiles: List[Dict[str, Any]],
    issues: List[str],
) -> Optional[Dict[str, int]]:
    """Make per-tile robber flags consistent with top-level robber coordinates.

    Source of truth: top-level `robber` coordinates when valid (`getRobberLocation` /
    snapshot). Per-tile flags are overwritten to match. If top-level is missing or
    invalid and exactly one tile claims the robber, infer top-level from that tile.
    """
    effective = robber_norm
    if effective is not None:
        rx = effective.get("x")
        ry = effective.get("y")
        if rx is None or ry is None or not _in_bounds(int(rx), int(ry)):
            issues.append("Invalid top-level robber coordinates; attempting inference from tiles")
            effective = None

    if effective is not None:
        rx, ry = int(effective["x"]), int(effective["y"])
        coord_set = {(t.get("x"), t.get("y")) for t in tiles}
        if (rx, ry) not in coord_set:
            issues.append(
                f"Robber top-level position ({rx},{ry}) has no matching tile entry; per-tile flags cleared"
            )
        mismatch = False
        for t in tiles:
            want = bool(t.get("x") == rx and t.get("y") == ry)
            if bool(t.get("robber")) != want:
                mismatch = True
            t["robber"] = want
        if mismatch:
            issues.append("Robber per-tile flags repaired to match top-level robber coordinates")
        return effective

    claiming = [(t.get("x"), t.get("y")) for t in tiles if t.get("robber")]
    if len(claiming) == 1:
        x, y = claiming[0]
        if x is not None and y is not None and _in_bounds(int(x), int(y)):
            issues.append(
                "Robber location inferred from single per-tile robber flag (top-level robber missing or invalid)"
            )
            inferred = {"x": int(x), "y": int(y)}
            for t in tiles:
                t["robber"] = bool(t.get("x") == inferred["x"] and t.get("y") == inferred["y"])
            return inferred

    if len(claiming) > 1:
        issues.append(
            "Conflicting robber flags on multiple tiles; clearing all (top-level robber missing or invalid)"
        )
    for t in tiles:
        t["robber"] = False
    return None


def _extract_board_from_live(board: Any) -> Dict[str, Any]:
    tiles_grid = _coerce_2d_grid(_call(board, "getTiles") or [])
    structures_grid = _coerce_3d_grid(_call(board, "getStructures") or [])
    roads_grid = _coerce_3d_grid(_call(board, "getRoads") or [])

    robber = _location_to_dict(_call(board, "getRobberLocation"))

    tiles: List[Dict[str, Any]] = []
    valid_tile_count = 0
    for x_coord, column in enumerate(tiles_grid):
        for y_coord, tile in enumerate(column or []):
            if tile is None:
                continue

            tile_type = _normalize_resource_name(_call(tile, "getType"))
            if tile_type is None:
                continue

            valid_tile_count += 1
            tiles.append(
                {
                    "x": x_coord,
                    "y": y_coord,
                    "tile_id": f"tile:{x_coord}:{y_coord}",
                    "type": tile_type,
                    "number": _safe_int(_call(tile, "getNumber")),
                    "robber": bool(_call(tile, "hasRobber", False)),
                    "adjacent_structure_vertices": _tile_adjacent_vertices(x_coord, y_coord),
                }
            )

    structures: List[Dict[str, Any]] = []
    for x_coord, column in enumerate(structures_grid):
        for y_coord, row in enumerate(column or []):
            for orientation, structure in enumerate(row or []):
                if structure is None:
                    continue

                owner_obj = _call(structure, "getOwner")
                if owner_obj is None:
                    continue

                owner = _player_identity(owner_obj)
                structures.append(
                    {
                        "x": x_coord,
                        "y": y_coord,
                        "orientation": orientation,
                        "structure_id": f"structure:{x_coord}:{y_coord}:{orientation}",
                        "type": _normalize_structure_type(_call(structure, "getType")),
                        "owner_id": owner["player_id"],
                        "owner_name": owner["player_name"],
                        "adjacent_tiles": _vertex_adjacent_tiles(x_coord, y_coord, orientation),
                        "incident_roads": _vertex_incident_roads(x_coord, y_coord, orientation),
                    }
                )

    roads: List[Dict[str, Any]] = []
    for x_coord, column in enumerate(roads_grid):
        for y_coord, row in enumerate(column or []):
            for orientation, road in enumerate(row or []):
                if road is None:
                    continue

                owner_obj = _call(road, "getOwner")
                if owner_obj is None:
                    continue

                owner = _player_identity(owner_obj)
                roads.append(
                    {
                        "x": x_coord,
                        "y": y_coord,
                        "orientation": orientation,
                        "road_id": f"road:{x_coord}:{y_coord}:{orientation}",
                        "owner_id": owner["player_id"],
                        "owner_name": owner["player_name"],
                        "endpoint_vertices": _road_endpoint_vertices(x_coord, y_coord, orientation),
                    }
                )

    result: Dict[str, Any] = {
        "robber": robber,
        "tiles": tiles,
        "structures": structures,
        "roads": roads,
        "board_counts": {
            "valid_tile_count": valid_tile_count,
            "occupied_structure_count": len(structures),
            "occupied_road_count": len(roads),
        },
        "adjacency_support": {
            "tile_to_vertices": "included",
            "structure_to_tiles_and_roads": "included",
            "road_to_endpoint_vertices": "included",
            "derivation_hooks": "coordinate_helpers",
            "trust_level": "coordinate_derived_match_java_grid_layout",
            "validation_recommendation": "compare_against_known_snapshots_or_board_unit_tests",
        },
        "engine_layout_policy": {
            "source_of_truth": "live_engine_output",
            "valid_mask_reference": ENGINE_BOARD_LAYOUT_ID,
            "out_of_mask_tiles_policy": "keep_and_flag",
        },
    }

    out_of_mask_tiles = [
        {"x": tile["x"], "y": tile["y"], "tile_id": tile["tile_id"]}
        for tile in tiles
        if (tile["x"], tile["y"]) not in VALID_TILE_COORDINATES
    ]
    if out_of_mask_tiles:
        result["board_counts"]["out_of_mask_tile_count"] = len(out_of_mask_tiles)
        result["board_counts"]["out_of_mask_tiles"] = out_of_mask_tiles

    live_notes: List[str] = []
    result["robber"] = _reconcile_robber_flags(robber, tiles, live_notes)
    if live_notes:
        result.setdefault("board_counts", {})["robber_reconcile_notes"] = live_notes
    return result


def _validate_and_normalize_snapshot(snapshot: Dict[str, Any], strict: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    normalized = dict(snapshot)
    normalized["players"] = _coerce_iterable_to_list(snapshot.get("players"))

    robber = snapshot.get("robber")
    robber_norm: Optional[Dict[str, int]] = None
    if isinstance(robber, dict):
        rx = _safe_int(robber.get("x"))
        ry = _safe_int(robber.get("y"))
        if rx is None or ry is None or not _in_bounds(rx, ry):
            issues.append("Invalid robber coordinates; set to None")
        else:
            robber_norm = {"x": rx, "y": ry}
    elif robber is not None:
        issues.append("Robber field must be dict or None")

    seen_tile_ids: Set[str] = set()
    normalized_tiles: List[Dict[str, Any]] = []
    for index, raw in enumerate(_coerce_iterable_to_list(snapshot.get("tiles"))):
        if not isinstance(raw, dict):
            issues.append(f"Tile[{index}] ignored: expected dict")
            continue
        x_coord = _safe_int(raw.get("x"))
        y_coord = _safe_int(raw.get("y"))
        if x_coord is None or y_coord is None or not _in_bounds(x_coord, y_coord):
            issues.append(f"Tile[{index}] ignored: invalid coordinates")
            continue

        tile_type = _normalize_resource_name(raw.get("type"))
        if tile_type == "UNKNOWN":
            issues.append(f"Tile[{index}] has unknown resource type")

        tile_id = f"tile:{x_coord}:{y_coord}"
        if tile_id in seen_tile_ids:
            msg = f"Duplicate tile_id {tile_id}; keeping first occurrence"
            issues.append(msg)
            continue
        seen_tile_ids.add(tile_id)

        normalized_tiles.append(
            {
                "x": x_coord,
                "y": y_coord,
                "tile_id": tile_id,
                "type": tile_type,
                "number": _safe_int(raw.get("number")),
                "robber": bool(raw.get("robber", False)),
                "adjacent_structure_vertices": _tile_adjacent_vertices(x_coord, y_coord),
            }
        )

    seen_structure_ids: Set[str] = set()
    normalized_structures: List[Dict[str, Any]] = []
    for index, raw in enumerate(_coerce_iterable_to_list(snapshot.get("structures"))):
        if not isinstance(raw, dict):
            issues.append(f"Structure[{index}] ignored: expected dict")
            continue

        x_coord = _safe_int(raw.get("x"))
        y_coord = _safe_int(raw.get("y"))
        orientation = _safe_int(raw.get("orientation"))
        if x_coord is None or y_coord is None or not _in_bounds(x_coord, y_coord):
            issues.append(f"Structure[{index}] ignored: invalid coordinates")
            continue
        if orientation is None or not (0 <= orientation < STRUCTURE_ORIENTATION_COUNT):
            issues.append(f"Structure[{index}] ignored: invalid orientation")
            continue

        owner = _owner_ref_from_snapshot(
            raw.get("owner_ref")
            or {
                "owner_id": raw.get("owner_id"),
                "owner_name": raw.get("owner_name") or raw.get("owner"),
            }
        )

        structure_id = f"structure:{x_coord}:{y_coord}:{orientation}"
        if structure_id in seen_structure_ids:
            issues.append(f"Duplicate structure_id {structure_id}; keeping first occurrence")
            continue
        seen_structure_ids.add(structure_id)

        normalized_structures.append(
            {
                "x": x_coord,
                "y": y_coord,
                "orientation": orientation,
                "structure_id": structure_id,
                "type": _normalize_structure_type(raw.get("type")),
                "owner_id": owner["owner_id"],
                "owner_name": owner["owner_name"],
                "adjacent_tiles": _vertex_adjacent_tiles(x_coord, y_coord, orientation),
                "incident_roads": _vertex_incident_roads(x_coord, y_coord, orientation),
            }
        )

    seen_road_ids: Set[str] = set()
    normalized_roads: List[Dict[str, Any]] = []
    for index, raw in enumerate(_coerce_iterable_to_list(snapshot.get("roads"))):
        if not isinstance(raw, dict):
            issues.append(f"Road[{index}] ignored: expected dict")
            continue

        x_coord = _safe_int(raw.get("x"))
        y_coord = _safe_int(raw.get("y"))
        orientation = _safe_int(raw.get("orientation"))
        if x_coord is None or y_coord is None or not _in_bounds(x_coord, y_coord):
            issues.append(f"Road[{index}] ignored: invalid coordinates")
            continue
        if orientation is None or not (0 <= orientation < ROAD_ORIENTATION_COUNT):
            issues.append(f"Road[{index}] ignored: invalid orientation")
            continue

        owner = _owner_ref_from_snapshot(
            raw.get("owner_ref")
            or {
                "owner_id": raw.get("owner_id"),
                "owner_name": raw.get("owner_name") or raw.get("owner"),
            }
        )

        road_id = f"road:{x_coord}:{y_coord}:{orientation}"
        if road_id in seen_road_ids:
            issues.append(f"Duplicate road_id {road_id}; keeping first occurrence")
            continue
        seen_road_ids.add(road_id)

        normalized_roads.append(
            {
                "x": x_coord,
                "y": y_coord,
                "orientation": orientation,
                "road_id": road_id,
                "owner_id": owner["owner_id"],
                "owner_name": owner["owner_name"],
                "endpoint_vertices": _road_endpoint_vertices(x_coord, y_coord, orientation),
            }
        )

    normalized["tiles"] = normalized_tiles
    normalized["structures"] = normalized_structures
    normalized["roads"] = normalized_roads
    normalized["robber"] = _reconcile_robber_flags(robber_norm, normalized_tiles, issues)

    if strict and issues:
        raise ValueError("Snapshot validation failed: " + "; ".join(issues))

    return normalized, issues


def _order_players_for_ml(
    players: List[Dict[str, Any]],
    current_player_id: Optional[str],
    current_player_name: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Current player first; remaining players sorted by (player_id, player_name)."""
    current = _select_current_player(players, current_player_id, current_player_name)

    def sort_key(p: Dict[str, Any]) -> Tuple[str, str]:
        return (p.get("player_id") or "", p.get("player_name") or "")

    if current is None:
        ordered = sorted(players, key=sort_key)
        return ordered, {
            "ordering_rule": "current_player_first_then_opponents_sorted_by_player_id_then_name",
            "current_player_index": None,
            "viewer_player_index": None,
            "ordered_player_ids": [p.get("player_id") for p in ordered],
            "ordered_player_names": [p.get("player_name") for p in ordered],
        }

    others = [p for p in players if p is not current]
    others_sorted = sorted(others, key=sort_key)
    ordered = [current] + others_sorted
    return ordered, {
        "ordering_rule": "current_player_first_then_opponents_sorted_by_player_id_then_name",
        "current_player_index": 0,
        "viewer_player_index": 0,
        "ordered_player_ids": [p.get("player_id") for p in ordered],
        "ordered_player_names": [p.get("player_name") for p in ordered],
    }


def _compute_state_id(state: Dict[str, Any]) -> str:
    """Deterministic id for a normalized state payload (replay, dedup, deltas)."""
    raw = json.dumps(state, sort_keys=True, default=str)
    data = json.loads(raw)
    turn_meta = data.get("turn_metadata")
    if isinstance(turn_meta, dict):
        turn_meta.pop("state_id", None)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _compute_position_id(state: Dict[str, Any]) -> str:
    """Deterministic id for board position + public turn anchor, excluding noisy metadata.

    Excludes validation notes and derived-feature placeholders so equivalent positions
    hash identically for replay/dedup workflows.
    """

    canonical = {
        "robber": state.get("robber"),
        "tiles": state.get("tiles", []),
        "structures": state.get("structures", []),
        "roads": state.get("roads", []),
        "turn_anchor": {
            "turn_index": (state.get("turn_metadata") or {}).get("turn_index"),
            "current_player_id": (state.get("turn_metadata") or {}).get("current_player_id"),
            "current_phase": (state.get("turn_metadata") or {}).get("current_phase"),
        },
        "players_public": {
            "all_players": [
                {
                    "player_id": player.get("player_id"),
                    "player_name": player.get("player_name"),
                    "total_resource_cards": player.get("total_resource_cards"),
                    "dev_total_cards": (player.get("development_cards") or {}).get("total_cards")
                    if isinstance(player.get("development_cards"), dict)
                    else None,
                    "knights_played_visible": player.get("knights_played_visible"),
                    "largest_army_flag": player.get("largest_army_flag"),
                    "ports": player.get("ports"),
                    "remaining_pieces": player.get("remaining_pieces"),
                    "board_presence": player.get("board_presence"),
                }
                for player in (state.get("players") or {}).get("all_players", [])
                if isinstance(player, dict)
            ]
        },
    }

    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _apply_state_ids(omniscient: Dict[str, Any], observed: Dict[str, Any]) -> None:
    omniscient_meta = omniscient.setdefault("turn_metadata", {})
    observed_meta = observed.setdefault("turn_metadata", {})

    omniscient_meta["state_id"] = _compute_state_id(omniscient)
    observed_meta["state_id"] = _compute_state_id(observed)
    omniscient_meta["position_id"] = _compute_position_id(omniscient)
    observed_meta["position_id"] = _compute_position_id(observed)


def _select_current_player(
    players: List[Dict[str, Any]],
    current_player_id: Optional[str],
    current_player_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    for summary in players:
        if current_player_id and summary.get("player_id") == current_player_id:
            return summary
    for summary in players:
        if current_player_name and summary.get("player_name") == current_player_name:
            return summary
    return players[0] if players else None


def _project_observed_players(
    players: List[Dict[str, Any]],
    viewer_player_id: Optional[str],
    reveal_private: bool,
) -> List[Dict[str, Any]]:
    # Public-info policy:
    # - Preserve opponent `total_resource_cards` as visible hand size in standard Catan.
    # - Preserve opponent dev-card `total_cards` as publicly trackable aggregate.
    # Hidden buckets (resource types and dev-card types) remain redacted.
    if reveal_private:
        output = []
        for player in players:
            clone = dict(player)
            clone["hidden_from_viewer"] = False
            output.append(clone)
        return output

    observed: List[Dict[str, Any]] = []
    for player in players:
        clone = dict(player)
        is_viewer = viewer_player_id is not None and player.get("player_id") == viewer_player_id
        if is_viewer:
            clone["hidden_from_viewer"] = False
            observed.append(clone)
            continue

        clone["resource_cards"] = None
        clone["development_cards"] = {
            "schema_version": DEV_CARD_SCHEMA_VERSION,
            "grouped_counts": {
                "knight": None,
                "progress": None,
                "victory_point": None,
            },
            "progress_subtype_counts": {
                "road_building": None,
                "year_of_plenty": None,
                "monopoly": None,
            },
            "total_cards": player.get("development_cards", {}).get("total_cards") if isinstance(player.get("development_cards"), dict) else None,
            "counting_assumption": "progress_subtypes_are_subset_of_grouped_progress",
        }
        clone["hidden_from_viewer"] = True
        observed.append(clone)

    return observed


def _aggregate_resource_totals(players: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    totals = {resource: 0 for resource in TRACKED_RESOURCES}
    has_full_visibility = True

    for summary in players:
        resources = summary.get("resource_cards")
        if not isinstance(resources, dict):
            has_full_visibility = False
            continue
        for resource in TRACKED_RESOURCES:
            totals[resource] += _safe_int(resources.get(resource), 0) or 0

    if not has_full_visibility:
        return {resource: None for resource in TRACKED_RESOURCES}
    return totals


def _assemble_state_payload(
    board_state: Dict[str, Any],
    players: List[Dict[str, Any]],
    turn_metadata: Dict[str, Any],
    public_game_state: Dict[str, Any],
    supply_state: Dict[str, Any],
    transition_context: Dict[str, Any],
    validation_issues: Optional[List[str]] = None,
    player_ordering_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resource_totals = _aggregate_resource_totals(players)
    total_resources = None
    if all(value is not None for value in resource_totals.values()):
        total_resources = sum(value for value in resource_totals.values() if value is not None)

    return {
        "turn_metadata": turn_metadata,
        "public_game_state": public_game_state,
        "supply_state": supply_state,
        "transition_context": transition_context,
        "robber": board_state["robber"],
        "tiles": board_state["tiles"],
        "structures": board_state["structures"],
        "roads": board_state["roads"],
        "board_counts": board_state.get("board_counts", {}),
        "adjacency_support": board_state.get("adjacency_support", {}),
        "players": {
            "all_players": players,
            "player_ordering": player_ordering_meta or {},
            "resources_in_play": resource_totals,
            "total_resources_in_play": total_resources,
            "total_known_development_cards_in_play": sum(
                (_safe_int(p.get("development_cards", {}).get("total_cards"), 0) or 0)
                for p in players
            ),
        },
        "derived_features": build_derived_features_placeholder(),
        "validation": {
            "issues": validation_issues or [],
        },
    }


def _build_output_payload(
    omniscient_state: Dict[str, Any],
    observed_state: Dict[str, Any],
    top_level_mirror: str = "observed",
) -> Dict[str, Any]:
    # Backward-compatible mirrors at top level default to observed_state (non-cheating for agents).
    if top_level_mirror not in ("observed", "omniscient"):
        top_level_mirror = "observed"
    mirror = observed_state if top_level_mirror == "observed" else omniscient_state
    omni_tm = omniscient_state.get("turn_metadata") or {}
    obs_tm = observed_state.get("turn_metadata") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "top_level_mirror": top_level_mirror,
        "state_ids": {
            "omniscient": omni_tm.get("state_id"),
            "observed": obs_tm.get("state_id"),
        },
        "position_ids": {
            "omniscient": omni_tm.get("position_id"),
            "observed": obs_tm.get("position_id"),
        },
        "encoding_metadata": {
            "vectorization_ready": True,
            "slot_ordering": _build_slot_ordering_metadata(),
            "resource_order": list(TRACKED_RESOURCES),
            "structure_type_order": list(STRUCTURE_TYPES),
            "dev_card_schema_version": DEV_CARD_SCHEMA_VERSION,
            "engine_board_layout": ENGINE_BOARD_LAYOUT_ID,
        },
        "omniscient_state": omniscient_state,
        "observed_state": observed_state,
        "robber": mirror.get("robber"),
        "tiles": mirror.get("tiles", []),
        "structures": mirror.get("structures", []),
        "roads": mirror.get("roads", []),
        "players": mirror.get("players", {}),
    }


def write_state_json(state: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    return output_path


def extract_board_state(board: Any) -> Dict[str, Any]:
    """Backward-compatible raw board extraction from live board object."""
    return _extract_board_from_live(board)


def extract_board_state_from_game(game: Any) -> Dict[str, Any]:
    board = _call(game, "getBoard")
    if board is None:
        raise ValueError("The supplied object does not expose a getBoard() method")
    return extract_board_state(board)


def extract_full_state_from_game(
    game: Any,
    players: Optional[Iterable[Any]] = None,
    current_player: Optional[Any] = None,
    current_player_name: Optional[str] = None,
    current_player_id: Optional[str] = None,
    reveal_private: bool = False,
    top_level_mirror: str = "observed",
) -> Dict[str, Any]:
    """Extract full state from a live game.

    `top_level_mirror`: ``\"observed\"`` (default, safe for agents) or ``\"omniscient\"``
    for backward-compatible top-level fields that mirror full information.
    """
    board_state = extract_board_state_from_game(game)
    player_summaries = _extract_players_from_game(game, board_state, players=players)

    selected_current_player = _player_identity(current_player) if current_player is not None else {"player_id": None, "player_name": None}

    turn_metadata = _extract_turn_metadata(game=game)
    turn_metadata["current_player_id"] = (
        current_player_id
        or selected_current_player.get("player_id")
        or turn_metadata.get("current_player_id")
    )
    turn_metadata["current_player_name"] = (
        current_player_name
        or selected_current_player.get("player_name")
        or turn_metadata.get("current_player_name")
    )

    current_summary = _select_current_player(
        player_summaries,
        turn_metadata.get("current_player_id"),
        turn_metadata.get("current_player_name"),
    )
    if current_summary is not None:
        turn_metadata["current_player_id"] = current_summary.get("player_id")
        turn_metadata["current_player_name"] = current_summary.get("player_name")

    ordered_players, player_ordering_meta = _order_players_for_ml(
        player_summaries,
        turn_metadata.get("current_player_id"),
        turn_metadata.get("current_player_name"),
    )

    public_game_state = _extract_public_game_state(game, board_state, ordered_players)
    supply_state = _extract_supply_state(game, ordered_players)
    transition_context = _extract_transition_context(game=game)

    omniscient = _assemble_state_payload(
        board_state=board_state,
        players=ordered_players,
        turn_metadata=turn_metadata,
        public_game_state=public_game_state,
        supply_state=supply_state,
        transition_context=transition_context,
        player_ordering_meta=player_ordering_meta,
    )

    viewer_player_id = turn_metadata.get("current_player_id")
    observed_players = _project_observed_players(
        ordered_players,
        viewer_player_id=viewer_player_id,
        reveal_private=reveal_private,
    )

    observed = _assemble_state_payload(
        board_state=board_state,
        players=observed_players,
        turn_metadata=turn_metadata,
        public_game_state=public_game_state,
        supply_state=supply_state,
        transition_context=transition_context,
        player_ordering_meta=player_ordering_meta,
    )

    _apply_state_ids(omniscient, observed)
    return _build_output_payload(omniscient, observed, top_level_mirror=top_level_mirror)


def extract_board_state_from_snapshot(
    snapshot: Dict[str, Any],
    current_player_name: Optional[str] = None,
    current_player_id: Optional[str] = None,
    reveal_private: bool = False,
    strict_validation: bool = False,
    top_level_mirror: str = "observed",
) -> Dict[str, Any]:
    """Load state from a JSON snapshot. Default top-level mirrors match ``observed_state``."""
    normalized_snapshot, issues = _validate_and_normalize_snapshot(snapshot, strict=strict_validation)

    board_state = {
        "robber": normalized_snapshot.get("robber"),
        "tiles": normalized_snapshot.get("tiles", []),
        "structures": normalized_snapshot.get("structures", []),
        "roads": normalized_snapshot.get("roads", []),
        "board_counts": {
            "valid_tile_count": len(normalized_snapshot.get("tiles", [])),
            "occupied_structure_count": len(normalized_snapshot.get("structures", [])),
            "occupied_road_count": len(normalized_snapshot.get("roads", [])),
        },
        "adjacency_support": {
            "tile_to_vertices": "included",
            "structure_to_tiles_and_roads": "included",
            "road_to_endpoint_vertices": "included",
            "derivation_hooks": "coordinate_helpers",
            "trust_level": "coordinate_derived_match_java_grid_layout",
            "validation_recommendation": "compare_against_known_snapshots_or_board_unit_tests",
        },
    }

    player_summaries = _extract_players_from_snapshot(normalized_snapshot.get("players", []), board_state)

    turn_metadata = _extract_turn_metadata(snapshot=normalized_snapshot)
    turn_metadata["current_player_id"] = current_player_id or turn_metadata.get("current_player_id")
    turn_metadata["current_player_name"] = current_player_name or turn_metadata.get("current_player_name")

    current_summary = _select_current_player(
        player_summaries,
        turn_metadata.get("current_player_id"),
        turn_metadata.get("current_player_name"),
    )
    if current_summary is not None:
        turn_metadata["current_player_id"] = current_summary.get("player_id")
        turn_metadata["current_player_name"] = current_summary.get("player_name")

    ordered_players, player_ordering_meta = _order_players_for_ml(
        player_summaries,
        turn_metadata.get("current_player_id"),
        turn_metadata.get("current_player_name"),
    )

    public_game_state = (
        normalized_snapshot.get("public_game_state")
        if isinstance(normalized_snapshot.get("public_game_state"), dict)
        else {
            "largest_army_owner": {"player_id": None, "player_name": None},
            "longest_road_owner": {"player_id": None, "player_name": None, "length": None},
            "visible_knight_counts": {
                summary.get("player_id") or summary.get("player_name"): summary.get("knights_played_visible")
                for summary in ordered_players
            },
        }
    )

    supply_state = (
        normalized_snapshot.get("supply_state")
        if isinstance(normalized_snapshot.get("supply_state"), dict)
        else {
            "bank_resource_cards_remaining": None,
            "development_cards_remaining": None,
            "player_piece_supply": {
                (summary.get("player_id") or summary.get("player_name")): summary.get("remaining_pieces")
                for summary in ordered_players
            },
        }
    )

    transition_context = _extract_transition_context(snapshot=normalized_snapshot)

    omniscient = _assemble_state_payload(
        board_state=board_state,
        players=ordered_players,
        turn_metadata=turn_metadata,
        public_game_state=public_game_state,
        supply_state=supply_state,
        transition_context=transition_context,
        validation_issues=issues,
        player_ordering_meta=player_ordering_meta,
    )

    observed_players = _project_observed_players(
        ordered_players,
        viewer_player_id=turn_metadata.get("current_player_id"),
        reveal_private=reveal_private,
    )

    observed = _assemble_state_payload(
        board_state=board_state,
        players=observed_players,
        turn_metadata=turn_metadata,
        public_game_state=public_game_state,
        supply_state=supply_state,
        transition_context=transition_context,
        validation_issues=issues,
        player_ordering_meta=player_ordering_meta,
    )

    _apply_state_ids(omniscient, observed)
    return _build_output_payload(omniscient, observed, top_level_mirror=top_level_mirror)


def load_snapshot(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Catan state into omniscient + observed ML-ready JSON."
    )
    parser.add_argument(
        "snapshot",
        nargs="?",
        help="Path to JSON snapshot. For live game objects, import this module and call extract_full_state_from_game().",
    )
    parser.add_argument(
        "--output",
        default="ai/board_state.json",
        help="Output JSON file path (default: ai/board_state.json).",
    )
    parser.add_argument(
        "--current-player",
        default=None,
        help="Current player name for observed projection.",
    )
    parser.add_argument(
        "--current-player-id",
        default=None,
        help="Current player stable ID for observed projection.",
    )
    parser.add_argument(
        "--reveal-private",
        action="store_true",
        help="Show full hidden opponent info in observed state (debug only).",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail on snapshot validation issues instead of normalizing and continuing.",
    )
    parser.add_argument(
        "--top-level-omniscient",
        action="store_true",
        help="Mirror omniscient_state at top level (offline ML only; default is observed).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the extracted state.")
    args = parser.parse_args()

    if not args.snapshot:
        parser.error(
            "Provide a snapshot path, or import this module and call extract_full_state_from_game()."
        )

    snapshot = load_snapshot(Path(args.snapshot))

    state = extract_board_state_from_snapshot(
        snapshot,
        current_player_name=args.current_player,
        current_player_id=args.current_player_id,
        reveal_private=args.reveal_private,
        strict_validation=args.strict_validation,
        top_level_mirror="omniscient" if args.top_level_omniscient else "observed",
    )

    output_path = write_state_json(state, Path(args.output))

    if args.pretty:
        print(json.dumps(state, indent=2, sort_keys=True))
    else:
        print(json.dumps(state, separators=(",", ":"), sort_keys=True))

    print(f"\nWrote state to {output_path}")


if __name__ == "__main__":
    main()
