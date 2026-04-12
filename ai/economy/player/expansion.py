from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from ai.economy.constants import (
    MAX_SINGLE_VERTEX_PIPS,
    PIP_WEIGHTS,
    TRACKED_RESOURCES,
    VALID_STRUCTURE_SLOTS,
    adjacent_vertices_for_vertex,
    vertex_adjacent_tiles,
    vertex_id,
    vertex_incident_roads,
)
from ai.economy.normalize import clamp01, score_from_cap


def _valid_structure_slots(slot_ordering: Optional[Dict[str, object]]) -> List[Tuple[int, int, int]]:
    if not isinstance(slot_ordering, dict):
        return list(VALID_STRUCTURE_SLOTS)
    raw_slots = slot_ordering.get("structure_slots")
    raw_valid = slot_ordering.get("structure_slot_valid")
    if not isinstance(raw_slots, list) or not isinstance(raw_valid, list):
        return list(VALID_STRUCTURE_SLOTS)
    output: List[Tuple[int, int, int]] = []
    for entry, is_valid in zip(raw_slots, raw_valid):
        if not is_valid or not isinstance(entry, dict):
            continue
        x_coord = entry.get("x")
        y_coord = entry.get("y")
        orientation = entry.get("orientation")
        if isinstance(x_coord, int) and isinstance(y_coord, int) and isinstance(orientation, int):
            output.append((x_coord, y_coord, orientation))
    return output or list(VALID_STRUCTURE_SLOTS)


def _local_vertex_ev(
    x_coord: int,
    y_coord: int,
    orientation: int,
    tile_map: Dict[Tuple[int, int], Dict[str, object]],
) -> float:
    total = 0.0
    for tile_coord in vertex_adjacent_tiles(x_coord, y_coord, orientation):
        tile = tile_map.get((tile_coord["x"], tile_coord["y"]))
        if tile is None:
            continue
        total += float(tile.get("pip_weight", 0.0))
    return total


def build_expansion_profile(
    state: Dict[str, object],
    player_id: Optional[str],
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    valid_slots = _valid_structure_slots(slot_ordering)
    tile_map = {
        (int(tile["x"]), int(tile["y"])): {
            **tile,
            "pip_weight": float(PIP_WEIGHTS.get(tile.get("number"), 0.0)) if isinstance(tile.get("number"), int) else 0.0,
        }
        for tile in state.get("tiles", [])
        if isinstance(tile, dict) and isinstance(tile.get("x"), int) and isinstance(tile.get("y"), int)
    }

    occupied_vertices = {
        (int(structure["x"]), int(structure["y"]), int(structure["orientation"])): structure
        for structure in state.get("structures", [])
        if isinstance(structure, dict)
        and isinstance(structure.get("x"), int)
        and isinstance(structure.get("y"), int)
        and isinstance(structure.get("orientation"), int)
    }
    occupied_neighbors: Set[Tuple[int, int, int]] = set()
    for x_coord, y_coord, orientation in occupied_vertices.keys():
        for neighbor in adjacent_vertices_for_vertex(x_coord, y_coord, orientation):
            occupied_neighbors.add((neighbor["x"], neighbor["y"], neighbor["orientation"]))

    owned_roads = {
        (int(road["x"]), int(road["y"]), int(road["orientation"]))
        for road in state.get("roads", [])
        if isinstance(road, dict)
        and road.get("owner_id") == player_id
        and isinstance(road.get("x"), int)
        and isinstance(road.get("y"), int)
        and isinstance(road.get("orientation"), int)
    }

    reachable_sites: List[Dict[str, object]] = []
    for x_coord, y_coord, orientation in valid_slots:
        key = (x_coord, y_coord, orientation)
        if key in occupied_vertices or key in occupied_neighbors:
            continue
        reachable = False
        for incident in vertex_incident_roads(x_coord, y_coord, orientation):
            road_key = (incident["x"], incident["y"], incident["orientation"])
            if road_key in owned_roads:
                reachable = True
                break
        if not reachable:
            continue
        local_ev = _local_vertex_ev(x_coord, y_coord, orientation, tile_map)
        reachable_sites.append(
            {
                "vertex_id": vertex_id(x_coord, y_coord, orientation),
                "ev_score": local_ev,
            }
        )
    reachable_sites.sort(key=lambda item: (-float(item["ev_score"]), str(item["vertex_id"])))

    owned_settlements = [
        structure
        for structure in state.get("structures", [])
        if isinstance(structure, dict)
        and structure.get("owner_id") == player_id
        and structure.get("type") == "settlement"
    ]
    upgrade_values = [
        _local_vertex_ev(int(structure["x"]), int(structure["y"]), int(structure["orientation"]), tile_map)
        for structure in owned_settlements
    ]

    top_reachable_ev = float(reachable_sites[0]["ev_score"]) if reachable_sites else 0.0
    expansion_score = clamp01(
        0.6 * score_from_cap(len(reachable_sites), 4.0)
        + 0.4 * score_from_cap(top_reachable_ev, MAX_SINGLE_VERTEX_PIPS)
    )
    return {
        "reachable_site_count": len(reachable_sites),
        "top_reachable_site_ev": top_reachable_ev,
        "best_reachable_site_ids": [site["vertex_id"] for site in reachable_sites[:3]],
        "score": expansion_score,
        "reachable_sites": reachable_sites[:5],
        "upgradeable_settlement_count": len(upgrade_values),
        "top_upgradeable_settlement_ev": max(upgrade_values) if upgrade_values else 0.0,
        "mean_upgradeable_settlement_ev": (sum(upgrade_values) / len(upgrade_values)) if upgrade_values else 0.0,
    }
