from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ai.economy.board.ports import port_alignment_score, port_fallback_score, port_for_vertex
from ai.economy.constants import (
    BUILD_RECIPES,
    MAX_PAIR_PIPS,
    MAX_SINGLE_VERTEX_PIPS,
    OPENING_ARCHETYPE_WEIGHTS,
    OPENING_BALANCE_WEIGHTS,
    TRACKED_RESOURCES,
    VALID_STRUCTURE_SLOTS,
    adjacent_vertices_for_vertex,
    vertex_adjacent_tiles,
    vertex_id,
)
from ai.economy.normalize import clamp01, diversity_score, evenness_score, score_from_cap
from ai.economy.tags import opening_candidate_tags


def _coerce_valid_structure_slots(slot_ordering: Optional[Dict[str, object]]) -> List[Tuple[int, int, int]]:
    if not isinstance(slot_ordering, dict):
        return list(VALID_STRUCTURE_SLOTS)
    raw_slots = slot_ordering.get("structure_slots")
    raw_valid = slot_ordering.get("structure_slot_valid")
    if not isinstance(raw_slots, list) or not isinstance(raw_valid, list):
        return list(VALID_STRUCTURE_SLOTS)
    valid_slots: List[Tuple[int, int, int]] = []
    for entry, is_valid in zip(raw_slots, raw_valid):
        if not is_valid or not isinstance(entry, dict):
            continue
        x_coord = entry.get("x")
        y_coord = entry.get("y")
        orientation = entry.get("orientation")
        if isinstance(x_coord, int) and isinstance(y_coord, int) and isinstance(orientation, int):
            valid_slots.append((x_coord, y_coord, orientation))
    return valid_slots or list(VALID_STRUCTURE_SLOTS)


def _recipe_coverage_score(pip_by_resource: Dict[str, float]) -> float:
    total = 0.0
    for recipe in BUILD_RECIPES.values():
        resources = list(recipe.keys())
        if not resources:
            continue
        coverage = sum(1.0 for resource in resources if pip_by_resource.get(resource, 0.0) > 0.0)
        total += coverage / float(len(resources))
    return total / float(len(BUILD_RECIPES))


def _expansion_frontier(
    x_coord: int,
    y_coord: int,
    orientation: int,
    tile_map: Dict[Tuple[int, int], Dict[str, object]],
    valid_slots: Sequence[Tuple[int, int, int]],
) -> Tuple[int, float]:
    valid_slot_set = set(valid_slots)
    first_ring = adjacent_vertices_for_vertex(x_coord, y_coord, orientation)
    second_ring: Set[Tuple[int, int, int]] = set()
    for neighbor in first_ring:
        nx = neighbor["x"]
        ny = neighbor["y"]
        no = neighbor["orientation"]
        for second in adjacent_vertices_for_vertex(nx, ny, no):
            key = (second["x"], second["y"], second["orientation"])
            if key == (x_coord, y_coord, orientation) or key not in valid_slot_set:
                continue
            second_ring.add(key)

    best_total = 0.0
    for sx, sy, so in second_ring:
        total = 0.0
        for tile_coord in vertex_adjacent_tiles(sx, sy, so):
            tile = tile_map.get((tile_coord["x"], tile_coord["y"]))
            if tile is None:
                continue
            total += float(tile.get("pip_weight", 0.0))
        if total > best_total:
            best_total = total
    return len(second_ring), score_from_cap(best_total, MAX_SINGLE_VERTEX_PIPS)


def _resource_component_score(resources: Iterable[str], pip_by_resource: Dict[str, float], cap: float = 5.0) -> float:
    resource_list = list(resources)
    if not resource_list:
        return 0.0
    total = 0.0
    for resource in resource_list:
        total += score_from_cap(float(pip_by_resource.get(resource, 0.0)), cap)
    return total / float(len(resource_list))


def _candidate_archetype_scores(
    pip_by_resource: Dict[str, float],
    ev_score_norm: float,
    expansion_frontier_score: float,
    upgrade_quality_score: float,
    port_name: Optional[str],
    total_pips: float,
) -> Dict[str, float]:
    road_resource = _resource_component_score(("BRICK", "LUMBER"), pip_by_resource)
    settlement_resource = _resource_component_score(("BRICK", "LUMBER", "WOOL", "GRAIN"), pip_by_resource)
    city_resource = clamp01(
        0.5 * score_from_cap(float(pip_by_resource.get("ORE", 0.0)), 5.0)
        + 0.5 * score_from_cap(float(pip_by_resource.get("GRAIN", 0.0)), 5.0)
    )
    dev_resource = _resource_component_score(("ORE", "GRAIN", "WOOL"), pip_by_resource)
    port_align = port_alignment_score(port_name, pip_by_resource, total_pips)
    dev_port_fallback = port_fallback_score(port_name, pip_by_resource, total_pips)

    return {
        "road": clamp01(
            OPENING_ARCHETYPE_WEIGHTS["road"]["resource_component"] * road_resource
            + OPENING_ARCHETYPE_WEIGHTS["road"]["expansion_frontier"] * expansion_frontier_score
            + OPENING_ARCHETYPE_WEIGHTS["road"]["port_alignment"] * port_align
        ),
        "settlement": clamp01(
            OPENING_ARCHETYPE_WEIGHTS["settlement"]["resource_component"] * settlement_resource
            + OPENING_ARCHETYPE_WEIGHTS["settlement"]["expansion_frontier"] * expansion_frontier_score
            + OPENING_ARCHETYPE_WEIGHTS["settlement"]["ev"] * ev_score_norm
        ),
        "city": clamp01(
            OPENING_ARCHETYPE_WEIGHTS["city"]["resource_component"] * city_resource
            + OPENING_ARCHETYPE_WEIGHTS["city"]["ev"] * ev_score_norm
            + OPENING_ARCHETYPE_WEIGHTS["city"]["upgrade_quality"] * upgrade_quality_score
        ),
        "dev": clamp01(
            OPENING_ARCHETYPE_WEIGHTS["dev"]["resource_component"] * dev_resource
            + OPENING_ARCHETYPE_WEIGHTS["dev"]["ev"] * ev_score_norm
            + OPENING_ARCHETYPE_WEIGHTS["dev"]["port_fallback"] * dev_port_fallback
        ),
    }


def _build_single_candidates(
    tiles: Sequence[Dict[str, object]],
    slot_ordering: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    tile_map = {
        (int(tile["x"]), int(tile["y"])): {
            **tile,
            "pip_weight": float(tile.get("pip_weight", 0.0)),
        }
        for tile in tiles
        if isinstance(tile, dict) and isinstance(tile.get("x"), int) and isinstance(tile.get("y"), int)
    }
    valid_slots = _coerce_valid_structure_slots(slot_ordering)
    candidates: List[Dict[str, object]] = []

    for x_coord, y_coord, orientation in valid_slots:
        adjacent_tile_coords = vertex_adjacent_tiles(x_coord, y_coord, orientation)
        pip_by_resource = {resource: 0.0 for resource in TRACKED_RESOURCES}
        total_pips = 0.0
        for tile_coord in adjacent_tile_coords:
            tile = tile_map.get((tile_coord["x"], tile_coord["y"]))
            if tile is None:
                continue
            resource = str(tile.get("type") or "").upper()
            pip_weight = float(tile.get("pip_weight", 0.0))
            total_pips += pip_weight
            if resource in pip_by_resource:
                pip_by_resource[resource] += pip_weight

        diversity = diversity_score(pip_by_resource)
        evenness = evenness_score(pip_by_resource.values())
        recipe_coverage = _recipe_coverage_score(pip_by_resource)
        balance = clamp01(
            OPENING_BALANCE_WEIGHTS["diversity"] * diversity
            + OPENING_BALANCE_WEIGHTS["evenness"] * evenness
            + OPENING_BALANCE_WEIGHTS["recipe_coverage"] * recipe_coverage
        )
        expansion_frontier_count, expansion_frontier_score = _expansion_frontier(
            x_coord=x_coord,
            y_coord=y_coord,
            orientation=orientation,
            tile_map=tile_map,
            valid_slots=valid_slots,
        )
        port_name = port_for_vertex(x_coord, y_coord, orientation)
        ev_score_norm = score_from_cap(total_pips, MAX_SINGLE_VERTEX_PIPS)
        upgrade_quality_score = clamp01(
            0.6 * ev_score_norm
            + 0.4 * score_from_cap(
                max((float(value) for value in pip_by_resource.values()), default=0.0),
                5.0,
            )
        )
        archetype_scores = _candidate_archetype_scores(
            pip_by_resource=pip_by_resource,
            ev_score_norm=ev_score_norm,
            expansion_frontier_score=expansion_frontier_score,
            upgrade_quality_score=upgrade_quality_score,
            port_name=port_name,
            total_pips=total_pips,
        )
        tags = opening_candidate_tags(
            total_pips=total_pips,
            balance_score=balance,
            port=port_name,
            port_alignment_score=port_alignment_score(port_name, pip_by_resource, total_pips),
            archetype_scores=archetype_scores,
        )
        candidates.append(
            {
                "vertex_id": vertex_id(x_coord, y_coord, orientation),
                "x": x_coord,
                "y": y_coord,
                "orientation": orientation,
                "adjacent_tiles": adjacent_tile_coords,
                "total_pips": total_pips,
                "pip_by_resource": pip_by_resource,
                "diversity_score": diversity,
                "evenness_score": evenness,
                "recipe_coverage_score": recipe_coverage,
                "balance_score": balance,
                "expansion_frontier_count": expansion_frontier_count,
                "expansion_frontier_score": expansion_frontier_score,
                "upgrade_quality_score": upgrade_quality_score,
                "port": port_name,
                "port_alignment_score": port_alignment_score(port_name, pip_by_resource, total_pips),
                "ev_score": total_pips,
                "ev_score_norm": ev_score_norm,
                "archetype_scores": archetype_scores,
                "tags": tags,
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item.get("ev_score", 0.0)),
            -float(item.get("balance_score", 0.0)),
            str(item.get("vertex_id") or ""),
        )
    )
    return candidates


def _build_pair_rankings(candidates: Sequence[Dict[str, object]], limit: int = 12) -> Dict[str, List[Dict[str, object]]]:
    candidate_by_id = {str(candidate["vertex_id"]): candidate for candidate in candidates}
    pairs: List[Dict[str, object]] = []
    for index, left in enumerate(candidates):
        left_key = (int(left["x"]), int(left["y"]), int(left["orientation"]))
        left_neighbors = {
            (neighbor["x"], neighbor["y"], neighbor["orientation"])
            for neighbor in adjacent_vertices_for_vertex(*left_key)
        }
        for right in candidates[index + 1:]:
            right_key = (int(right["x"]), int(right["y"]), int(right["orientation"]))
            if right_key == left_key or right_key in left_neighbors:
                continue
            combined_pips = {
                resource: float(left["pip_by_resource"].get(resource, 0.0)) + float(right["pip_by_resource"].get(resource, 0.0))
                for resource in TRACKED_RESOURCES
            }
            total_pips = float(left["total_pips"]) + float(right["total_pips"])
            balance = clamp01(
                OPENING_BALANCE_WEIGHTS["diversity"] * diversity_score(combined_pips)
                + OPENING_BALANCE_WEIGHTS["evenness"] * evenness_score(combined_pips.values())
                + OPENING_BALANCE_WEIGHTS["recipe_coverage"] * _recipe_coverage_score(combined_pips)
            )
            pairs.append(
                {
                    "pair_id": f"{left['vertex_id']}|{right['vertex_id']}",
                    "vertex_ids": [left["vertex_id"], right["vertex_id"]],
                    "total_pips": total_pips,
                    "total_pips_norm": score_from_cap(total_pips, MAX_PAIR_PIPS),
                    "balance_score": balance,
                    "pip_by_resource": combined_pips,
                    "ports": [left.get("port"), right.get("port")],
                }
            )
    pairs.sort(
        key=lambda item: (
            -float(item.get("total_pips", 0.0)),
            -float(item.get("balance_score", 0.0)),
            str(item.get("pair_id") or ""),
        )
    )
    return {
        "top_by_ev": pairs[:limit],
        "top_by_balance": sorted(
            pairs,
            key=lambda item: (
                -float(item.get("balance_score", 0.0)),
                -float(item.get("total_pips", 0.0)),
                str(item.get("pair_id") or ""),
            ),
        )[:limit],
    }


def build_opening_analysis(
    tiles: Sequence[Dict[str, object]],
    slot_ordering: Optional[Dict[str, object]] = None,
    limit: int = 12,
) -> Dict[str, object]:
    candidates = _build_single_candidates(tiles=tiles, slot_ordering=slot_ordering)
    rankings = {
        "top_by_ev": candidates[:limit],
        "top_by_balance": sorted(
            candidates,
            key=lambda item: (
                -float(item.get("balance_score", 0.0)),
                -float(item.get("ev_score", 0.0)),
                str(item.get("vertex_id") or ""),
            ),
        )[:limit],
        "top_by_archetype": {
            archetype: sorted(
                candidates,
                key=lambda item: (
                    -float((item.get("archetype_scores") or {}).get(archetype, 0.0)),
                    -float(item.get("ev_score", 0.0)),
                    str(item.get("vertex_id") or ""),
                ),
            )[:limit]
            for archetype in OPENING_ARCHETYPE_WEIGHTS.keys()
        },
        "pairs": _build_pair_rankings(candidates, limit=limit),
    }
    return {
        "opening_candidates": candidates,
        "opening_rankings": rankings,
    }
