from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ai.economy.board.openings import build_opening_analysis
from ai.economy.board.ports import summarize_port_opportunity
from ai.economy.constants import (
    PIP_WEIGHTS,
    STANDARD_RESOURCE_BASELINE_PIPS,
    TRACKED_RESOURCES,
)
from ai.economy.normalize import clamp01, safe_div, score_from_cap
from ai.economy.tags import resource_environment_tags


def _tile_vertices(tile: Dict[str, object]) -> set[Tuple[int, int, int]]:
    x_coord = int(tile["x"])
    y_coord = int(tile["y"])
    return {
        (x_coord, y_coord, 0),
        (x_coord, y_coord, 1),
        (x_coord + 1, y_coord + 1, 1),
        (x_coord - 1, y_coord - 1, 0),
        (x_coord, y_coord + 1, 1),
        (x_coord, y_coord - 1, 0),
    }


def _tile_neighbors(tiles: Sequence[Dict[str, object]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    keyed_tiles = {
        (int(tile["x"]), int(tile["y"])): tile
        for tile in tiles
        if isinstance(tile, dict) and isinstance(tile.get("x"), int) and isinstance(tile.get("y"), int)
    }
    vertices = {coord: _tile_vertices(tile) for coord, tile in keyed_tiles.items()}
    neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {coord: [] for coord in keyed_tiles.keys()}
    items = list(keyed_tiles.keys())
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            if len(vertices[left].intersection(vertices[right])) >= 2:
                neighbors[left].append(right)
                neighbors[right].append(left)
    return neighbors


def _resource_specific_tags(resource: str, scarcity: float, abundance: float, concentration: float) -> List[str]:
    tags: List[str] = []
    prefix = resource.lower()
    if scarcity >= 0.6:
        tags.append(f"{prefix}_scarce")
    if abundance >= 0.6:
        tags.append(f"{prefix}_abundant")
    if concentration >= 0.65:
        tags.append(f"{prefix}_concentrated")
    return tags


def build_board_profile(
    tiles: Sequence[Dict[str, object]],
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    normalized_tiles: List[Dict[str, object]] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        x_coord = tile.get("x")
        y_coord = tile.get("y")
        if not isinstance(x_coord, int) or not isinstance(y_coord, int):
            continue
        number = tile.get("number")
        pip_weight = float(PIP_WEIGHTS.get(number, 0.0)) if isinstance(number, int) else 0.0
        normalized_tiles.append({**tile, "pip_weight": pip_weight})

    total_productive_pips = sum(
        float(tile["pip_weight"])
        for tile in normalized_tiles
        if str(tile.get("type") or "").upper() in TRACKED_RESOURCES
    )

    by_resource: Dict[str, Dict[str, object]] = {}
    for resource in TRACKED_RESOURCES:
        resource_tiles = [tile for tile in normalized_tiles if str(tile.get("type") or "").upper() == resource]
        pip_values = [float(tile["pip_weight"]) for tile in resource_tiles]
        pip_total = sum(pip_values)
        tile_count = len(resource_tiles)
        baseline = STANDARD_RESOURCE_BASELINE_PIPS[resource]
        pip_share = safe_div(pip_total, total_productive_pips)
        scarcity_score = clamp01(max(0.0, baseline - pip_total) / baseline) if baseline > 0.0 else 0.0
        abundance_score = clamp01(max(0.0, pip_total - baseline) / baseline) if baseline > 0.0 else 0.0
        mean_pips = safe_div(pip_total, float(tile_count))
        strong_number_share = safe_div(
            sum(float(tile["pip_weight"]) for tile in resource_tiles if tile.get("number") in {6, 8}),
            pip_total,
        )
        concentration_hhi = sum((safe_div(value, pip_total) ** 2) for value in pip_values if pip_total > 0.0)
        concentration_score = 0.0
        if tile_count > 1:
            min_hhi = 1.0 / float(tile_count)
            concentration_score = clamp01((concentration_hhi - min_hhi) / (1.0 - min_hhi))
        elif tile_count == 1:
            concentration_score = 1.0
        number_token_counts: Dict[str, int] = {}
        for tile in resource_tiles:
            key = str(tile.get("number"))
            number_token_counts[key] = number_token_counts.get(key, 0) + 1
        by_resource[resource] = {
            "resource": resource,
            "tile_count": tile_count,
            "pip_total": pip_total,
            "pip_share": pip_share,
            "baseline_pip_total": baseline,
            "scarcity_score": scarcity_score,
            "abundance_score": abundance_score,
            "mean_pips_per_tile": mean_pips,
            "strong_number_share": strong_number_share,
            "concentration_hhi": concentration_hhi,
            "concentration_score": concentration_score,
            "token_quality_score": score_from_cap(mean_pips, 5.0),
            "number_token_counts": number_token_counts,
            "tags": _resource_specific_tags(resource, scarcity_score, abundance_score, concentration_score),
        }

    board_tags = resource_environment_tags({resource: dict(stats) for resource, stats in by_resource.items()})
    strongest_resources = sorted(
        by_resource.keys(),
        key=lambda resource: (-float(by_resource[resource]["pip_total"]), resource),
    )
    weakest_resources = sorted(
        by_resource.keys(),
        key=lambda resource: (float(by_resource[resource]["pip_total"]), resource),
    )

    desert_tile = next((tile for tile in normalized_tiles if str(tile.get("type") or "").upper() == "DESERT"), None)
    neighbors = _tile_neighbors(normalized_tiles)
    desert_neighbor_summary = {
        "neighbor_count": 0,
        "neighbor_total_pips": 0.0,
        "neighbor_resource_pips": {resource: 0.0 for resource in TRACKED_RESOURCES},
        "centrality_score": 0.0,
        "initial_robber_safe": True,
        "opening_blocked_pips": 0.0,
    }
    if desert_tile is not None:
        desert_coord = (int(desert_tile["x"]), int(desert_tile["y"]))
        desert_neighbors = neighbors.get(desert_coord, [])
        desert_neighbor_summary["neighbor_count"] = len(desert_neighbors)
        desert_neighbor_summary["centrality_score"] = clamp01(len(desert_neighbors) / 6.0)
        for coord in desert_neighbors:
            tile = next(
                (candidate for candidate in normalized_tiles if int(candidate["x"]) == coord[0] and int(candidate["y"]) == coord[1]),
                None,
            )
            if tile is None:
                continue
            resource = str(tile.get("type") or "").upper()
            pip_weight = float(tile.get("pip_weight", 0.0))
            desert_neighbor_summary["neighbor_total_pips"] += pip_weight
            if resource in desert_neighbor_summary["neighbor_resource_pips"]:
                desert_neighbor_summary["neighbor_resource_pips"][resource] += pip_weight
        desert_neighbor_summary["tile"] = {
            "x": int(desert_tile["x"]),
            "y": int(desert_tile["y"]),
        }

    opening_analysis = build_opening_analysis(tiles=normalized_tiles, slot_ordering=slot_ordering)
    opening_candidates = opening_analysis["opening_candidates"]

    return {
        "resource_environment": {
            "by_resource": by_resource,
            "board_tags": board_tags,
            "strongest_resources": strongest_resources,
            "weakest_resources": weakest_resources,
        },
        "desert_robber": desert_neighbor_summary,
        "port_opportunity": summarize_port_opportunity(opening_candidates),
        "opening_candidates": opening_candidates,
        "opening_rankings": opening_analysis["opening_rankings"],
    }
