from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ai.economy.constants import (
    ABUNDANCE_DEVIATION_CAP,
    AVERAGE_TILE_PIPS,
    PIP_WEIGHTS,
    RESOURCE_RATIO_SPREAD_CAP,
    ROBBER_FRAGILE_CONCENTRATION_THRESHOLD,
    ROBBER_FRAGILE_STRENGTH_THRESHOLD,
    SCARCITY_DEVIATION_CAP,
    STANDARD_RESOURCE_BASELINE_PIPS,
    STANDARD_RESOURCE_BASELINE_TILE_COUNTS,
    STRONG_NUMBER_TOKENS,
    TRACKED_RESOURCES,
)
from ai.economy.normalize import (
    abundance_score,
    clamp01,
    concentration_score_from_hhi,
    safe_div,
    scarcity_score,
    score_from_cap,
    token_quality_score,
)
from ai.economy.tags import desert_robber_tags, resource_environment_tags, resource_tags_from_stats


Tile = Dict[str, object]
Coord = Tuple[int, int]


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _resource_name(value: object) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in set(TRACKED_RESOURCES) | {"DESERT"} else "UNKNOWN"


def _tile_vertices(tile: Tile) -> set[Tuple[int, int, int]]:
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


def _tile_neighbors(tiles: Sequence[Tile]) -> Dict[Coord, List[Coord]]:
    keyed_tiles = {
        (int(tile["x"]), int(tile["y"])): tile
        for tile in tiles
        if isinstance(tile.get("x"), int) and isinstance(tile.get("y"), int)
    }
    vertices = {coord: _tile_vertices(tile) for coord, tile in keyed_tiles.items()}
    neighbors: Dict[Coord, List[Coord]] = {coord: [] for coord in keyed_tiles.keys()}
    coords = list(keyed_tiles.keys())
    for index, left in enumerate(coords):
        for right in coords[index + 1:]:
            if len(vertices[left].intersection(vertices[right])) >= 2:
                neighbors[left].append(right)
                neighbors[right].append(left)
    return {coord: sorted(values) for coord, values in neighbors.items()}


def _normalize_tiles(tiles: Sequence[Tile]) -> List[Tile]:
    normalized_tiles: List[Tile] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        x_coord = _safe_int(tile.get("x"))
        y_coord = _safe_int(tile.get("y"))
        if x_coord is None or y_coord is None:
            continue
        number = _safe_int(tile.get("number"))
        resource = _resource_name(tile.get("type"))
        pip_weight = float(PIP_WEIGHTS.get(number, 0.0)) if number is not None else 0.0
        normalized_tiles.append(
            {
                "x": x_coord,
                "y": y_coord,
                "tile_id": tile.get("tile_id") or f"tile:{x_coord}:{y_coord}",
                "type": resource,
                "number": number,
                "robber": bool(tile.get("robber", False)),
                "pip_weight": pip_weight,
            }
        )
    normalized_tiles.sort(key=lambda item: (int(item["x"]), int(item["y"])))
    return normalized_tiles


def _round(value: float) -> float:
    return round(float(value), 6)


def _board_resource_environment(normalized_tiles: Sequence[Tile]) -> Dict[str, object]:
    total_productive_pips = sum(
        float(tile["pip_weight"])
        for tile in normalized_tiles
        if _resource_name(tile.get("type")) in TRACKED_RESOURCES
    )

    by_resource: Dict[str, Dict[str, object]] = {}
    for resource in TRACKED_RESOURCES:
        resource_tiles = [tile for tile in normalized_tiles if _resource_name(tile.get("type")) == resource]
        pip_values = [float(tile["pip_weight"]) for tile in resource_tiles]
        pip_total = sum(pip_values)
        tile_count = len(resource_tiles)
        baseline_pips = STANDARD_RESOURCE_BASELINE_PIPS[resource]
        baseline_tile_count = STANDARD_RESOURCE_BASELINE_TILE_COUNTS[resource]
        baseline_ratio = safe_div(pip_total, baseline_pips)
        scarcity_deviation = max(0.0, 1.0 - baseline_ratio)
        abundance_deviation = max(0.0, baseline_ratio - 1.0)
        concentration_hhi = sum((safe_div(value, pip_total) ** 2) for value in pip_values if pip_total > 0.0)
        mean_pips = safe_div(pip_total, float(tile_count))

        number_token_counts: Dict[str, int] = {}
        for tile in resource_tiles:
            key = str(tile.get("number"))
            number_token_counts[key] = number_token_counts.get(key, 0) + 1

        stats: Dict[str, object] = {
            "resource": resource,
            "tile_count": tile_count,
            "pip_total": _round(pip_total),
            "pip_share": _round(safe_div(pip_total, total_productive_pips)),
            "baseline_pip_total": _round(baseline_pips),
            "baseline_tile_count": baseline_tile_count,
            "baseline_ratio": _round(baseline_ratio),
            "scarcity_deviation": _round(scarcity_deviation),
            "abundance_deviation": _round(abundance_deviation),
            "scarcity_score": _round(scarcity_score(pip_total, baseline_pips, SCARCITY_DEVIATION_CAP)),
            "abundance_score": _round(abundance_score(pip_total, baseline_pips, ABUNDANCE_DEVIATION_CAP)),
            "mean_pips_per_tile": _round(mean_pips),
            "strong_number_share": _round(
                safe_div(
                    sum(float(tile["pip_weight"]) for tile in resource_tiles if tile.get("number") in STRONG_NUMBER_TOKENS),
                    pip_total,
                )
            ),
            "concentration_hhi": _round(concentration_hhi),
            "concentration_score": _round(concentration_score_from_hhi(concentration_hhi, tile_count)),
            "token_quality_score": _round(token_quality_score(mean_pips)),
            "number_token_counts": dict(sorted(number_token_counts.items())),
            "tags": [],
        }
        stats["tags"] = resource_tags_from_stats(resource, stats)
        by_resource[resource] = stats

    baseline_total = sum(STANDARD_RESOURCE_BASELINE_PIPS.values())
    ratios = [float(stats["baseline_ratio"]) for stats in by_resource.values()]
    ratio_spread = (max(ratios) - min(ratios)) if ratios else 0.0
    resource_environment: Dict[str, object] = {
        "by_resource": by_resource,
        "strongest_resources": sorted(
            TRACKED_RESOURCES,
            key=lambda resource: (-float(by_resource[resource]["pip_total"]), resource),
        ),
        "weakest_resources": sorted(
            TRACKED_RESOURCES,
            key=lambda resource: (float(by_resource[resource]["pip_total"]), resource),
        ),
        "total_productive_pips": _round(total_productive_pips),
        "baseline_total_pips": _round(baseline_total),
        "expected_average_tile_pips": _round(AVERAGE_TILE_PIPS),
        "resource_ratio_spread": _round(ratio_spread),
        "board_balance_score": _round(1.0 - score_from_cap(ratio_spread, RESOURCE_RATIO_SPREAD_CAP)),
        "board_polarization_score": _round(score_from_cap(ratio_spread, RESOURCE_RATIO_SPREAD_CAP)),
        "board_tags": [],
    }
    resource_environment["board_tags"] = resource_environment_tags(resource_environment)
    return resource_environment


def _same_location(left: Optional[Dict[str, int]], right: Optional[Dict[str, int]]) -> bool:
    return bool(left and right and left.get("x") == right.get("x") and left.get("y") == right.get("y"))


def _build_desert_robber(
    normalized_tiles: Sequence[Tile],
    resource_environment: Dict[str, object],
    robber: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    desert_tile = next((tile for tile in normalized_tiles if _resource_name(tile.get("type")) == "DESERT"), None)
    desert_location = (
        {"x": int(desert_tile["x"]), "y": int(desert_tile["y"])}
        if desert_tile is not None
        else None
    )
    flagged_robber_tile = next((tile for tile in normalized_tiles if bool(tile.get("robber"))), None)
    robber_location = robber or (
        {"x": int(flagged_robber_tile["x"]), "y": int(flagged_robber_tile["y"])}
        if flagged_robber_tile is not None
        else desert_location
    )

    tiles_by_coord = {(int(tile["x"]), int(tile["y"])): tile for tile in normalized_tiles}
    neighbors = _tile_neighbors(normalized_tiles)
    desert_coord = (desert_location["x"], desert_location["y"]) if desert_location else None
    desert_neighbors = neighbors.get(desert_coord, []) if desert_coord is not None else []
    neighbor_resource_pips = {resource: 0.0 for resource in TRACKED_RESOURCES}
    neighbor_total_pips = 0.0
    for coord in desert_neighbors:
        tile = tiles_by_coord.get(coord)
        if tile is None:
            continue
        resource = _resource_name(tile.get("type"))
        pip_weight = float(tile.get("pip_weight", 0.0))
        neighbor_total_pips += pip_weight
        if resource in neighbor_resource_pips:
            neighbor_resource_pips[resource] += pip_weight

    by_resource = resource_environment.get("by_resource", {})
    robber_fragility_by_resource = {resource: 0.0 for resource in TRACKED_RESOURCES}
    strong_concentrated_resources: List[str] = []
    if isinstance(by_resource, dict):
        for resource, stats in sorted(by_resource.items()):
            if not isinstance(stats, dict):
                continue
            concentration = float(stats.get("concentration_score", 0.0))
            baseline_ratio = float(stats.get("baseline_ratio", 0.0))
            fragility_score = concentration if baseline_ratio >= ROBBER_FRAGILE_STRENGTH_THRESHOLD else 0.0
            robber_fragility_by_resource[resource] = _round(fragility_score)
            if fragility_score >= ROBBER_FRAGILE_CONCENTRATION_THRESHOLD:
                strong_concentrated_resources.append(resource)

    desert_robber: Dict[str, object] = {
        "desert_location": desert_location,
        "robber_location": robber_location,
        "robber_on_desert": _same_location(robber_location, desert_location),
        "desert_neighbor_count": len(desert_neighbors),
        "desert_neighbor_total_pips": _round(neighbor_total_pips),
        "desert_neighbor_resource_pips": {
            resource: _round(value)
            for resource, value in sorted(neighbor_resource_pips.items())
        },
        "desert_centrality_score": _round(clamp01(len(desert_neighbors) / 6.0)),
        "robber_fragility_by_resource": dict(sorted(robber_fragility_by_resource.items())),
        "strong_concentrated_resources": strong_concentrated_resources,
        "robber_fragile_resources": list(strong_concentrated_resources),
        "tags": [],
    }
    desert_robber["tags"] = desert_robber_tags(desert_robber)
    return desert_robber


def _port_opportunity(port_data: object = None) -> Dict[str, object]:
    """Return a stable phase-1 port block without inferring unavailable port geometry."""
    if not port_data:
        return {
            "available": False,
            "ports": {},
            "notes": ["port_geometry_unavailable_from_extracted_board_state"],
        }
    if isinstance(port_data, dict):
        return {
            "available": True,
            "ports": dict(sorted(port_data.items())),
            "notes": [],
        }
    if isinstance(port_data, list):
        return {
            "available": True,
            "ports": {"items": port_data},
            "notes": [],
        }
    return {
        "available": False,
        "ports": {},
        "notes": ["unsupported_port_payload"],
    }


def build_board_profile(
    tiles: Sequence[Tile],
    robber: Optional[Dict[str, int]] = None,
    port_data: object = None,
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build deterministic pre-settlement economy metrics from extracted board tiles."""
    del slot_ordering
    normalized_tiles = _normalize_tiles(tiles)
    resource_environment = _board_resource_environment(normalized_tiles)
    desert_robber = _build_desert_robber(
        normalized_tiles=normalized_tiles,
        resource_environment=resource_environment,
        robber=robber,
    )
    resource_environment["board_tags"] = sorted(
        set(resource_environment_tags(resource_environment) + desert_robber_tags(desert_robber))
    )
    return {
        "resource_environment": resource_environment,
        "desert_robber": desert_robber,
        "port_opportunity": _port_opportunity(port_data),
    }
