from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ai.economy.constants import (
    BUILD_RECIPES,
    MAX_PAIR_PIPS,
    MAX_SINGLE_VERTEX_PIPS,
    PIP_WEIGHTS,
    TRACKED_RESOURCES,
    VALID_STRUCTURE_SLOTS,
    adjacent_vertices_for_vertex,
    vertex_adjacent_tiles,
    vertex_id,
)
from ai.economy.normalize import clamp01, diversity_score, evenness_score, safe_div, score_from_cap


Tile = Dict[str, object]
VertexKey = Tuple[int, int, int]

PAIR_SCORE_WEIGHTS = {
    "ev": 0.24,
    "recipe": 0.18,
    "synergy": 0.16,
    "diversity": 0.12,
    "evenness": 0.10,
    "scarcity": 0.10,
    "expansion": 0.06,
    "port": 0.04,
}

VERTEX_SCORE_WEIGHTS = {
    "ev": 0.28,
    "recipe": 0.18,
    "synergy": 0.14,
    "scarcity": 0.12,
    "diversity": 0.10,
    "evenness": 0.10,
    "expansion": 0.08,
    "port": 0.00,
}


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _resource_name(value: object) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in TRACKED_RESOURCES else "UNKNOWN"


def _normalize_tiles(tiles: Sequence[Tile]) -> List[Tile]:
    normalized: List[Tile] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        x_coord = _safe_int(tile.get("x"))
        y_coord = _safe_int(tile.get("y"))
        if x_coord is None or y_coord is None:
            continue
        number = _safe_int(tile.get("number"))
        pip_weight = float(tile.get("pip_weight", PIP_WEIGHTS.get(number, 0.0))) if number is not None else 0.0
        normalized.append(
            {
                "x": x_coord,
                "y": y_coord,
                "tile_id": tile.get("tile_id") or f"tile:{x_coord}:{y_coord}",
                "type": str(tile.get("type") or "").strip().upper(),
                "number": number,
                "pip_weight": pip_weight,
            }
        )
    normalized.sort(key=lambda item: (int(item["x"]), int(item["y"])))
    return normalized


def _coerce_valid_structure_slots(slot_ordering: Optional[Dict[str, object]]) -> List[VertexKey]:
    if not isinstance(slot_ordering, dict):
        return list(VALID_STRUCTURE_SLOTS)
    raw_slots = slot_ordering.get("structure_slots")
    raw_valid = slot_ordering.get("structure_slot_valid")
    if not isinstance(raw_slots, list) or not isinstance(raw_valid, list):
        return list(VALID_STRUCTURE_SLOTS)
    valid_slots: List[VertexKey] = []
    for entry, is_valid in zip(raw_slots, raw_valid):
        if not is_valid or not isinstance(entry, dict):
            continue
        x_coord = _safe_int(entry.get("x"))
        y_coord = _safe_int(entry.get("y"))
        orientation = _safe_int(entry.get("orientation"))
        if x_coord is not None and y_coord is not None and orientation is not None:
            valid_slots.append((x_coord, y_coord, orientation))
    return valid_slots or list(VALID_STRUCTURE_SLOTS)


def _occupied_or_blocked_slots(structures: Sequence[Dict[str, object]]) -> Set[VertexKey]:
    blocked: Set[VertexKey] = set()
    for structure in structures:
        if not isinstance(structure, dict):
            continue
        x_coord = _safe_int(structure.get("x"))
        y_coord = _safe_int(structure.get("y"))
        orientation = _safe_int(structure.get("orientation"))
        if x_coord is None or y_coord is None or orientation is None:
            continue
        key = (x_coord, y_coord, orientation)
        blocked.add(key)
        for neighbor in adjacent_vertices_for_vertex(*key):
            blocked.add((neighbor["x"], neighbor["y"], neighbor["orientation"]))
    return blocked


def _board_resource_stats(board_economy: Optional[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    by_resource = {}
    if isinstance(board_economy, dict):
        resource_environment = board_economy.get("resource_environment", {})
        if isinstance(resource_environment, dict):
            raw_by_resource = resource_environment.get("by_resource", {})
            if isinstance(raw_by_resource, dict):
                by_resource = raw_by_resource

    output: Dict[str, Dict[str, float]] = {}
    for resource in TRACKED_RESOURCES:
        raw = by_resource.get(resource, {}) if isinstance(by_resource, dict) else {}
        output[resource] = {
            "scarcity_score": float(raw.get("scarcity_score", 0.0)) if isinstance(raw, dict) else 0.0,
            "abundance_score": float(raw.get("abundance_score", 0.0)) if isinstance(raw, dict) else 0.0,
            "concentration_score": float(raw.get("concentration_score", 0.0)) if isinstance(raw, dict) else 0.0,
            "robber_fragility_score": 0.0,
        }

    desert_robber = board_economy.get("desert_robber", {}) if isinstance(board_economy, dict) else {}
    fragility = desert_robber.get("robber_fragility_by_resource", {}) if isinstance(desert_robber, dict) else {}
    if isinstance(fragility, dict):
        for resource in TRACKED_RESOURCES:
            output[resource]["robber_fragility_score"] = float(fragility.get(resource, 0.0))
    return output


def _recipe_scores(pip_by_resource: Dict[str, float]) -> Dict[str, float]:
    recipe_scores: Dict[str, float] = {}
    for recipe_name, recipe in BUILD_RECIPES.items():
        total_need = sum(recipe.values())
        if total_need <= 0:
            recipe_scores[recipe_name] = 0.0
            continue
        weighted_support = 0.0
        for resource, need in recipe.items():
            support = score_from_cap(float(pip_by_resource.get(resource, 0.0)), 5.0)
            weighted_support += float(need) * support
        recipe_scores[recipe_name] = _round(weighted_support / float(total_need))
    return recipe_scores


def _recipe_coverage_score(pip_by_resource: Dict[str, float]) -> float:
    scores = _recipe_scores(pip_by_resource)
    if not scores:
        return 0.0
    return _round(sum(scores.values()) / float(len(scores)))


def _scarcity_capture_score(
    pip_by_resource: Dict[str, float],
    board_resource_stats: Dict[str, Dict[str, float]],
    total_pips: float,
) -> float:
    if total_pips <= 0.0:
        return 0.0
    weighted_scarcity = sum(
        float(pip_by_resource.get(resource, 0.0)) * board_resource_stats[resource]["scarcity_score"]
        for resource in TRACKED_RESOURCES
    )
    scarce_share = safe_div(weighted_scarcity, total_pips)
    scarce_pip_pressure = score_from_cap(weighted_scarcity, 10.0)
    return _round(clamp01(0.65 * scarce_share + 0.35 * scarce_pip_pressure))


def _synergy_score(pip_by_resource: Dict[str, float], recipe_scores: Dict[str, float]) -> float:
    road_pair = min(
        score_from_cap(float(pip_by_resource.get("BRICK", 0.0)), 5.0),
        score_from_cap(float(pip_by_resource.get("LUMBER", 0.0)), 5.0),
    )
    city_pair = min(
        score_from_cap(float(pip_by_resource.get("ORE", 0.0)), 5.0),
        score_from_cap(float(pip_by_resource.get("GRAIN", 0.0)), 5.0),
    )
    dev_triangle = min(
        score_from_cap(float(pip_by_resource.get("ORE", 0.0)), 5.0),
        score_from_cap(float(pip_by_resource.get("GRAIN", 0.0)), 5.0),
        score_from_cap(float(pip_by_resource.get("WOOL", 0.0)), 5.0),
    )
    recipe_average = sum(recipe_scores.values()) / float(len(recipe_scores)) if recipe_scores else 0.0
    return _round(
        clamp01(
            0.42 * recipe_average
            + 0.20 * evenness_score(pip_by_resource.values())
            + 0.16 * road_pair
            + 0.12 * city_pair
            + 0.10 * dev_triangle
        )
    )


def _tile_exposure_fragility(tile_exposure: Dict[str, float], total_pips: float) -> Tuple[float, float, float]:
    if total_pips <= 0.0:
        return 0.0, 0.0, 0.0
    values = [max(0.0, float(value)) for value in tile_exposure.values()]
    if not values:
        return 0.0, 0.0, 0.0
    max_loss = max(values)
    max_loss_share = safe_div(max_loss, total_pips)
    hhi = sum((safe_div(value, total_pips) ** 2) for value in values)
    fragility = clamp01(0.70 * max_loss_share + 0.30 * hhi)
    return _round(fragility), _round(max_loss), _round(max_loss_share)


def _local_robber_fragility_score(
    pip_by_resource: Dict[str, float],
    board_resource_stats: Dict[str, Dict[str, float]],
) -> float:
    total_pips = sum(max(0.0, value) for value in pip_by_resource.values())
    if total_pips <= 0.0:
        return 0.0
    weighted = sum(
        float(pip_by_resource.get(resource, 0.0)) * board_resource_stats[resource]["robber_fragility_score"]
        for resource in TRACKED_RESOURCES
    )
    return _round(safe_div(weighted, total_pips))


def _expansion_frontier(
    x_coord: int,
    y_coord: int,
    orientation: int,
    tile_map: Dict[Tuple[int, int], Tile],
    valid_slots: Sequence[VertexKey],
    blocked_slots: Set[VertexKey],
) -> Tuple[int, float]:
    valid_slot_set = set(valid_slots)
    first_ring = adjacent_vertices_for_vertex(x_coord, y_coord, orientation)
    second_ring: Set[VertexKey] = set()
    for neighbor in first_ring:
        for second in adjacent_vertices_for_vertex(neighbor["x"], neighbor["y"], neighbor["orientation"]):
            key = (second["x"], second["y"], second["orientation"])
            if key == (x_coord, y_coord, orientation) or key not in valid_slot_set or key in blocked_slots:
                continue
            second_ring.add(key)

    best_total = 0.0
    for sx, sy, so in second_ring:
        total = 0.0
        for tile_coord in vertex_adjacent_tiles(sx, sy, so):
            tile = tile_map.get((tile_coord["x"], tile_coord["y"]))
            if tile is not None:
                total += float(tile.get("pip_weight", 0.0))
        best_total = max(best_total, total)
    return len(second_ring), _round(score_from_cap(best_total, MAX_SINGLE_VERTEX_PIPS))


def _port_alignment_placeholder() -> float:
    return 0.0


def _opening_score(
    ev_score_norm: float,
    diversity: float,
    evenness: float,
    recipe_coverage: float,
    scarcity_capture: float,
    synergy: float,
    expansion: float,
    port_alignment: float,
    robber_fragility: float,
    weights: Dict[str, float],
) -> float:
    positive = (
        weights["ev"] * ev_score_norm
        + weights["recipe"] * recipe_coverage
        + weights["synergy"] * synergy
        + weights["scarcity"] * scarcity_capture
        + weights["diversity"] * diversity
        + weights["evenness"] * evenness
        + weights["expansion"] * expansion
        + weights["port"] * port_alignment
    )
    return _round(clamp01(positive - 0.10 * robber_fragility))


def _opening_tags(
    ev_score_norm: float,
    recipe_scores: Dict[str, float],
    scarcity_capture: float,
    synergy: float,
    robber_fragility: float,
    opening_score: float,
) -> List[str]:
    tags: List[str] = []
    if ev_score_norm >= 0.67:
        tags.append("high_ev")
    if opening_score >= 0.70:
        tags.append("strong_opening_economy")
    if scarcity_capture >= 0.45:
        tags.append("scarcity_capture")
    if synergy >= 0.60:
        tags.append("synergistic")
    if robber_fragility >= 0.55:
        tags.append("robber_fragile")
    for recipe_name, score in sorted(recipe_scores.items()):
        if score >= 0.65:
            tags.append(f"{recipe_name}_support")
    return sorted(set(tags))


def _build_single_candidates(
    tiles: Sequence[Tile],
    board_economy: Optional[Dict[str, object]] = None,
    structures: Sequence[Dict[str, object]] = (),
    slot_ordering: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    normalized_tiles = _normalize_tiles(tiles)
    tile_map = {
        (int(tile["x"]), int(tile["y"])): tile
        for tile in normalized_tiles
    }
    valid_slots = _coerce_valid_structure_slots(slot_ordering)
    blocked_slots = _occupied_or_blocked_slots(structures)
    board_stats = _board_resource_stats(board_economy)
    candidates: List[Dict[str, object]] = []

    for x_coord, y_coord, orientation in valid_slots:
        vertex_key = (x_coord, y_coord, orientation)
        if vertex_key in blocked_slots:
            continue

        adjacent_tiles: List[Dict[str, object]] = []
        pip_by_resource = {resource: 0.0 for resource in TRACKED_RESOURCES}
        tile_exposure: Dict[str, float] = {}
        total_pips = 0.0
        for tile_coord in vertex_adjacent_tiles(x_coord, y_coord, orientation):
            tile = tile_map.get((tile_coord["x"], tile_coord["y"]))
            if tile is None:
                continue
            resource = _resource_name(tile.get("type"))
            pip_weight = float(tile.get("pip_weight", 0.0))
            adjacent_tiles.append(
                {
                    "x": int(tile["x"]),
                    "y": int(tile["y"]),
                    "tile_id": tile["tile_id"],
                    "type": tile.get("type"),
                    "number": tile.get("number"),
                    "pip_weight": _round(pip_weight),
                }
            )
            total_pips += pip_weight
            tile_exposure[str(tile["tile_id"])] = tile_exposure.get(str(tile["tile_id"]), 0.0) + pip_weight
            if resource in pip_by_resource:
                pip_by_resource[resource] += pip_weight

        if not adjacent_tiles:
            continue

        diversity = _round(diversity_score(pip_by_resource))
        evenness = _round(evenness_score(pip_by_resource.values()))
        recipe_scores = _recipe_scores(pip_by_resource)
        recipe_coverage = _recipe_coverage_score(pip_by_resource)
        scarcity_capture = _scarcity_capture_score(pip_by_resource, board_stats, total_pips)
        synergy = _synergy_score(pip_by_resource, recipe_scores)
        tile_fragility, key_tile_pips, key_tile_loss_share = _tile_exposure_fragility(tile_exposure, total_pips)
        board_fragility = _local_robber_fragility_score(pip_by_resource, board_stats)
        robber_fragility = _round(clamp01(0.70 * tile_fragility + 0.30 * board_fragility))
        expansion_count, expansion_score = _expansion_frontier(
            x_coord=x_coord,
            y_coord=y_coord,
            orientation=orientation,
            tile_map=tile_map,
            valid_slots=valid_slots,
            blocked_slots=blocked_slots,
        )
        port_alignment = _port_alignment_placeholder()
        ev_norm = _round(score_from_cap(total_pips, MAX_SINGLE_VERTEX_PIPS))
        score = _opening_score(
            ev_score_norm=ev_norm,
            diversity=diversity,
            evenness=evenness,
            recipe_coverage=recipe_coverage,
            scarcity_capture=scarcity_capture,
            synergy=synergy,
            expansion=expansion_score,
            port_alignment=port_alignment,
            robber_fragility=robber_fragility,
            weights=VERTEX_SCORE_WEIGHTS,
        )

        rounded_pips = {resource: _round(pip_by_resource[resource]) for resource in TRACKED_RESOURCES}
        candidates.append(
            {
                "vertex_id": vertex_id(x_coord, y_coord, orientation),
                "x": x_coord,
                "y": y_coord,
                "orientation": orientation,
                "adjacent_tiles": adjacent_tiles,
                "ev_score": _round(total_pips),
                "ev_score_norm": ev_norm,
                "total_pips": _round(total_pips),
                "resource_mix": rounded_pips,
                "pip_by_resource": rounded_pips,
                "diversity_score": diversity,
                "evenness_score": evenness,
                "recipe_coverage_by_recipe": recipe_scores,
                "recipe_coverage_score": recipe_coverage,
                "scarcity_capture_score": scarcity_capture,
                "synergy_score": synergy,
                "robber_fragility_score": robber_fragility,
                "key_tile_pips": key_tile_pips,
                "key_tile_loss_share": key_tile_loss_share,
                "expansion_frontier_count": expansion_count,
                "expansion_frontier_score": expansion_score,
                "port": None,
                "port_alignment_score": port_alignment,
                "opening_score": score,
                "tags": _opening_tags(
                    ev_score_norm=ev_norm,
                    recipe_scores=recipe_scores,
                    scarcity_capture=scarcity_capture,
                    synergy=synergy,
                    robber_fragility=robber_fragility,
                    opening_score=score,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item.get("opening_score", 0.0)),
            -float(item.get("ev_score", 0.0)),
            str(item.get("vertex_id") or ""),
        )
    )
    return candidates


def _distance_rule_compatible(left: Dict[str, object], right: Dict[str, object]) -> bool:
    left_key = (int(left["x"]), int(left["y"]), int(left["orientation"]))
    right_key = (int(right["x"]), int(right["y"]), int(right["orientation"]))
    if left_key == right_key:
        return False
    left_neighbors = {
        (neighbor["x"], neighbor["y"], neighbor["orientation"])
        for neighbor in adjacent_vertices_for_vertex(*left_key)
    }
    return right_key not in left_neighbors


def _combine_tile_exposure(left: Dict[str, object], right: Dict[str, object]) -> Dict[str, float]:
    exposure: Dict[str, float] = {}
    for candidate in (left, right):
        for tile in candidate.get("adjacent_tiles", []):
            if not isinstance(tile, dict):
                continue
            tile_id = str(tile.get("tile_id"))
            exposure[tile_id] = exposure.get(tile_id, 0.0) + float(tile.get("pip_weight", 0.0))
    return exposure


def _build_opening_pairs(
    candidates: Sequence[Dict[str, object]],
    board_economy: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    board_stats = _board_resource_stats(board_economy)
    pairs: List[Dict[str, object]] = []
    ordered = sorted(candidates, key=lambda item: str(item.get("vertex_id") or ""))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if not _distance_rule_compatible(left, right):
                continue

            combined_pips = {
                resource: float((left.get("pip_by_resource") or {}).get(resource, 0.0))
                + float((right.get("pip_by_resource") or {}).get(resource, 0.0))
                for resource in TRACKED_RESOURCES
            }
            total_pips = sum(combined_pips.values())
            recipe_scores = _recipe_scores(combined_pips)
            recipe_coverage = _recipe_coverage_score(combined_pips)
            diversity = _round(diversity_score(combined_pips))
            evenness = _round(evenness_score(combined_pips.values()))
            scarcity_capture = _scarcity_capture_score(combined_pips, board_stats, total_pips)
            synergy = _synergy_score(combined_pips, recipe_scores)
            tile_fragility, key_tile_pips, key_tile_loss_share = _tile_exposure_fragility(
                _combine_tile_exposure(left, right),
                total_pips,
            )
            board_fragility = _local_robber_fragility_score(combined_pips, board_stats)
            robber_fragility = _round(clamp01(0.70 * tile_fragility + 0.30 * board_fragility))
            expansion_score = _round(
                max(
                    float(left.get("expansion_frontier_score", 0.0)),
                    float(right.get("expansion_frontier_score", 0.0)),
                )
            )
            port_alignment = _port_alignment_placeholder()
            total_norm = _round(score_from_cap(total_pips, MAX_PAIR_PIPS))
            pair_score = _opening_score(
                ev_score_norm=total_norm,
                diversity=diversity,
                evenness=evenness,
                recipe_coverage=recipe_coverage,
                scarcity_capture=scarcity_capture,
                synergy=synergy,
                expansion=expansion_score,
                port_alignment=port_alignment,
                robber_fragility=robber_fragility,
                weights=PAIR_SCORE_WEIGHTS,
            )
            rounded_pips = {resource: _round(combined_pips[resource]) for resource in TRACKED_RESOURCES}
            pairs.append(
                {
                    "pair_id": f"{left['vertex_id']}|{right['vertex_id']}",
                    "vertex_ids": [left["vertex_id"], right["vertex_id"]],
                    "total_pips": _round(total_pips),
                    "total_pips_norm": total_norm,
                    "resource_mix": rounded_pips,
                    "pip_by_resource": rounded_pips,
                    "diversity_score": diversity,
                    "evenness_score": evenness,
                    "recipe_coverage_by_recipe": recipe_scores,
                    "recipe_coverage_score": recipe_coverage,
                    "scarcity_capture_score": scarcity_capture,
                    "synergy_score": synergy,
                    "robber_fragility_score": robber_fragility,
                    "key_tile_pips": key_tile_pips,
                    "key_tile_loss_share": key_tile_loss_share,
                    "expansion_frontier_score": expansion_score,
                    "port_alignment_score": port_alignment,
                    "opening_pair_score": pair_score,
                    "tags": _opening_tags(
                        ev_score_norm=total_norm,
                        recipe_scores=recipe_scores,
                        scarcity_capture=scarcity_capture,
                        synergy=synergy,
                        robber_fragility=robber_fragility,
                        opening_score=pair_score,
                    ),
                }
            )
    pairs.sort(
        key=lambda item: (
            -float(item.get("opening_pair_score", 0.0)),
            -float(item.get("total_pips", 0.0)),
            str(item.get("pair_id") or ""),
        )
    )
    return pairs


def _top(items: Sequence[Dict[str, object]], key_fields: Iterable[str], limit: int) -> List[Dict[str, object]]:
    fields = list(key_fields)
    return sorted(
        items,
        key=lambda item: tuple([-float(item.get(field, 0.0)) for field in fields] + [str(item.get("pair_id") or item.get("vertex_id") or "")]),
    )[:limit]


def build_opening_analysis(
    tiles: Sequence[Tile],
    slot_ordering: Optional[Dict[str, object]] = None,
    limit: int = 12,
    board_economy: Optional[Dict[str, object]] = None,
    structures: Sequence[Dict[str, object]] = (),
) -> Dict[str, object]:
    """Score legal opening vertices and two-settlement opening pairs.

    The board resource environment remains the upstream source of global scarcity
    and concentration. This module only asks which legal vertices and vertex pairs
    can actually access that economy.
    """
    candidates = _build_single_candidates(
        tiles=tiles,
        board_economy=board_economy,
        structures=structures,
        slot_ordering=slot_ordering,
    )
    pairs = _build_opening_pairs(candidates, board_economy=board_economy)
    rankings = {
        "top_vertices_by_score": candidates[:limit],
        "top_vertices_by_ev": _top(candidates, ("ev_score", "opening_score"), limit),
        "top_vertices_by_recipe_coverage": _top(candidates, ("recipe_coverage_score", "opening_score"), limit),
        "top_vertices_by_synergy": _top(candidates, ("synergy_score", "opening_score"), limit),
        "top_pairs_by_score": pairs[:limit],
        "top_pairs_by_ev": _top(pairs, ("total_pips", "opening_pair_score"), limit),
        "top_pairs_by_recipe_coverage": _top(pairs, ("recipe_coverage_score", "opening_pair_score"), limit),
        "top_pairs_by_synergy": _top(pairs, ("synergy_score", "opening_pair_score"), limit),
        "top_by_ev": _top(candidates, ("ev_score", "opening_score"), limit),
        "top_by_balance": _top(candidates, ("evenness_score", "opening_score"), limit),
        "pairs": {
            "top_by_score": pairs[:limit],
            "top_by_ev": _top(pairs, ("total_pips", "opening_pair_score"), limit),
            "top_by_balance": _top(pairs, ("evenness_score", "opening_pair_score"), limit),
        },
    }
    return {
        "schema_version": "opening_economy_v1",
        "opening_candidates": candidates,
        "opening_pairs": pairs,
        "opening_rankings": rankings,
        "scoring_notes": {
            "board_resource_environment_source": "board_economy.resource_environment",
            "port_alignment": "placeholder_until_board_port_geometry_is_extracted",
            "pair_distance_rule": "candidate vertices must not be identical or adjacent",
        },
    }
