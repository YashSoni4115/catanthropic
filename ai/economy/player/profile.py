from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ai.economy.constants import BUILD_RECIPES, PIP_WEIGHTS, TRACKED_RESOURCES
from ai.economy.normalize import (
    clamp01,
    concentration_score_from_hhi,
    diversity_score,
    evenness_score,
    safe_div,
    score_from_cap,
)
from ai.economy.player.expansion import build_expansion_profile
from ai.economy.player.pressure import build_player_pressure


ResourceMap = Dict[str, float]
TileCoord = Tuple[int, int]


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


def _empty_resource_map() -> ResourceMap:
    return {resource: 0.0 for resource in TRACKED_RESOURCES}


def _player_list(state: Dict[str, object]) -> List[Dict[str, object]]:
    players = state.get("players")
    if isinstance(players, dict):
        raw_players = players.get("all_players", [])
    elif isinstance(players, list):
        raw_players = players
    else:
        raw_players = []
    output = [player for player in raw_players if isinstance(player, dict)]
    seen = {
        str(player.get("player_id") or player.get("id") or player.get("player_name") or player.get("name"))
        for player in output
    }
    raw_structures = state.get("structures", [])
    if isinstance(raw_structures, list):
        for structure in raw_structures:
            if not isinstance(structure, dict):
                continue
            owner_id = structure.get("owner_id")
            owner_name = structure.get("owner_name")
            key = str(owner_id or owner_name)
            if key == "None" or key in seen:
                continue
            seen.add(key)
            output.append({"player_id": owner_id, "player_name": owner_name})
    return output


def _player_identity(player: Dict[str, object], fallback_index: int) -> Dict[str, Optional[str]]:
    player_id = player.get("player_id")
    player_name = player.get("player_name")
    if player_id is None:
        player_id = player.get("id") or player.get("name")
    if player_name is None:
        player_name = player.get("name")
    if player_id is None and player_name is None:
        player_id = f"unknown:{fallback_index}"
    return {
        "player_id": str(player_id) if player_id is not None else None,
        "player_name": str(player_name) if player_name is not None else None,
    }


def _tile_map(state: Dict[str, object]) -> Dict[TileCoord, Dict[str, object]]:
    output: Dict[TileCoord, Dict[str, object]] = {}
    raw_tiles = state.get("tiles", [])
    if not isinstance(raw_tiles, list):
        return output
    for tile in raw_tiles:
        if not isinstance(tile, dict):
            continue
        x_coord = _safe_int(tile.get("x"))
        y_coord = _safe_int(tile.get("y"))
        if x_coord is None or y_coord is None:
            continue
        number = _safe_int(tile.get("number"))
        output[(x_coord, y_coord)] = {
            "x": x_coord,
            "y": y_coord,
            "type": str(tile.get("type") or "").strip().upper(),
            "number": number,
            "pip_weight": float(tile.get("pip_weight", PIP_WEIGHTS.get(number, 0.0))) if number is not None else 0.0,
        }
    return output


def _robber_coord(state: Dict[str, object]) -> Optional[TileCoord]:
    robber = state.get("robber")
    if isinstance(robber, dict):
        x_coord = _safe_int(robber.get("x"))
        y_coord = _safe_int(robber.get("y"))
        if x_coord is not None and y_coord is not None:
            return (x_coord, y_coord)
    tiles = state.get("tiles", [])
    if isinstance(tiles, list):
        for tile in tiles:
            if not isinstance(tile, dict) or not tile.get("robber"):
                continue
            x_coord = _safe_int(tile.get("x"))
            y_coord = _safe_int(tile.get("y"))
            if x_coord is not None and y_coord is not None:
                return (x_coord, y_coord)
    return None


def _player_structures(state: Dict[str, object], player_id: Optional[str], player_name: Optional[str]) -> List[Dict[str, object]]:
    raw_structures = state.get("structures", [])
    if not isinstance(raw_structures, list):
        return []
    structures: List[Dict[str, object]] = []
    for structure in raw_structures:
        if not isinstance(structure, dict):
            continue
        owner_id = structure.get("owner_id")
        owner_name = structure.get("owner_name")
        if player_id is not None and str(owner_id) == player_id:
            structures.append(structure)
        elif owner_id is None and player_name is not None and str(owner_name) == player_name:
            structures.append(structure)
    return structures


def _structure_multiplier(structure: Dict[str, object]) -> float:
    return 2.0 if str(structure.get("type") or "").strip().lower() == "city" else 1.0


def _structure_id(structure: Dict[str, object]) -> str:
    existing = structure.get("structure_id")
    if existing is not None:
        return str(existing)
    return f"structure:{structure.get('x')}:{structure.get('y')}:{structure.get('orientation')}"


def _production_from_structures(
    structures: Sequence[Dict[str, object]],
    tiles_by_coord: Dict[TileCoord, Dict[str, object]],
    robber: Optional[TileCoord],
) -> Dict[str, object]:
    expected = _empty_resource_map()
    settlement_weighted = _empty_resource_map()
    city_weighted = _empty_resource_map()
    blocked = _empty_resource_map()
    production_counts = {resource: 0 for resource in TRACKED_RESOURCES}
    blocked_counts = {resource: 0 for resource in TRACKED_RESOURCES}
    contributing_structures: List[Dict[str, object]] = []

    for structure in structures:
        multiplier = _structure_multiplier(structure)
        structure_expected = _empty_resource_map()
        structure_blocked = _empty_resource_map()
        adjacent_tiles = structure.get("adjacent_tiles", [])
        if not isinstance(adjacent_tiles, list):
            adjacent_tiles = []

        for adjacent in adjacent_tiles:
            if not isinstance(adjacent, dict):
                continue
            x_coord = _safe_int(adjacent.get("x"))
            y_coord = _safe_int(adjacent.get("y"))
            if x_coord is None or y_coord is None:
                continue
            tile = tiles_by_coord.get((x_coord, y_coord))
            if tile is None:
                continue
            resource = _resource_name(tile.get("type"))
            if resource not in expected:
                continue
            pip_weight = float(tile.get("pip_weight", 0.0)) * multiplier
            if pip_weight <= 0.0:
                continue

            expected[resource] += pip_weight
            structure_expected[resource] += pip_weight
            production_counts[resource] += 1
            if multiplier > 1.0:
                city_weighted[resource] += pip_weight
            else:
                settlement_weighted[resource] += pip_weight
            if robber == (x_coord, y_coord):
                blocked[resource] += pip_weight
                structure_blocked[resource] += pip_weight
                blocked_counts[resource] += 1

        contributing_structures.append(
            {
                "structure_id": _structure_id(structure),
                "type": "city" if multiplier > 1.0 else "settlement",
                "expected_income": _rounded_resource_map(structure_expected),
                "blocked_income": _rounded_resource_map(structure_blocked),
                "total_expected_income": _round(sum(structure_expected.values())),
                "total_blocked_income": _round(sum(structure_blocked.values())),
            }
        )

    return {
        "expected": expected,
        "settlement_weighted": settlement_weighted,
        "city_weighted": city_weighted,
        "blocked": blocked,
        "production_counts": production_counts,
        "blocked_counts": blocked_counts,
        "contributing_structures": contributing_structures,
    }


def _rounded_resource_map(values: Dict[str, float]) -> ResourceMap:
    return {resource: _round(values.get(resource, 0.0)) for resource in TRACKED_RESOURCES}


def _board_resource_stats(board_economy: Optional[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    by_resource: Dict[str, object] = {}
    if isinstance(board_economy, dict):
        resource_environment = board_economy.get("resource_environment", {})
        if isinstance(resource_environment, dict):
            raw_by_resource = resource_environment.get("by_resource", {})
            if isinstance(raw_by_resource, dict):
                by_resource = raw_by_resource

    output: Dict[str, Dict[str, float]] = {}
    for resource in TRACKED_RESOURCES:
        stats = by_resource.get(resource, {})
        output[resource] = {
            "scarcity_score": float(stats.get("scarcity_score", 0.0)) if isinstance(stats, dict) else 0.0,
            "abundance_score": float(stats.get("abundance_score", 0.0)) if isinstance(stats, dict) else 0.0,
        }
    return output


def _scarcity_capture_score(expected: ResourceMap, board_resource_stats: Dict[str, Dict[str, float]]) -> float:
    total = sum(expected.values())
    if total <= 0.0:
        return 0.0
    weighted = sum(expected[resource] * board_resource_stats[resource]["scarcity_score"] for resource in TRACKED_RESOURCES)
    return _round(clamp01(0.65 * safe_div(weighted, total) + 0.35 * score_from_cap(weighted, 10.0)))


def _concentration_score(expected: ResourceMap) -> float:
    total = sum(expected.values())
    if total <= 0.0:
        return 0.0
    hhi = sum((safe_div(value, total) ** 2) for value in expected.values() if value > 0.0)
    active_count = sum(1 for value in expected.values() if value > 0.0)
    return _round(concentration_score_from_hhi(hhi, active_count))


def _orientation_scores(expected: ResourceMap) -> Dict[str, float]:
    brick = float(expected.get("BRICK", 0.0))
    wool = float(expected.get("WOOL", 0.0))
    ore = float(expected.get("ORE", 0.0))
    grain = float(expected.get("GRAIN", 0.0))
    lumber = float(expected.get("LUMBER", 0.0))
    return {
        "road": _round(score_from_cap(min(brick, lumber), 5.0)),
        "settlement": _round(score_from_cap(min(brick, lumber, wool, grain), 5.0)),
        "city": _round(score_from_cap(min(ore / 3.0, grain / 2.0), 2.5)),
        "dev": _round(score_from_cap(min(ore, grain, wool), 5.0)),
    }


def _top_orientations(orientation_scores: Dict[str, float], limit: int = 2) -> List[str]:
    ordered = sorted(orientation_scores.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _ in ordered[:limit]]


def _recipe_bottlenecks(expected: ResourceMap, orientation_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    plan_resources = {
        "road": BUILD_RECIPES["road"],
        "settlement": BUILD_RECIPES["settlement"],
        "city": BUILD_RECIPES["city"],
        "dev": BUILD_RECIPES["dev"],
    }
    resource_pressure = {resource: 0.0 for resource in TRACKED_RESOURCES}
    for plan_name in _top_orientations(orientation_scores, limit=2):
        recipe = plan_resources[plan_name]
        for resource, amount in recipe.items():
            if plan_name == "city":
                support = score_from_cap(safe_div(expected.get(resource, 0.0), float(amount)), 2.5)
            else:
                support = score_from_cap(expected.get(resource, 0.0), 5.0)
            resource_pressure[resource] = max(resource_pressure[resource], 1.0 - support)
    return {
        resource: {"severity": _round(severity)}
        for resource, severity in sorted(resource_pressure.items())
        if severity > 0.0
    }


def _strong_weak_resources(expected: ResourceMap) -> Tuple[List[str], List[str]]:
    positive = {resource: value for resource, value in expected.items() if value > 0.0}
    strongest = sorted(TRACKED_RESOURCES, key=lambda resource: (-expected.get(resource, 0.0), resource))
    if not positive:
        return [], list(TRACKED_RESOURCES)
    weakest = sorted(TRACKED_RESOURCES, key=lambda resource: (expected.get(resource, 0.0), resource))
    return strongest[:2], weakest[:2]


def _robber_burden(expected: ResourceMap, blocked: ResourceMap, bottlenecks: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    total = sum(expected.values())
    blocked_total = sum(blocked.values())
    blocked_resources = [resource for resource in TRACKED_RESOURCES if blocked.get(resource, 0.0) > 0.0]
    key_bottleneck_blocked = any(
        resource in bottlenecks and float(bottlenecks[resource].get("severity", 0.0)) >= 0.45
        for resource in blocked_resources
    )
    return {
        "currently_blocked": blocked_total > 0.0,
        "blocked_pips": _round(blocked_total),
        "blocked_share": _round(safe_div(blocked_total, total)),
        "blocked_resources": blocked_resources,
        "key_bottleneck_blocked": key_bottleneck_blocked,
    }


def _tags(
    expected: ResourceMap,
    orientation_scores: Dict[str, float],
    top_orientations: List[str],
    bottlenecks: Dict[str, Dict[str, float]],
    diversity: float,
    evenness: float,
    concentration: float,
    robber_burden: Dict[str, object],
) -> List[str]:
    tags: List[str] = []
    for plan in top_orientations[:2]:
        if orientation_scores.get(plan, 0.0) > 0.0:
            tags.append(f"{plan}_favored")
    for resource, detail in bottlenecks.items():
        if float(detail.get("severity", 0.0)) >= 0.55:
            tags.append(f"{resource.lower()}_bottleneck")
    if diversity >= 0.8 and evenness >= 0.65:
        tags.append("balanced_economy")
    if concentration >= 0.65:
        tags.append("concentrated_economy")
    strongest, weakest = _strong_weak_resources(expected)
    for resource in strongest[:1]:
        tags.append(f"{resource.lower()}_strong")
    for resource in weakest[:1]:
        tags.append(f"{resource.lower()}_weak")
    if bool(robber_burden.get("currently_blocked")):
        tags.append("currently_blocked")
    if bool(robber_burden.get("key_bottleneck_blocked")):
        tags.append("bottleneck_blocked")
    return sorted(set(tags))


def _rank_maps(profiles: Sequence[Dict[str, object]]) -> Dict[str, object]:
    player_ids = [str(profile["player_id"]) for profile in profiles if profile.get("player_id") is not None]
    totals = {str(profile["player_id"]): float(profile.get("total_expected_income", 0.0)) for profile in profiles if profile.get("player_id") is not None}
    by_resource = {
        resource: {
            str(profile["player_id"]): float((profile.get("expected_income") or {}).get(resource, 0.0))
            for profile in profiles
            if profile.get("player_id") is not None
        }
        for resource in TRACKED_RESOURCES
    }
    total_sum = sum(totals.values())
    resource_sums = {resource: sum(values.values()) for resource, values in by_resource.items()}

    def rank_for(values: Dict[str, float]) -> Dict[str, int]:
        ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        return {player_id: rank for rank, (player_id, _) in enumerate(ordered, start=1)}

    total_ranks = rank_for(totals)
    resource_ranks = {resource: rank_for(values) for resource, values in by_resource.items()}
    leaders_by_resource = {}
    for resource, values in by_resource.items():
        leader_id, leader_value = max(values.items(), key=lambda item: (item[1], item[0])) if values else (None, 0.0)
        leaders_by_resource[resource] = {"player_id": leader_id, "value": _round(leader_value)}
    overall_leader_id, overall_leader_value = max(totals.items(), key=lambda item: (item[1], item[0])) if totals else (None, 0.0)

    relative_by_player: Dict[str, Dict[str, object]] = {}
    for player_id in player_ids:
        total_value = totals.get(player_id, 0.0)
        resource_metrics = {}
        for resource in TRACKED_RESOURCES:
            value = by_resource[resource].get(player_id, 0.0)
            leader_value = float(leaders_by_resource[resource]["value"])
            resource_metrics[resource] = {
                "rank": resource_ranks[resource].get(player_id),
                "table_share": _round(safe_div(value, resource_sums[resource])),
                "leader_gap": _round(max(0.0, leader_value - value)),
            }
        relative_by_player[player_id] = {
            "rank": total_ranks.get(player_id),
            "share": _round(safe_div(total_value, total_sum)),
            "leader_gap_value": _round(max(0.0, overall_leader_value - total_value)),
            "total_income_rank": total_ranks.get(player_id),
            "total_table_share": _round(safe_div(total_value, total_sum)),
            "leader_gap": {
                "total_expected_income": _round(max(0.0, overall_leader_value - total_value)),
            },
            "resource_ranks": resource_metrics,
        }
    return {
        "relative_by_player": relative_by_player,
        "table_leaders": {
            "overall": {"player_id": overall_leader_id, "value": _round(overall_leader_value)},
            "by_resource": leaders_by_resource,
        },
    }


def _build_player_profile(
    state: Dict[str, object],
    player: Dict[str, object],
    fallback_index: int,
    board_economy: Optional[Dict[str, object]],
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    identity = _player_identity(player, fallback_index)
    player_id = identity["player_id"]
    player_name = identity["player_name"]
    structures = _player_structures(state, player_id, player_name)
    production = _production_from_structures(structures, _tile_map(state), _robber_coord(state))
    expected = production["expected"]
    blocked = production["blocked"]
    board_stats = _board_resource_stats(board_economy)

    total_expected = sum(expected.values())
    total_blocked = sum(blocked.values())
    diversity = _round(diversity_score(expected))
    evenness = _round(evenness_score(expected.values()))
    concentration = _concentration_score(expected)
    orientation_scores = _orientation_scores(expected)
    top_orientations = _top_orientations(orientation_scores)
    bottlenecks = _recipe_bottlenecks(expected, orientation_scores)
    scarcity_capture = _scarcity_capture_score(expected, board_stats)
    robber_burden = _robber_burden(expected, blocked, bottlenecks)
    strongest, weakest = _strong_weak_resources(expected)

    profile: Dict[str, object] = {
        "player_id": player_id,
        "player_name": player_name,
        "hidden_from_viewer": bool(player.get("hidden_from_viewer", False)),
        "structure_count": len(structures),
        "settlement_count": sum(1 for structure in structures if _structure_multiplier(structure) <= 1.0),
        "city_count": sum(1 for structure in structures if _structure_multiplier(structure) > 1.0),
        "expected_income": _rounded_resource_map(expected),
        "expected_production_by_resource": _rounded_resource_map(expected),
        "total_expected_income": _round(total_expected),
        "expected_production_total": _round(total_expected),
        "production_counts": dict(production["production_counts"]),
        "settlement_weighted_income": _rounded_resource_map(production["settlement_weighted"]),
        "city_weighted_income": _rounded_resource_map(production["city_weighted"]),
        "blocked_income": _rounded_resource_map(blocked),
        "blocked_production_by_resource": _rounded_resource_map(blocked),
        "total_blocked_income": _round(total_blocked),
        "blocked_production_total": _round(total_blocked),
        "blocked_counts": dict(production["blocked_counts"]),
        "contributing_structures": production["contributing_structures"],
        "diversity_score": diversity,
        "evenness_score": evenness,
        "balance_score": evenness,
        "concentration_score": concentration,
        "bottlenecks": bottlenecks,
        "scarcity_capture_score": scarcity_capture,
        "strong_resources": strongest,
        "weak_resources": weakest,
        "orientation_scores": orientation_scores,
        "top_orientations": top_orientations,
        "favored_plan": top_orientations[0] if top_orientations else None,
        "robber_burden": robber_burden,
        "robber_burden_score": robber_burden["blocked_share"],
        "resource_ranks": {},
        "total_income_rank": None,
        "leader_gap": {},
        "relative_table_metrics": {},
        "tags": _tags(
            expected=expected,
            orientation_scores=orientation_scores,
            top_orientations=top_orientations,
            bottlenecks=bottlenecks,
            diversity=diversity,
            evenness=evenness,
            concentration=concentration,
            robber_burden=robber_burden,
        ),
    }
    expansion = build_expansion_profile(
        state=state,
        player_id=player_id,
        player_name=player_name,
        player_profile=profile,
        board_economy=board_economy,
        slot_ordering=slot_ordering,
    )
    profile["expansion"] = expansion
    profile["expansion_potential"] = {
        "reachable_site_count": expansion.get("reachable_site_count", 0),
        "top_reachable_site_ev": expansion.get("top_reachable_site_ev", 0.0),
        "best_reachable_site_ids": [
            site.get("vertex_id")
            for site in expansion.get("top_reachable_sites", [])
            if isinstance(site, dict)
        ],
        "score": expansion.get("frontier_score", 0.0),
    }
    profile["upgrade_potential"] = {
        "upgradeable_settlement_count": expansion.get("upgradeable_settlement_count", 0),
        "top_upgradeable_settlement_score": expansion.get("best_upgradeable_settlement_score", 0.0),
        "city_upside_score": expansion.get("city_upside_score", 0.0),
    }
    profile["tags"] = sorted(set(list(profile.get("tags", [])) + list(expansion.get("tags", []))))
    return profile


def build_player_economy_v1(
    state: Dict[str, object],
    board_economy: Optional[Dict[str, object]] = None,
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build public, deterministic post-placement player economy profiles."""
    players = _player_list(state)
    profiles = [
        _build_player_profile(
            state=state,
            player=player,
            fallback_index=index,
            board_economy=board_economy,
            slot_ordering=slot_ordering,
        )
        for index, player in enumerate(players)
    ]
    relative = _rank_maps(profiles)
    for profile in profiles:
        player_id = profile.get("player_id")
        if player_id is None:
            continue
        relative_metrics = relative["relative_by_player"].get(str(player_id), {})
        profile["resource_ranks"] = relative_metrics.get("resource_ranks", {})
        profile["total_income_rank"] = relative_metrics.get("total_income_rank")
        profile["leader_gap"] = relative_metrics.get("leader_gap", {})
        profile["relative_table_metrics"] = {
            "rank": relative_metrics.get("rank"),
            "share": relative_metrics.get("share", 0.0),
            "leader_gap": relative_metrics.get("leader_gap_value", 0.0),
            "total_income_rank": profile["total_income_rank"],
            "total_table_share": relative_metrics.get("total_table_share", 0.0),
            "leader_gap_detail": profile["leader_gap"],
            "by_resource": profile["resource_ranks"],
        }
    pressure = build_player_pressure(profiles)
    for profile in profiles:
        player_id = profile.get("player_id")
        if player_id is None:
            continue
        pressure_block = dict((pressure.get("by_player") or {}).get(str(player_id), {}))
        profile["pressure"] = pressure_block
        profile["tags"] = sorted(set(list(profile.get("tags", [])) + list(pressure_block.get("tags", []))))
    profiles.sort(
        key=lambda profile: (
            int(profile.get("total_income_rank") or 999),
            str(profile.get("player_id") or ""),
        )
    )
    return {
        "schema_version": "player_economy_v1",
        "by_player": profiles,
        "table_leaders": relative["table_leaders"],
        "table_pressure": pressure.get("table_pressure", {}),
    }


def build_player_profiles(
    state: Dict[str, object],
    board_economy: Optional[Dict[str, object]] = None,
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Backward-compatible alias for older callers."""
    return build_player_economy_v1(state=state, board_economy=board_economy, slot_ordering=slot_ordering)
