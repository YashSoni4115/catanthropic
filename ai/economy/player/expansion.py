from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from ai.economy.constants import (
    BUILD_RECIPES,
    MAX_SINGLE_VERTEX_PIPS,
    PIP_WEIGHTS,
    TRACKED_RESOURCES,
    VALID_STRUCTURE_SLOTS,
    adjacent_vertices_for_vertex,
    road_endpoint_vertices,
    vertex_adjacent_tiles,
    vertex_id,
    vertex_incident_roads,
)
from ai.economy.normalize import clamp01, diversity_score, evenness_score, safe_div, score_from_cap


VertexKey = Tuple[int, int, int]
RoadKey = Tuple[int, int, int]
TileCoord = Tuple[int, int]
ResourceMap = Dict[str, float]


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _empty_resource_map() -> ResourceMap:
    return {resource: 0.0 for resource in TRACKED_RESOURCES}


def _valid_structure_slots(slot_ordering: Optional[Dict[str, object]]) -> List[VertexKey]:
    if not isinstance(slot_ordering, dict):
        return list(VALID_STRUCTURE_SLOTS)
    raw_slots = slot_ordering.get("structure_slots")
    raw_valid = slot_ordering.get("structure_slot_valid")
    if not isinstance(raw_slots, list) or not isinstance(raw_valid, list):
        return list(VALID_STRUCTURE_SLOTS)
    output: List[VertexKey] = []
    for entry, is_valid in zip(raw_slots, raw_valid):
        if not is_valid or not isinstance(entry, dict):
            continue
        x_coord = _safe_int(entry.get("x"))
        y_coord = _safe_int(entry.get("y"))
        orientation = _safe_int(entry.get("orientation"))
        if x_coord is not None and y_coord is not None and orientation is not None:
            output.append((x_coord, y_coord, orientation))
    return output or list(VALID_STRUCTURE_SLOTS)


def _tile_map(state: Dict[str, object]) -> Dict[TileCoord, Dict[str, object]]:
    output: Dict[TileCoord, Dict[str, object]] = {}
    tiles = state.get("tiles", [])
    if not isinstance(tiles, list):
        return output
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        x_coord = _safe_int(tile.get("x"))
        y_coord = _safe_int(tile.get("y"))
        number = _safe_int(tile.get("number"))
        if x_coord is None or y_coord is None:
            continue
        output[(x_coord, y_coord)] = {
            "x": x_coord,
            "y": y_coord,
            "tile_id": tile.get("tile_id") or f"tile:{x_coord}:{y_coord}",
            "type": str(tile.get("type") or "").strip().upper(),
            "number": number,
            "pip_weight": float(tile.get("pip_weight", PIP_WEIGHTS.get(number, 0.0))) if number is not None else 0.0,
        }
    return output


def _vertex_from_structure(structure: Dict[str, object]) -> Optional[VertexKey]:
    x_coord = _safe_int(structure.get("x"))
    y_coord = _safe_int(structure.get("y"))
    orientation = _safe_int(structure.get("orientation"))
    if x_coord is None or y_coord is None or orientation is None:
        return None
    return (x_coord, y_coord, orientation)


def _road_key(road: Dict[str, object]) -> Optional[RoadKey]:
    x_coord = _safe_int(road.get("x"))
    y_coord = _safe_int(road.get("y"))
    orientation = _safe_int(road.get("orientation"))
    if x_coord is None or y_coord is None or orientation is None:
        return None
    return (x_coord, y_coord, orientation)


def _road_endpoints(road: Dict[str, object]) -> List[VertexKey]:
    raw_endpoints = road.get("endpoint_vertices")
    endpoints: List[VertexKey] = []
    if isinstance(raw_endpoints, list):
        for endpoint in raw_endpoints:
            if not isinstance(endpoint, dict):
                continue
            x_coord = _safe_int(endpoint.get("x"))
            y_coord = _safe_int(endpoint.get("y"))
            orientation = _safe_int(endpoint.get("orientation"))
            if x_coord is not None and y_coord is not None and orientation is not None:
                endpoints.append((x_coord, y_coord, orientation))
    if endpoints:
        return endpoints
    key = _road_key(road)
    if key is None:
        return []
    return [
        (endpoint["x"], endpoint["y"], endpoint["orientation"])
        for endpoint in road_endpoint_vertices(*key)
    ]


def _owned_structures(state: Dict[str, object], player_id: Optional[str], player_name: Optional[str]) -> List[Dict[str, object]]:
    structures = state.get("structures", [])
    if not isinstance(structures, list):
        return []
    output: List[Dict[str, object]] = []
    for structure in structures:
        if not isinstance(structure, dict):
            continue
        owner_id = structure.get("owner_id")
        owner_name = structure.get("owner_name")
        if player_id is not None and str(owner_id) == str(player_id):
            output.append(structure)
        elif owner_id is None and player_name is not None and str(owner_name) == str(player_name):
            output.append(structure)
    return output


def _road_sets(state: Dict[str, object], player_id: Optional[str], player_name: Optional[str]) -> Dict[str, object]:
    roads = state.get("roads", [])
    occupied_roads: Set[RoadKey] = set()
    owned_roads: Set[RoadKey] = set()
    owned_endpoints: Set[VertexKey] = set()
    if not isinstance(roads, list):
        return {"occupied_roads": occupied_roads, "owned_roads": owned_roads, "owned_endpoints": owned_endpoints}
    for road in roads:
        if not isinstance(road, dict):
            continue
        key = _road_key(road)
        if key is None:
            continue
        occupied_roads.add(key)
        owner_id = road.get("owner_id")
        owner_name = road.get("owner_name")
        owns = (player_id is not None and str(owner_id) == str(player_id)) or (
            owner_id is None and player_name is not None and str(owner_name) == str(player_name)
        )
        if owns:
            owned_roads.add(key)
            owned_endpoints.update(_road_endpoints(road))
    return {"occupied_roads": occupied_roads, "owned_roads": owned_roads, "owned_endpoints": owned_endpoints}


def _occupied_vertices(state: Dict[str, object]) -> Dict[VertexKey, Dict[str, object]]:
    structures = state.get("structures", [])
    output: Dict[VertexKey, Dict[str, object]] = {}
    if not isinstance(structures, list):
        return output
    for structure in structures:
        if not isinstance(structure, dict):
            continue
        key = _vertex_from_structure(structure)
        if key is not None:
            output[key] = structure
    return output


def _distance_blocked_vertices(occupied: Dict[VertexKey, Dict[str, object]]) -> Set[VertexKey]:
    blocked = set(occupied.keys())
    for key in occupied.keys():
        for neighbor in adjacent_vertices_for_vertex(*key):
            blocked.add((neighbor["x"], neighbor["y"], neighbor["orientation"]))
    return blocked


def _network_vertices(
    owned_structures: Sequence[Dict[str, object]],
    owned_road_endpoints: Set[VertexKey],
) -> Set[VertexKey]:
    output = set(owned_road_endpoints)
    for structure in owned_structures:
        key = _vertex_from_structure(structure)
        if key is not None:
            output.add(key)
    return output


def _resource_mix_for_vertex(
    key: VertexKey,
    tiles_by_coord: Dict[TileCoord, Dict[str, object]],
) -> Tuple[ResourceMap, List[Dict[str, object]]]:
    mix = _empty_resource_map()
    adjacent_tiles: List[Dict[str, object]] = []
    for tile_coord in vertex_adjacent_tiles(*key):
        tile = tiles_by_coord.get((tile_coord["x"], tile_coord["y"]))
        if tile is None:
            continue
        resource = str(tile.get("type") or "").strip().upper()
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
        if resource in mix:
            mix[resource] += pip_weight
    return mix, adjacent_tiles


def _recipe_coverage_score(resource_mix: ResourceMap) -> float:
    scores = []
    for recipe in BUILD_RECIPES.values():
        if not recipe:
            continue
        scores.append(
            sum(1.0 for resource in recipe if resource_mix.get(resource, 0.0) > 0.0)
            / float(len(recipe))
        )
    return _round(sum(scores) / float(len(scores))) if scores else 0.0


def _board_stats(board_economy: Optional[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    raw_by_resource = {}
    if isinstance(board_economy, dict):
        resource_environment = board_economy.get("resource_environment", {})
        if isinstance(resource_environment, dict) and isinstance(resource_environment.get("by_resource"), dict):
            raw_by_resource = resource_environment["by_resource"]
    fragility = {}
    if isinstance(board_economy, dict):
        desert = board_economy.get("desert_robber", {})
        if isinstance(desert, dict) and isinstance(desert.get("robber_fragility_by_resource"), dict):
            fragility = desert["robber_fragility_by_resource"]
    output: Dict[str, Dict[str, float]] = {}
    for resource in TRACKED_RESOURCES:
        stats = raw_by_resource.get(resource, {}) if isinstance(raw_by_resource, dict) else {}
        output[resource] = {
            "scarcity_score": float(stats.get("scarcity_score", 0.0)) if isinstance(stats, dict) else 0.0,
            "robber_fragility_score": float(fragility.get(resource, 0.0)) if isinstance(fragility, dict) else 0.0,
        }
    return output


def _scarcity_capture_score(resource_mix: ResourceMap, stats: Dict[str, Dict[str, float]]) -> float:
    total = sum(resource_mix.values())
    if total <= 0.0:
        return 0.0
    weighted = sum(resource_mix[resource] * stats[resource]["scarcity_score"] for resource in TRACKED_RESOURCES)
    return _round(clamp01(0.65 * safe_div(weighted, total) + 0.35 * score_from_cap(weighted, 10.0)))


def _robber_fragility_score(resource_mix: ResourceMap, adjacent_tiles: Sequence[Dict[str, object]], stats: Dict[str, Dict[str, float]]) -> float:
    total = sum(resource_mix.values())
    if total <= 0.0:
        return 0.0
    max_tile_pips = max((float(tile.get("pip_weight", 0.0)) for tile in adjacent_tiles), default=0.0)
    tile_fragility = safe_div(max_tile_pips, total)
    board_fragility = safe_div(
        sum(resource_mix[resource] * stats[resource]["robber_fragility_score"] for resource in TRACKED_RESOURCES),
        total,
    )
    return _round(clamp01(0.70 * tile_fragility + 0.30 * board_fragility))


def _player_expected(player_profile: Optional[Dict[str, object]]) -> ResourceMap:
    raw = player_profile.get("expected_income", {}) if isinstance(player_profile, dict) else {}
    return {
        resource: float(raw.get(resource, 0.0)) if isinstance(raw, dict) else 0.0
        for resource in TRACKED_RESOURCES
    }


def _player_bottlenecks(player_profile: Optional[Dict[str, object]]) -> Dict[str, float]:
    raw = player_profile.get("bottlenecks", {}) if isinstance(player_profile, dict) else {}
    output = {resource: 0.0 for resource in TRACKED_RESOURCES}
    if isinstance(raw, dict):
        for resource, detail in raw.items():
            if resource in output and isinstance(detail, dict):
                output[resource] = float(detail.get("severity", 0.0))
    return output


def _orientation_scores(expected: ResourceMap) -> Dict[str, float]:
    return {
        "road": _round(score_from_cap(min(expected["BRICK"], expected["LUMBER"]), 5.0)),
        "settlement": _round(score_from_cap(min(expected["BRICK"], expected["LUMBER"], expected["WOOL"], expected["GRAIN"]), 5.0)),
        "city": _round(score_from_cap(min(expected["ORE"] / 3.0, expected["GRAIN"] / 2.0), 2.5)),
        "dev": _round(score_from_cap(min(expected["ORE"], expected["GRAIN"], expected["WOOL"]), 5.0)),
    }


def _synergy_with_player(resource_mix: ResourceMap, player_profile: Optional[Dict[str, object]]) -> float:
    total = sum(resource_mix.values())
    if total <= 0.0:
        return 0.0
    current = _player_expected(player_profile)
    bottlenecks = _player_bottlenecks(player_profile)
    bottleneck_fit = safe_div(sum(resource_mix[resource] * bottlenecks[resource] for resource in TRACKED_RESOURCES), total)
    current_diversity = diversity_score(current)
    combined = {resource: current[resource] + resource_mix[resource] for resource in TRACKED_RESOURCES}
    diversity_uplift = max(0.0, diversity_score(combined) - current_diversity)
    current_orientation = _orientation_scores(current)
    combined_orientation = _orientation_scores(combined)
    orientation_uplift = max(
        0.0,
        max(combined_orientation.values(), default=0.0) - max(current_orientation.values(), default=0.0),
    )
    return _round(clamp01(0.55 * bottleneck_fit + 0.25 * score_from_cap(diversity_uplift, 0.4) + 0.20 * score_from_cap(orientation_uplift, 0.4)))


def _empty_incident_road_to_network(
    site: VertexKey,
    network_vertices: Set[VertexKey],
    occupied_roads: Set[RoadKey],
) -> bool:
    for incident in vertex_incident_roads(*site):
        road_key = (incident["x"], incident["y"], incident["orientation"])
        if road_key in occupied_roads:
            continue
        endpoints = {
            (endpoint["x"], endpoint["y"], endpoint["orientation"])
            for endpoint in road_endpoint_vertices(*road_key)
        }
        if site in endpoints and any(endpoint in network_vertices for endpoint in endpoints if endpoint != site):
            return True
    return False


def _two_road_plausible(
    site: VertexKey,
    network_vertices: Set[VertexKey],
    occupied_roads: Set[RoadKey],
) -> bool:
    for neighbor in adjacent_vertices_for_vertex(*site):
        neighbor_key = (neighbor["x"], neighbor["y"], neighbor["orientation"])
        if _empty_incident_road_to_network(neighbor_key, network_vertices, occupied_roads):
            road_between = next(
                (
                    (road["x"], road["y"], road["orientation"])
                    for road in vertex_incident_roads(*site)
                    if neighbor_key in {
                        (endpoint["x"], endpoint["y"], endpoint["orientation"])
                        for endpoint in road_endpoint_vertices(road["x"], road["y"], road["orientation"])
                    }
                ),
                None,
            )
            if road_between is not None and road_between not in occupied_roads:
                return True
    return False


def _reachability_tier(
    site: VertexKey,
    owned_roads: Set[RoadKey],
    occupied_roads: Set[RoadKey],
    network_vertices: Set[VertexKey],
) -> Optional[str]:
    for incident in vertex_incident_roads(*site):
        road_key = (incident["x"], incident["y"], incident["orientation"])
        if road_key in owned_roads:
            return "connected"
    if _empty_incident_road_to_network(site, network_vertices, occupied_roads):
        return "one_road"
    if _two_road_plausible(site, network_vertices, occupied_roads):
        return "two_road"
    return None


def _site_score(
    resource_mix: ResourceMap,
    adjacent_tiles: Sequence[Dict[str, object]],
    board_stats: Dict[str, Dict[str, float]],
    player_profile: Optional[Dict[str, object]],
    reachability_tier: str,
) -> Dict[str, float]:
    total = sum(resource_mix.values())
    ev_norm = score_from_cap(total, MAX_SINGLE_VERTEX_PIPS)
    diversity = diversity_score(resource_mix)
    evenness = evenness_score(resource_mix.values())
    recipe = _recipe_coverage_score(resource_mix)
    scarcity = _scarcity_capture_score(resource_mix, board_stats)
    robber = _robber_fragility_score(resource_mix, adjacent_tiles, board_stats)
    synergy = _synergy_with_player(resource_mix, player_profile)
    tier_bonus = {"connected": 1.0, "one_road": 0.82, "two_road": 0.64}.get(reachability_tier, 0.0)
    raw_score = (
        0.24 * ev_norm
        + 0.16 * recipe
        + 0.15 * synergy
        + 0.13 * diversity
        + 0.10 * evenness
        + 0.10 * scarcity
        + 0.12 * tier_bonus
        - 0.08 * robber
    )
    return {
        "ev_score": _round(total),
        "ev_score_norm": _round(ev_norm),
        "diversity_score": _round(diversity),
        "evenness_score": _round(evenness),
        "recipe_coverage_score": recipe,
        "scarcity_capture_score": scarcity,
        "robber_fragility_score": robber,
        "synergy_with_player_score": synergy,
        "reachability_score": _round(tier_bonus),
        "site_score": _round(clamp01(raw_score)),
    }


def _upgradeable_settlements(
    owned_structures: Sequence[Dict[str, object]],
    tiles_by_coord: Dict[TileCoord, Dict[str, object]],
    board_stats: Dict[str, Dict[str, float]],
    player_profile: Optional[Dict[str, object]],
    limit: int,
) -> Dict[str, object]:
    city_orientation = float((player_profile or {}).get("orientation_scores", {}).get("city", 0.0)) if isinstance(player_profile, dict) else 0.0
    settlements = [
        structure
        for structure in owned_structures
        if str(structure.get("type") or "settlement").strip().lower() == "settlement"
    ]
    upgrades: List[Dict[str, object]] = []
    for structure in settlements:
        key = _vertex_from_structure(structure)
        if key is None:
            continue
        resource_mix, adjacent_tiles = _resource_mix_for_vertex(key, tiles_by_coord)
        metrics = _site_score(resource_mix, adjacent_tiles, board_stats, player_profile, "connected")
        ore_grain = min(score_from_cap(resource_mix["ORE"], 5.0), score_from_cap(resource_mix["GRAIN"], 5.0))
        upgrade_score = clamp01(0.62 * metrics["site_score"] + 0.23 * city_orientation + 0.15 * ore_grain)
        upgrades.append(
            {
                "structure_id": structure.get("structure_id") or f"structure:{key[0]}:{key[1]}:{key[2]}",
                "vertex_id": vertex_id(*key),
                "ev_score": metrics["ev_score"],
                "resource_mix": {resource: _round(resource_mix[resource]) for resource in TRACKED_RESOURCES},
                "upgrade_score": _round(upgrade_score),
                "ore_grain_fit_score": _round(ore_grain),
            }
        )
    upgrades.sort(key=lambda item: (-float(item["upgrade_score"]), str(item["vertex_id"])))
    best = float(upgrades[0]["upgrade_score"]) if upgrades else 0.0
    avg = safe_div(sum(float(item["upgrade_score"]) for item in upgrades), len(upgrades))
    city_upside = clamp01(0.65 * best + 0.35 * city_orientation)
    return {
        "upgradeable_settlement_count": len(upgrades),
        "top_upgradeable_settlements": upgrades[:limit],
        "best_upgradeable_settlement_score": _round(best),
        "average_upgradeable_settlement_score": _round(avg),
        "city_upside_score": _round(city_upside),
        "ore_grain_upgrade_path_score": _round(city_orientation),
    }


def _expansion_tags(frontier_score: float, boxed_in_score: float, city_upside_score: float, top_site_score: float) -> List[str]:
    tags: List[str] = []
    if boxed_in_score >= 0.75:
        tags.append("boxed_in")
    if boxed_in_score <= 0.35 and frontier_score >= 0.35:
        tags.append("open_frontier")
    if frontier_score >= 0.65:
        tags.append("strong_expansion")
    if frontier_score <= 0.25:
        tags.append("weak_expansion")
    if city_upside_score >= 0.60:
        tags.append("city_upside_strong")
    if top_site_score >= 0.60:
        tags.append("settlement_followup_strong")
    return sorted(set(tags))


def build_expansion_profile(
    state: Dict[str, object],
    player_id: Optional[str] = None,
    player_name: Optional[str] = None,
    player_profile: Optional[Dict[str, object]] = None,
    board_economy: Optional[Dict[str, object]] = None,
    slot_ordering: Optional[Dict[str, object]] = None,
    limit: int = 5,
) -> Dict[str, object]:
    """Evaluate deterministic settlement follow-up and city-upgrade potential."""
    if player_id is None and isinstance(player_profile, dict):
        raw_id = player_profile.get("player_id")
        player_id = str(raw_id) if raw_id is not None else None
    if player_name is None and isinstance(player_profile, dict):
        raw_name = player_profile.get("player_name")
        player_name = str(raw_name) if raw_name is not None else None

    tiles_by_coord = _tile_map(state)
    valid_slots = _valid_structure_slots(slot_ordering)
    occupied = _occupied_vertices(state)
    illegal_by_distance = _distance_blocked_vertices(occupied)
    owned_structures = _owned_structures(state, player_id, player_name)
    road_state = _road_sets(state, player_id, player_name)
    occupied_roads = road_state["occupied_roads"]
    owned_roads = road_state["owned_roads"]
    network_vertices = _network_vertices(owned_structures, road_state["owned_endpoints"])
    board_stats = _board_stats(board_economy)

    reachable_sites: List[Dict[str, object]] = []
    for site in valid_slots:
        if site in illegal_by_distance:
            continue
        tier = _reachability_tier(site, owned_roads, occupied_roads, network_vertices)
        if tier is None:
            continue
        resource_mix, adjacent_tiles = _resource_mix_for_vertex(site, tiles_by_coord)
        if not adjacent_tiles:
            continue
        metrics = _site_score(resource_mix, adjacent_tiles, board_stats, player_profile, tier)
        reachable_sites.append(
            {
                "vertex_id": vertex_id(*site),
                "x": site[0],
                "y": site[1],
                "orientation": site[2],
                "reachability_tier": tier,
                "adjacent_tiles": adjacent_tiles,
                "resource_mix": {resource: _round(resource_mix[resource]) for resource in TRACKED_RESOURCES},
                **metrics,
            }
        )
    reachable_sites.sort(key=lambda item: (-float(item["site_score"]), -float(item["ev_score"]), str(item["vertex_id"])))

    top_sites = reachable_sites[:limit]
    top_site_score = float(top_sites[0]["site_score"]) if top_sites else 0.0
    top_site_ev = float(top_sites[0]["ev_score"]) if top_sites else 0.0
    average_top_score = safe_div(sum(float(site["site_score"]) for site in top_sites), len(top_sites))
    combined_top_mix = _empty_resource_map()
    for site in top_sites:
        mix = site.get("resource_mix", {})
        if isinstance(mix, dict):
            for resource in TRACKED_RESOURCES:
                combined_top_mix[resource] += float(mix.get(resource, 0.0))
    frontier_diversity = diversity_score(combined_top_mix)
    frontier_recipe = _recipe_coverage_score(combined_top_mix)
    settlement_orientation = float((player_profile or {}).get("orientation_scores", {}).get("settlement", 0.0)) if isinstance(player_profile, dict) else 0.0
    road_orientation = float((player_profile or {}).get("orientation_scores", {}).get("road", 0.0)) if isinstance(player_profile, dict) else 0.0
    plan_continuity = max(settlement_orientation, road_orientation)
    frontier_score = clamp01(
        0.25 * score_from_cap(len(reachable_sites), 6.0)
        + 0.25 * top_site_score
        + 0.18 * average_top_score
        + 0.13 * frontier_diversity
        + 0.11 * frontier_recipe
        + 0.08 * plan_continuity
    )
    boxed_in_score = clamp01(1.0 - score_from_cap(len(reachable_sites), 4.0))
    upgrade = _upgradeable_settlements(owned_structures, tiles_by_coord, board_stats, player_profile, limit)
    tags = _expansion_tags(
        frontier_score=frontier_score,
        boxed_in_score=boxed_in_score,
        city_upside_score=float(upgrade["city_upside_score"]),
        top_site_score=top_site_score,
    )
    return {
        "schema_version": "player_expansion_v1",
        "reachable_site_count": len(reachable_sites),
        "top_reachable_sites": top_sites,
        "top_reachable_site_score": _round(top_site_score),
        "top_reachable_site_ev": _round(top_site_ev),
        "average_top_reachable_site_score": _round(average_top_score),
        "frontier_score": _round(frontier_score),
        "boxed_in_score": _round(boxed_in_score),
        "appears_boxed_in": boxed_in_score >= 0.75,
        "appears_open": boxed_in_score <= 0.35 and frontier_score >= 0.35,
        "frontier_resource_diversity_score": _round(frontier_diversity),
        "frontier_recipe_coverage_score": _round(frontier_recipe),
        "plan_continuity_score": _round(plan_continuity),
        **upgrade,
        "tags": tags,
    }
