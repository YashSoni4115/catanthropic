from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ai.economy.constants import TRACKED_RESOURCES
from ai.economy.normalize import clamp01, safe_div, score_from_cap


TIER_SPEED = {
    "connected": 0,
    "one_road": 1,
    "two_road": 2,
}


def _round(value: float) -> float:
    return round(float(value), 6)


def _player_id(profile: Dict[str, object]) -> Optional[str]:
    value = profile.get("player_id")
    return str(value) if value is not None else None


def _top_sites(profile: Dict[str, object], limit: int = 5) -> List[Dict[str, object]]:
    expansion = profile.get("expansion", {})
    sites = expansion.get("top_reachable_sites", []) if isinstance(expansion, dict) else []
    return [site for site in sites[:limit] if isinstance(site, dict)]


def _site_key(site: Dict[str, object]) -> Optional[str]:
    value = site.get("vertex_id")
    return str(value) if value is not None else None


def _tier_rank(site: Dict[str, object]) -> int:
    return TIER_SPEED.get(str(site.get("reachability_tier") or ""), 9)


def _site_score(site: Dict[str, object]) -> float:
    return float(site.get("site_score", 0.0))


def _site_ev(site: Dict[str, object]) -> float:
    return float(site.get("ev_score", 0.0))


def _resource_map(profile: Dict[str, object]) -> Dict[str, float]:
    raw = profile.get("expected_income", {})
    return {
        resource: float(raw.get(resource, 0.0)) if isinstance(raw, dict) else 0.0
        for resource in TRACKED_RESOURCES
    }


def _normalized_resource_vector(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0.0:
        return {resource: 0.0 for resource in TRACKED_RESOURCES}
    return {resource: safe_div(max(0.0, values.get(resource, 0.0)), total) for resource in TRACKED_RESOURCES}


def _resource_overlap(left: Dict[str, float], right: Dict[str, float]) -> float:
    left_norm = _normalized_resource_vector(left)
    right_norm = _normalized_resource_vector(right)
    return _round(sum(min(left_norm[resource], right_norm[resource]) for resource in TRACKED_RESOURCES))


def _bottleneck_resources(profile: Dict[str, object], threshold: float = 0.45) -> List[str]:
    bottlenecks = profile.get("bottlenecks", {})
    if not isinstance(bottlenecks, dict):
        return []
    return sorted(
        resource
        for resource, detail in bottlenecks.items()
        if resource in TRACKED_RESOURCES
        and isinstance(detail, dict)
        and float(detail.get("severity", 0.0)) >= threshold
    )


def _frontier_score(profile: Dict[str, object]) -> float:
    expansion = profile.get("expansion", {})
    return float(expansion.get("frontier_score", 0.0)) if isinstance(expansion, dict) else 0.0


def _boxed_in_score(profile: Dict[str, object]) -> float:
    expansion = profile.get("expansion", {})
    return float(expansion.get("boxed_in_score", 0.0)) if isinstance(expansion, dict) else 0.0


def _city_upside_score(profile: Dict[str, object]) -> float:
    expansion = profile.get("expansion", {})
    return float(expansion.get("city_upside_score", 0.0)) if isinstance(expansion, dict) else 0.0


def _total_income(profile: Dict[str, object]) -> float:
    return float(profile.get("total_expected_income", profile.get("expected_production_total", 0.0)))


def _rank(profile: Dict[str, object]) -> int:
    value = profile.get("total_income_rank")
    try:
        return int(value) if value is not None else 999
    except (TypeError, ValueError):
        return 999


def _site_index_by_player(profiles: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    output: Dict[str, Dict[str, Dict[str, object]]] = {}
    for profile in profiles:
        player_id = _player_id(profile)
        if player_id is None:
            continue
        output[player_id] = {}
        for site in _top_sites(profile):
            key = _site_key(site)
            if key is not None:
                output[player_id][key] = site
    return output


def _contested_expansion(
    profile: Dict[str, object],
    opponents: Sequence[Dict[str, object]],
    site_index: Dict[str, Dict[str, Dict[str, object]]],
) -> Dict[str, object]:
    player_id = _player_id(profile)
    player_sites = _top_sites(profile)
    contested: List[Dict[str, object]] = []
    equal_or_faster = 0
    player_ahead = 0
    rival_scores: Dict[str, float] = {}

    for site in player_sites:
        key = _site_key(site)
        if key is None:
            continue
        player_speed = _tier_rank(site)
        rivals: List[Dict[str, object]] = []
        for opponent in opponents:
            rival_id = _player_id(opponent)
            if rival_id is None or rival_id == player_id:
                continue
            rival_site = site_index.get(rival_id, {}).get(key)
            if rival_site is None:
                continue
            rival_speed = _tier_rank(rival_site)
            equal_fast = rival_speed <= player_speed
            if equal_fast:
                equal_or_faster += 1
            else:
                player_ahead += 1
            pressure = clamp01(
                0.48 * _site_score(site)
                + 0.32 * score_from_cap(_site_ev(site), 12.0)
                + 0.20 * (1.0 if equal_fast else 0.45)
            )
            rival_scores[rival_id] = rival_scores.get(rival_id, 0.0) + pressure
            rivals.append(
                {
                    "player_id": rival_id,
                    "reachability_tier": rival_site.get("reachability_tier"),
                    "site_score": _round(_site_score(rival_site)),
                    "equal_or_faster": equal_fast,
                }
            )
        if rivals:
            contested.append(
                {
                    "vertex_id": key,
                    "site_score": _round(_site_score(site)),
                    "ev_score": _round(_site_ev(site)),
                    "reachability_tier": site.get("reachability_tier"),
                    "rivals": sorted(rivals, key=lambda item: (str(item["player_id"]), str(item["reachability_tier"]))),
                }
            )

    contested_score = clamp01(
        0.55 * safe_div(len(contested), max(1, len(player_sites)))
        + 0.45 * safe_div(sum(_site_score(site) for site in contested), max(1.0, sum(_site_score(site) for site in player_sites)))
    )
    race_pressure = clamp01(
        0.70 * safe_div(equal_or_faster, max(1, len(player_sites)))
        + 0.30 * contested_score
    )
    strongest_rival = None
    if rival_scores:
        strongest_rival = max(rival_scores.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "contested_site_count": len(contested),
        "contested_top_site_fraction": _round(safe_div(len(contested), len(player_sites))),
        "top_contested_sites": sorted(contested, key=lambda item: (-float(item["site_score"]), str(item["vertex_id"])))[:5],
        "contested_frontier_score": _round(contested_score),
        "race_pressure_score": _round(race_pressure),
        "opponent_equal_or_faster_site_count": equal_or_faster,
        "player_ahead_site_count": player_ahead,
        "strongest_rival_by_expansion_race": strongest_rival,
    }


def _block_pressure(contested: Dict[str, object]) -> Dict[str, object]:
    vulnerable = 0
    strongest_blocker: Optional[str] = None
    blocker_scores: Dict[str, float] = {}
    for site in contested.get("top_contested_sites", []):
        if not isinstance(site, dict):
            continue
        tier = str(site.get("reachability_tier") or "")
        if tier == "connected":
            continue
        vulnerable += 1
        site_score = float(site.get("site_score", 0.0))
        for rival in site.get("rivals", []):
            if not isinstance(rival, dict):
                continue
            rival_id = rival.get("player_id")
            if rival_id is None:
                continue
            blocker_scores[str(rival_id)] = blocker_scores.get(str(rival_id), 0.0) + site_score
    if blocker_scores:
        strongest_blocker = max(blocker_scores.items(), key=lambda item: (item[1], item[0]))[0]
    block_score = clamp01(
        0.65 * safe_div(vulnerable, max(1, int(contested.get("contested_site_count", 0))))
        + 0.35 * float(contested.get("contested_frontier_score", 0.0))
    )
    return {
        "block_threat_score": _round(block_score),
        "vulnerable_frontier_count": vulnerable,
        "appears_cuttable": block_score >= 0.55 and vulnerable > 0,
        "strongest_blocker_rival": strongest_blocker,
    }


def _resource_competition(profile: Dict[str, object], opponents: Sequence[Dict[str, object]]) -> Dict[str, object]:
    own_resources = _resource_map(profile)
    own_bottlenecks = set(_bottleneck_resources(profile))
    own_strong = [str(resource) for resource in profile.get("strong_resources", []) if resource in TRACKED_RESOURCES]
    scarce_capture = float(profile.get("scarcity_capture_score", 0.0))
    rival_scores: Dict[str, float] = {}
    bottleneck_pressure = 0.0
    strong_line_pressure = 0.0

    for opponent in opponents:
        rival_id = _player_id(opponent)
        if rival_id is None:
            continue
        overlap = _resource_overlap(own_resources, _resource_map(opponent))
        rival_bottlenecks = set(_bottleneck_resources(opponent))
        bottleneck_overlap = safe_div(len(own_bottlenecks.intersection(rival_bottlenecks)), max(1, len(own_bottlenecks)))
        strong_overlap = 0.0
        if own_strong:
            opponent_resources = _resource_map(opponent)
            strong_overlap = safe_div(
                sum(1.0 for resource in own_strong if opponent_resources.get(resource, 0.0) > 0.0),
                len(own_strong),
            )
        score = clamp01(0.48 * overlap + 0.27 * bottleneck_overlap + 0.25 * strong_overlap)
        rival_scores[rival_id] = score
        bottleneck_pressure = max(bottleneck_pressure, bottleneck_overlap)
        strong_line_pressure = max(strong_line_pressure, strong_overlap)

    resource_score = max(rival_scores.values(), default=0.0)
    scarce_pressure = clamp01(resource_score * scarce_capture)
    strongest_rival = max(rival_scores.items(), key=lambda item: (item[1], item[0]))[0] if rival_scores else None
    return {
        "resource_competition_score": _round(resource_score),
        "resource_overlap_by_rival": {player_id: _round(score) for player_id, score in sorted(rival_scores.items())},
        "bottleneck_resource_pressure_score": _round(bottleneck_pressure),
        "scarce_capture_pressure_score": _round(scarce_pressure),
        "strong_resource_line_contested": strong_line_pressure >= 0.5 and resource_score >= 0.35,
        "strongest_resource_competition_rival": strongest_rival,
    }


def _leader_pressure(profile: Dict[str, object], opponents: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = _total_income(profile)
    max_opponent_total = max((_total_income(opponent) for opponent in opponents), default=0.0)
    max_opponent_frontier = max((_frontier_score(opponent) for opponent in opponents), default=0.0)
    own_frontier = _frontier_score(profile)
    is_leader = _rank(profile) == 1
    leader_gap_pressure = score_from_cap(max(0.0, max_opponent_total - total), 8.0)
    opponent_frontier_pressure = score_from_cap(max(0.0, max_opponent_frontier - own_frontier), 1.0)
    strong_now_fragile_later = is_leader and opponent_frontier_pressure >= 0.35
    weak_now_latent = (not is_leader) and own_frontier >= 0.65 and _total_income(profile) <= max_opponent_total
    leader_pressure = clamp01(
        0.45 * (opponent_frontier_pressure if is_leader else leader_gap_pressure)
        + 0.25 * _boxed_in_score(profile)
        + 0.20 * score_from_cap(max_opponent_frontier, 1.0)
        + 0.10 * (1.0 if strong_now_fragile_later else 0.0)
    )
    frontier_rival = None
    if opponents:
        frontier_rival = max(opponents, key=lambda opponent: (_frontier_score(opponent), str(_player_id(opponent) or "")))
    return {
        "is_public_economy_leader": is_leader,
        "leader_gap_pressure": _round(leader_gap_pressure),
        "opponents_stronger_combined_frontier": max_opponent_frontier > own_frontier,
        "strong_now_fragile_later": strong_now_fragile_later,
        "weak_now_latent_expansion": weak_now_latent,
        "leader_pressure_score": _round(leader_pressure),
        "strongest_frontier_rival": _player_id(frontier_rival) if frontier_rival is not None else None,
    }


def _tags(pressure: Dict[str, object]) -> List[str]:
    tags: List[str] = []
    if float(pressure.get("contested_frontier_score", 0.0)) >= 0.55:
        tags.append("highly_contested")
    race = float(pressure.get("race_pressure_score", 0.0))
    if race >= 0.55:
        tags.append("race_behind")
    elif int(pressure.get("player_ahead_site_count", 0)) > int(pressure.get("opponent_equal_or_faster_site_count", 0)):
        tags.append("race_ahead")
    if bool(pressure.get("appears_cuttable")):
        tags.append("cutoff_risk")
    if bool(pressure.get("is_public_economy_leader")) and float(pressure.get("leader_pressure_score", 0.0)) >= 0.45:
        tags.append("leader_under_pressure")
    if bool(pressure.get("weak_now_latent_expansion")):
        tags.append("latent_expansion_threat")
    if bool(pressure.get("strong_resource_line_contested")):
        tags.append("resource_line_contested")
    return sorted(set(tags))


def _compose_pressure(profile: Dict[str, object], opponents: Sequence[Dict[str, object]], site_index: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, object]:
    contested = _contested_expansion(profile, opponents, site_index)
    block = _block_pressure(contested)
    resources = _resource_competition(profile, opponents)
    leader = _leader_pressure(profile, opponents)
    top_rival = (
        contested.get("strongest_rival_by_expansion_race")
        or block.get("strongest_blocker_rival")
        or resources.get("strongest_resource_competition_rival")
        or leader.get("strongest_frontier_rival")
    )
    pressure = {
        "schema_version": "player_pressure_v1",
        **contested,
        **block,
        **resources,
        **leader,
        "top_rival_player_id": top_rival,
        "tags": [],
    }
    pressure["tags"] = _tags(pressure)
    return pressure


def build_player_pressure(profiles: List[Dict[str, object]]) -> Dict[str, object]:
    """Build deterministic opponent-interaction pressure for player economy profiles."""
    site_index = _site_index_by_player(profiles)
    by_player: Dict[str, Dict[str, object]] = {}
    for profile in profiles:
        player_id = _player_id(profile)
        if player_id is None:
            continue
        opponents = [opponent for opponent in profiles if _player_id(opponent) != player_id]
        by_player[player_id] = _compose_pressure(profile, opponents, site_index)

    frontier_leader = max(
        profiles,
        key=lambda profile: (_frontier_score(profile), str(_player_id(profile) or "")),
        default=None,
    )
    pressure_leader_id = None
    pressure_leader_score = 0.0
    if by_player:
        pressure_leader_id, pressure_block = max(
            by_player.items(),
            key=lambda item: (
                float(item[1].get("leader_pressure_score", 0.0))
                + float(item[1].get("race_pressure_score", 0.0))
                + float(item[1].get("block_threat_score", 0.0)),
                item[0],
            ),
        )
        pressure_leader_score = (
            float(pressure_block.get("leader_pressure_score", 0.0))
            + float(pressure_block.get("race_pressure_score", 0.0))
            + float(pressure_block.get("block_threat_score", 0.0))
        )
    return {
        "schema_version": "player_pressure_table_v1",
        "by_player": by_player,
        "table_pressure": {
            "frontier_leader_player_id": _player_id(frontier_leader) if frontier_leader is not None else None,
            "highest_pressure_player_id": pressure_leader_id,
            "highest_pressure_composite_score": _round(pressure_leader_score),
        },
    }
