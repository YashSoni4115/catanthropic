from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ai.economy.constants import EXPECTED_PRODUCTION_CAP, PORT_RESOURCE_MAP, TRACKED_RESOURCES
from ai.economy.normalize import clamp01, diversity_score, evenness_score, rank_with_gaps, score_from_cap
from ai.economy.player.expansion import build_expansion_profile
from ai.economy.player.pressure import build_player_pressure
from ai.economy.tags import player_economy_tags


def _orientation_scores(expected_by_resource: Dict[str, float]) -> Dict[str, float]:
    brick = float(expected_by_resource.get("BRICK", 0.0))
    wool = float(expected_by_resource.get("WOOL", 0.0))
    ore = float(expected_by_resource.get("ORE", 0.0))
    grain = float(expected_by_resource.get("GRAIN", 0.0))
    lumber = float(expected_by_resource.get("LUMBER", 0.0))
    return {
        "road": score_from_cap(min(brick, lumber), 5.0),
        "settlement": score_from_cap(min(brick, lumber, wool, grain), 5.0),
        "city": score_from_cap(min(ore / 3.0 if ore > 0.0 else 0.0, grain / 2.0 if grain > 0.0 else 0.0), 2.5),
        "dev": score_from_cap(min(ore, grain, wool), 5.0),
    }


def _bottlenecks(expected_by_resource: Dict[str, float], orientation_scores: Dict[str, float]) -> List[Dict[str, float | str]]:
    recipes = {
        "road": ("BRICK", "LUMBER"),
        "settlement": ("BRICK", "LUMBER", "WOOL", "GRAIN"),
        "city": ("ORE", "GRAIN"),
        "dev": ("ORE", "GRAIN", "WOOL"),
    }
    top_plans = sorted(orientation_scores.items(), key=lambda item: (-item[1], item[0]))[:2]
    severity_by_resource: Dict[str, float] = {}
    for plan_name, _ in top_plans:
        components = recipes[plan_name]
        for resource in components:
            severity = 1.0 - score_from_cap(float(expected_by_resource.get(resource, 0.0)), 5.0)
            severity_by_resource[resource] = max(severity_by_resource.get(resource, 0.0), severity)
    ordered = sorted(severity_by_resource.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"resource": resource, "severity": severity}
        for resource, severity in ordered[:3]
    ]


def _port_leverage_score(ports: Dict[str, bool], expected_by_resource: Dict[str, float], total_expected: float, balance_score: float) -> float:
    if not isinstance(ports, dict):
        return 0.0
    leverage_scores: List[float] = []
    imbalance = 1.0 - balance_score
    for port_name, owned in ports.items():
        if not owned:
            continue
        if port_name == "three_to_one":
            leverage_scores.append(clamp01(0.6 * score_from_cap(total_expected, EXPECTED_PRODUCTION_CAP) + 0.4 * imbalance))
            continue
        target_resource = PORT_RESOURCE_MAP.get(port_name)
        if target_resource is None:
            continue
        target_pips = float(expected_by_resource.get(target_resource, 0.0))
        leverage_scores.append(clamp01(0.7 * score_from_cap(target_pips, 5.0) + 0.3 * imbalance))
    return max(leverage_scores) if leverage_scores else 0.0


def _build_base_profile(
    state: Dict[str, object],
    player: Dict[str, object],
    slot_ordering: Optional[Dict[str, object]],
) -> Dict[str, object]:
    player_id = player.get("player_id")
    player_structures = [
        structure
        for structure in state.get("structures", [])
        if isinstance(structure, dict) and structure.get("owner_id") == player_id
    ]
    tile_map = {
        (int(tile["x"]), int(tile["y"])): tile
        for tile in state.get("tiles", [])
        if isinstance(tile, dict) and isinstance(tile.get("x"), int) and isinstance(tile.get("y"), int)
    }
    expected_by_resource = {resource: 0.0 for resource in TRACKED_RESOURCES}
    blocked_by_resource = {resource: 0.0 for resource in TRACKED_RESOURCES}
    robber_xy = None
    robber = state.get("robber")
    if isinstance(robber, dict) and isinstance(robber.get("x"), int) and isinstance(robber.get("y"), int):
        robber_xy = (int(robber["x"]), int(robber["y"]))

    for structure in player_structures:
        structure_type = str(structure.get("type") or "settlement")
        multiplier = 2.0 if structure_type == "city" else 1.0
        for adjacent in structure.get("adjacent_tiles", []) if isinstance(structure.get("adjacent_tiles"), list) else []:
            if not isinstance(adjacent, dict):
                continue
            key = (adjacent.get("x"), adjacent.get("y"))
            if not (isinstance(key[0], int) and isinstance(key[1], int)):
                continue
            tile = tile_map.get((key[0], key[1]))
            if tile is None:
                continue
            resource = str(tile.get("type") or "").upper()
            number = tile.get("number")
            if resource not in expected_by_resource or not isinstance(number, int):
                continue
            pip_weight = {
                2: 1.0,
                3: 2.0,
                4: 3.0,
                5: 4.0,
                6: 5.0,
                8: 5.0,
                9: 4.0,
                10: 3.0,
                11: 2.0,
                12: 1.0,
            }.get(number, 0.0) * multiplier
            expected_by_resource[resource] += pip_weight
            if robber_xy == (key[0], key[1]):
                blocked_by_resource[resource] += pip_weight

    expected_total = sum(expected_by_resource.values())
    blocked_total = sum(blocked_by_resource.values())
    balance = evenness_score(expected_by_resource.values())
    orientation_scores = _orientation_scores(expected_by_resource)
    favored_plan = max(orientation_scores.items(), key=lambda item: (item[1], item[0]))[0]
    bottlenecks = _bottlenecks(expected_by_resource, orientation_scores)
    expansion = build_expansion_profile(state=state, player_id=player_id, slot_ordering=slot_ordering)
    port_leverage = _port_leverage_score(player.get("ports", {}), expected_by_resource, expected_total, balance)
    robber_burden = 0.0 if expected_total <= 0.0 else blocked_total / expected_total
    tags = player_economy_tags(
        favored_plan=favored_plan,
        expected_by_resource=expected_by_resource,
        bottlenecks=bottlenecks,
        port_leverage_score=port_leverage,
        robber_burden_score=robber_burden,
    )
    return {
        "player_id": player_id,
        "player_name": player.get("player_name"),
        "hidden_from_viewer": bool(player.get("hidden_from_viewer", False)),
        "victory_points_visible": player.get("victory_points_visible"),
        "total_resource_cards": player.get("total_resource_cards"),
        "expected_production_total": expected_total,
        "expected_production_by_resource": expected_by_resource,
        "blocked_production_total": blocked_total,
        "blocked_production_by_resource": blocked_by_resource,
        "diversity_score": diversity_score(expected_by_resource),
        "balance_score": balance,
        "orientation_scores": orientation_scores,
        "favored_plan": favored_plan,
        "bottlenecks": bottlenecks,
        "port_leverage_score": port_leverage,
        "expansion_potential": {
            "reachable_site_count": expansion.get("reachable_site_count", 0),
            "top_reachable_site_ev": expansion.get("top_reachable_site_ev", 0.0),
            "best_reachable_site_ids": expansion.get("best_reachable_site_ids", []),
            "score": expansion.get("score", 0.0),
        },
        "upgrade_potential": {
            "upgradeable_settlement_count": expansion.get("upgradeable_settlement_count", 0),
            "top_upgradeable_settlement_ev": expansion.get("top_upgradeable_settlement_ev", 0.0),
            "mean_upgradeable_settlement_ev": expansion.get("mean_upgradeable_settlement_ev", 0.0),
        },
        "robber_burden_score": robber_burden,
        "ports": dict(player.get("ports", {})) if isinstance(player.get("ports"), dict) else {},
        "pressure": {},
        "relative_table_metrics": {},
        "tags": tags,
    }


def build_player_profiles(
    state: Dict[str, object],
    board_economy: Optional[Dict[str, object]] = None,
    slot_ordering: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    del board_economy
    players = (
        state.get("players", {}).get("all_players", [])
        if isinstance(state.get("players"), dict)
        else []
    )
    base_profiles = [
        _build_base_profile(state=state, player=player, slot_ordering=slot_ordering)
        for player in players
        if isinstance(player, dict)
    ]
    total_expected = {
        str(profile["player_id"]): float(profile["expected_production_total"])
        for profile in base_profiles
        if profile.get("player_id") is not None
    }
    total_rankings = rank_with_gaps(total_expected)
    by_resource_rankings = {
        resource: rank_with_gaps(
            {
                str(profile["player_id"]): float(profile["expected_production_by_resource"].get(resource, 0.0))
                for profile in base_profiles
                if profile.get("player_id") is not None
            }
        )
        for resource in TRACKED_RESOURCES
    }
    pressure = build_player_pressure(base_profiles)

    for profile in base_profiles:
        player_id = profile.get("player_id")
        if player_id is None:
            continue
        player_key = str(player_id)
        relative_metrics = dict(total_rankings.get(player_key, {}))
        relative_metrics["by_resource"] = {
            resource: dict(by_resource_rankings[resource].get(player_key, {}))
            for resource in TRACKED_RESOURCES
        }
        profile["relative_table_metrics"] = relative_metrics
        profile["pressure"] = dict((pressure.get("by_player") or {}).get(player_key, {}))
        profile["tags"] = sorted(set(list(profile.get("tags", [])) + list((profile["pressure"] or {}).get("tags", []))))

    return {
        "by_player": base_profiles,
        "table_leaders": pressure.get("table_leaders", {}),
    }
