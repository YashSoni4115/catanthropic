from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ai.economy.constants import BUILD_RECIPES, TRACKED_RESOURCES
from ai.economy.normalize import clamp01, score_from_cap, top_keys
from ai.economy.tags import trade_tags


def _actual_affordability(resource_cards: Dict[str, int]) -> Dict[str, bool]:
    output: Dict[str, bool] = {}
    for recipe_name, recipe in BUILD_RECIPES.items():
        output[recipe_name] = all(int(resource_cards.get(resource, 0)) >= amount for resource, amount in recipe.items())
    return output


def _bottleneck_map(bottlenecks: List[Dict[str, float | str]]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for item in bottlenecks:
        resource = item.get("resource")
        severity = item.get("severity")
        if isinstance(resource, str):
            output[resource] = float(severity or 0.0)
    return output


def build_trade_profile(
    state: Dict[str, object],
    board_economy: Dict[str, object],
    player_economy: Dict[str, object],
    belief_economy: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    board_resource_environment = (
        ((board_economy.get("resource_environment") or {}).get("by_resource", {}))
        if isinstance(board_economy.get("resource_environment"), dict)
        else {}
    )
    player_profiles = list(player_economy.get("by_player", [])) if isinstance(player_economy.get("by_player"), list) else []
    state_players = (
        state.get("players", {}).get("all_players", [])
        if isinstance(state.get("players"), dict)
        else []
    )
    state_player_by_id = {
        str(player.get("player_id")): player
        for player in state_players
        if isinstance(player, dict) and player.get("player_id") is not None
    }
    belief_by_player = (
        belief_economy.get("by_player", {})
        if isinstance(belief_economy, dict) and isinstance(belief_economy.get("by_player"), dict)
        else {}
    )

    by_player: Dict[str, Dict[str, object]] = {}
    for profile in player_profiles:
        player_id = profile.get("player_id")
        if player_id is None:
            continue
        player_key = str(player_id)
        expected_by_resource = dict(profile.get("expected_production_by_resource", {}))
        strongest_resource = max(expected_by_resource.items(), key=lambda item: (item[1], item[0]))[0] if expected_by_resource else None
        weakest_resource = min(expected_by_resource.items(), key=lambda item: (item[1], item[0]))[0] if expected_by_resource else None
        bottlenecks = _bottleneck_map(list(profile.get("bottlenecks", [])))
        need_scores: Dict[str, float] = {}
        surplus_scores: Dict[str, float] = {}
        for resource in TRACKED_RESOURCES:
            scarcity = float((board_resource_environment.get(resource) or {}).get("scarcity_score", 0.0))
            abundance = float((board_resource_environment.get(resource) or {}).get("abundance_score", 0.0))
            production_strength = score_from_cap(float(expected_by_resource.get(resource, 0.0)), 5.0)
            bottleneck = float(bottlenecks.get(resource, 0.0))
            weakest_bonus = 0.15 if weakest_resource == resource else 0.0
            strongest_bonus = 0.15 if strongest_resource == resource else 0.0
            need_scores[resource] = clamp01(0.45 * bottleneck + 0.35 * scarcity + 0.20 * weakest_bonus)
            surplus_scores[resource] = clamp01(max(0.0, 0.6 * production_strength + 0.2 * abundance + 0.2 * strongest_bonus - 0.7 * need_scores[resource]))

        state_player = state_player_by_id.get(player_key, {})
        resource_cards = state_player.get("resource_cards") if isinstance(state_player.get("resource_cards"), dict) else None
        hidden = bool(state_player.get("hidden_from_viewer", False))
        affordability: Dict[str, object] = {
            "road": None,
            "settlement": None,
            "city": None,
            "dev": None,
        }
        if resource_cards is not None and not hidden:
            affordability = dict(_actual_affordability(resource_cards))
        elif player_key in belief_by_player:
            affordability = dict((belief_by_player[player_key] or {}).get("p_can_afford", {}))

        need_top2 = top_keys(need_scores, limit=2, min_score=0.1)
        offer_top2 = top_keys(surplus_scores, limit=2, min_score=0.1)
        tags = trade_tags(need_top2, offer_top2, strongest_resource, weakest_resource)

        by_player[player_key] = {
            "player_id": player_id,
            "player_name": profile.get("player_name"),
            "need_scores": need_scores,
            "surplus_scores": surplus_scores,
            "need_top2": need_top2,
            "offer_top2": offer_top2,
            "strongest_resource": strongest_resource,
            "weakest_resource": weakest_resource,
            "affordability": affordability,
            "relative_table_metrics": dict(profile.get("relative_table_metrics", {})),
            "tags": tags,
        }

    leader_by_resource: Dict[str, Optional[str]] = {}
    top_buyers_by_resource: Dict[str, List[str]] = {}
    top_sellers_by_resource: Dict[str, List[str]] = {}
    for resource in TRACKED_RESOURCES:
        leader = max(
            by_player.values(),
            key=lambda item: (float((item.get("relative_table_metrics", {}).get("by_resource", {}).get(resource, {}) or {}).get("share", 0.0)), str(item.get("player_id") or "")),
            default=None,
        )
        leader_by_resource[resource] = str(leader.get("player_id")) if leader is not None else None
        ordered_buyers = sorted(
            by_player.values(),
            key=lambda item: (-float((item.get("need_scores") or {}).get(resource, 0.0)), str(item.get("player_id") or "")),
        )
        ordered_sellers = sorted(
            by_player.values(),
            key=lambda item: (-float((item.get("surplus_scores") or {}).get(resource, 0.0)), str(item.get("player_id") or "")),
        )
        top_buyers_by_resource[resource] = [str(item.get("player_id")) for item in ordered_buyers[:2] if item.get("player_id") is not None]
        top_sellers_by_resource[resource] = [str(item.get("player_id")) for item in ordered_sellers[:2] if item.get("player_id") is not None]

    scarce_resources = sorted(
        TRACKED_RESOURCES,
        key=lambda resource: (-float((board_resource_environment.get(resource) or {}).get("scarcity_score", 0.0)), resource),
    )[:2]
    abundant_resources = sorted(
        TRACKED_RESOURCES,
        key=lambda resource: (-float((board_resource_environment.get(resource) or {}).get("abundance_score", 0.0)), resource),
    )[:2]

    return {
        "table_market": {
            "leader_by_resource": leader_by_resource,
            "scarce_resources": scarce_resources,
            "abundant_resources": abundant_resources,
            "top_buyers_by_resource": top_buyers_by_resource,
            "top_sellers_by_resource": top_sellers_by_resource,
        },
        "by_player": by_player,
    }
