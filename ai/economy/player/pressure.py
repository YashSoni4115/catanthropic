from __future__ import annotations

from typing import Dict, List

from ai.economy.constants import TRACKED_RESOURCES
from ai.economy.normalize import clamp01, rank_with_gaps, score_from_cap


def build_player_pressure(
    profiles: List[Dict[str, object]],
) -> Dict[str, object]:
    total_expected = {
        str(profile.get("player_id")): float(profile.get("expected_production_total", 0.0))
        for profile in profiles
        if profile.get("player_id") is not None
    }
    vp_visible = {
        str(profile.get("player_id")): float(profile.get("victory_points_visible", 0) or 0)
        for profile in profiles
        if profile.get("player_id") is not None
    }
    resource_leaders = {
        resource: max(
            (
                (str(profile.get("player_id")), float((profile.get("expected_production_by_resource") or {}).get(resource, 0.0)))
                for profile in profiles
                if profile.get("player_id") is not None
            ),
            key=lambda item: (item[1], item[0]),
            default=(None, 0.0),
        )
        for resource in TRACKED_RESOURCES
    }
    total_ranks = rank_with_gaps(total_expected)
    vp_ranks = rank_with_gaps(vp_visible)

    by_player: Dict[str, Dict[str, object]] = {}
    max_reachable_score = max((float((profile.get("expansion_potential") or {}).get("score", 0.0)) for profile in profiles), default=0.0)
    max_total_expected = max(total_expected.values(), default=0.0)
    max_vp = max(vp_visible.values(), default=0.0)
    for profile in profiles:
        player_id = profile.get("player_id")
        if player_id is None:
            continue
        player_key = str(player_id)
        expansion_score = float((profile.get("expansion_potential") or {}).get("score", 0.0))
        expected_total = float(profile.get("expected_production_total", 0.0))
        visible_vp = float(profile.get("victory_points_visible", 0) or 0)
        contested = [
            resource.lower()
            for resource in TRACKED_RESOURCES
            if resource_leaders[resource][0] is not None
            and resource_leaders[resource][0] != player_key
            and float((profile.get("expected_production_by_resource") or {}).get(resource, 0.0)) > 0.0
        ]
        pressure_score = clamp01(
            0.45 * score_from_cap(max_total_expected - expected_total, 8.0)
            + 0.30 * score_from_cap(max_reachable_score - expansion_score, 1.0)
            + 0.25 * score_from_cap(max_vp - visible_vp, 4.0)
        )
        tags: List[str] = []
        if contested:
            tags.append(f"contested_{contested[0]}")
        if pressure_score >= 0.55:
            tags.append("public_pressure_high")
        by_player[player_key] = {
            "resource_competition": contested,
            "expansion_race_gap": max_reachable_score - expansion_score,
            "leader_gap_expected_production": max_total_expected - expected_total,
            "leader_gap_vp_visible": max_vp - visible_vp,
            "pressure_score": pressure_score,
            "tags": tags,
        }

    return {
        "by_player": by_player,
        "table_leaders": {
            "total_expected_production": total_ranks,
            "visible_victory_points": vp_ranks,
            "by_resource": {
                resource: {
                    "player_id": leader_id,
                    "value": leader_value,
                }
                for resource, (leader_id, leader_value) in resource_leaders.items()
            },
        },
    }
