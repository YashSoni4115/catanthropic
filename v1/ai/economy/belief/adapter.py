from __future__ import annotations

import math
from typing import Any, Dict

from ai.economy.constants import TRACKED_RESOURCES
from ai.economy.tags import belief_tags


def _confidence_from_support(support_size: int) -> float:
    if support_size <= 1:
        return 1.0
    return 1.0 / (1.0 + math.log1p(max(0, support_size - 1)))


def _summary_from_snapshot(player_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    expected_counts = {
        resource: float((summary.get("expected_counts") or {}).get(resource, 0.0))
        for resource in TRACKED_RESOURCES
    }
    p_at_least_one = {
        resource: float((summary.get("p_at_least_one") or {}).get(resource, 0.0))
        for resource in TRACKED_RESOURCES
    }
    affordability = {
        "road": float((summary.get("p_can_afford") or {}).get("road", 0.0)),
        "settlement": float((summary.get("p_can_afford") or {}).get("settlement", 0.0)),
        "city": float((summary.get("p_can_afford") or {}).get("city", 0.0)),
        "dev": float((summary.get("p_can_afford") or {}).get("dev_card", (summary.get("p_can_afford") or {}).get("dev", 0.0))),
    }
    support_size = int(summary.get("support_size", 0) or 0)
    return {
        "player_id": player_id,
        "known_total": summary.get("known_total"),
        "support_size": support_size,
        "confidence_score": _confidence_from_support(support_size),
        "expected_hidden_resource_counts": expected_counts,
        "p_at_least_one": p_at_least_one,
        "p_can_afford": affordability,
        "tags": belief_tags(p_at_least_one, affordability),
    }


def build_belief_economy_v1(belief_source: Any = None) -> Dict[str, Any]:
    if belief_source is None:
        return {
            "available": False,
            "tracker_version": "resource_belief_tracker_v1",
            "by_player": {},
        }

    snapshot: Dict[str, Any]
    if hasattr(belief_source, "snapshot") and callable(getattr(belief_source, "snapshot")):
        snapshot = belief_source.snapshot()
    elif isinstance(belief_source, dict):
        snapshot = belief_source
    else:
        return {
            "available": False,
            "tracker_version": "resource_belief_tracker_v1",
            "by_player": {},
        }

    by_player = {
        str(player_id): _summary_from_snapshot(str(player_id), summary)
        for player_id, summary in snapshot.items()
        if isinstance(summary, dict)
    }
    return {
        "available": bool(by_player),
        "tracker_version": "resource_belief_tracker_v1",
        "by_player": by_player,
    }
