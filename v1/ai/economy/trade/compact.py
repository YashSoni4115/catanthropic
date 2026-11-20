from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ai.economy.constants import RESOURCE_SHORT
from ai.economy.normalize import rounded


def _compact_resources(resources: Iterable[str]) -> List[str]:
    return [RESOURCE_SHORT.get(resource, resource[:1]) for resource in resources]


def build_trade_context_v1(
    economy_state: Dict[str, object],
    viewer_player_id: str,
    counterpart_ids: Optional[List[str]] = None,
) -> Dict[str, object]:
    trade_profile = economy_state.get("trade_profile", {}) if isinstance(economy_state.get("trade_profile"), dict) else {}
    by_player = trade_profile.get("by_player", {}) if isinstance(trade_profile.get("by_player"), dict) else {}
    table_market = trade_profile.get("table_market", {}) if isinstance(trade_profile.get("table_market"), dict) else {}
    viewer = by_player.get(viewer_player_id, {}) if isinstance(by_player, dict) else {}

    filtered_ids = set(counterpart_ids or [])
    counterparties: List[Dict[str, object]] = []
    for player_id, profile in sorted(by_player.items()):
        if player_id == viewer_player_id:
            continue
        if filtered_ids and player_id not in filtered_ids:
            continue
        relative = profile.get("relative_table_metrics", {}) if isinstance(profile.get("relative_table_metrics"), dict) else {}
        counterparties.append(
            {
                "player_id": player_id,
                "rank": int(relative.get("rank", 0) or 0),
                "gap": rounded(float(relative.get("leader_gap", 0.0) or 0.0), 2),
                "need_top2": _compact_resources(profile.get("need_top2", [])),
                "offer_top2": _compact_resources(profile.get("offer_top2", [])),
                "tags": list(profile.get("tags", []))[:5],
            }
        )

    leaders = {
        RESOURCE_SHORT.get(resource, resource[:1]): player_id
        for resource, player_id in sorted((table_market.get("leader_by_resource") or {}).items())
        if player_id is not None
    }

    viewer_profile = {
        "need_top2": _compact_resources(viewer.get("need_top2", [])),
        "offer_top2": _compact_resources(viewer.get("offer_top2", [])),
        "tags": list(viewer.get("tags", []))[:5],
    }

    summary_tags = sorted(set(viewer_profile["tags"] + [tag for cp in counterparties for tag in cp.get("tags", [])]))[:10]

    return {
        "schema_version": "trade_context_v1",
        "viewer": viewer_player_id,
        "turn_index": ((economy_state.get("context") or {}).get("turn_index") if isinstance(economy_state.get("context"), dict) else None),
        "leaders": leaders,
        "scarce": _compact_resources(table_market.get("scarce_resources", [])),
        "abundant": _compact_resources(table_market.get("abundant_resources", [])),
        "viewer_profile": viewer_profile,
        "counterparties": counterparties,
        "summary_tags": summary_tags,
    }
