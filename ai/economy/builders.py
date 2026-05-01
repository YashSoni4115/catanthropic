from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from ai.economy.board.openings import build_opening_analysis
from ai.economy.board.profile import build_board_profile
from ai.economy.player.expansion import build_expansion_profile
from ai.economy.player.profile import build_player_economy_v1 as _build_player_economy_from_state
from ai.economy.tags import board_economy_tags


_BOARD_PROFILE_CACHE: Dict[str, Dict[str, object]] = {}


def _extract_slot_ordering(state_payload: Dict[str, Any]) -> Optional[Dict[str, object]]:
    encoding = state_payload.get("encoding_metadata")
    if not isinstance(encoding, dict):
        return None
    slot_ordering = encoding.get("slot_ordering")
    return slot_ordering if isinstance(slot_ordering, dict) else None


def _select_state_view(state_payload: Dict[str, Any], view: str) -> Dict[str, Any]:
    if view == "omniscient" and isinstance(state_payload.get("omniscient_state"), dict):
        return state_payload["omniscient_state"]
    if view == "observed" and isinstance(state_payload.get("observed_state"), dict):
        return state_payload["observed_state"]
    if isinstance(state_payload.get("observed_state"), dict):
        return state_payload["observed_state"]
    if isinstance(state_payload.get("omniscient_state"), dict):
        return state_payload["omniscient_state"]
    return state_payload


def _coerce_board_block(board_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, object]], Optional[str]]:
    if isinstance(board_input.get("observed_state"), dict) or isinstance(board_input.get("omniscient_state"), dict):
        state = _select_state_view(board_input, "observed")
        return state, _extract_slot_ordering(board_input), str(board_input.get("schema_version")) if board_input.get("schema_version") is not None else None
    return board_input, _extract_slot_ordering(board_input), str(board_input.get("schema_version")) if board_input.get("schema_version") is not None else None


def _board_fingerprint(board_state: Dict[str, Any]) -> str:
    canonical = {
        "tiles": sorted(
            [
                {
                    "x": _safe_int(tile.get("x")),
                    "y": _safe_int(tile.get("y")),
                    "type": str(tile.get("type") or "").strip().upper(),
                    "number": tile.get("number"),
                }
                for tile in board_state.get("tiles", [])
                if isinstance(tile, dict)
            ],
            key=lambda item: (str(item["x"]), str(item["y"]), str(item["type"]), str(item["number"])),
        )
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_robber(state: Dict[str, Any]) -> Optional[Dict[str, int]]:
    robber = state.get("robber")
    if not isinstance(robber, dict):
        return None
    x_coord = _safe_int(robber.get("x"))
    y_coord = _safe_int(robber.get("y"))
    if x_coord is None or y_coord is None:
        return None
    return {"x": x_coord, "y": y_coord}


def _extract_board_port_data(state: Dict[str, Any]) -> object:
    for key in ("board_ports", "port_geometry", "ports_by_vertex"):
        value = state.get(key)
        if value:
            return value
    return None


def build_board_economy_v1(board_state: Dict[str, Any]) -> Dict[str, object]:
    state, slot_ordering, _ = _coerce_board_block(board_state)
    fingerprint = _board_fingerprint(state)
    port_data = _extract_board_port_data(state)
    port_blob = json.dumps(port_data, sort_keys=True, default=str, separators=(",", ":"))
    cache_key = hashlib.sha256(f"{fingerprint}:{port_blob}".encode("utf-8")).hexdigest()
    cached = _BOARD_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)
    board_profile = build_board_profile(
        tiles=state.get("tiles", []) if isinstance(state.get("tiles"), list) else [],
        robber=_extract_robber(state),
        port_data=port_data,
        slot_ordering=slot_ordering,
    )
    _BOARD_PROFILE_CACHE[cache_key] = board_profile
    return deepcopy(board_profile)


def build_opening_economy_v1(
    board_state: Dict[str, Any],
    board_economy: Optional[Dict[str, object]] = None,
    limit: int = 12,
) -> Dict[str, object]:
    """Build deterministic legal opening vertex and pair scoring from extracted state."""
    state, slot_ordering, _ = _coerce_board_block(board_state)
    resolved_board_economy = board_economy if isinstance(board_economy, dict) else build_board_economy_v1(state)
    return build_opening_analysis(
        tiles=state.get("tiles", []) if isinstance(state.get("tiles"), list) else [],
        structures=state.get("structures", []) if isinstance(state.get("structures"), list) else [],
        slot_ordering=slot_ordering,
        limit=limit,
        board_economy=resolved_board_economy,
    )


def build_player_economy_v1(
    state_payload: Dict[str, Any],
    view: str = "observed",
    board_economy: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build deterministic post-placement player economy from an extracted state payload."""
    source_view = view if view in {"board_only", "observed", "omniscient"} else "observed"
    state = _select_state_view(state_payload, "observed" if source_view == "board_only" else source_view)
    if source_view == "board_only":
        return {
            "schema_version": "player_economy_v1",
            "by_player": [],
            "table_leaders": {"overall": {"player_id": None, "value": 0.0}, "by_resource": {}},
            "table_pressure": {
                "frontier_leader_player_id": None,
                "highest_pressure_player_id": None,
                "highest_pressure_composite_score": 0.0,
            },
        }
    resolved_board_economy = board_economy if isinstance(board_economy, dict) else build_board_economy_v1(state)
    return _build_player_economy_from_state(
        state=state,
        board_economy=resolved_board_economy,
        slot_ordering=_extract_slot_ordering(state_payload),
    )


def build_player_expansion_economy_v1(
    state_payload: Dict[str, Any],
    player_id: str,
    view: str = "observed",
    board_economy: Optional[Dict[str, object]] = None,
    player_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build expansion analysis for one player from an extracted state payload."""
    source_view = view if view in {"observed", "omniscient"} else "observed"
    state = _select_state_view(state_payload, source_view)
    resolved_board_economy = board_economy if isinstance(board_economy, dict) else build_board_economy_v1(state)
    resolved_profile = player_profile
    if resolved_profile is None:
        player_economy = _build_player_economy_from_state(
            state=state,
            board_economy=resolved_board_economy,
            slot_ordering=_extract_slot_ordering(state_payload),
        )
        resolved_profile = next(
            (
                profile
                for profile in player_economy.get("by_player", [])
                if isinstance(profile, dict) and str(profile.get("player_id")) == str(player_id)
            ),
            None,
        )
    return build_expansion_profile(
        state=state,
        player_id=str(player_id),
        player_profile=resolved_profile,
        board_economy=resolved_board_economy,
        slot_ordering=_extract_slot_ordering(state_payload),
    )


def build_player_pressure_economy_v1(
    state_payload: Dict[str, Any],
    view: str = "observed",
    board_economy: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build table-relative player pressure analysis from an extracted state payload."""
    player_economy = build_player_economy_v1(
        state_payload=state_payload,
        view=view,
        board_economy=board_economy,
    )
    return {
        "schema_version": "player_pressure_table_v1",
        "by_player": {
            str(profile.get("player_id")): profile.get("pressure", {})
            for profile in player_economy.get("by_player", [])
            if isinstance(profile, dict) and profile.get("player_id") is not None
        },
        "table_pressure": player_economy.get("table_pressure", {}),
    }


def build_economy_state_v1(
    state_payload: Dict[str, Any],
    view: str = "observed",
    include_belief: bool = False,
    include_trade: bool = False,
    belief_source: Any = None,
) -> Dict[str, object]:
    del belief_source
    source_view = view if view in {"board_only", "observed", "omniscient"} else "observed"
    state = _select_state_view(state_payload, "observed" if source_view == "board_only" else source_view)
    board_economy = build_board_economy_v1(state)
    player_economy = build_player_economy_v1(state_payload, view=source_view, board_economy=board_economy)
    fingerprint = _board_fingerprint(state)

    turn_metadata = state.get("turn_metadata", {}) if isinstance(state.get("turn_metadata"), dict) else {}
    context = {
        "extractor_schema_version": state_payload.get("schema_version"),
        "source_view": source_view,
        "state_id": turn_metadata.get("state_id"),
        "position_id": turn_metadata.get("position_id"),
        "turn_index": turn_metadata.get("turn_index"),
        "current_player_id": turn_metadata.get("current_player_id"),
        "board_fingerprint": fingerprint,
    }

    belief_economy: Dict[str, object] = {
        "available": False,
        "requested": bool(include_belief),
        "reason": "not_built_in_player_economy_phase",
    }
    trade_profile: Dict[str, object] = {
        "available": False,
        "requested": bool(include_trade),
        "reason": "not_built_in_player_economy_phase",
    }

    player_tags = [
        tag
        for player in player_economy.get("by_player", [])
        if isinstance(player, dict)
        for tag in player.get("tags", [])
    ]
    summary_tags = sorted(set(board_economy_tags(board_economy) + player_tags))
    return {
        "schema_version": "economy_state_v1",
        "context": context,
        "board_economy": board_economy,
        "player_economy": player_economy,
        "belief_economy": belief_economy,
        "trade_profile": trade_profile,
        "summary_tags": summary_tags,
    }


def build_trade_context_v1(
    economy_state: Dict[str, object],
    viewer_player_id: str,
    counterpart_ids: Optional[list[str]] = None,
) -> Dict[str, object]:
    from ai.economy.trade.compact import build_trade_context_v1 as _build_trade_context_v1

    return _build_trade_context_v1(
        economy_state=economy_state,
        viewer_player_id=viewer_player_id,
        counterpart_ids=counterpart_ids,
    )
