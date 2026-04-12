from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from ai.economy.board.profile import build_board_profile
from ai.economy.belief.adapter import build_belief_economy_v1
from ai.economy.player.profile import build_player_profiles
from ai.economy.trade.compact import build_trade_context_v1 as _build_trade_context_v1
from ai.economy.trade.profile import build_trade_profile


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
                    "x": tile.get("x"),
                    "y": tile.get("y"),
                    "type": tile.get("type"),
                    "number": tile.get("number"),
                }
                for tile in board_state.get("tiles", [])
                if isinstance(tile, dict)
            ],
            key=lambda item: (item["x"], item["y"]),
        )
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_board_economy_v1(board_state: Dict[str, Any]) -> Dict[str, object]:
    state, slot_ordering, _ = _coerce_board_block(board_state)
    fingerprint = _board_fingerprint(state)
    cached = _BOARD_PROFILE_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    board_profile = build_board_profile(
        tiles=state.get("tiles", []) if isinstance(state.get("tiles"), list) else [],
        slot_ordering=slot_ordering,
    )
    _BOARD_PROFILE_CACHE[fingerprint] = board_profile
    return board_profile


def build_economy_state_v1(
    state_payload: Dict[str, Any],
    view: str = "observed",
    include_belief: bool = False,
    include_trade: bool = True,
    belief_source: Any = None,
) -> Dict[str, object]:
    source_view = view if view in {"board_only", "observed", "omniscient"} else "observed"
    state = _select_state_view(state_payload, "observed" if source_view == "board_only" else source_view)
    slot_ordering = _extract_slot_ordering(state_payload)
    board_economy = build_board_economy_v1(state_payload)
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

    player_economy: Dict[str, object]
    if source_view == "board_only":
        player_economy = {
            "by_player": [],
            "table_leaders": {},
        }
    else:
        player_economy = build_player_profiles(
            state=state,
            board_economy=board_economy,
            slot_ordering=slot_ordering,
        )

    belief_economy = build_belief_economy_v1(belief_source if include_belief else None)
    trade_profile = (
        build_trade_profile(
            state=state,
            board_economy=board_economy,
            player_economy=player_economy,
            belief_economy=belief_economy,
        )
        if include_trade and source_view != "board_only"
        else {"table_market": {}, "by_player": {}}
    )

    summary_tags = sorted(
        set(
            list(((board_economy.get("resource_environment") or {}).get("board_tags", []) if isinstance(board_economy.get("resource_environment"), dict) else []))
            + [
                tag
                for player in player_economy.get("by_player", [])
                if isinstance(player, dict)
                for tag in player.get("tags", [])
            ]
        )
    )
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
    return _build_trade_context_v1(
        economy_state=economy_state,
        viewer_player_id=viewer_player_id,
        counterpart_ids=counterpart_ids,
    )
