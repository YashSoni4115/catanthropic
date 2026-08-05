"""Streaming opening placement scorer for Java self-play integration.

Reads JSON snapshots from stdin (one per line), runs the value-model-based
opening evaluator, and writes ranked (settlement, road) placements to stdout.

Protocol (mirrors infer_win_model_v1.py):
  - Input:  JSON line with ``"snapshot"`` containing board state (tiles,
            structures, roads, players) as emitted by Java SelfPlayGameRunner.
  - Output: JSON line with ``"placements"`` list of ``{sx,sy,so, rx,ry,ro, score}``
            ordered best-first, plus ``"policy"`` metadata.

Usage::

    python -u ai/selfplay/opening_scorer_v1.py \\
        --model-dir ai/models/value_model_economy_v1 \\
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``from ai.…`` imports work when
# this script is launched as a subprocess from Java.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from typing import Any, Dict, List, Optional


def _build_state_payload_from_snapshot(request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Java-emitted snapshot request into the extracted-state format
    expected by the opening evaluator.

    The Java runner emits snapshots with ``tiles``, ``structures``, ``roads``,
    ``players`` at the top level inside a ``"snapshot"`` key.  The evaluator
    expects ``observed_state`` wrapping those fields, plus ``encoding_metadata``
    with slot orderings for the vectorizer.
    """
    from ai.board_state_extractor import _build_slot_ordering_metadata

    snapshot = request.get("snapshot", request)
    turn_metadata = request.get("turn_metadata", snapshot.get("turn_metadata", {}))
    if not isinstance(turn_metadata, dict):
        turn_metadata = {}

    # Build turn_metadata from snapshot fields if missing
    if "current_player_id" not in turn_metadata:
        turn_metadata["current_player_id"] = snapshot.get("current_player_id", "0")
    if "current_player_name" not in turn_metadata:
        turn_metadata["current_player_name"] = snapshot.get("current_player_name", "P0")
    if "turn_index" not in turn_metadata:
        turn_metadata["turn_index"] = snapshot.get("turn_index", -1)
    if "current_phase" not in turn_metadata:
        turn_metadata["current_phase"] = snapshot.get("current_phase", "setup")

    # Build a minimal extracted-state payload
    observed_state = {
        "turn_metadata": turn_metadata,
        "tiles": snapshot.get("tiles", []),
        "structures": snapshot.get("structures", []),
        "roads": snapshot.get("roads", []),
        "players": {
            "all_players": snapshot.get("players", []),
        },
        "robber": snapshot.get("robber"),
    }

    return {
        "turn_metadata": turn_metadata,
        "observed_state": observed_state,
        "encoding_metadata": {
            "slot_ordering": _build_slot_ordering_metadata(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming opening scorer for self-play")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--model-name", default="hgb", help="Model to use (hgb or logreg)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to return")
    parser.add_argument("--value-weight", type=float, default=0.7)
    parser.add_argument("--opening-weight", type=float, default=0.3)
    parser.add_argument("--sample-topk", type=int, default=0,
                        help="If > 0, return top-k candidates for stochastic sampling by caller")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    # Lazy import to avoid slow startup until first request
    from ai.decision.opening_value_evaluator_v1 import (
        ValueModelOracle,
        score_opening_settlement_road_v1,
    )

    # Pre-warm model load
    try:
        _oracle = ValueModelOracle(str(model_dir), model_name=args.model_name)
        print(f"[opening_scorer] model loaded: {model_dir} vec_len={_oracle.expected_vector_length} "
              f"economy={_oracle.economy_features_enabled}", file=sys.stderr)
    except Exception as e:
        print(f"[opening_scorer] FATAL: cannot load model: {e}", file=sys.stderr)
        return 1

    effective_top_k = max(args.top_k, args.sample_topk) if args.sample_topk > 0 else args.top_k

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            state_payload = _build_state_payload_from_snapshot(request)

            current_player_name = None
            tm = request.get("turn_metadata", {})
            if isinstance(tm, dict):
                current_player_name = tm.get("current_player_name")

            result = score_opening_settlement_road_v1(
                state_payload=state_payload,
                model_dir=str(model_dir),
                model_name=args.model_name,
                top_k=effective_top_k,
                value_weight=args.value_weight,
                opening_weight=args.opening_weight,
                current_player_name=current_player_name,
            )

            placements: List[Dict[str, Any]] = []
            for c in result.candidates:
                placements.append({
                    "sx": c.settlement_x,
                    "sy": c.settlement_y,
                    "so": c.settlement_orientation,
                    "rx": c.road_x,
                    "ry": c.road_y,
                    "ro": c.road_orientation,
                    "vertex_id": c.vertex_id,
                    "road_id": c.road_id,
                    "value_model_score": c.value_model_score,
                    "opening_economy_score": c.opening_economy_score,
                    "blended_score": c.blended_score,
                    "rank": c.rank,
                })

            response: Dict[str, Any] = {
                "placements": placements,
                "policy": "value_model",
                "num_legal": result.num_legal_candidates,
                "economy_features_enabled": result.economy_features_enabled,
            }
            if result.notes:
                response["notes"] = result.notes

        except Exception as exc:
            response = {
                "placements": [],
                "policy": "value_model",
                "error": str(exc),
            }

        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
