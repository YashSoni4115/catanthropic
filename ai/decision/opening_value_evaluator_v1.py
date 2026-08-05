"""Opening placement evaluator using the production value model.

Supports two evaluation modes:

- **settlement** (legacy): scores each legal settlement vertex independently.
- **settlement_road** (default): scores each legal (settlement, road) pair,
  reflecting the real Catan opening action of placing one settlement and one
  road in a single turn.

Usage (Python)::

    from ai.decision.opening_value_evaluator_v1 import (
        score_opening_candidates_v1,
        score_opening_settlement_road_v1,
    )
    # Settlement-only
    results = score_opening_candidates_v1(state_payload, model_dir="ai/models/value_model_economy_v1")
    # Settlement + road (recommended)
    results = score_opening_settlement_road_v1(state_payload, model_dir="ai/models/value_model_economy_v1")

Usage (CLI)::

    # Settlement + road (default)
    python -m ai.decision.opening_value_evaluator_v1 \\
        --input ai/selfplay/data/selfplay_guided_1000games_v3_extracted.jsonl \\
        --model-dir ai/models/value_model_economy_v1 \\
        --top-k 10

    # Settlement-only (legacy)
    python -m ai.decision.opening_value_evaluator_v1 \\
        --input ... --model-dir ... --mode settlement

Approximations:
- State projection adds a settlement and road to the observed_state without
  running the Java engine.  Adjacency-derived features are recomputed by the
  vectorizer from the modified structure/road lists.
- The projection does NOT simulate resource grants, opponent responses, or
  future turns.
- This is intentionally narrow to opening placement evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ai.economy.board.openings import _build_single_candidates
from ai.economy.constants import road_id, vertex_incident_roads
from ai.vectorizer_v1 import StateVectorizerV1


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Structured result for a single-settlement opening candidate."""
    rank: int
    vertex_id: str
    x: int
    y: int
    orientation: int
    value_model_score: float
    opening_economy_score: float
    blended_score: float
    total_pips: float
    resource_mix: Dict[str, float]
    tags: List[str]
    detail: Dict[str, Any]


@dataclass
class SettlementRoadCandidateResult:
    """Structured result for a settlement+road opening candidate."""
    rank: int
    vertex_id: str
    settlement_x: int
    settlement_y: int
    settlement_orientation: int
    road_id: str
    road_x: int
    road_y: int
    road_orientation: int
    value_model_score: float
    opening_economy_score: float
    blended_score: float
    total_pips: float
    resource_mix: Dict[str, float]
    tags: List[str]


@dataclass
class EvaluationResult:
    """Top-level evaluation output."""
    mode: str
    candidates: List[Any]
    model_dir: str
    economy_features_enabled: bool
    vector_length: int
    value_weight: float
    opening_weight: float
    num_legal_candidates: int
    current_player_name: Optional[str]
    notes: List[str]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class ValueModelOracle:
    """Loads a trained value model and provides score(x) -> p(win)."""

    def __init__(self, model_dir: str, model_name: str = "hgb"):
        self.model_dir = Path(model_dir).resolve()
        model_path = self.model_dir / f"model_{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with model_path.open("rb") as f:
            self.model = pickle.load(f)

        config_path = self.model_dir / "config.json"
        self.config: Dict[str, Any] = {}
        if config_path.exists():
            with config_path.open() as f:
                self.config = json.load(f)

        split_path = self.model_dir / "split_summary.json"
        self.split_summary: Dict[str, Any] = {}
        if split_path.exists():
            with split_path.open() as f:
                self.split_summary = json.load(f)

        self.economy_features_enabled = bool(self.config.get("economy_features_enabled", False))
        self.expected_vector_length = int(self.split_summary.get("vector_length", 0))
        self.model_name = model_name

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Return p(win) for a batch of feature vectors."""
        proba = self.model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)


# ---------------------------------------------------------------------------
# State projection
# ---------------------------------------------------------------------------

def _project_settlement(
    state_payload: Dict[str, Any],
    sx: int, sy: int, so: int,
    owner_id: str, owner_name: str,
) -> Dict[str, Any]:
    """Return a copy of state_payload with a new settlement added."""
    projected = copy.copy(state_payload)
    observed = state_payload.get("observed_state")
    if isinstance(observed, dict):
        projected["observed_state"] = obs = copy.copy(observed)
    else:
        obs = projected
    obs["structures"] = list(obs.get("structures") or []) + [{
        "x": sx, "y": sy, "orientation": so,
        "structure_id": f"structure:{sx}:{sy}:{so}",
        "type": "settlement",
        "owner_id": owner_id, "owner_name": owner_name,
        "adjacent_tiles": [], "incident_roads": [],
    }]
    return projected


def _project_settlement_road(
    state_payload: Dict[str, Any],
    sx: int, sy: int, so: int,
    rx: int, ry: int, ro: int,
    owner_id: str, owner_name: str,
) -> Dict[str, Any]:
    """Return a copy with both a new settlement and a new road added."""
    projected = copy.copy(state_payload)
    observed = state_payload.get("observed_state")
    if isinstance(observed, dict):
        projected["observed_state"] = obs = copy.copy(observed)
    else:
        obs = projected
    obs["structures"] = list(obs.get("structures") or []) + [{
        "x": sx, "y": sy, "orientation": so,
        "structure_id": f"structure:{sx}:{sy}:{so}",
        "type": "settlement",
        "owner_id": owner_id, "owner_name": owner_name,
        "adjacent_tiles": [], "incident_roads": [],
    }]
    obs["roads"] = list(obs.get("roads") or []) + [{
        "x": rx, "y": ry, "orientation": ro,
        "road_id": road_id(rx, ry, ro),
        "owner_id": owner_id, "owner_name": owner_name,
    }]
    return projected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_player(
    state_payload: Dict[str, Any],
    current_player_name: Optional[str],
    notes: List[str],
) -> Tuple[str, str]:
    """Return (owner_id, current_player_name)."""
    observed = state_payload.get("observed_state", state_payload)
    turn_meta = state_payload.get("turn_metadata", observed.get("turn_metadata", {}))
    if not isinstance(turn_meta, dict):
        turn_meta = {}
    if current_player_name is None:
        current_player_name = str(turn_meta.get("current_player_name") or "")
    if not current_player_name:
        current_player_name = "P0"
        notes.append("current_player_name not found; defaulting to P0")
    return f"name:{current_player_name}", current_player_name


def _build_oracle_and_vectorizer(
    model_dir: str, model_name: str,
) -> Tuple[ValueModelOracle, StateVectorizerV1]:
    oracle = ValueModelOracle(model_dir, model_name=model_name)
    vectorizer = StateVectorizerV1(
        adjacency_features_enabled=True,
        economy_features_enabled=oracle.economy_features_enabled,
    )
    return oracle, vectorizer


def _occupied_road_set(observed: Dict[str, Any]) -> Set[Tuple[int, int, int]]:
    """Return the set of already-occupied road slots."""
    occupied: Set[Tuple[int, int, int]] = set()
    for r in observed.get("roads", []):
        if isinstance(r, dict):
            try:
                occupied.add((int(r["x"]), int(r["y"]), int(r["orientation"])))
            except (KeyError, TypeError, ValueError):
                pass
    return occupied


def _get_opening_candidates(
    state_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (single_candidates, board_economy)."""
    observed = state_payload.get("observed_state", state_payload)
    from ai.economy import build_economy_state_v1
    eco_state = build_economy_state_v1(
        state_payload, view="observed", include_trade=False, include_belief=False,
    )
    board_economy = eco_state.get("board_economy")
    encoding = state_payload.get("encoding_metadata", {})
    candidates = _build_single_candidates(
        tiles=observed.get("tiles", []),
        board_economy=board_economy,
        structures=observed.get("structures", []),
        slot_ordering=encoding.get("slot_ordering"),
    )
    return candidates, board_economy


def _empty_result(
    mode: str, model_dir: str, oracle: ValueModelOracle,
    vw: float, ow: float, n: int, player: Optional[str], notes: List[str],
) -> EvaluationResult:
    return EvaluationResult(
        mode=mode, candidates=[], model_dir=str(model_dir),
        economy_features_enabled=oracle.economy_features_enabled,
        vector_length=oracle.expected_vector_length,
        value_weight=vw, opening_weight=ow,
        num_legal_candidates=n, current_player_name=player, notes=notes,
    )


# ---------------------------------------------------------------------------
# Settlement-only scoring (legacy)
# ---------------------------------------------------------------------------

def score_opening_candidates_v1(
    state_payload: Dict[str, Any],
    model_dir: str,
    model_name: str = "hgb",
    top_k: int = 10,
    value_weight: float = 0.7,
    opening_weight: float = 0.3,
    current_player_name: Optional[str] = None,
) -> EvaluationResult:
    """Score legal opening settlement candidates (settlement-only mode)."""
    notes: List[str] = []
    owner_id, player = _resolve_player(state_payload, current_player_name, notes)
    oracle, vectorizer = _build_oracle_and_vectorizer(model_dir, model_name)
    opening_candidates, _ = _get_opening_candidates(state_payload)

    if not opening_candidates:
        notes.append("No legal opening vertices found")
        return _empty_result("settlement", model_dir, oracle, value_weight, opening_weight, 0, player, notes)

    vectors: List[np.ndarray] = []
    valid: List[Dict[str, Any]] = []
    for cand in opening_candidates:
        cx, cy, co = int(cand["x"]), int(cand["y"]), int(cand["orientation"])
        projected = _project_settlement(state_payload, cx, cy, co, owner_id, player)
        try:
            vec = vectorizer.vectorize(projected)
        except Exception as e:
            notes.append(f"Vectorization failed for {cand['vertex_id']}: {e}")
            continue
        x_arr = np.asarray(vec.x, dtype=np.float32)
        if oracle.expected_vector_length > 0 and x_arr.shape[0] != oracle.expected_vector_length:
            notes.append(f"Vector length mismatch for {cand['vertex_id']}: {x_arr.shape[0]} != {oracle.expected_vector_length}")
            continue
        vectors.append(x_arr)
        valid.append(cand)

    if not vectors:
        notes.append("No candidates produced valid feature vectors")
        return _empty_result("settlement", model_dir, oracle, value_weight, opening_weight, len(opening_candidates), player, notes)

    p_win = oracle.score_batch(np.vstack(vectors))
    w_total = value_weight + opening_weight
    results: List[CandidateResult] = []
    for idx, cand in enumerate(valid):
        os_ = float(cand.get("opening_score", 0.0))
        vm = float(p_win[idx])
        results.append(CandidateResult(
            rank=0, vertex_id=str(cand["vertex_id"]),
            x=int(cand["x"]), y=int(cand["y"]), orientation=int(cand["orientation"]),
            value_model_score=round(vm, 6), opening_economy_score=round(os_, 6),
            blended_score=round((value_weight * vm + opening_weight * os_) / w_total, 6),
            total_pips=float(cand.get("total_pips", 0.0)),
            resource_mix={k: float(v) for k, v in (cand.get("resource_mix") or {}).items()},
            tags=list(cand.get("tags") or []), detail=cand,
        ))
    results.sort(key=lambda r: (-r.blended_score, -r.value_model_score, r.vertex_id))
    for i, r in enumerate(results):
        r.rank = i + 1
    return EvaluationResult(
        mode="settlement", candidates=results[:top_k], model_dir=str(model_dir),
        economy_features_enabled=oracle.economy_features_enabled,
        vector_length=oracle.expected_vector_length, value_weight=value_weight,
        opening_weight=opening_weight, num_legal_candidates=len(opening_candidates),
        current_player_name=player, notes=notes,
    )


# ---------------------------------------------------------------------------
# Settlement + road scoring
# ---------------------------------------------------------------------------

def score_opening_settlement_road_v1(
    state_payload: Dict[str, Any],
    model_dir: str,
    model_name: str = "hgb",
    top_k: int = 10,
    value_weight: float = 0.7,
    opening_weight: float = 0.3,
    current_player_name: Optional[str] = None,
) -> EvaluationResult:
    """Score legal opening (settlement, road) pairs using the value model.

    For each legal settlement vertex, enumerates its incident road edges
    via ``vertex_incident_roads``, filters out already-occupied roads,
    projects both the settlement and the road onto the observed state,
    vectorizes, and scores with the trained value model.

    The deterministic ``opening_economy_score`` is inherited from the
    settlement vertex (the economy subsystem does not have road-aware
    opening scoring).  The value model, however, sees the road in the
    projected state and differentiates road directions.
    """
    notes: List[str] = []
    owner_id, player = _resolve_player(state_payload, current_player_name, notes)
    oracle, vectorizer = _build_oracle_and_vectorizer(model_dir, model_name)
    opening_candidates, _ = _get_opening_candidates(state_payload)

    if not opening_candidates:
        notes.append("No legal opening vertices found")
        return _empty_result("settlement_road", model_dir, oracle, value_weight, opening_weight, 0, player, notes)

    observed = state_payload.get("observed_state", state_payload)
    occupied_roads = _occupied_road_set(observed)

    # Enumerate (settlement, road) pairs
    pair_infos: List[Tuple[Dict[str, Any], int, int, int]] = []
    for cand in opening_candidates:
        sx, sy, so = int(cand["x"]), int(cand["y"]), int(cand["orientation"])
        for edge in vertex_incident_roads(sx, sy, so):
            rk = (edge["x"], edge["y"], edge["orientation"])
            if rk not in occupied_roads:
                pair_infos.append((cand, rk[0], rk[1], rk[2]))

    if not pair_infos:
        notes.append("No legal settlement+road pairs found")
        return _empty_result("settlement_road", model_dir, oracle, value_weight, opening_weight, len(opening_candidates), player, notes)

    # Project + vectorize all pairs
    vectors: List[np.ndarray] = []
    valid_pairs: List[Tuple[Dict[str, Any], int, int, int]] = []
    for cand, rx, ry, ro in pair_infos:
        sx, sy, so = int(cand["x"]), int(cand["y"]), int(cand["orientation"])
        projected = _project_settlement_road(state_payload, sx, sy, so, rx, ry, ro, owner_id, player)
        try:
            vec = vectorizer.vectorize(projected)
        except Exception as e:
            notes.append(f"Vectorization failed for {cand['vertex_id']}+{road_id(rx,ry,ro)}: {e}")
            continue
        x_arr = np.asarray(vec.x, dtype=np.float32)
        if oracle.expected_vector_length > 0 and x_arr.shape[0] != oracle.expected_vector_length:
            continue
        vectors.append(x_arr)
        valid_pairs.append((cand, rx, ry, ro))

    if not vectors:
        notes.append("No pairs produced valid feature vectors")
        return _empty_result("settlement_road", model_dir, oracle, value_weight, opening_weight, len(pair_infos), player, notes)

    p_win = oracle.score_batch(np.vstack(vectors))
    w_total = value_weight + opening_weight

    results: List[SettlementRoadCandidateResult] = []
    for idx, (cand, rx, ry, ro) in enumerate(valid_pairs):
        os_ = float(cand.get("opening_score", 0.0))
        vm = float(p_win[idx])
        results.append(SettlementRoadCandidateResult(
            rank=0, vertex_id=str(cand["vertex_id"]),
            settlement_x=int(cand["x"]), settlement_y=int(cand["y"]),
            settlement_orientation=int(cand["orientation"]),
            road_id=road_id(rx, ry, ro), road_x=rx, road_y=ry, road_orientation=ro,
            value_model_score=round(vm, 6), opening_economy_score=round(os_, 6),
            blended_score=round((value_weight * vm + opening_weight * os_) / w_total, 6),
            total_pips=float(cand.get("total_pips", 0.0)),
            resource_mix={k: float(v) for k, v in (cand.get("resource_mix") or {}).items()},
            tags=list(cand.get("tags") or []),
        ))

    results.sort(key=lambda r: (-r.blended_score, -r.value_model_score, r.vertex_id, r.road_id))
    for i, r in enumerate(results):
        r.rank = i + 1
    return EvaluationResult(
        mode="settlement_road", candidates=results[:top_k], model_dir=str(model_dir),
        economy_features_enabled=oracle.economy_features_enabled,
        vector_length=oracle.expected_vector_length, value_weight=value_weight,
        opening_weight=opening_weight, num_legal_candidates=len(pair_infos),
        current_player_name=player, notes=notes,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_setup_state(input_path: Path, max_scan: int = 500) -> Optional[Dict[str, Any]]:
    """Scan the input for an early-game state suitable for opening evaluation."""
    best: Optional[Dict[str, Any]] = None
    best_turn: int = 999999
    with input_path.open() as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            obs = payload.get("observed_state", payload)
            tm = payload.get("turn_metadata", obs.get("turn_metadata", {}))
            turn_index = tm.get("turn_index", 999999)
            if isinstance(turn_index, int) and turn_index < best_turn:
                best = payload
                best_turn = turn_index
    return best


def _format_settlement_results(result: EvaluationResult) -> str:
    lines: List[str] = []
    lines.append(f"Opening Evaluation — settlement mode ({result.num_legal_candidates} legal vertices)")
    lines.append(f"Model: {result.model_dir}  economy={result.economy_features_enabled}  vec_len={result.vector_length}")
    lines.append(f"Blend: value_weight={result.value_weight}  opening_weight={result.opening_weight}")
    lines.append(f"Player: {result.current_player_name}")
    for note in result.notes:
        lines.append(f"  NOTE: {note}")
    lines.append("")
    lines.append(f"{'Rank':>4}  {'Vertex':<14} {'Blend':>7} {'VM p(win)':>9} {'Opening':>8} {'Pips':>6}  Resources                    Tags")
    lines.append("-" * 110)
    for c in result.candidates:
        res = " ".join(f"{k[0]}={v:.1f}" for k, v in sorted(c.resource_mix.items()) if v > 0)
        tags = ", ".join(c.tags) if c.tags else ""
        lines.append(
            f"{c.rank:>4}  {c.vertex_id:<14} {c.blended_score:>7.4f} {c.value_model_score:>9.4f} "
            f"{c.opening_economy_score:>8.4f} {c.total_pips:>6.1f}  {res:<28s} {tags}"
        )
    return "\n".join(lines)


def _format_settlement_road_results(result: EvaluationResult) -> str:
    lines: List[str] = []
    lines.append(f"Opening Evaluation — settlement+road mode ({result.num_legal_candidates} legal pairs)")
    lines.append(f"Model: {result.model_dir}  economy={result.economy_features_enabled}  vec_len={result.vector_length}")
    lines.append(f"Blend: value_weight={result.value_weight}  opening_weight={result.opening_weight}")
    lines.append(f"Player: {result.current_player_name}")
    for note in result.notes:
        lines.append(f"  NOTE: {note}")
    lines.append("")
    lines.append(f"{'Rank':>4}  {'Settlement':<14} {'Road':<14} {'Blend':>7} {'VM p(win)':>9} {'Opening':>8} {'Pips':>6}  Resources                    Tags")
    lines.append("-" * 125)
    for c in result.candidates:
        res = " ".join(f"{k[0]}={v:.1f}" for k, v in sorted(c.resource_mix.items()) if v > 0)
        tags = ", ".join(c.tags) if c.tags else ""
        lines.append(
            f"{c.rank:>4}  {c.vertex_id:<14} {c.road_id:<14} {c.blended_score:>7.4f} {c.value_model_score:>9.4f} "
            f"{c.opening_economy_score:>8.4f} {c.total_pips:>6.1f}  {res:<28s} {tags}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate opening placement candidates with the value model",
    )
    parser.add_argument("--input", required=True, help="Extracted-state JSONL file")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--model-name", default="hgb", help="Model to use (hgb or logreg)")
    parser.add_argument(
        "--mode", choices=["settlement", "settlement_road"], default="settlement_road",
        help="Candidate mode: settlement-only or settlement+road pair (default: settlement_road)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--value-weight", type=float, default=0.7)
    parser.add_argument("--opening-weight", type=float, default=0.3)
    parser.add_argument("--state-index", type=int, default=None,
                        help="Use the Nth line (0-based) instead of auto-detecting")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    if args.state_index is not None:
        state_payload = None
        with input_path.open() as f:
            for i, line in enumerate(f):
                if i == args.state_index:
                    state_payload = json.loads(line.strip())
                    break
        if state_payload is None:
            print(f"state_index {args.state_index} out of range", file=sys.stderr)
            return 1
    else:
        state_payload = _find_setup_state(input_path)
        if state_payload is None:
            print("No state found in first 500 lines", file=sys.stderr)
            return 1

    if args.mode == "settlement":
        result = score_opening_candidates_v1(
            state_payload, model_dir=args.model_dir, model_name=args.model_name,
            top_k=args.top_k, value_weight=args.value_weight,
            opening_weight=args.opening_weight,
        )
    else:
        result = score_opening_settlement_road_v1(
            state_payload, model_dir=args.model_dir, model_name=args.model_name,
            top_k=args.top_k, value_weight=args.value_weight,
            opening_weight=args.opening_weight,
        )

    if args.json:
        output = {
            "mode": result.mode,
            "candidates": [asdict(c) for c in result.candidates],
            "model_dir": result.model_dir,
            "economy_features_enabled": result.economy_features_enabled,
            "vector_length": result.vector_length,
            "value_weight": result.value_weight,
            "opening_weight": result.opening_weight,
            "num_legal_candidates": result.num_legal_candidates,
            "current_player_name": result.current_player_name,
            "notes": result.notes,
        }
        for c in output["candidates"]:
            c.pop("detail", None)
        print(json.dumps(output, indent=2))
    else:
        if result.mode == "settlement":
            print(_format_settlement_results(result))
        else:
            print(_format_settlement_road_results(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
