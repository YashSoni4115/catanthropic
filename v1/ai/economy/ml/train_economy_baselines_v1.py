"""Economy-feature ML experiment: compare economy vs baseline vs combined features.

Usage (single cap):
    python -m ai.economy.ml.train_economy_baselines_v1 \\
        --input ai/selfplay/data/selfplay_guided_1000games_v3_extracted.jsonl \\
        --output-dir ai/economy/ml/results/experiment_v1

Usage (staged caps):
    python -m ai.economy.ml.train_economy_baselines_v1 \\
        --input ai/selfplay/data/selfplay_guided_1000games_v3_extracted.jsonl \\
        --output-dir ai/economy/ml/results/experiment_v1 \\
        --row-caps 50000 100000 200000

Modes compared:
    baseline   - hand-crafted game-state features (VP, resources, relative position)
    economy    - flattened economy subsystem features only
    combined   - baseline + economy concatenated

Models:
    logreg     - L2-regularized logistic regression with StandardScaler
    hgb        - HistGradientBoostingClassifier

Task:
    Binary classification: does the current player eventually win this game?

Split:
    Deterministic game-level split (80/10/10) — no row-level leakage.

Observed-safe:
    Economy features built from observed_state only. Never touches omniscient_state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

RESOURCE_ORDER = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER")
PHASE_ORDER = ("setup", "roll", "main", "trade", "build", "unknown")
STAGE_KEYS = ("early", "mid", "late")
VALID_SPLITS = {"train", "val", "test"}
FEATURE_MODES = ("baseline", "economy", "combined")
MODEL_NAMES = ("logreg", "hgb")

# ═══════════════════════════════════════════════════════════════════════
# Row record
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Row:
    game_id: str
    split: str
    y_win: int
    x_baseline: np.ndarray
    x_economy: np.ndarray
    baseline_names: List[str]
    economy_names: List[str]
    row_index_in_game: int


# ═══════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════════════════════════════


def _deterministic_split(game_id: str, seed: str, ratios: Tuple[int, int, int]) -> str:
    digest = hashlib.md5(f"{seed}:{game_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


# ═══════════════════════════════════════════════════════════════════════
# Baseline features (mirrors train_win_model_v1 selfplay_raw features)
# ═══════════════════════════════════════════════════════════════════════

BASELINE_FEATURE_NAMES: List[str] = []


def _build_baseline_feature_names() -> List[str]:
    names = ["turn_index"]
    names += [f"phase_{p}" for p in PHASE_ORDER]
    names += ["roll"]
    names += ["current_vp", "current_total_resources"]
    names += [f"current_{r}" for r in RESOURCE_ORDER]
    for i in range(3):
        names += [f"opp{i}_vp", f"opp{i}_total_resources"]
    names += ["vp_gap_vs_max_opp", "vp_gap_vs_mean_opp", "res_gap_vs_mean_opp"]
    return names


BASELINE_FEATURE_NAMES = _build_baseline_feature_names()


def _safe_resource(player: Dict[str, Any], resource: str) -> float:
    resources = player.get("resources", {})
    if isinstance(resources, dict):
        v = resources.get(resource)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _total_resources(player: Dict[str, Any]) -> float:
    return sum(_safe_resource(player, r) for r in RESOURCE_ORDER)


def _extract_baseline_features(payload: Dict[str, Any]) -> Optional[Tuple[np.ndarray, int]]:
    """Extract the same baseline features as train_win_model_v1._selfplay_raw_to_row.

    Returns (feature_vector, y_win) or None if the row is unusable.
    """
    turn_meta = payload.get("turn_metadata", {})
    if not isinstance(turn_meta, dict):
        return None

    # Player state can be top-level or inside observed_state
    players = payload.get("player_state")
    if not isinstance(players, list) or len(players) == 0:
        return None

    current_name = str(turn_meta.get("current_player_name") or "")
    winner_name = str(payload.get("winner_player_name") or "")
    if not winner_name:
        return None

    current: Optional[Dict[str, Any]] = None
    others: List[Dict[str, Any]] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        name = str(p.get("player_name") or "")
        if name == current_name and current is None:
            current = p
        else:
            others.append(p)
    if current is None:
        return None

    others.sort(key=lambda p: str(p.get("player_name") or ""))

    turn_index = turn_meta.get("turn_index")
    if not isinstance(turn_index, int):
        turn_index = payload.get("row_index_in_game", -1)
        if not isinstance(turn_index, int):
            turn_index = -1

    roll = payload.get("roll")
    roll_val = float(roll) if isinstance(roll, (int, float)) else -1.0

    current_vp = float(current.get("victory_points") or 0.0)
    current_res = _total_resources(current)

    x: List[float] = []
    x.append(float(turn_index))

    # Phase one-hot
    phase = str(turn_meta.get("current_phase") or "unknown").lower()
    x.extend(1.0 if phase == p else 0.0 for p in PHASE_ORDER)

    x.append(roll_val)
    x.append(current_vp)
    x.append(current_res)
    for r in RESOURCE_ORDER:
        x.append(_safe_resource(current, r))

    opp_vps: List[float] = []
    opp_res: List[float] = []
    for opp in others[:3]:
        vp = float(opp.get("victory_points") or 0.0)
        res = _total_resources(opp)
        opp_vps.append(vp)
        opp_res.append(res)
        x.extend([vp, res])
    while len(opp_vps) < 3:
        opp_vps.append(0.0)
        opp_res.append(0.0)
        x.extend([0.0, 0.0])

    max_opp_vp = max(opp_vps)
    mean_opp_vp = sum(opp_vps) / len(opp_vps)
    mean_opp_res = sum(opp_res) / len(opp_res)
    x.extend([current_vp - max_opp_vp, current_vp - mean_opp_vp, current_res - mean_opp_res])

    y_win = 1 if current_name == winner_name else 0
    return np.asarray(x, dtype=np.float32), y_win


# ═══════════════════════════════════════════════════════════════════════
# Economy features — with board/opening economy caching
# ═══════════════════════════════════════════════════════════════════════


class _EconomyCache:
    """In-memory cache for static economy components keyed by board fingerprint.

    Cached (static per board layout):
        - board_economy: resource profiles, scarcity, concentration, pip totals
        - opening_economy: vertex/pair scores for legal openings

    NOT cached (dynamic per turn):
        - player_economy: depends on current structures / roads
        - trade_profile: depends on current player resources / positions
        - belief: depends on hidden-info tracker state
        - expansion / pressure: depend on current board state

    Cache keys use the same SHA-256 board fingerprint as builders.py.
    """

    def __init__(self) -> None:
        self._board: Dict[str, Dict[str, object]] = {}
        self._opening: Dict[str, Dict[str, object]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get_board_economy(
        self,
        state: Dict[str, Any],
        fingerprint: str,
    ) -> Dict[str, object]:
        cached = self._board.get(fingerprint)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        from ai.economy.builders import build_board_economy_v1
        result = build_board_economy_v1(state)
        self._board[fingerprint] = result
        return result

    def get_opening_economy(
        self,
        state: Dict[str, Any],
        fingerprint: str,
        board_economy: Dict[str, object],
    ) -> Dict[str, object]:
        cached = self._opening.get(fingerprint)
        if cached is not None:
            return cached
        from ai.economy.builders import build_opening_economy_v1
        result = build_opening_economy_v1(state, board_economy=board_economy)
        self._opening[fingerprint] = result
        return result

    def stats(self) -> Dict[str, int]:
        return {
            "board_cache_size": len(self._board),
            "opening_cache_size": len(self._opening),
            "hits": self.hits,
            "misses": self.misses,
        }


def _extract_economy_features(
    payload: Dict[str, Any],
    current_player_id: Optional[str],
    cache: Optional[_EconomyCache] = None,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Build economy state from an extracted state payload and flatten features.

    When *cache* is provided, board_economy and opening_economy are looked up
    by board fingerprint instead of being recomputed each row.  Player economy,
    trade, expansion, and pressure are always computed fresh (they are dynamic).
    """
    from ai.economy.builders import (
        _board_fingerprint,
        _select_state_view,
        build_board_economy_v1,
        build_player_economy_v1,
    )
    from ai.economy.trade.profile import build_trade_profile as _build_trade_profile

    # Resolve observed state view
    if isinstance(payload.get("observed_state"), dict):
        state = _select_state_view(payload, "observed")
    elif isinstance(payload.get("tiles"), list):
        state = payload
    else:
        return None

    if current_player_id is None:
        return None

    try:
        fingerprint = _board_fingerprint(state)

        # Board economy — static, cacheable
        if cache is not None:
            board_economy = cache.get_board_economy(state, fingerprint)
        else:
            board_economy = build_board_economy_v1(state)

        # Player economy — dynamic, always recomputed
        player_economy = build_player_economy_v1(
            payload, view="observed", board_economy=board_economy,
        )

        # Trade profile — dynamic
        belief_economy: Dict[str, object] = {
            "available": False, "requested": False, "reason": "not_requested",
        }
        raw_trade = _build_trade_profile(
            state=state,
            board_economy=board_economy,
            player_economy=player_economy,
            belief_economy=None,
        )
        trade_profile: Dict[str, object] = {"available": True, "requested": True, **raw_trade}

        # Assemble economy state dict (same schema as build_economy_state_v1)
        turn_metadata = state.get("turn_metadata", {}) if isinstance(state.get("turn_metadata"), dict) else {}
        from ai.economy.tags import board_economy_tags
        player_tags = [
            tag
            for player in player_economy.get("by_player", [])
            if isinstance(player, dict)
            for tag in player.get("tags", [])
        ]
        eco: Dict[str, object] = {
            "schema_version": "economy_state_v1",
            "context": {
                "source_view": "observed",
                "board_fingerprint": fingerprint,
                "turn_index": turn_metadata.get("turn_index"),
                "current_player_id": turn_metadata.get("current_player_id"),
            },
            "board_economy": board_economy,
            "player_economy": player_economy,
            "belief_economy": belief_economy,
            "trade_profile": trade_profile,
            "summary_tags": sorted(set(board_economy_tags(board_economy) + player_tags)),
        }

    except Exception:
        return None

    from ai.economy.ml.flatten_v1 import flatten_economy_features_v1

    try:
        features = flatten_economy_features_v1(eco, str(current_player_id))
    except Exception:
        return None

    # Deterministic key ordering
    sorted_keys = sorted(features.keys())
    values = np.asarray([features[k] for k in sorted_keys], dtype=np.float32)
    return values, sorted_keys


# ═══════════════════════════════════════════════════════════════════════
# Row builder
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LoadStats:
    """Timing and cache stats from _build_rows."""
    total_time: float = 0.0
    feature_gen_time: float = 0.0
    rows_loaded: int = 0
    rows_skipped: int = 0
    cache_stats: Dict[str, int] = field(default_factory=dict)


def _build_rows(
    input_path: Path,
    split_seed: str,
    split_ratios: Tuple[int, int, int],
    max_rows: Optional[int],
    winner_only: bool,
) -> Tuple[List[Row], LoadStats]:
    """Read JSONL, extract both feature sets, assign splits.

    Returns (rows, load_stats) with timing and cache diagnostics.
    """
    rows: List[Row] = []
    economy_names: Optional[List[str]] = None
    skipped = 0
    cache = _EconomyCache()
    t0 = time.time()
    t_feat = 0.0  # cumulative feature-generation time

    for i, payload in enumerate(_iter_jsonl(input_path)):
        if max_rows is not None and len(rows) >= max_rows:
            break

        if winner_only and str(payload.get("termination_reason") or "") != "winner":
            skipped += 1
            continue

        t_row_start = time.time()

        # Baseline features
        baseline_result = _extract_baseline_features(payload)
        if baseline_result is None:
            skipped += 1
            continue
        x_baseline, y_win = baseline_result

        # Economy features — resolve player ID to match economy state format
        turn_meta = payload.get("turn_metadata", {})
        current_player_name = (
            turn_meta.get("current_player_name") if isinstance(turn_meta, dict) else None
        )
        # Economy subsystem uses "name:X" IDs; turn_metadata has numeric id + string name
        current_player_id = f"name:{current_player_name}" if current_player_name else None
        econ_result = _extract_economy_features(payload, current_player_id, cache=cache)

        t_feat += time.time() - t_row_start

        if econ_result is None:
            skipped += 1
            continue
        x_economy, econ_names = econ_result
        if economy_names is None:
            economy_names = econ_names

        # Split
        game_id = str(payload.get("game_id") or f"unknown_{i}")
        split = _deterministic_split(game_id, split_seed, split_ratios)

        row_index = payload.get("row_index_in_game")
        if not isinstance(row_index, int):
            row_index = -1

        rows.append(Row(
            game_id=game_id,
            split=split,
            y_win=y_win,
            x_baseline=x_baseline,
            x_economy=x_economy,
            baseline_names=BASELINE_FEATURE_NAMES,
            economy_names=econ_names,
            row_index_in_game=row_index,
        ))

        if (len(rows) % 5000) == 0:
            elapsed = time.time() - t0
            cs = cache.stats()
            print(
                f"  ... {len(rows)} rows loaded ({elapsed:.1f}s, "
                f"{skipped} skipped, cache h/m={cs['hits']}/{cs['misses']})"
            )

    elapsed = time.time() - t0
    cs = cache.stats()
    print(f"  Loaded {len(rows)} rows in {elapsed:.1f}s ({skipped} skipped)")
    print(
        f"  Cache: {cs['board_cache_size']} boards cached, "
        f"{cs['hits']} hits / {cs['misses']} misses"
    )

    stats = LoadStats(
        total_time=elapsed,
        feature_gen_time=t_feat,
        rows_loaded=len(rows),
        rows_skipped=skipped,
        cache_stats=cs,
    )
    return rows, stats


# ═══════════════════════════════════════════════════════════════════════
# Matrix assembly
# ═══════════════════════════════════════════════════════════════════════


def _assemble(
    rows: Sequence[Row],
    split: str,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[Row], List[str]]:
    selected = [r for r in rows if r.split == split]
    if not selected:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64), [], []

    if mode == "baseline":
        X = np.stack([r.x_baseline for r in selected])
        names = selected[0].baseline_names
    elif mode == "economy":
        X = np.stack([r.x_economy for r in selected])
        names = selected[0].economy_names
    elif mode == "combined":
        X = np.hstack([
            np.stack([r.x_baseline for r in selected]),
            np.stack([r.x_economy for r in selected]),
        ])
        names = selected[0].baseline_names + selected[0].economy_names
    else:
        raise ValueError(f"Unknown mode: {mode}")

    y = np.asarray([r.y_win for r in selected], dtype=np.int64)
    return X.astype(np.float32), y, selected, names


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════


def _binary_metrics(y_true: np.ndarray, p_win: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"n": 0}
    y_pred = (p_win >= 0.5).astype(np.int64)
    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_true, p_win, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_true, p_win)), 4),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = round(float(roc_auc_score(y_true, p_win)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, p_win)), 4)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _game_max_index(rows: Sequence[Row]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        if r.row_index_in_game < 0:
            continue
        if r.game_id not in out or r.row_index_in_game > out[r.game_id]:
            out[r.game_id] = r.row_index_in_game
    return out


def _stage_for_row(row: Row, max_idx: Dict[str, int]) -> str:
    mx = max_idx.get(row.game_id, -1)
    if row.row_index_in_game < 0 or mx <= 0:
        return "mid"
    progress = row.row_index_in_game / mx
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "mid"
    return "late"


def _metrics_with_stage(
    y_true: np.ndarray,
    p_win: np.ndarray,
    rows: Sequence[Row],
    max_idx: Dict[str, int],
) -> Dict[str, Any]:
    base = _binary_metrics(y_true, p_win)
    if y_true.size == 0:
        return base
    stage_indices: Dict[str, List[int]] = {k: [] for k in STAGE_KEYS}
    for i, r in enumerate(rows):
        stage_indices[_stage_for_row(r, max_idx)].append(i)

    stage_metrics: Dict[str, Any] = {}
    for stage in STAGE_KEYS:
        idx = stage_indices[stage]
        if not idx:
            stage_metrics[stage] = {"n": 0}
        else:
            stage_metrics[stage] = _binary_metrics(y_true[idx], p_win[idx])
    base["by_stage"] = stage_metrics
    return base


# ═══════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════


def _build_model(name: str, seed: int) -> Any:
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(solver="saga", max_iter=500, C=1.0, random_state=seed)),
        ])
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6, max_iter=300, learning_rate=0.05,
            min_samples_leaf=40, random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Feature importance
# ═══════════════════════════════════════════════════════════════════════


def _feature_importance(model: Any, feature_names: List[str], top_k: int = 30) -> List[Dict[str, Any]]:
    estimator = model
    if hasattr(model, "named_steps") and "logreg" in model.named_steps:
        estimator = model.named_steps["logreg"]

    importances: Optional[np.ndarray] = None
    source = "unknown"

    if hasattr(estimator, "coef_") and isinstance(estimator.coef_, np.ndarray):
        importances = np.abs(estimator.coef_[0])
        source = "abs_coef"
    elif hasattr(estimator, "feature_importances_") and isinstance(estimator.feature_importances_, np.ndarray):
        importances = estimator.feature_importances_
        source = "feature_importances_"

    if importances is None:
        return []

    top_idx = np.argsort(importances)[::-1][:top_k]
    out: List[Dict[str, Any]] = []
    for idx in top_idx:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        out.append({
            "rank": len(out) + 1,
            "feature": name,
            "importance": round(float(importances[idx]), 6),
            "source": source,
        })
    return out


def _group_importance(importances: List[Dict[str, Any]]) -> Dict[str, float]:
    """Group feature importances by economy block prefix."""
    groups: Dict[str, float] = {}
    for item in importances:
        name = item.get("feature", "")
        imp = float(item.get("importance", 0.0))
        if name.startswith("board.opening"):
            group = "opening"
        elif name.startswith("board."):
            group = "board"
        elif name.startswith("self.expansion"):
            group = "expansion"
        elif name.startswith("self.pressure"):
            group = "pressure"
        elif name.startswith("self.trade") or name.startswith("self.trade_"):
            group = "trade"
        elif name.startswith("belief."):
            group = "belief"
        elif name.startswith("self.plan."):
            group = "player_plan"
        elif name.startswith("self."):
            group = "player"
        else:
            group = "baseline"
        groups[group] = groups.get(group, 0.0) + imp
    return {k: round(v, 6) for k, v in sorted(groups.items(), key=lambda x: -x[1])}


# ═══════════════════════════════════════════════════════════════════════
# Comparison printer
# ═══════════════════════════════════════════════════════════════════════


def _fmt(v: Any, w: int = 7) -> str:
    return f"{v:{w}.4f}" if v is not None else f"{'n/a':>{w}}"


def _print_comparison(all_results: Dict[str, Dict[str, Dict[str, Any]]], label: str = "") -> None:
    """Print a concise comparison table across modes and models."""
    title = f"EXPERIMENT COMPARISON SUMMARY{f'  [{label}]' if label else ''}"
    print("\n" + "=" * 105)
    print(title)
    print("=" * 105)

    header = (
        f"{'mode':<12} {'model':<8} "
        f"{'v.AUC':>7} {'v.PRAUC':>7} {'v.LL':>8} {'v.Acc':>6} {'v.F1':>6} | "
        f"{'t.AUC':>7} {'t.LL':>8} | "
        f"{'early':>7} {'mid':>7} {'late':>7}"
    )
    print(header)
    print("-" * 105)

    for mode in FEATURE_MODES:
        for model_name in list(MODEL_NAMES) + ["prior"]:
            key = f"{mode}/{model_name}"
            result = all_results.get(key)
            if result is None:
                continue
            val = result.get("val", {})
            tst = result.get("test", {})

            stage = val.get("by_stage", {})
            early_auc = (stage.get("early") or {}).get("roc_auc")
            mid_auc = (stage.get("mid") or {}).get("roc_auc")
            late_auc = (stage.get("late") or {}).get("roc_auc")

            line = (
                f"{mode:<12} {model_name:<8} "
                f"{_fmt(val.get('roc_auc'))} {_fmt(val.get('pr_auc'))} "
                f"{_fmt(val.get('log_loss'), 8)} {_fmt(val.get('accuracy'), 6)} "
                f"{_fmt(val.get('f1'), 6)} | "
                f"{_fmt(tst.get('roc_auc'))} {_fmt(tst.get('log_loss'), 8)} | "
                f"{_fmt(early_auc)} {_fmt(mid_auc)} {_fmt(late_auc)}"
            )
            print(line)
    print("=" * 105)

    # Economy lift summary
    for model_name in MODEL_NAMES:
        baseline_key = f"baseline/{model_name}"
        combined_key = f"combined/{model_name}"
        economy_key = f"economy/{model_name}"
        b_val = (all_results.get(baseline_key) or {}).get("val", {})
        c_val = (all_results.get(combined_key) or {}).get("val", {})
        e_val = (all_results.get(economy_key) or {}).get("val", {})
        b_auc = b_val.get("roc_auc")
        c_auc = c_val.get("roc_auc")
        e_auc = e_val.get("roc_auc")
        if b_auc is not None and c_auc is not None:
            delta = c_auc - b_auc
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {model_name} economy lift: combined AUC - baseline AUC = {delta:+.4f} {direction}")
        if e_auc is not None and b_auc is not None:
            delta_e = e_auc - b_auc
            direction_e = "↑" if delta_e > 0 else "↓" if delta_e < 0 else "="
            print(f"  {model_name} economy-only vs baseline:      {delta_e:+.4f} {direction_e}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Split summary helpers
# ═══════════════════════════════════════════════════════════════════════


def _compute_split_summary(rows: Sequence[Row]) -> Dict[str, Dict[str, Any]]:
    games_by_split: Dict[str, set] = {s: set() for s in VALID_SPLITS}
    rows_by_split: Dict[str, int] = {s: 0 for s in VALID_SPLITS}
    pos_by_split: Dict[str, int] = {s: 0 for s in VALID_SPLITS}
    for r in rows:
        games_by_split[r.split].add(r.game_id)
        rows_by_split[r.split] += 1
        pos_by_split[r.split] += r.y_win

    all_games: set = set()
    for s in VALID_SPLITS:
        overlap = all_games & games_by_split[s]
        if overlap:
            raise RuntimeError(f"Game-level split leakage: {len(overlap)} games in multiple splits")
        all_games.update(games_by_split[s])

    return {
        s: {
            "games": len(games_by_split[s]),
            "rows": rows_by_split[s],
            "positive_rate": round(pos_by_split[s] / rows_by_split[s], 4) if rows_by_split[s] > 0 else None,
        }
        for s in VALID_SPLITS
    }


# ═══════════════════════════════════════════════════════════════════════
# Main experiment (single cap)
# ═══════════════════════════════════════════════════════════════════════


def run_experiment(
    input_path: Path,
    output_dir: Path,
    split_seed: str = "economy_experiment_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
    max_rows: Optional[int] = None,
    winner_only: bool = True,
    seed: int = 42,
    top_k_importance: int = 30,
    models: Sequence[str] = MODEL_NAMES,
    rows_preloaded: Optional[Tuple[List[Row], LoadStats]] = None,
) -> Dict[str, Any]:
    """Run the full 3-mode comparison experiment.

    If *rows_preloaded* is provided, skip data loading and use the given rows
    (sliced to max_rows).  This enables staged-cap runs that load data once.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_t0 = time.time()

    cap_label = f"{max_rows}" if max_rows else "full"
    print(f"\n{'━' * 80}")
    print(f"  Experiment cap={cap_label}")
    print(f"{'━' * 80}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Max rows: {max_rows or 'unlimited'}")
    print(f"Winner only: {winner_only}")
    print(f"Seed: {seed}")
    print()

    # ── Load data ─────────────────────────────────────────────────────
    if rows_preloaded is not None:
        all_rows, load_stats = rows_preloaded
        if max_rows is not None and len(all_rows) > max_rows:
            rows = all_rows[:max_rows]
        else:
            rows = all_rows
        print(f"  Using {len(rows)} pre-loaded rows (of {len(all_rows)} available)")
    else:
        print("Loading data and building features...")
        rows, load_stats = _build_rows(input_path, split_seed, split_ratios, max_rows, winner_only)

    if not rows:
        raise RuntimeError("No usable rows after loading and filtering")

    # ── Split summary ─────────────────────────────────────────────────
    split_summary = _compute_split_summary(rows)

    print(f"\nSplit summary:")
    for s in ("train", "val", "test"):
        info = split_summary[s]
        print(f"  {s}: {info['games']} games, {info['rows']} rows, pos_rate={info['positive_rate']}")
    print(f"  Baseline features: {len(rows[0].baseline_names)}")
    print(f"  Economy features:  {len(rows[0].economy_names)}")
    print(f"  Combined features: {len(rows[0].baseline_names) + len(rows[0].economy_names)}")

    max_idx = _game_max_index(rows)
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_importances: Dict[str, Any] = {}
    train_timings: Dict[str, float] = {}

    # ── Train / eval for each mode × model ────────────────────────────
    for mode in FEATURE_MODES:
        X_train, y_train, train_rows, feat_names = _assemble(rows, "train", mode)
        X_val, y_val, val_rows, _ = _assemble(rows, "val", mode)
        X_test, y_test, test_rows, _ = _assemble(rows, "test", mode)

        if X_train.size == 0 or X_val.size == 0:
            print(f"  WARNING: {mode} has empty train or val; skipping")
            continue

        print(f"\n── Mode: {mode} ({X_train.shape[1]} features) ──")

        # Prior baseline
        prior = DummyClassifier(strategy="prior")
        prior.fit(X_train, y_train)
        p_val_prior = _predict(prior, X_val)
        p_test_prior = _predict(prior, X_test) if X_test.size > 0 else np.array([])
        all_results[f"{mode}/prior"] = {
            "val": _metrics_with_stage(y_val, p_val_prior, val_rows, max_idx),
            "test": _metrics_with_stage(y_test, p_test_prior, test_rows, max_idx) if X_test.size > 0 else {},
        }

        for model_name in models:
            key = f"{mode}/{model_name}"
            print(f"  Training {key}...", end=" ", flush=True)
            t0 = time.time()

            model = _build_model(model_name, seed)
            model.fit(X_train, y_train)

            p_train = _predict(model, X_train)
            p_val = _predict(model, X_val)
            p_test = _predict(model, X_test) if X_test.size > 0 else np.array([])

            train_m = _metrics_with_stage(y_train, p_train, train_rows, max_idx)
            val_m = _metrics_with_stage(y_val, p_val, val_rows, max_idx)
            test_m = _metrics_with_stage(y_test, p_test, test_rows, max_idx) if X_test.size > 0 else {}
            all_results[key] = {"train": train_m, "val": val_m, "test": test_m}

            elapsed = time.time() - t0
            train_timings[key] = round(elapsed, 2)
            auc_str = f"AUC={val_m.get('roc_auc', 'n/a')}"
            ll_str = f"logloss={val_m.get('log_loss', 'n/a')}"
            print(f"{auc_str}, {ll_str} ({elapsed:.1f}s)")

            # Feature importance
            fi = _feature_importance(model, feat_names, top_k_importance)
            grouped = _group_importance(fi) if fi else {}
            all_importances[key] = {"top_features": fi, "by_group": grouped}

            # Save model
            model_path = output_dir / f"model_{mode}_{model_name}.pkl"
            with model_path.open("wb") as fh:
                pickle.dump(model, fh)

    experiment_elapsed = time.time() - experiment_t0
    total_train_time = sum(train_timings.values())

    # ── Runtime summary ───────────────────────────────────────────────
    runtime_summary = {
        "cap": max_rows,
        "rows_used": len(rows),
        "total_wall_clock_s": round(experiment_elapsed, 2),
        "feature_generation_s": round(load_stats.feature_gen_time, 2),
        "training_s": round(total_train_time, 2),
        "overhead_s": round(experiment_elapsed - total_train_time, 2),
        "per_model_training_s": train_timings,
        "cache_stats": load_stats.cache_stats,
        "rows_skipped": load_stats.rows_skipped,
    }

    # ── Save artifacts ────────────────────────────────────────────────
    _save_json(output_dir / "metrics.json", all_results)
    _save_json(output_dir / "feature_importance.json", all_importances)
    _save_json(output_dir / "split_summary.json", split_summary)
    _save_json(output_dir / "runtime.json", runtime_summary)
    _save_json(output_dir / "feature_names.json", {
        "baseline": BASELINE_FEATURE_NAMES,
        "economy": rows[0].economy_names if rows else [],
    })
    _save_json(output_dir / "config.json", {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "split_seed": split_seed,
        "split_ratios": list(split_ratios),
        "max_rows": max_rows,
        "winner_only": winner_only,
        "seed": seed,
        "models": list(models),
        "top_k_importance": top_k_importance,
        "baseline_feature_count": len(BASELINE_FEATURE_NAMES),
        "economy_feature_count": len(rows[0].economy_names) if rows else 0,
    })

    # ── Print comparison ──────────────────────────────────────────────
    _print_comparison(all_results, label=cap_label)

    # ── Print top feature importances ─────────────────────────────────
    for mode in ("economy", "combined"):
        for model_name in models:
            key = f"{mode}/{model_name}"
            fi = all_importances.get(key, {})
            top = fi.get("top_features", [])[:15]
            grouped = fi.get("by_group", {})
            if top:
                print(f"\nTop features ({key}):")
                for item in top:
                    print(f"  {item['rank']:>3}. {item['feature']:<45} {item['importance']:.6f}")
            if grouped:
                print(f"  Grouped importance ({key}):")
                for group, imp in grouped.items():
                    print(f"    {group:<20} {imp:.6f}")

    # ── Print runtime ─────────────────────────────────────────────────
    print(f"\n  Runtime: {experiment_elapsed:.1f}s total "
          f"(feature gen {load_stats.feature_gen_time:.1f}s, "
          f"training {total_train_time:.1f}s)")

    print(f"\nArtifacts saved to: {output_dir}")

    return {
        "metrics": all_results,
        "runtime": runtime_summary,
        "split_summary": split_summary,
        "importances": all_importances,
    }


# ═══════════════════════════════════════════════════════════════════════
# Staged-cap experiment
# ═══════════════════════════════════════════════════════════════════════


def run_staged_experiment(
    input_path: Path,
    output_dir: Path,
    row_caps: List[Optional[int]],
    split_seed: str = "economy_experiment_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
    winner_only: bool = True,
    seed: int = 42,
    top_k_importance: int = 30,
    models: Sequence[str] = MODEL_NAMES,
) -> None:
    """Run experiments at increasing row caps, loading data once.

    - Loads rows up to max(row_caps) once.
    - For each cap, slices rows[:cap] and runs the full experiment.
    - Saves per-cap artifacts in sub-directories.
    - Saves a cross-cap comparison summary at the top level.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine effective max to load
    numeric_caps = [c for c in row_caps if c is not None]
    load_cap = max(numeric_caps) if numeric_caps else None
    has_full = None in row_caps

    effective_load_cap = None if has_full else load_cap

    print("=" * 80)
    print("STAGED EXPERIMENT")
    print(f"  Caps: {[c if c is not None else 'full' for c in row_caps]}")
    print(f"  Loading up to: {'unlimited' if effective_load_cap is None else effective_load_cap} rows")
    print("=" * 80)

    # Load all data once
    staged_t0 = time.time()
    print("\nLoading data (shared across all caps)...")
    all_rows, load_stats = _build_rows(
        input_path, split_seed, split_ratios, effective_load_cap, winner_only,
    )
    if not all_rows:
        raise RuntimeError("No usable rows after loading and filtering")

    print(f"\n  Total rows available: {len(all_rows)}")

    # Run per cap
    cross_cap_summary: List[Dict[str, Any]] = []

    for cap in row_caps:
        cap_label = str(cap) if cap is not None else "full"
        cap_dir = output_dir / f"cap_{cap_label}"

        result = run_experiment(
            input_path=input_path,
            output_dir=cap_dir,
            split_seed=split_seed,
            split_ratios=split_ratios,
            max_rows=cap,
            winner_only=winner_only,
            seed=seed,
            top_k_importance=top_k_importance,
            models=models,
            rows_preloaded=(all_rows, load_stats),
        )

        # Extract concise summary for cross-cap table
        metrics = result["metrics"]
        runtime = result["runtime"]
        split_s = result["split_summary"]

        entry: Dict[str, Any] = {
            "cap": cap,
            "rows_used": runtime["rows_used"],
            "games_train": split_s.get("train", {}).get("games", 0),
            "games_val": split_s.get("val", {}).get("games", 0),
            "games_test": split_s.get("test", {}).get("games", 0),
            "total_time_s": runtime["total_wall_clock_s"],
            "feature_gen_s": runtime["feature_generation_s"],
            "training_s": runtime["training_s"],
            "cache_hits": runtime.get("cache_stats", {}).get("hits", 0),
            "cache_misses": runtime.get("cache_stats", {}).get("misses", 0),
            "cache_boards": runtime.get("cache_stats", {}).get("board_cache_size", 0),
        }
        for mode in FEATURE_MODES:
            for model_name in models:
                key = f"{mode}/{model_name}"
                val_m = (metrics.get(key) or {}).get("val", {})
                tst_m = (metrics.get(key) or {}).get("test", {})
                entry[f"{key}/val_auc"] = val_m.get("roc_auc")
                entry[f"{key}/val_logloss"] = val_m.get("log_loss")
                entry[f"{key}/test_auc"] = tst_m.get("roc_auc")
                entry[f"{key}/test_logloss"] = tst_m.get("log_loss")
                # Late-game val AUC
                val_late = (val_m.get("by_stage") or {}).get("late") or {}
                entry[f"{key}/val_late_auc"] = val_late.get("roc_auc")
        cross_cap_summary.append(entry)

    # ── Save cross-cap summary ────────────────────────────────────────
    _save_json(output_dir / "cross_cap_summary.json", cross_cap_summary)

    staged_elapsed = time.time() - staged_t0

    # ── Print cross-cap comparison ────────────────────────────────────
    _print_cross_cap_summary(cross_cap_summary, models)

    print(f"\nTotal staged experiment time: {staged_elapsed:.1f}s")
    print(f"Cross-cap summary saved to: {output_dir / 'cross_cap_summary.json'}")


def _print_cross_cap_summary(
    summary: List[Dict[str, Any]],
    models: Sequence[str] = MODEL_NAMES,
) -> None:
    W = 130

    def _cap_str(entry: Dict[str, Any]) -> str:
        cap = entry.get("cap")
        return str(cap) if cap is not None else "full"

    # ── Section 1: Infrastructure ─────────────────────────────────────
    print("\n" + "=" * W)
    print("CROSS-CAP SUMMARY")
    print("=" * W)
    print(f"\n  {'cap':>10} {'rows':>8} {'tr/va/te':>10} {'feat_s':>7} {'train_s':>8} {'total_s':>8} | {'cache_boards':>12} {'hit/miss':>12}")
    print("  " + "-" * (W - 2))
    for e in summary:
        games_str = f"{e.get('games_train',0)}/{e.get('games_val',0)}/{e.get('games_test',0)}"
        hm = f"{e.get('cache_hits',0)}/{e.get('cache_misses',0)}"
        print(
            f"  {_cap_str(e):>10} {e.get('rows_used',0):>8} {games_str:>10} "
            f"{e.get('feature_gen_s',0):>7.1f} {e.get('training_s',0):>8.1f} "
            f"{e.get('total_time_s',0):>8.1f} | {e.get('cache_boards',0):>12} {hm:>12}"
        )

    # ── Section 2: Val AUC + LogLoss ──────────────────────────────────
    best_model = models[0] if models else "hgb"
    for mn in models:
        print(f"\n  ── Val metrics ({mn}) ──")
        print(f"  {'cap':>10} {'base_AUC':>9} {'econ_AUC':>9} {'comb_AUC':>9} | {'base_LL':>9} {'econ_LL':>9} {'comb_LL':>9}")
        print("  " + "-" * 72)
        for e in summary:
            vals = {}
            for mode in FEATURE_MODES:
                key = f"{mode}/{mn}"
                vals[f"{mode}_auc"] = e.get(f"{key}/val_auc")
                vals[f"{mode}_ll"] = e.get(f"{key}/val_logloss")
            print(
                f"  {_cap_str(e):>10} "
                f"{_fmt(vals['baseline_auc'], 9)} {_fmt(vals['economy_auc'], 9)} {_fmt(vals['combined_auc'], 9)} | "
                f"{_fmt(vals['baseline_ll'], 9)} {_fmt(vals['economy_ll'], 9)} {_fmt(vals['combined_ll'], 9)}"
            )

    # ── Section 3: Test AUC + LogLoss ─────────────────────────────────
    for mn in models:
        print(f"\n  ── Test metrics ({mn}) ──")
        print(f"  {'cap':>10} {'base_AUC':>9} {'econ_AUC':>9} {'comb_AUC':>9} | {'base_LL':>9} {'econ_LL':>9} {'comb_LL':>9}")
        print("  " + "-" * 72)
        for e in summary:
            vals = {}
            for mode in FEATURE_MODES:
                key = f"{mode}/{mn}"
                vals[f"{mode}_auc"] = e.get(f"{key}/test_auc")
                vals[f"{mode}_ll"] = e.get(f"{key}/test_logloss")
            print(
                f"  {_cap_str(e):>10} "
                f"{_fmt(vals['baseline_auc'], 9)} {_fmt(vals['economy_auc'], 9)} {_fmt(vals['combined_auc'], 9)} | "
                f"{_fmt(vals['baseline_ll'], 9)} {_fmt(vals['economy_ll'], 9)} {_fmt(vals['combined_ll'], 9)}"
            )

    # ── Section 4: Late-game val AUC ──────────────────────────────────
    print(f"\n  ── Late-game val AUC ──")
    for mn in models:
        print(f"  {mn}:  ", end="")
        print(f"{'cap':>10} {'base':>7} {'econ':>7} {'comb':>7}")
        for e in summary:
            b = e.get(f"baseline/{mn}/val_late_auc")
            ec = e.get(f"economy/{mn}/val_late_auc")
            co = e.get(f"combined/{mn}/val_late_auc")
            print(f"          {_cap_str(e):>10} {_fmt(b)} {_fmt(ec)} {_fmt(co)}")

    # ── Section 5: AUC lift ───────────────────────────────────────────
    print(f"\n  ── AUC lift (economy & combined over baseline) ──")
    for mn in models:
        print(f"  {mn}:")
        print(f"    {'cap':>10} {'econ-base_v':>12} {'comb-base_v':>12} | {'econ-base_t':>12} {'comb-base_t':>12}")
        for e in summary:
            bv = e.get(f"baseline/{mn}/val_auc")
            ev = e.get(f"economy/{mn}/val_auc")
            cv = e.get(f"combined/{mn}/val_auc")
            bt = e.get(f"baseline/{mn}/test_auc")
            et = e.get(f"economy/{mn}/test_auc")
            ct = e.get(f"combined/{mn}/test_auc")

            def _delta(a, b):
                if a is not None and b is not None:
                    d = a - b
                    arrow = "↑" if d > 0 else "↓" if d < 0 else "="
                    return f"{d:+.4f} {arrow}"
                return "n/a"

            print(
                f"    {_cap_str(e):>10} "
                f"{_delta(ev, bv):>12} {_delta(cv, bv):>12} | "
                f"{_delta(et, bt):>12} {_delta(ct, bt):>12}"
            )
    print("\n" + "=" * W)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Economy-feature ML experiment: baseline vs economy vs combined",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL with extracted state payloads")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write model artifacts and metrics")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap total rows for single-cap run (default: unlimited)")
    parser.add_argument("--row-caps", type=int, nargs="+", default=None,
                        help="Run staged experiments at these row caps "
                             "(e.g. --row-caps 50000 100000 200000). "
                             "Loads data once, then runs each cap in sequence. "
                             "Overrides --max-rows when provided.")
    parser.add_argument("--include-full", action="store_true", default=False,
                        help="With --row-caps, also run an uncapped experiment as the last stage")
    parser.add_argument("--winner-only", action="store_true", default=True,
                        help="Only use rows from winner-terminated games (default: True)")
    parser.add_argument("--no-winner-only", action="store_false", dest="winner_only",
                        help="Include all games regardless of termination")
    parser.add_argument("--models", default="hgb,logreg",
                        help="Comma-separated model list: hgb,logreg")
    parser.add_argument("--split-seed", default="economy_experiment_v1")
    parser.add_argument("--split-ratios", default="80,10,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-importance", type=int, default=30)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    split_parts = [int(x.strip()) for x in args.split_ratios.split(",")]
    if len(split_parts) != 3 or sum(split_parts) != 100:
        print("ERROR: --split-ratios must be three integers summing to 100", file=sys.stderr)
        return 1

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in selected_models:
        if m not in MODEL_NAMES:
            print(f"ERROR: unknown model '{m}'. Supported: {MODEL_NAMES}", file=sys.stderr)
            return 1

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    ratios = tuple(split_parts)

    # ── Staged-cap mode ───────────────────────────────────────────────
    if args.row_caps is not None:
        caps: List[Optional[int]] = sorted(args.row_caps)
        if args.include_full:
            caps.append(None)

        run_staged_experiment(
            input_path=input_path,
            output_dir=output_dir,
            row_caps=caps,
            split_seed=args.split_seed,
            split_ratios=ratios,
            winner_only=args.winner_only,
            seed=args.seed,
            top_k_importance=args.top_k_importance,
            models=selected_models,
        )
        return 0

    # ── Single-cap mode (original behavior) ───────────────────────────
    run_experiment(
        input_path=input_path,
        output_dir=output_dir,
        split_seed=args.split_seed,
        split_ratios=ratios,
        max_rows=args.max_rows,
        winner_only=args.winner_only,
        seed=args.seed,
        top_k_importance=args.top_k_importance,
        models=selected_models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())