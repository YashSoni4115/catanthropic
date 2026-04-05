from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai.dataset_builder_v1 import VALID_SPLITS, iter_state_payloads, load_split_manifest
from ai.trajectory_dataset_builder_v1 import iter_trajectory_rows


DEFAULT_SPLIT_RATIOS = (80, 10, 10)
PHASE_ORDER = ["setup", "roll", "main", "trade", "build", "unknown"]
RESOURCE_ORDER = ["BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"]
STAGE_KEYS = ["early", "mid", "late"]


@dataclass
class RowRecord:
    game_id: str
    split: str
    y_win: int
    x: np.ndarray
    position_id: Optional[str]
    row_index_in_game: int


def _parse_split_ratios(raw: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("split ratios must be train,val,test")
    ratios = tuple(int(p) for p in parts)
    if sum(ratios) != 100:
        raise ValueError("split ratios must sum to 100")
    if any(r < 0 for r in ratios):
        raise ValueError("split ratios cannot be negative")
    return ratios  # type: ignore[return-value]


def _deterministic_split_for_game(game_id: str, split_seed: str, ratios: Tuple[int, int, int]) -> str:
    train_ratio, val_ratio, _ = ratios
    digest = hashlib.md5(f"{split_seed}:{game_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _detect_input_format(input_path: Path) -> str:
    for payload in _iter_jsonl(input_path):
        if isinstance(payload.get("x"), list):
            return "rows"
        if isinstance(payload.get("player_state"), list) and isinstance(payload.get("turn_metadata"), dict):
            return "selfplay_raw"
        return "states"
    raise ValueError(f"No JSON objects found in input: {input_path}")


def _winner_only_payloads(payloads: Iterable[Dict[str, Any]], winner_only: bool) -> Iterator[Dict[str, Any]]:
    for payload in payloads:
        if winner_only and str(payload.get("termination_reason") or "") != "winner":
            continue
        yield payload


def _phase_onehot(phase_raw: Any) -> List[float]:
    phase = str(phase_raw or "unknown").lower()
    out = [1.0 if phase == item else 0.0 for item in PHASE_ORDER]
    if sum(out) == 0.0:
        out[-1] = 1.0
    return out


def _player_total_resources(player_row: Dict[str, Any]) -> float:
    resources = player_row.get("resources") if isinstance(player_row.get("resources"), dict) else {}
    total = 0.0
    for resource in RESOURCE_ORDER:
        value = resources.get(resource)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _safe_resource_value(player_row: Dict[str, Any], resource: str) -> float:
    resources = player_row.get("resources") if isinstance(player_row.get("resources"), dict) else {}
    value = resources.get(resource)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _selfplay_raw_to_row(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    turn_meta = payload.get("turn_metadata") if isinstance(payload.get("turn_metadata"), dict) else {}
    players = payload.get("player_state") if isinstance(payload.get("player_state"), list) else []
    if len(players) == 0:
        return None

    current_name = str(turn_meta.get("current_player_name") or "")
    winner_name = str(payload.get("winner_player_name") or "")
    if not winner_name:
        return None

    current_player: Optional[Dict[str, Any]] = None
    others: List[Dict[str, Any]] = []
    for entry in players:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("player_name") or "")
        if name == current_name and current_player is None:
            current_player = entry
        else:
            others.append(entry)

    if current_player is None:
        return None

    others.sort(key=lambda row: str(row.get("player_name") or ""))

    turn_index = turn_meta.get("turn_index")
    if not isinstance(turn_index, int):
        row_idx = payload.get("row_index_in_game")
        turn_index = int(row_idx) if isinstance(row_idx, int) else -1

    latest_roll = payload.get("roll")
    roll_value = float(latest_roll) if isinstance(latest_roll, (int, float)) else -1.0

    current_vp = float(current_player.get("victory_points") or 0.0)
    current_total_res = _player_total_resources(current_player)

    x: List[float] = []
    x.append(float(turn_index))
    x.extend(_phase_onehot(turn_meta.get("current_phase")))
    x.append(roll_value)

    x.append(current_vp)
    x.append(current_total_res)
    for resource in RESOURCE_ORDER:
        x.append(_safe_resource_value(current_player, resource))

    opp_vps: List[float] = []
    opp_res: List[float] = []
    for opponent in others[:3]:
        opp_vp = float(opponent.get("victory_points") or 0.0)
        opp_total_res = _player_total_resources(opponent)
        opp_vps.append(opp_vp)
        opp_res.append(opp_total_res)
        x.append(opp_vp)
        x.append(opp_total_res)
    while len(opp_vps) < 3:
        opp_vps.append(0.0)
        opp_res.append(0.0)
        x.extend([0.0, 0.0])

    max_opp_vp = max(opp_vps) if opp_vps else 0.0
    mean_opp_vp = float(sum(opp_vps) / len(opp_vps)) if opp_vps else 0.0
    mean_opp_res = float(sum(opp_res) / len(opp_res)) if opp_res else 0.0
    x.append(current_vp - max_opp_vp)
    x.append(current_vp - mean_opp_vp)
    x.append(current_total_res - mean_opp_res)

    return {
        "game_id": payload.get("game_id"),
        "split": payload.get("split"),
        "position_id": payload.get("position_id"),
        "row_index_in_game": payload.get("row_index_in_game"),
        "x": x,
        "y_win": 1 if current_name == winner_name else 0,
    }


def _iter_selfplay_raw_rows(input_path: Path, winner_only: bool) -> Iterator[Dict[str, Any]]:
    for payload in _iter_jsonl(input_path):
        if winner_only and str(payload.get("termination_reason") or "") != "winner":
            continue
        row = _selfplay_raw_to_row(payload)
        if row is not None:
            yield row


def _row_to_record(
    row: Dict[str, Any],
    split_seed: str,
    split_ratios: Tuple[int, int, int],
) -> Optional[RowRecord]:
    x = row.get("x")
    y = row.get("y_win")
    if not isinstance(x, list):
        return None
    if not isinstance(y, int) or y not in (0, 1):
        return None

    game_id = str(row.get("game_id") or "__unknown_game__")
    split = str(row.get("split") or "").strip().lower()
    if split not in VALID_SPLITS:
        split = _deterministic_split_for_game(game_id, split_seed, split_ratios)

    position_id = row.get("position_id")
    row_index = row.get("row_index_in_game")
    if not isinstance(row_index, int):
        row_index = row.get("trajectory_row_index_in_game")
    if not isinstance(row_index, int):
        row_index = -1

    try:
        features = np.asarray(x, dtype=np.float32)
    except Exception:
        return None

    return RowRecord(
        game_id=game_id,
        split=split,
        y_win=y,
        x=features,
        position_id=str(position_id) if position_id is not None else None,
        row_index_in_game=row_index,
    )


def _game_rng(game_id: str, seed: int) -> random.Random:
    digest = hashlib.md5(f"{seed}:{game_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


def _reservoir_add(
    sample: List[RowRecord],
    seen_count: int,
    cap: int,
    item: RowRecord,
    rng: random.Random,
) -> None:
    if len(sample) < cap:
        sample.append(item)
        return
    replacement_index = rng.randint(0, seen_count - 1)
    if replacement_index < cap:
        sample[replacement_index] = item


def _sample_records(
    row_iter: Iterable[Dict[str, Any]],
    split_seed: str,
    split_ratios: Tuple[int, int, int],
    max_rows_per_game: Optional[int],
    seed: int,
    dedupe_position_id: bool,
) -> List[RowRecord]:
    sampled_by_game: Dict[str, List[RowRecord]] = {}
    seen_count_by_game: Dict[str, int] = {}
    position_seen_by_game: Dict[str, set] = {}
    rng_by_game: Dict[str, random.Random] = {}

    for row in row_iter:
        record = _row_to_record(row, split_seed=split_seed, split_ratios=split_ratios)
        if record is None:
            continue

        if dedupe_position_id and record.position_id is not None:
            seen_positions = position_seen_by_game.setdefault(record.game_id, set())
            if record.position_id in seen_positions:
                continue
            seen_positions.add(record.position_id)

        if max_rows_per_game is None:
            sampled_by_game.setdefault(record.game_id, []).append(record)
            continue

        if max_rows_per_game <= 0:
            continue

        sample = sampled_by_game.setdefault(record.game_id, [])
        seen_count = seen_count_by_game.get(record.game_id, 0) + 1
        seen_count_by_game[record.game_id] = seen_count
        if record.game_id not in rng_by_game:
            rng_by_game[record.game_id] = _game_rng(record.game_id, seed)
        _reservoir_add(
            sample=sample,
            seen_count=seen_count,
            cap=max_rows_per_game,
            item=record,
            rng=rng_by_game[record.game_id],
        )

    output: List[RowRecord] = []
    for game_id in sorted(sampled_by_game.keys()):
        output.extend(sampled_by_game[game_id])
    return output


def _compute_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    n_total = len(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return np.ones(n_total, dtype=np.float32)
    w_pos = n_total / (2.0 * positives)
    w_neg = n_total / (2.0 * negatives)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def _parse_stage_weights(raw: str) -> Dict[str, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("stage weights must be three comma-separated floats: early,mid,late")
    values = [float(v) for v in parts]
    if any(v < 0.0 for v in values):
        raise ValueError("stage weights must be non-negative")
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("stage weights must sum to > 0")
    norm = [v / total for v in values]
    return {
        "early": norm[0],
        "mid": norm[1],
        "late": norm[2],
    }


def _slice_by_split(records: Sequence[RowRecord], split: str) -> Tuple[np.ndarray, np.ndarray, List[RowRecord]]:
    selected = [r for r in records if r.split == split]
    if not selected:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    X = np.stack([r.x for r in selected]).astype(np.float32)
    y = np.asarray([r.y_win for r in selected], dtype=np.int64)
    return X, y, selected


def _binary_metrics(y_true: np.ndarray, p_win: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"n": 0}

    y_pred = (p_win >= 0.5).astype(np.int64)
    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
        "log_loss": float(log_loss(y_true, p_win, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_win)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    unique = np.unique(y_true)
    if unique.size >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p_win))
        out["pr_auc"] = float(average_precision_score(y_true, p_win))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _game_max_row_index(records: Sequence[RowRecord]) -> Dict[str, int]:
    output: Dict[str, int] = {}
    for record in records:
        idx = record.row_index_in_game
        if idx < 0:
            continue
        current = output.get(record.game_id)
        if current is None or idx > current:
            output[record.game_id] = idx
    return output


def _stage_for_record(record: RowRecord, max_idx_by_game: Dict[str, int]) -> str:
    max_idx = max_idx_by_game.get(record.game_id, -1)
    if record.row_index_in_game < 0 or max_idx <= 0:
        return "mid"
    progress = float(record.row_index_in_game) / float(max_idx)
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "mid"
    return "late"


def _metrics_with_stage(
    y_true: np.ndarray,
    p_win: np.ndarray,
    records: Sequence[RowRecord],
    max_idx_by_game: Dict[str, int],
    stage_weights: Dict[str, float],
) -> Dict[str, Any]:
    base = _binary_metrics(y_true, p_win)
    if y_true.size == 0:
        return base

    stage_indices: Dict[str, List[int]] = {k: [] for k in STAGE_KEYS}
    for idx, record in enumerate(records):
        stage = _stage_for_record(record, max_idx_by_game)
        stage_indices[stage].append(idx)

    stage_metrics: Dict[str, Any] = {}
    weighted_log_loss = 0.0
    used_weight = 0.0
    for stage in STAGE_KEYS:
        indices = stage_indices[stage]
        if not indices:
            stage_metrics[stage] = {"n": 0}
            continue
        stage_y = y_true[indices]
        stage_p = p_win[indices]
        metrics = _binary_metrics(stage_y, stage_p)
        stage_metrics[stage] = metrics
        if metrics.get("log_loss") is not None:
            weight = float(stage_weights.get(stage, 0.0))
            weighted_log_loss += weight * float(metrics["log_loss"])
            used_weight += weight

    if used_weight > 0.0:
        stage_balanced_log_loss = weighted_log_loss / used_weight
    else:
        stage_balanced_log_loss = None

    base["stage_metrics"] = stage_metrics
    base["stage_balanced_log_loss"] = stage_balanced_log_loss
    base["stage_weights"] = dict(stage_weights)
    return base


def _build_model(model_name: str, seed: int) -> Any:
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        solver="saga",
                        max_iter=400,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=40,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _extract_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba must return shape [n_samples, 2]")
    return np.asarray(proba[:, 1], dtype=np.float64)


def _unpack_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps") and "logreg" in model.named_steps:
        return model.named_steps["logreg"]
    return model


def _top_feature_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_features: int,
    seed: int,
    permutation_sample_size: int,
    permutation_repeats: int,
) -> List[Dict[str, Any]]:
    if X_val.size == 0:
        return []

    estimator = _unpack_estimator(model)
    importances: Optional[np.ndarray] = None
    source = "unknown"

    if hasattr(estimator, "coef_"):
        coef = getattr(estimator, "coef_")
        if isinstance(coef, np.ndarray) and coef.ndim == 2 and coef.shape[0] >= 1:
            importances = np.abs(coef[0])
            source = "abs_coef"
    elif hasattr(estimator, "feature_importances_"):
        imp = getattr(estimator, "feature_importances_")
        if isinstance(imp, np.ndarray):
            importances = imp
            source = "feature_importances_"

    if importances is None:
        sample_n = min(max(1, permutation_sample_size), X_val.shape[0])
        idx = np.random.default_rng(seed).choice(X_val.shape[0], size=sample_n, replace=False)
        result = permutation_importance(
            model,
            X_val[idx],
            y_val[idx],
            scoring="neg_log_loss",
            n_repeats=max(1, permutation_repeats),
            random_state=seed,
            n_jobs=1,
        )
        importances = np.asarray(result.importances_mean, dtype=np.float64)
        source = "permutation_neg_log_loss"

    top_indices = np.argsort(importances)[::-1][: max(1, max_features)]
    output: List[Dict[str, Any]] = []
    for idx in top_indices:
        output.append(
            {
                "feature_index": int(idx),
                "feature_name": f"feature_{int(idx)}",
                "importance": float(importances[idx]),
                "source": source,
            }
        )
    return output


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _split_summary(records: Sequence[RowRecord]) -> Dict[str, Any]:
    games_by_split: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    row_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    positive_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    game_to_split: Dict[str, str] = {}
    leakage_games: List[str] = []
    for r in records:
        games_by_split[r.split].add(r.game_id)
        row_counts[r.split] += 1
        positive_counts[r.split] += int(r.y_win)
        existing = game_to_split.get(r.game_id)
        if existing is None:
            game_to_split[r.game_id] = r.split
        elif existing != r.split:
            leakage_games.append(r.game_id)

    summary: Dict[str, Any] = {
        "rows": row_counts,
        "games": {k: len(v) for k, v in games_by_split.items()},
        "positive_rate": {
            k: (positive_counts[k] / row_counts[k] if row_counts[k] > 0 else None)
            for k in row_counts
        },
        "game_split_leakage_count": len(set(leakage_games)),
    }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V1 current-player eventual win model")
    parser.add_argument("--input", required=True, help="Input JSONL (raw states or vectorized rows)")
    parser.add_argument("--output-dir", required=True, help="Directory to write model artifacts")
    parser.add_argument("--input-format", choices=["auto", "states", "rows", "selfplay_raw"], default="auto")
    parser.add_argument("--models", default="hgb,logreg", help="Comma-separated model list: hgb,logreg")
    parser.add_argument("--winner-only", action="store_true", help="Keep only rows from winner-ended games")
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional split manifest JSON used when building from raw states",
    )
    parser.add_argument("--split-seed", default="win_model_v1", help="Seed for deterministic split assignment")
    parser.add_argument("--split-ratios", default="80,10,10", help="train,val,test ratios")
    parser.add_argument(
        "--max-rows-per-game",
        type=int,
        default=128,
        help="Reservoir cap per game to control dataset size (default: 128)",
    )
    parser.add_argument("--dedupe-position-id", action="store_true", help="Drop duplicate position_id per game")
    parser.add_argument("--disable-drop-consecutive-duplicates", action="store_true")
    parser.add_argument("--balance-classes", action="store_true", help="Use balanced sample weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-importance", type=int, default=64)
    parser.add_argument(
        "--stage-weights",
        default="0.5,0.3,0.2",
        help="Weights for stage-balanced log loss as early,mid,late",
    )
    parser.add_argument(
        "--skip-feature-importance",
        action="store_true",
        help="Skip feature importance computation to reduce runtime",
    )
    parser.add_argument(
        "--perm-importance-sample-size",
        type=int,
        default=2000,
        help="Validation rows to sample for permutation importance (used when needed)",
    )
    parser.add_argument(
        "--perm-importance-repeats",
        type=int,
        default=3,
        help="Permutation repeats for feature importance (used when needed)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ratios = _parse_split_ratios(args.split_ratios)
    stage_weights = _parse_stage_weights(args.stage_weights)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in selected_models:
        if name not in {"hgb", "logreg"}:
            raise ValueError(f"Unsupported model in --models: {name}")

    input_format = args.input_format
    if input_format == "auto":
        input_format = _detect_input_format(input_path)

    split_manifest = load_split_manifest(Path(args.split_manifest)) if args.split_manifest else None

    if input_format == "states":
        payloads = _winner_only_payloads(
            iter_state_payloads(input_path),
            winner_only=bool(args.winner_only),
        )
        row_iter = iter_trajectory_rows(
            payloads=payloads,
            split_manifest=split_manifest,
            split_seed=args.split_seed,
            split_ratios=split_ratios,
            drop_unlabeled=True,
            drop_consecutive_duplicate_positions=not args.disable_drop_consecutive_duplicates,
            skip_malformed=True,
        )
    elif input_format == "rows":
        row_iter = _iter_jsonl(input_path)
    else:
        row_iter = _iter_selfplay_raw_rows(input_path=input_path, winner_only=bool(args.winner_only))

    records = _sample_records(
        row_iter=row_iter,
        split_seed=args.split_seed,
        split_ratios=split_ratios,
        max_rows_per_game=args.max_rows_per_game,
        seed=args.seed,
        dedupe_position_id=bool(args.dedupe_position_id),
    )

    if not records:
        raise RuntimeError("No usable training records after filtering/sampling")

    split_info = _split_summary(records)
    if split_info.get("game_split_leakage_count", 0) > 0:
        raise RuntimeError("Game-level split leakage detected; fix split assignment before training")

    X_train, y_train, train_records = _slice_by_split(records, "train")
    X_val, y_val, val_records = _slice_by_split(records, "val")
    X_test, y_test, test_records = _slice_by_split(records, "test")

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("One or more splits are empty after filtering/sampling")

    train_sample_weight: Optional[np.ndarray] = None
    if args.balance_classes:
        train_sample_weight = _compute_balanced_sample_weight(y_train)

    model_metrics: Dict[str, Any] = {}
    model_paths: Dict[str, str] = {}
    feature_importance_paths: Dict[str, str] = {}
    max_idx_by_game = _game_max_row_index(records)

    for model_name in selected_models:
        model = _build_model(model_name, seed=args.seed)
        fit_kwargs: Dict[str, Any] = {}
        if train_sample_weight is not None:
            if isinstance(model, Pipeline):
                fit_kwargs["logreg__sample_weight"] = train_sample_weight
            else:
                fit_kwargs["sample_weight"] = train_sample_weight
        model.fit(X_train, y_train, **fit_kwargs)

        p_train = _extract_probabilities(model, X_train)
        p_val = _extract_probabilities(model, X_val)
        p_test = _extract_probabilities(model, X_test)

        model_metrics[model_name] = {
            "train": _metrics_with_stage(y_train, p_train, train_records, max_idx_by_game, stage_weights),
            "val": _metrics_with_stage(y_val, p_val, val_records, max_idx_by_game, stage_weights),
            "test": _metrics_with_stage(y_test, p_test, test_records, max_idx_by_game, stage_weights),
        }

        model_path = output_dir / f"model_{model_name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        model_paths[model_name] = str(model_path)

        if args.skip_feature_importance:
            feature_importance = []
        else:
            feature_importance = _top_feature_importance(
                model=model,
                X_val=X_val,
                y_val=y_val,
                max_features=args.top_k_importance,
                seed=args.seed,
                permutation_sample_size=args.perm_importance_sample_size,
                permutation_repeats=args.perm_importance_repeats,
            )
        fi_path = output_dir / f"feature_importance_{model_name}.json"
        _save_json(fi_path, {"model": model_name, "top_features": feature_importance})
        feature_importance_paths[model_name] = str(fi_path)

    prior = DummyClassifier(strategy="prior")
    prior.fit(X_train, y_train)
    p_train_prior = _extract_probabilities(prior, X_train)
    p_val_prior = _extract_probabilities(prior, X_val)
    p_test_prior = _extract_probabilities(prior, X_test)
    model_metrics["baseline_prior"] = {
        "train": _metrics_with_stage(y_train, p_train_prior, train_records, max_idx_by_game, stage_weights),
        "val": _metrics_with_stage(y_val, p_val_prior, val_records, max_idx_by_game, stage_weights),
        "test": _metrics_with_stage(y_test, p_test_prior, test_records, max_idx_by_game, stage_weights),
    }

    best_model = min(
        selected_models,
        key=lambda name: float(model_metrics[name]["val"].get("stage_balanced_log_loss") or model_metrics[name]["val"]["log_loss"]),
    )

    config_payload = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "input_format": input_format,
        "models": selected_models,
        "winner_only": bool(args.winner_only),
        "split_seed": args.split_seed,
        "split_ratios": split_ratios,
        "max_rows_per_game": args.max_rows_per_game,
        "dedupe_position_id": bool(args.dedupe_position_id),
        "drop_consecutive_duplicates": not args.disable_drop_consecutive_duplicates,
        "balance_classes": bool(args.balance_classes),
        "seed": args.seed,
        "stage_weights": stage_weights,
        "skip_feature_importance": bool(args.skip_feature_importance),
        "perm_importance_sample_size": args.perm_importance_sample_size,
        "perm_importance_repeats": args.perm_importance_repeats,
    }

    _save_json(output_dir / "config.json", config_payload)
    _save_json(
        output_dir / "split_summary.json",
        {
            **split_info,
            "num_records": len(records),
            "vector_length": int(records[0].x.shape[0]),
        },
    )
    _save_json(
        output_dir / "metrics.json",
        {
            "best_model": best_model,
            "metrics": model_metrics,
            "model_paths": model_paths,
            "feature_importance_paths": feature_importance_paths,
        },
    )

    print(f"Records used: {len(records)}")
    print(f"Train/Val/Test rows: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    print(f"Best model (val log_loss): {best_model}")
    print(f"Artifacts written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
