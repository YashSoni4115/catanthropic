"""Train robber_hotspot_v1 — 19-way multiclass classifier over next-robber tile.

Label extraction
----------------
Consecutive-row scan per game over the extracted self-play JSONL.  When the
top-level ``observed_state.robber`` (x, y) changes between two rows, we treat
the *previous* row's ``observed_state`` as the input and the *current* row's
robber (x, y) as the label.  We skip:

  * the initial desert placement (previous robber == None)
  * no-op moves (new position equals previous position)
  * placements landing on a tile not present in the previous row's tile list
    (schema drift)

Split by game (80/10/10) via deterministic md5-hash of ``game_id`` to avoid
row-level leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai.economy.robber_hotspot_features import (
    INPUT_DIM,
    NUM_TILES,
    build_tile_index,
    extract_input_vector,
    feature_schema,
)


VALID_SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = (80, 10, 10)


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _split_for_game(game_id: str, seed: str, ratios: Tuple[int, int, int]) -> str:
    train_ratio, val_ratio, _ = ratios
    digest = hashlib.md5(f"{seed}:{game_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def iter_robber_placements(input_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield one record per robber-placement event.

    Each record has keys:
      game_id, x (label vec of length INPUT_DIM), y_tile_idx (0..18),
      turn_index, active_player_id
    """
    prev_by_game: Dict[str, Dict[str, Any]] = {}
    history_by_game: Dict[str, Dict[str, Any]] = {}

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            observed = row.get("observed_state") or {}
            if not observed:
                continue
            game_id = row.get("game_id") or observed.get("turn_metadata", {}).get("game_id")
            if not game_id:
                continue
            game_id = str(game_id)
            robber = observed.get("robber")
            if not isinstance(robber, dict):
                prev_by_game.pop(game_id, None)
                continue
            cur_key = (int(robber.get("x", -1)), int(robber.get("y", -1)))
            prev_entry = prev_by_game.get(game_id)

            # Track per-game history
            hist = history_by_game.setdefault(
                game_id,
                {"times_robbed": {}, "targeted_by_active": {}},
            )

            if prev_entry is not None:
                prev_key = prev_entry["key"]
                prev_obs = prev_entry["observed"]
                if cur_key != prev_key and cur_key != (-1, -1):
                    # Confirm target tile exists in previous state's tile list
                    tile_index_map = build_tile_index(prev_obs)
                    if cur_key in tile_index_map:
                        y_idx = tile_index_map[cur_key]
                        # Build feature vector using the STATE BEFORE the move
                        active_pid_prev = (
                            (prev_obs.get("turn_metadata") or {}).get("current_player_id")
                            or (prev_obs.get("turn_metadata") or {}).get("current_player_name")
                        )
                        active_targeted = hist["targeted_by_active"].get(
                            active_pid_prev, set()
                        ) if active_pid_prev else set()
                        try:
                            x_vec, _ = extract_input_vector(
                                prev_obs,
                                times_robbed_by_tile=dict(hist["times_robbed"]),
                                active_targeted_tiles=set(active_targeted),
                            )
                        except Exception:
                            x_vec = None
                        if x_vec is not None:
                            yield {
                                "game_id": game_id,
                                "x": x_vec,
                                "y_tile_idx": int(y_idx),
                                "turn_index": int(
                                    (prev_obs.get("turn_metadata") or {}).get("turn_index") or 0
                                ),
                                "active_player_id": active_pid_prev,
                            }
                        # Update history (counted after emitting the sample so
                        # the sample reflects state *before* this placement).
                        hist["times_robbed"][cur_key] = hist["times_robbed"].get(cur_key, 0) + 1
                        if active_pid_prev is not None:
                            hist["targeted_by_active"].setdefault(active_pid_prev, set()).add(
                                cur_key
                            )

            prev_by_game[game_id] = {"key": cur_key, "observed": observed}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=8,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=seed,
    )


def _build_logreg(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=200,
                    C=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-proba, axis=1)[:, :k]
    hit = 0
    class_lookup = {int(c): idx for idx, c in enumerate(classes)}
    for i, yt in enumerate(y_true):
        yt_idx = class_lookup.get(int(yt))
        if yt_idx is None:
            continue
        if yt_idx in order[i]:
            hit += 1
    return hit / y_true.size


def _uniform_baseline_metrics(y_true: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"n": 0}
    p_uniform = 1.0 / NUM_TILES
    ll = -np.log(p_uniform) * np.ones(y_true.size)
    return {
        "n": int(y_true.size),
        "top1_acc": float((y_true == 0).mean() * 0.0 + 1.0 / NUM_TILES),  # ≈1/19 in expectation
        "top3_acc": 3.0 / NUM_TILES,
        "log_loss": float(ll.mean()),
    }


def _model_metrics(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    all_labels: List[int],
) -> Dict[str, float]:
    if X.size == 0:
        return {"n": 0}
    proba = model.predict_proba(X)
    y_pred = model.classes_[np.argmax(proba, axis=1)]
    top1 = float(accuracy_score(y, y_pred))
    top3 = _top_k_accuracy(y, proba, model.classes_, k=3)
    try:
        ll = float(log_loss(y, proba, labels=list(model.classes_)))
    except ValueError:
        ll = float("nan")
    return {
        "n": int(y.size),
        "top1_acc": top1,
        "top3_acc": top3,
        "log_loss": ll,
    }


def train(
    input_path: Path,
    output_dir: Path,
    split_ratios: Tuple[int, int, int],
    split_seed: str,
    seed: int,
    max_records: Optional[int],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting robber-placement events from {input_path} ...")
    records: List[Dict[str, Any]] = []
    for rec in iter_robber_placements(input_path):
        records.append(rec)
        if max_records is not None and len(records) >= max_records:
            break
    print(f"  → {len(records):,} robber-placement events extracted")
    if not records:
        raise RuntimeError("No robber-placement events found in input file")

    # Assign splits by game_id
    for rec in records:
        rec["split"] = _split_for_game(rec["game_id"], split_seed, split_ratios)

    X_by_split: Dict[str, List[List[float]]] = {s: [] for s in VALID_SPLITS}
    y_by_split: Dict[str, List[int]] = {s: [] for s in VALID_SPLITS}
    games_by_split: Dict[str, set] = {s: set() for s in VALID_SPLITS}
    for rec in records:
        X_by_split[rec["split"]].append(rec["x"])
        y_by_split[rec["split"]].append(rec["y_tile_idx"])
        games_by_split[rec["split"]].add(rec["game_id"])

    X_train = np.asarray(X_by_split["train"], dtype=np.float32)
    y_train = np.asarray(y_by_split["train"], dtype=np.int32)
    X_val = np.asarray(X_by_split["val"], dtype=np.float32)
    y_val = np.asarray(y_by_split["val"], dtype=np.int32)
    X_test = np.asarray(X_by_split["test"], dtype=np.float32)
    y_test = np.asarray(y_by_split["test"], dtype=np.int32)

    print(
        f"  splits — train: {X_train.shape[0]:,} rows / {len(games_by_split['train'])} games; "
        f"val: {X_val.shape[0]:,} / {len(games_by_split['val'])}; "
        f"test: {X_test.shape[0]:,} / {len(games_by_split['test'])}"
    )

    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        raise RuntimeError("Empty split after assignment — need more games or wider ratios")

    all_labels = sorted(set(y_train.tolist()) | set(y_val.tolist()) | set(y_test.tolist()))

    print("Training HistGradientBoostingClassifier ...")
    hgb = _build_hgb(seed=seed)
    hgb.fit(X_train, y_train)
    print("Training LogisticRegression (multinomial) ...")
    lr = _build_logreg(seed=seed)
    lr.fit(X_train, y_train)

    metrics: Dict[str, Any] = {"hgb": {}, "logreg": {}, "uniform_baseline": {}}
    for split_name, X_s, y_s in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        metrics["hgb"][split_name] = _model_metrics(hgb, X_s, y_s, all_labels)
        metrics["logreg"][split_name] = _model_metrics(lr, X_s, y_s, all_labels)
        metrics["uniform_baseline"][split_name] = _uniform_baseline_metrics(y_s)

    # Best model = highest val top-1 accuracy
    best_model_name = max(
        ("hgb", "logreg"),
        key=lambda n: metrics[n]["val"].get("top1_acc", 0.0),
    )

    # ── Save artifacts ────────────────────────────────────────────────
    with (output_dir / "model_hgb.pkl").open("wb") as f:
        pickle.dump(hgb, f)
    with (output_dir / "model_logreg.pkl").open("wb") as f:
        pickle.dump(lr, f)

    split_summary = {
        "num_records": len(records),
        "rows": {s: len(y_by_split[s]) for s in VALID_SPLITS},
        "games": {s: len(games_by_split[s]) for s in VALID_SPLITS},
        "positive_rate": {  # per project convention, we log top-1 accuracy here
            s: metrics["hgb"][s].get("top1_acc", 0.0) for s in VALID_SPLITS
        },
        "vector_length": INPUT_DIM,
        "num_classes": NUM_TILES,
        "game_split_leakage_count": 0,  # split-by-game guarantees zero
    }
    with (output_dir / "split_summary.json").open("w") as f:
        json.dump(split_summary, f, indent=2, sort_keys=True)

    with (output_dir / "metrics.json").open("w") as f:
        json.dump({"best_model": best_model_name, "metrics": metrics}, f, indent=2, sort_keys=True)

    with (output_dir / "feature_schema.json").open("w") as f:
        json.dump({"names": feature_schema(), "input_dim": INPUT_DIM}, f, indent=2)

    with (output_dir / "config.json").open("w") as f:
        json.dump(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "split_ratios": list(split_ratios),
                "split_seed": split_seed,
                "seed": seed,
                "max_records": max_records,
                "input_dim": INPUT_DIM,
                "num_classes": NUM_TILES,
            },
            f, indent=2, sort_keys=True,
        )

    print(f"\nBest model (val top1): {best_model_name}")
    print(f"Artifacts written to: {output_dir}")
    return {
        "best_model": best_model_name,
        "metrics": metrics,
        "split_summary": split_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_split_ratios(raw: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in raw.split(",")]
    if len(parts) != 3 or sum(parts) != 100:
        raise ValueError("split ratios must be three ints summing to 100")
    return parts[0], parts[1], parts[2]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train robber_hotspot_v1")
    parser.add_argument("--input", required=True, help="Path to extracted self-play JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-ratios", default="80,10,10")
    parser.add_argument("--split-seed", default="robber_hotspot_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional cap on total placement events (for smoke tests)")
    args = parser.parse_args(argv)

    train(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        split_ratios=_parse_split_ratios(args.split_ratios),
        split_seed=args.split_seed,
        seed=args.seed,
        max_records=args.max_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
