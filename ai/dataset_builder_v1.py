from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from ai.vectorizer_v1 import StateVectorizerV1, VectorizedState, make_training_row

VALID_SPLITS = {"train", "val", "test"}


def _nested_get(payload: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_non_null(payload: Dict[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _nested_get(payload, path)
        if value is not None:
            return value
    return None


def _extract_game_id(payload: Dict[str, Any], override_game_id: Optional[str]) -> Optional[str]:
    if override_game_id is not None:
        return override_game_id
    value = _first_non_null(
        payload,
        [
            ("game_id",),
            ("metadata", "game_id"),
            ("trajectory", "game_id"),
            ("context", "game_id"),
        ],
    )
    return str(value) if value is not None else None


def _extract_turn_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    observed = payload.get("observed_state")
    if isinstance(observed, dict):
        turn_meta = observed.get("turn_metadata")
        if isinstance(turn_meta, dict):
            return turn_meta
    return {}


def _extract_row_index_in_game(payload: Dict[str, Any], fallback_index: int) -> int:
    turn_meta = _extract_turn_metadata(payload)
    turn_index = turn_meta.get("turn_index")
    if isinstance(turn_index, int):
        return turn_index

    explicit = _first_non_null(
        payload,
        [
            ("row_index_in_game",),
            ("metadata", "row_index_in_game"),
            ("trajectory", "row_index_in_game"),
        ],
    )
    if isinstance(explicit, int):
        return explicit
    return fallback_index


def _extract_current_player_identity(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    turn_meta = _extract_turn_metadata(payload)
    current_player_id = turn_meta.get("current_player_id")
    current_player_name = turn_meta.get("current_player_name")
    return (
        str(current_player_id) if current_player_id is not None else None,
        str(current_player_name) if current_player_name is not None else None,
    )


def _extract_winner_identity(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    winner_id = _first_non_null(
        payload,
        [
            ("winner_player_id",),
            ("winner_id",),
            ("outcome", "winner_player_id"),
            ("outcome", "winner_id"),
            ("result", "winner_player_id"),
            ("result", "winner_id"),
            ("trajectory", "winner_player_id"),
            ("trajectory", "winner_id"),
        ],
    )
    winner_name = _first_non_null(
        payload,
        [
            ("winner_player_name",),
            ("winner_name",),
            ("outcome", "winner_player_name"),
            ("outcome", "winner_name"),
            ("result", "winner_player_name"),
            ("result", "winner_name"),
            ("trajectory", "winner_player_name"),
            ("trajectory", "winner_name"),
        ],
    )
    return (
        str(winner_id) if winner_id is not None else None,
        str(winner_name) if winner_name is not None else None,
    )


def _derive_y_win(
    payload: Dict[str, Any],
    global_label_win: Optional[int],
    allow_global_label_fallback: bool,
) -> Optional[int]:
    existing_label = _first_non_null(
        payload,
        [
            ("y_win",),
            ("label", "y_win"),
            ("outcome", "y_win"),
            ("result", "y_win"),
        ],
    )
    if isinstance(existing_label, bool):
        return int(existing_label)
    if isinstance(existing_label, int) and existing_label in (0, 1):
        return existing_label

    current_player_id, current_player_name = _extract_current_player_identity(payload)
    winner_player_id, winner_player_name = _extract_winner_identity(payload)

    if current_player_id is not None and winner_player_id is not None:
        return 1 if current_player_id == winner_player_id else 0
    if current_player_name is not None and winner_player_name is not None:
        return 1 if current_player_name == winner_player_name else 0

    if allow_global_label_fallback and global_label_win in (0, 1):
        return global_label_win
    return None


def _deterministic_split_for_game(game_id: str, split_seed: str, ratios: Tuple[int, int, int]) -> str:
    train_ratio, val_ratio, _ = ratios
    digest = hashlib.md5(f"{split_seed}:{game_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _parse_split_ratios(raw: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("split ratios must be three comma-separated integers (train,val,test)")
    try:
        ratios = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError("split ratios must be integers") from exc
    if sum(ratios) != 100:
        raise ValueError("split ratios must sum to 100")
    if any(r < 0 for r in ratios):
        raise ValueError("split ratios must be non-negative")
    return ratios  # type: ignore[return-value]


def _normalize_phase_filter(value: Optional[str]) -> Optional[Set[str]]:
    if value is None:
        return None
    phases = {item.strip().lower() for item in value.split(",") if item.strip()}
    return phases if phases else None


def _validate_row(row: Dict[str, Any], expected_schema_hash: Optional[str], expected_vector_length: Optional[int]) -> Tuple[bool, Optional[str], Optional[int]]:
    x = row.get("x")
    vector_length = row.get("vector_length")
    schema_hash = row.get("feature_schema_hash")

    if not isinstance(x, list):
        return False, expected_schema_hash, expected_vector_length
    if not isinstance(vector_length, int) or vector_length != len(x):
        return False, expected_schema_hash, expected_vector_length
    if not isinstance(schema_hash, str) or not schema_hash:
        return False, expected_schema_hash, expected_vector_length

    next_expected_hash = expected_schema_hash if expected_schema_hash is not None else schema_hash
    next_expected_length = expected_vector_length if expected_vector_length is not None else vector_length

    if schema_hash != next_expected_hash:
        return False, expected_schema_hash, expected_vector_length
    if vector_length != next_expected_length:
        return False, expected_schema_hash, expected_vector_length

    return True, next_expected_hash, next_expected_length


def load_split_manifest(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)

    if not isinstance(content, dict):
        raise ValueError("Split manifest must be a JSON object")

    mapping: Dict[str, str] = {}
    for key, value in content.items():
        game_key = str(key)
        split_value = str(value).strip().lower()
        if split_value not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{value}' for game_id '{game_key}'")
        mapping[game_key] = split_value
    return mapping


def iter_state_payloads(input_path: Path) -> Iterator[Dict[str, Any]]:
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if isinstance(payload, dict):
                    yield payload
        return

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)

        if isinstance(content, dict):
            yield content
            return

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    yield item
            return

        raise ValueError("JSON input must be an object or a list of objects")

    raise ValueError("Unsupported input format. Use .json or .jsonl")


def iter_rows(
    payloads: Iterable[Dict[str, Any]],
    game_id: Optional[str] = None,
    split: Optional[str] = None,
    label_win: Optional[int] = None,
    allow_global_label_fallback: bool = False,
    adjacency_features_enabled: bool = True,
    split_by_game: bool = True,
    split_seed: str = "dataset_builder_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
    split_manifest: Optional[Dict[str, str]] = None,
    dedupe_by_position_id: bool = False,
    phase_filter: Optional[Set[str]] = None,
    drop_unlabeled: bool = True,
    skip_malformed: bool = True,
) -> Iterator[Dict[str, Any]]:
    vectorizer = StateVectorizerV1(adjacency_features_enabled=adjacency_features_enabled)
    seen_positions: Set[str] = set()
    expected_schema_hash: Optional[str] = None
    expected_vector_length: Optional[int] = None

    for payload_index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            if skip_malformed:
                continue
            raise ValueError("Payload must be a dictionary")

        turn_meta = _extract_turn_metadata(payload)
        if phase_filter is not None:
            current_phase = str(turn_meta.get("current_phase") or "").strip().lower()
            if current_phase not in phase_filter:
                continue

        try:
            vectorized: VectorizedState = vectorizer.vectorize(payload)
        except Exception:
            if skip_malformed:
                continue
            raise

        resolved_game_id = _extract_game_id(payload, game_id)
        resolved_label = _derive_y_win(
            payload,
            global_label_win=label_win,
            allow_global_label_fallback=allow_global_label_fallback,
        )
        if resolved_label is None and drop_unlabeled:
            continue

        row = make_training_row(
            state_payload=payload,
            vectorized=vectorized,
            label_win=resolved_label,
            game_id=resolved_game_id,
            split=None,
        )

        row["adjacency_features_enabled"] = adjacency_features_enabled
        row["vector_length"] = len(vectorized.x)
        if not row.get("feature_schema_hash"):
            row["feature_schema_hash"] = vectorized.metadata.get("feature_schema_hash")
        row["row_index_in_game"] = _extract_row_index_in_game(payload, fallback_index=payload_index)

        if split is not None:
            row["split"] = split
        elif split_manifest is not None and resolved_game_id is not None and resolved_game_id in split_manifest:
            row["split"] = split_manifest[resolved_game_id]
        elif split_by_game and resolved_game_id is not None:
            row["split"] = _deterministic_split_for_game(resolved_game_id, split_seed, split_ratios)

        position_id = row.get("position_id")
        if dedupe_by_position_id and isinstance(position_id, str):
            if position_id in seen_positions:
                continue
            seen_positions.add(position_id)

        is_valid, expected_schema_hash, expected_vector_length = _validate_row(
            row,
            expected_schema_hash=expected_schema_hash,
            expected_vector_length=expected_vector_length,
        )
        if not is_valid:
            if skip_malformed:
                continue
            raise ValueError("Row validation failed (schema hash/vector length consistency)")

        yield row


def build_rows(
    payloads: Iterable[Dict[str, Any]],
    game_id: Optional[str] = None,
    split: Optional[str] = None,
    label_win: Optional[int] = None,
    allow_global_label_fallback: bool = False,
    adjacency_features_enabled: bool = True,
    split_by_game: bool = True,
    split_seed: str = "dataset_builder_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
    split_manifest: Optional[Dict[str, str]] = None,
    dedupe_by_position_id: bool = False,
    phase_filter: Optional[Set[str]] = None,
    drop_unlabeled: bool = True,
    skip_malformed: bool = True,
) -> List[Dict[str, Any]]:
    return list(
        iter_rows(
            payloads=payloads,
            game_id=game_id,
            split=split,
            label_win=label_win,
            allow_global_label_fallback=allow_global_label_fallback,
            adjacency_features_enabled=adjacency_features_enabled,
            split_by_game=split_by_game,
            split_seed=split_seed,
            split_ratios=split_ratios,
            split_manifest=split_manifest,
            dedupe_by_position_id=dedupe_by_position_id,
            phase_filter=phase_filter,
            drop_unlabeled=drop_unlabeled,
            skip_malformed=skip_malformed,
        )
    )


def write_rows_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build vectorized training rows from extracted Catan states")
    parser.add_argument("--input", required=True, help="Path to input state file (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="Path to output JSONL rows")
    parser.add_argument("--game-id", default=None, help="Optional game id override for all rows")
    parser.add_argument("--split", default=None, help="Manual split override (train/val/test) for all rows")
    parser.add_argument(
        "--label-win",
        type=int,
        choices=[0, 1],
        default=None,
        help="Global fallback label only; prefer outcome-derived row labels",
    )
    parser.add_argument(
        "--allow-global-label-fallback",
        action="store_true",
        help="Allow --label-win fallback when outcome metadata is missing",
    )
    parser.add_argument(
        "--disable-adjacency-features",
        action="store_true",
        help="Disable adjacency-derived engineered production features",
    )
    parser.add_argument(
        "--disable-split-by-game",
        action="store_true",
        help="Disable deterministic split assignment by game_id",
    )
    parser.add_argument(
        "--split-seed",
        default="dataset_builder_v1",
        help="Seed namespace used for deterministic split-by-game assignment",
    )
    parser.add_argument(
        "--split-ratios",
        default="80,10,10",
        help="Split ratios as train,val,test percentages summing to 100",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional JSON file mapping game_id -> split (train|val|test)",
    )
    parser.add_argument(
        "--dedupe-by-position-id",
        action="store_true",
        help="Skip duplicate rows with the same position_id",
    )
    parser.add_argument(
        "--phase-filter",
        default=None,
        help="Optional comma-separated phase allow-list (e.g. main,build)",
    )
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep rows that cannot derive y_win (default drops them)",
    )
    parser.add_argument(
        "--fail-on-malformed",
        action="store_true",
        help="Fail fast on malformed payloads/rows (default skips malformed)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    split_ratios = _parse_split_ratios(args.split_ratios)
    phase_filter = _normalize_phase_filter(args.phase_filter)
    split_manifest = load_split_manifest(Path(args.split_manifest)) if args.split_manifest else None

    rows = iter_rows(
        payloads=iter_state_payloads(input_path),
        game_id=args.game_id,
        split=args.split,
        label_win=args.label_win,
        allow_global_label_fallback=args.allow_global_label_fallback,
        adjacency_features_enabled=not args.disable_adjacency_features,
        split_by_game=not args.disable_split_by_game,
        split_seed=args.split_seed,
        split_ratios=split_ratios,
        split_manifest=split_manifest,
        dedupe_by_position_id=args.dedupe_by_position_id,
        phase_filter=phase_filter,
        drop_unlabeled=not args.keep_unlabeled,
        skip_malformed=not args.fail_on_malformed,
    )
    count = write_rows_jsonl(rows, output_path)
    print(f"Wrote {count} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
