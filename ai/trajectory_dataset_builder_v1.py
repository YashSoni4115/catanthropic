from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ai.dataset_builder_v1 import iter_rows, iter_state_payloads, load_split_manifest, write_rows_jsonl


def iter_trajectory_rows(
    payloads: Iterable[Dict[str, Any]],
    game_id: Optional[str] = None,
    split: Optional[str] = None,
    label_win: Optional[int] = None,
    allow_global_label_fallback: bool = False,
    adjacency_features_enabled: bool = True,
    split_by_game: bool = True,
    split_seed: str = "trajectory_dataset_builder_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
    split_manifest: Optional[Dict[str, str]] = None,
    dedupe_by_position_id: bool = False,
    phase_filter: Optional[Set[str]] = None,
    drop_unlabeled: bool = True,
    skip_malformed: bool = True,
    drop_consecutive_duplicate_positions: bool = True,
) -> Iterator[Dict[str, Any]]:
    last_position_by_game: Dict[str, Optional[str]] = {}
    trajectory_index_by_game: Dict[str, int] = defaultdict(int)

    for row in iter_rows(
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
    ):
        resolved_game_id = str(row.get("game_id") or "__unknown_game__")
        position_id = row.get("position_id")

        if drop_consecutive_duplicate_positions:
            last_position = last_position_by_game.get(resolved_game_id)
            if isinstance(position_id, str) and position_id == last_position:
                continue
            last_position_by_game[resolved_game_id] = position_id if isinstance(position_id, str) else None

        row["trajectory_row_index_in_game"] = trajectory_index_by_game[resolved_game_id]
        trajectory_index_by_game[resolved_game_id] += 1
        row["trajectory_dataset_version"] = "trajectory_v1.0.0"
        yield row


def build_trajectory_rows(
    payloads: Iterable[Dict[str, Any]],
    game_id: Optional[str] = None,
    split: Optional[str] = None,
    label_win: Optional[int] = None,
    allow_global_label_fallback: bool = False,
    adjacency_features_enabled: bool = True,
    split_by_game: bool = True,
    split_seed: str = "trajectory_dataset_builder_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
    split_manifest: Optional[Dict[str, str]] = None,
    dedupe_by_position_id: bool = False,
    phase_filter: Optional[Set[str]] = None,
    drop_unlabeled: bool = True,
    skip_malformed: bool = True,
    drop_consecutive_duplicate_positions: bool = True,
) -> List[Dict[str, Any]]:
    return list(
        iter_trajectory_rows(
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
            drop_consecutive_duplicate_positions=drop_consecutive_duplicate_positions,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build trajectory-aware training rows from extracted Catan states")
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
    parser.add_argument("--split-seed", default="trajectory_dataset_builder_v1", help="Seed for deterministic split-by-game assignment")
    parser.add_argument("--split-ratios", default="80,10,10", help="Split ratios as train,val,test percentages")
    parser.add_argument("--split-manifest", default=None, help="Optional JSON file mapping game_id -> split (train|val|test)")
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
    parser.add_argument(
        "--keep-consecutive-duplicate-positions",
        action="store_true",
        help="Keep consecutive duplicate `position_id` rows within each game",
    )
    return parser


def _parse_split_ratios(raw: str) -> Tuple[int, int, int]:
    from ai.dataset_builder_v1 import _parse_split_ratios as parse_split_ratios

    return parse_split_ratios(raw)


def _normalize_phase_filter(value: Optional[str]) -> Optional[Set[str]]:
    if value is None:
        return None
    phases = {item.strip().lower() for item in value.split(",") if item.strip()}
    return phases if phases else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    split_ratios = _parse_split_ratios(args.split_ratios)
    phase_filter = _normalize_phase_filter(args.phase_filter)
    split_manifest = load_split_manifest(Path(args.split_manifest)) if args.split_manifest else None

    rows = iter_trajectory_rows(
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
        drop_consecutive_duplicate_positions=not args.keep_consecutive_duplicate_positions,
    )
    count = write_rows_jsonl(rows, output_path)
    print(f"Wrote {count} trajectory rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
