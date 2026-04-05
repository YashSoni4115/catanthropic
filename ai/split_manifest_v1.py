from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ai.dataset_builder_v1 import (
    _deterministic_split_for_game,
    _parse_split_ratios,
    _extract_game_id,
    iter_state_payloads,
)


def collect_game_ids(payloads: Iterable[dict], override_game_id: Optional[str] = None) -> List[str]:
    game_ids: Set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        game_id = _extract_game_id(payload, override_game_id)
        if game_id is not None:
            game_ids.add(game_id)
    return sorted(game_ids)


def build_split_manifest(
    game_ids: Iterable[str],
    split_seed: str = "split_manifest_v1",
    split_ratios: Tuple[int, int, int] = (80, 10, 10),
) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for game_id in sorted(set(game_ids)):
        manifest[game_id] = _deterministic_split_for_game(game_id, split_seed, split_ratios)
    return manifest


def write_split_manifest(manifest: Dict[str, str], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return len(manifest)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic game-level split manifest")
    parser.add_argument("--input", required=True, help="Path to input state file (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="Path to output split manifest JSON")
    parser.add_argument("--game-id", default=None, help="Optional override game id for all payloads")
    parser.add_argument("--split-seed", default="split_manifest_v1", help="Seed used for deterministic split assignment")
    parser.add_argument("--split-ratios", default="80,10,10", help="Split ratios as train,val,test percentages")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    split_ratios = _parse_split_ratios(args.split_ratios)

    game_ids = collect_game_ids(iter_state_payloads(input_path), override_game_id=args.game_id)
    manifest = build_split_manifest(game_ids, split_seed=args.split_seed, split_ratios=split_ratios)
    count = write_split_manifest(manifest, output_path)

    print(f"Wrote split manifest for {count} games to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
