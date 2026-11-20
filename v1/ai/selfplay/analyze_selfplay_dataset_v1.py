from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRACKED_RESOURCES = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER")
EXPECTED_TILE_COUNT = 19


@dataclass
class GameStats:
    game_id: str
    row_count: int = 0
    parseable_rows: int = 0
    terminal_row_count: int = 0
    simulation_error_rows: int = 0
    termination_reason_counts: Counter = field(default_factory=Counter)
    terminal_reason_counts: Counter = field(default_factory=Counter)
    winner_name_counts: Counter = field(default_factory=Counter)
    winner_id_counts: Counter = field(default_factory=Counter)
    terminal_winner_empty_rows: int = 0
    missing_observed_rows: int = 0
    missing_encoding_rows: int = 0
    missing_snapshot_rows: int = 0
    leakage_rows: int = 0
    malformed_rows: int = 0
    label_usable_rows: int = 0
    duplicate_position_rows: int = 0
    max_turn_index: int = -1
    winner_distribution_sampled: bool = False


@dataclass
class FirstPassResult:
    total_rows: int
    parse_failures: int
    parse_failure_examples: List[Dict[str, Any]]
    unique_games: int
    game_stats: Dict[str, GameStats]
    phase_counts: Counter
    action_counts: Counter
    multi_action_turn_counts: Counter
    winner_name_counts: Counter
    winner_seat_counts: Counter
    turn_index_counts: Counter
    duplicate_position_id_count: int
    rows_per_game: Dict[str, int]
    required_field_counts: Counter
    snapshot_check_counts: Counter
    leakage_check_counts: Counter
    label_validity_counts: Counter


@dataclass
class GameQuality:
    game_id: str
    canonical_winner_name: str
    canonical_winner_id: str
    winner_ended: bool
    truncated: bool
    simulation_error: bool
    winner_changes_within_game: bool
    no_consistent_winner: bool
    winner_backfill_missing_rows: int
    game_train_usable_base: bool
    terminal_reason: str


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _non_empty_str(value: Any) -> str:
    s = _safe_str(value).strip()
    return s


def _split_actions(action_taken: Any) -> List[str]:
    raw = _safe_str(action_taken)
    if not raw:
        return []
    return [part for part in raw.split("+") if part]


def _extract_current_player(turn_metadata: Dict[str, Any]) -> Tuple[str, str]:
    current_id = _non_empty_str(turn_metadata.get("current_player_id"))
    current_name = _non_empty_str(turn_metadata.get("current_player_name"))
    return current_id, current_name


def _extract_winner_identity(row: Dict[str, Any]) -> Tuple[str, str]:
    winner_id = _non_empty_str(row.get("winner_player_id"))
    winner_name = _non_empty_str(row.get("winner_player_name"))
    return winner_id, winner_name


def _bool_is_terminal(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return False


def _has_usable_label(row: Dict[str, Any]) -> bool:
    turn_meta = row.get("turn_metadata")
    if not isinstance(turn_meta, dict):
        return False
    current_id, current_name = _extract_current_player(turn_meta)
    winner_id, winner_name = _extract_winner_identity(row)
    if current_id and winner_id:
        return True
    if current_name and winner_name:
        return True
    return False


def _negative_resource_detected(row: Dict[str, Any]) -> bool:
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, dict):
        return False

    players = snapshot.get("players")
    if not isinstance(players, list):
        return False

    for player in players:
        if not isinstance(player, dict):
            continue
        resource_cards = player.get("resource_cards")
        if isinstance(resource_cards, dict):
            for resource in TRACKED_RESOURCES:
                value = resource_cards.get(resource)
                if isinstance(value, (int, float)) and value < 0:
                    return True
    return False


def _remaining_piece_impossible(row: Dict[str, Any]) -> bool:
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    players = snapshot.get("players")
    if not isinstance(players, list):
        return False

    for player in players:
        if not isinstance(player, dict):
            continue
        pieces = player.get("remaining_pieces")
        if not isinstance(pieces, dict):
            continue

        roads_remaining = pieces.get("roads_remaining")
        settlements_remaining = pieces.get("settlements_remaining")
        cities_remaining = pieces.get("cities_remaining")
        roads_placed = pieces.get("roads_placed_reported")
        settlements_placed = pieces.get("settlements_placed_reported")
        cities_placed = pieces.get("cities_placed_reported")

        if isinstance(roads_remaining, int) and not (0 <= roads_remaining <= 15):
            return True
        if isinstance(settlements_remaining, int) and not (0 <= settlements_remaining <= 5):
            return True
        if isinstance(cities_remaining, int) and not (0 <= cities_remaining <= 4):
            return True

        if isinstance(roads_placed, int) and not (0 <= roads_placed <= 15):
            return True
        if isinstance(settlements_placed, int) and not (0 <= settlements_placed <= 5):
            return True
        if isinstance(cities_placed, int) and not (0 <= cities_placed <= 4):
            return True

        if isinstance(roads_remaining, int) and isinstance(roads_placed, int):
            if roads_remaining + roads_placed != 15:
                return True
        if isinstance(settlements_remaining, int) and isinstance(settlements_placed, int):
            if settlements_remaining + settlements_placed != 5:
                return True
        if isinstance(cities_remaining, int) and isinstance(cities_placed, int):
            if cities_remaining + cities_placed != 4:
                return True

    return False


def _snapshot_board_checks(row: Dict[str, Any]) -> Dict[str, bool]:
    checks = {
        "invalid_tile_count": False,
        "duplicate_tile_coordinates": False,
        "invalid_robber_count": False,
        "robber_mismatch": False,
        "duplicate_structure_occupancy": False,
        "duplicate_road_occupancy": False,
        "owner_id_not_found": False,
    }

    snapshot = row.get("snapshot")
    if not isinstance(snapshot, dict):
        return checks

    players = snapshot.get("players")
    known_player_ids = set()
    if isinstance(players, list):
        for player in players:
            if isinstance(player, dict):
                pid = _non_empty_str(player.get("player_id"))
                if pid:
                    known_player_ids.add(pid)

    tiles = snapshot.get("tiles")
    if isinstance(tiles, list):
        if len(tiles) != EXPECTED_TILE_COUNT:
            checks["invalid_tile_count"] = True

        tile_coords = []
        robber_tile_count = 0
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            x = tile.get("x")
            y = tile.get("y")
            tile_coords.append((x, y))
            if tile.get("robber") is True:
                robber_tile_count += 1

        if len(tile_coords) != len(set(tile_coords)):
            checks["duplicate_tile_coordinates"] = True

        if robber_tile_count != 1:
            checks["invalid_robber_count"] = True

        robber = snapshot.get("robber")
        if isinstance(robber, dict):
            robber_xy = (robber.get("x"), robber.get("y"))
            flagged = [
                (tile.get("x"), tile.get("y"))
                for tile in tiles
                if isinstance(tile, dict) and tile.get("robber") is True
            ]
            if len(flagged) == 1 and flagged[0] != robber_xy:
                checks["robber_mismatch"] = True
            if len(flagged) == 0:
                checks["robber_mismatch"] = True

    structures = snapshot.get("structures")
    if isinstance(structures, list):
        structure_keys = []
        for structure in structures:
            if not isinstance(structure, dict):
                continue
            structure_keys.append((structure.get("x"), structure.get("y"), structure.get("orientation")))
            owner_id = _non_empty_str(structure.get("owner_id"))
            if owner_id and owner_id not in known_player_ids:
                checks["owner_id_not_found"] = True
        if len(structure_keys) != len(set(structure_keys)):
            checks["duplicate_structure_occupancy"] = True

    roads = snapshot.get("roads")
    if isinstance(roads, list):
        road_keys = []
        for road in roads:
            if not isinstance(road, dict):
                continue
            road_keys.append((road.get("x"), road.get("y"), road.get("orientation")))
            owner_id = _non_empty_str(road.get("owner_id"))
            if owner_id and owner_id not in known_player_ids:
                checks["owner_id_not_found"] = True
        if len(road_keys) != len(set(road_keys)):
            checks["duplicate_road_occupancy"] = True

    return checks


def _observed_state_leakage_checks(row: Dict[str, Any]) -> Dict[str, bool]:
    leaks = {
        "opponent_resource_type_leak": False,
        "opponent_dev_grouped_leak": False,
        "opponent_dev_subtype_leak": False,
    }

    observed_state = row.get("observed_state")
    if not isinstance(observed_state, dict):
        return leaks

    turn_meta = observed_state.get("turn_metadata")
    if not isinstance(turn_meta, dict):
        return leaks

    viewer_id = _non_empty_str(turn_meta.get("current_player_id"))
    viewer_name = _non_empty_str(turn_meta.get("current_player_name"))

    players_block = observed_state.get("players")
    if not isinstance(players_block, dict):
        return leaks

    all_players = players_block.get("all_players")
    if not isinstance(all_players, list):
        return leaks

    for player in all_players:
        if not isinstance(player, dict):
            continue

        pid = _non_empty_str(player.get("player_id"))
        pname = _non_empty_str(player.get("player_name"))
        is_viewer = (viewer_id and pid == viewer_id) or (viewer_name and pname == viewer_name)
        if is_viewer:
            continue

        resource_cards = player.get("resource_cards")
        if isinstance(resource_cards, dict):
            for resource in TRACKED_RESOURCES:
                value = resource_cards.get(resource)
                if isinstance(value, (int, float)) and value >= 0:
                    leaks["opponent_resource_type_leak"] = True
                    break

        dev_cards = player.get("development_cards")
        if isinstance(dev_cards, dict):
            grouped = dev_cards.get("grouped_counts")
            if isinstance(grouped, dict):
                for key in ("knight", "progress", "victory_point"):
                    value = grouped.get(key)
                    if isinstance(value, (int, float)) and value >= 0:
                        leaks["opponent_dev_grouped_leak"] = True
                        break

            subtypes = dev_cards.get("progress_subtypes")
            if isinstance(subtypes, dict):
                for key in ("road_building", "year_of_plenty", "monopoly"):
                    value = subtypes.get(key)
                    if isinstance(value, (int, float)) and value >= 0:
                        leaks["opponent_dev_subtype_leak"] = True
                        break

    return leaks


def _compute_row_flags(row: Dict[str, Any]) -> Dict[str, Any]:
    game_id = _non_empty_str(row.get("game_id"))
    position_id = _non_empty_str(row.get("position_id"))

    turn_meta = row.get("turn_metadata")
    has_turn_metadata = isinstance(turn_meta, dict)

    winner_id, winner_name = _extract_winner_identity(row)
    termination_reason = _non_empty_str(row.get("termination_reason"))
    simulation_error = _non_empty_str(row.get("simulation_error"))

    has_snapshot = isinstance(row.get("snapshot"), dict)
    has_observed_state = isinstance(row.get("observed_state"), dict)
    has_encoding_metadata = isinstance(row.get("encoding_metadata"), dict)

    snapshot_checks = _snapshot_board_checks(row)
    leakage_checks = _observed_state_leakage_checks(row)

    negative_resources = _negative_resource_detected(row)
    impossible_remaining_pieces = _remaining_piece_impossible(row)

    missing_or_malformed_game_id = game_id == ""
    missing_or_malformed_position_id = position_id == ""
    missing_or_malformed_turn_metadata = not has_turn_metadata
    missing_or_malformed_winner = winner_name == "" and winner_id == ""
    missing_or_malformed_termination_reason = termination_reason == ""

    label_usable = _has_usable_label(row)

    critical_schema_corruption = (
        missing_or_malformed_game_id
        or missing_or_malformed_turn_metadata
        or not has_observed_state
        or not has_encoding_metadata
    )

    critical_state_corruption = (
        snapshot_checks["invalid_tile_count"]
        or snapshot_checks["duplicate_tile_coordinates"]
        or snapshot_checks["invalid_robber_count"]
        or snapshot_checks["robber_mismatch"]
        or snapshot_checks["duplicate_structure_occupancy"]
        or snapshot_checks["duplicate_road_occupancy"]
        or negative_resources
    )

    leakage_present = any(leakage_checks.values())

    return {
        "game_id": game_id,
        "position_id": position_id,
        "row_index_in_game": _safe_int(row.get("row_index_in_game")),
        "turn_index": _safe_int(turn_meta.get("turn_index")) if has_turn_metadata else None,
        "termination_reason": termination_reason,
        "simulation_error": simulation_error,
        "is_terminal": _bool_is_terminal(row.get("is_terminal")),
        "winner_id": winner_id,
        "winner_name": winner_name,
        "missing_or_malformed_game_id": missing_or_malformed_game_id,
        "missing_or_malformed_position_id": missing_or_malformed_position_id,
        "missing_or_malformed_turn_metadata": missing_or_malformed_turn_metadata,
        "missing_or_malformed_winner": missing_or_malformed_winner,
        "missing_or_malformed_termination_reason": missing_or_malformed_termination_reason,
        "missing_snapshot": not has_snapshot,
        "missing_observed_state": not has_observed_state,
        "missing_encoding_metadata": not has_encoding_metadata,
        "label_usable": label_usable,
        "terminal_winner_empty": _bool_is_terminal(row.get("is_terminal")) and termination_reason == "winner" and not (winner_name or winner_id),
        "snapshot_invalid_tile_count": snapshot_checks["invalid_tile_count"],
        "snapshot_duplicate_tile_coordinates": snapshot_checks["duplicate_tile_coordinates"],
        "snapshot_invalid_robber_count": snapshot_checks["invalid_robber_count"],
        "snapshot_robber_mismatch": snapshot_checks["robber_mismatch"],
        "snapshot_duplicate_structure_occupancy": snapshot_checks["duplicate_structure_occupancy"],
        "snapshot_duplicate_road_occupancy": snapshot_checks["duplicate_road_occupancy"],
        "snapshot_owner_id_not_found": snapshot_checks["owner_id_not_found"],
        "negative_resource_counts": negative_resources,
        "impossible_remaining_piece_values": impossible_remaining_pieces,
        "leak_resource_type_counts": leakage_checks["opponent_resource_type_leak"],
        "leak_dev_grouped_counts": leakage_checks["opponent_dev_grouped_leak"],
        "leak_dev_subtype_counts": leakage_checks["opponent_dev_subtype_leak"],
        "leakage_present": leakage_present,
        "critical_schema_corruption": critical_schema_corruption,
        "critical_state_corruption": critical_state_corruption,
    }


def _rows_per_game_summary(rows_per_game: Dict[str, int]) -> Dict[str, Any]:
    values = list(rows_per_game.values())
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }

    sorted_values = sorted(values)
    p95_index = max(0, int(len(sorted_values) * 0.95) - 1)
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(round(statistics.mean(values), 4)),
        "median": float(round(statistics.median(values), 4)),
        "p95": float(sorted_values[p95_index]),
    }


def _read_jsonl_lines(path: Path, max_rows: Optional[int]) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_rows is not None and line_number > max_rows:
                break
            yield line_number, line


def first_pass(input_path: Path, max_rows: Optional[int], parse_failure_limit: int = 25) -> FirstPassResult:
    parse_failures = 0
    parse_failure_examples: List[Dict[str, Any]] = []

    game_stats: Dict[str, GameStats] = {}
    phase_counts: Counter = Counter()
    action_counts: Counter = Counter()
    multi_action_turn_counts: Counter = Counter()
    winner_name_counts: Counter = Counter()
    winner_seat_counts: Counter = Counter()
    turn_index_counts: Counter = Counter()
    rows_per_game: Dict[str, int] = defaultdict(int)
    required_field_counts: Counter = Counter()
    snapshot_check_counts: Counter = Counter()
    leakage_check_counts: Counter = Counter()
    label_validity_counts: Counter = Counter()

    seen_position_ids = set()
    duplicate_position_id_count = 0

    total_rows = 0

    for line_number, line in _read_jsonl_lines(input_path, max_rows=max_rows):
        total_rows += 1
        stripped = line.strip()
        if not stripped:
            parse_failures += 1
            if len(parse_failure_examples) < parse_failure_limit:
                parse_failure_examples.append({"line_number": line_number, "error": "empty_line"})
            continue

        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            parse_failures += 1
            if len(parse_failure_examples) < parse_failure_limit:
                parse_failure_examples.append(
                    {
                        "line_number": line_number,
                        "error": "json_decode_error",
                        "message": str(exc),
                    }
                )
            continue

        if not isinstance(row, dict):
            parse_failures += 1
            if len(parse_failure_examples) < parse_failure_limit:
                parse_failure_examples.append(
                    {
                        "line_number": line_number,
                        "error": "row_not_object",
                    }
                )
            continue

        flags = _compute_row_flags(row)

        for key in (
            "missing_or_malformed_game_id",
            "missing_or_malformed_position_id",
            "missing_or_malformed_turn_metadata",
            "missing_or_malformed_winner",
            "missing_or_malformed_termination_reason",
            "missing_snapshot",
            "missing_observed_state",
            "missing_encoding_metadata",
        ):
            if flags.get(key):
                required_field_counts[key] += 1

        for key in (
            "snapshot_invalid_tile_count",
            "snapshot_duplicate_tile_coordinates",
            "snapshot_invalid_robber_count",
            "snapshot_robber_mismatch",
            "snapshot_duplicate_structure_occupancy",
            "snapshot_duplicate_road_occupancy",
            "snapshot_owner_id_not_found",
            "negative_resource_counts",
            "impossible_remaining_piece_values",
        ):
            if flags.get(key):
                snapshot_check_counts[key] += 1

        for key in (
            "leak_resource_type_counts",
            "leak_dev_grouped_counts",
            "leak_dev_subtype_counts",
        ):
            if flags.get(key):
                leakage_check_counts[key] += 1

        label_validity_counts["label_usable_rows" if flags["label_usable"] else "label_missing_rows"] += 1
        if flags["terminal_winner_empty"]:
            label_validity_counts["terminal_winner_empty_rows"] += 1

        game_id = flags["game_id"]
        if not game_id:
            game_id = "__missing_game_id__"

        game = game_stats.setdefault(game_id, GameStats(game_id=game_id))
        game.row_count += 1
        game.parseable_rows += 1
        rows_per_game[game_id] += 1

        if flags["critical_schema_corruption"] or flags["critical_state_corruption"]:
            game.malformed_rows += 1
        if flags["missing_observed_state"]:
            game.missing_observed_rows += 1
        if flags["missing_encoding_metadata"]:
            game.missing_encoding_rows += 1
        if flags["missing_snapshot"]:
            game.missing_snapshot_rows += 1
        if flags["leakage_present"]:
            game.leakage_rows += 1
        if flags["label_usable"]:
            game.label_usable_rows += 1

        termination_reason = flags["termination_reason"] or "__missing__"
        game.termination_reason_counts[termination_reason] += 1

        if flags["is_terminal"]:
            game.terminal_row_count += 1
            game.terminal_reason_counts[termination_reason] += 1
            if flags["terminal_winner_empty"]:
                game.terminal_winner_empty_rows += 1

        if flags["simulation_error"]:
            game.simulation_error_rows += 1

        winner_name = flags["winner_name"] or ""
        winner_id = flags["winner_id"] or ""
        game.winner_name_counts[winner_name] += 1
        game.winner_id_counts[winner_id] += 1

        if winner_name:
            winner_name_counts[winner_name] += 1

        position_id = flags["position_id"]
        if position_id:
            if position_id in seen_position_ids:
                duplicate_position_id_count += 1
                game.duplicate_position_rows += 1
            else:
                seen_position_ids.add(position_id)

        turn_index = flags["turn_index"]
        if isinstance(turn_index, int):
            turn_index_counts[turn_index] += 1
            if turn_index > game.max_turn_index:
                game.max_turn_index = turn_index

        turn_metadata = row.get("turn_metadata")
        if isinstance(turn_metadata, dict):
            phase = _non_empty_str(turn_metadata.get("current_phase")).lower() or "unknown"
            phase_counts[phase] += 1

        action_parts = _split_actions(row.get("action_taken"))
        for action in action_parts:
            action_counts[action] += 1
        multi_action_turn_counts[len(action_parts)] += 1

        if flags["is_terminal"] and not game.winner_distribution_sampled:
            snapshot = row.get("snapshot")
            winner_for_game = flags["winner_name"]
            if isinstance(snapshot, dict) and winner_for_game:
                players = snapshot.get("players")
                if isinstance(players, list):
                    for seat_index, player in enumerate(players):
                        if not isinstance(player, dict):
                            continue
                        if _non_empty_str(player.get("player_name")) == winner_for_game:
                            winner_seat_counts[f"seat_{seat_index}"] += 1
                            break
            game.winner_distribution_sampled = True

    return FirstPassResult(
        total_rows=total_rows,
        parse_failures=parse_failures,
        parse_failure_examples=parse_failure_examples,
        unique_games=len(game_stats),
        game_stats=game_stats,
        phase_counts=phase_counts,
        action_counts=action_counts,
        multi_action_turn_counts=multi_action_turn_counts,
        winner_name_counts=winner_name_counts,
        winner_seat_counts=winner_seat_counts,
        turn_index_counts=turn_index_counts,
        duplicate_position_id_count=duplicate_position_id_count,
        rows_per_game=dict(rows_per_game),
        required_field_counts=required_field_counts,
        snapshot_check_counts=snapshot_check_counts,
        leakage_check_counts=leakage_check_counts,
        label_validity_counts=label_validity_counts,
    )


def _determine_canonical_winner(game: GameStats) -> Tuple[str, str]:
    winner_name_non_empty = {
        k: v for k, v in game.winner_name_counts.items() if k
    }
    winner_id_non_empty = {
        k: v for k, v in game.winner_id_counts.items() if k
    }

    canonical_name = ""
    canonical_id = ""

    if winner_name_non_empty:
        canonical_name = max(winner_name_non_empty.items(), key=lambda kv: kv[1])[0]
    if winner_id_non_empty:
        canonical_id = max(winner_id_non_empty.items(), key=lambda kv: kv[1])[0]

    return canonical_name, canonical_id


def build_game_quality(game_stats: Dict[str, GameStats]) -> Dict[str, GameQuality]:
    quality: Dict[str, GameQuality] = {}
    for game_id, stats in game_stats.items():
        terminal_reason = ""
        if stats.terminal_reason_counts:
            terminal_reason = max(stats.terminal_reason_counts.items(), key=lambda kv: kv[1])[0]

        winner_ended = terminal_reason == "winner"
        truncated = terminal_reason.startswith("truncated")
        simulation_error = stats.simulation_error_rows > 0

        winner_name_non_empty = {k for k in stats.winner_name_counts.keys() if k}
        winner_id_non_empty = {k for k in stats.winner_id_counts.keys() if k}
        winner_changes_within_game = len(winner_name_non_empty) > 1 or len(winner_id_non_empty) > 1
        no_consistent_winner = winner_ended and (len(winner_name_non_empty) != 1 and len(winner_id_non_empty) != 1)

        canonical_name, canonical_id = _determine_canonical_winner(stats)

        winner_backfill_missing_rows = 0
        if winner_ended and (canonical_name or canonical_id):
            if canonical_name:
                winner_backfill_missing_rows = stats.row_count - stats.winner_name_counts.get(canonical_name, 0)
            elif canonical_id:
                winner_backfill_missing_rows = stats.row_count - stats.winner_id_counts.get(canonical_id, 0)

        game_train_usable_base = (
            winner_ended
            and not simulation_error
            and not winner_changes_within_game
            and not no_consistent_winner
            and bool(canonical_name or canonical_id)
            and winner_backfill_missing_rows == 0
        )

        quality[game_id] = GameQuality(
            game_id=game_id,
            canonical_winner_name=canonical_name,
            canonical_winner_id=canonical_id,
            winner_ended=winner_ended,
            truncated=truncated,
            simulation_error=simulation_error,
            winner_changes_within_game=winner_changes_within_game,
            no_consistent_winner=no_consistent_winner,
            winner_backfill_missing_rows=winner_backfill_missing_rows,
            game_train_usable_base=game_train_usable_base,
            terminal_reason=terminal_reason,
        )

    return quality


def second_pass_and_write_outputs(
    input_path: Path,
    max_rows: Optional[int],
    game_quality: Dict[str, GameQuality],
    row_flags_csv_path: Path,
) -> Dict[str, Any]:
    row_flags_csv_path.parent.mkdir(parents=True, exist_ok=True)

    exclusion_counts = Counter()
    leak_rows = 0
    malformed_rows = 0
    train_usable_rows = 0
    train_usable_games = set()
    leaked_games = set()

    fieldnames = [
        "line_number",
        "game_id",
        "position_id",
        "row_index_in_game",
        "turn_index",
        "termination_reason",
        "is_terminal",
        "label_usable",
        "missing_snapshot",
        "missing_observed_state",
        "missing_encoding_metadata",
        "critical_schema_corruption",
        "critical_state_corruption",
        "leak_resource_type_counts",
        "leak_dev_grouped_counts",
        "leak_dev_subtype_counts",
        "leakage_present",
        "exclude_reason_primary",
        "exclude_reason_all",
        "train_usable",
    ]

    with row_flags_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for line_number, line in _read_jsonl_lines(input_path, max_rows=max_rows):
            stripped = line.strip()
            if not stripped:
                writer.writerow(
                    {
                        "line_number": line_number,
                        "game_id": "",
                        "position_id": "",
                        "exclude_reason_primary": "parse_failure",
                        "exclude_reason_all": "parse_failure",
                        "train_usable": False,
                    }
                )
                exclusion_counts["parse_failure"] += 1
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                writer.writerow(
                    {
                        "line_number": line_number,
                        "game_id": "",
                        "position_id": "",
                        "exclude_reason_primary": "parse_failure",
                        "exclude_reason_all": "parse_failure",
                        "train_usable": False,
                    }
                )
                exclusion_counts["parse_failure"] += 1
                continue

            if not isinstance(row, dict):
                writer.writerow(
                    {
                        "line_number": line_number,
                        "game_id": "",
                        "position_id": "",
                        "exclude_reason_primary": "parse_failure",
                        "exclude_reason_all": "parse_failure",
                        "train_usable": False,
                    }
                )
                exclusion_counts["parse_failure"] += 1
                continue

            flags = _compute_row_flags(row)
            game_id = flags["game_id"] or "__missing_game_id__"
            gq = game_quality.get(game_id)

            reasons = []
            if gq is None or not gq.game_train_usable_base:
                if gq is not None and gq.truncated:
                    reasons.append("game_truncated")
                elif gq is not None and gq.simulation_error:
                    reasons.append("game_simulation_error")
                else:
                    reasons.append("game_not_winner_ended_or_inconsistent")

            if not flags["label_usable"]:
                reasons.append("missing_valid_label")

            if flags["missing_observed_state"]:
                reasons.append("missing_observed_state")

            if flags["leakage_present"]:
                reasons.append("observed_state_leakage")
                leak_rows += 1
                leaked_games.add(game_id)

            if flags["critical_schema_corruption"] or flags["critical_state_corruption"]:
                reasons.append("critical_schema_or_state_corruption")
                malformed_rows += 1

            train_usable = len(reasons) == 0
            if train_usable:
                train_usable_rows += 1
                train_usable_games.add(game_id)
            else:
                exclusion_counts[reasons[0]] += 1

            writer.writerow(
                {
                    "line_number": line_number,
                    "game_id": game_id,
                    "position_id": flags["position_id"],
                    "row_index_in_game": flags["row_index_in_game"],
                    "turn_index": flags["turn_index"],
                    "termination_reason": flags["termination_reason"],
                    "is_terminal": flags["is_terminal"],
                    "label_usable": flags["label_usable"],
                    "missing_snapshot": flags["missing_snapshot"],
                    "missing_observed_state": flags["missing_observed_state"],
                    "missing_encoding_metadata": flags["missing_encoding_metadata"],
                    "critical_schema_corruption": flags["critical_schema_corruption"],
                    "critical_state_corruption": flags["critical_state_corruption"],
                    "leak_resource_type_counts": flags["leak_resource_type_counts"],
                    "leak_dev_grouped_counts": flags["leak_dev_grouped_counts"],
                    "leak_dev_subtype_counts": flags["leak_dev_subtype_counts"],
                    "leakage_present": flags["leakage_present"],
                    "exclude_reason_primary": reasons[0] if reasons else "",
                    "exclude_reason_all": "|".join(reasons),
                    "train_usable": train_usable,
                }
            )

    return {
        "train_usable_rows": train_usable_rows,
        "train_usable_games": len(train_usable_games),
        "excluded_by_primary_reason": dict(exclusion_counts),
        "rows_excluded_due_to_leakage": leak_rows,
        "rows_excluded_due_to_malformed_schema_or_state": malformed_rows,
        "leaked_game_count": len(leaked_games),
    }


def write_game_outputs(
    game_stats: Dict[str, GameStats],
    game_quality: Dict[str, GameQuality],
    game_json_path: Path,
    game_csv_path: Path,
) -> None:
    game_json_path.parent.mkdir(parents=True, exist_ok=True)
    game_csv_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for game_id in sorted(game_stats.keys()):
        stats = game_stats[game_id]
        quality = game_quality[game_id]
        records.append(
            {
                "game_id": game_id,
                "rows": stats.row_count,
                "terminal_rows": stats.terminal_row_count,
                "terminal_reason": quality.terminal_reason,
                "winner_ended": quality.winner_ended,
                "truncated": quality.truncated,
                "simulation_error": quality.simulation_error,
                "winner_changes_within_game": quality.winner_changes_within_game,
                "no_consistent_winner": quality.no_consistent_winner,
                "winner_backfill_missing_rows": quality.winner_backfill_missing_rows,
                "canonical_winner_name": quality.canonical_winner_name,
                "canonical_winner_id": quality.canonical_winner_id,
                "missing_observed_rows": stats.missing_observed_rows,
                "missing_snapshot_rows": stats.missing_snapshot_rows,
                "missing_encoding_rows": stats.missing_encoding_rows,
                "leakage_rows": stats.leakage_rows,
                "malformed_rows": stats.malformed_rows,
                "label_usable_rows": stats.label_usable_rows,
                "duplicate_position_rows": stats.duplicate_position_rows,
                "game_train_usable_base": quality.game_train_usable_base,
            }
        )

    with game_json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    if records:
        with game_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def build_summary(
    input_path: Path,
    first: FirstPassResult,
    game_quality: Dict[str, GameQuality],
    second_pass_summary: Dict[str, Any],
) -> Dict[str, Any]:
    games = list(game_quality.values())

    games_ending_winner = sum(1 for g in games if g.winner_ended)
    games_ending_truncation = sum(1 for g in games if g.truncated)
    games_with_simulation_errors = sum(1 for g in games if g.simulation_error)
    games_with_winner_changes = sum(1 for g in games if g.winner_changes_within_game)
    games_with_no_consistent_winner = sum(1 for g in games if g.no_consistent_winner)

    terminal_winner_empty_rows = sum(
        stats.terminal_winner_empty_rows for stats in first.game_stats.values()
    )
    winner_backfill_missing_games = sum(
        1 for g in games if g.winner_backfill_missing_rows > 0
    )

    turn_dist = {
        "min": min(first.turn_index_counts.keys()) if first.turn_index_counts else None,
        "max": max(first.turn_index_counts.keys()) if first.turn_index_counts else None,
        "unique_turn_indices": len(first.turn_index_counts),
    }

    action_total = sum(first.action_counts.values())
    multi_action_rows = sum(
        count for actions_per_turn, count in first.multi_action_turn_counts.items() if actions_per_turn > 1
    )

    summary = {
        "input_file": str(input_path),
        "basic_file_stats": {
            "total_rows": first.total_rows,
            "total_unique_games": first.unique_games,
            "rows_per_game_summary": _rows_per_game_summary(first.rows_per_game),
            "file_parse_failures": first.parse_failures,
            "parse_failure_examples": first.parse_failure_examples,
        },
        "required_field_validity": {
            "missing_or_malformed_game_id_rows": first.required_field_counts.get("missing_or_malformed_game_id", 0),
            "missing_or_malformed_position_id_rows": first.required_field_counts.get("missing_or_malformed_position_id", 0),
            "missing_or_malformed_turn_metadata_rows": first.required_field_counts.get("missing_or_malformed_turn_metadata", 0),
            "missing_or_malformed_winner_rows": first.required_field_counts.get("missing_or_malformed_winner", 0),
            "missing_or_malformed_termination_reason_rows": first.required_field_counts.get(
                "missing_or_malformed_termination_reason", 0
            ),
            "missing_snapshot_rows": first.required_field_counts.get("missing_snapshot", 0),
            "missing_observed_state_rows": first.required_field_counts.get("missing_observed_state", 0),
            "missing_encoding_metadata_rows": first.required_field_counts.get("missing_encoding_metadata", 0),
        },
        "game_level_validity": {
            "games_ending_with_winner": games_ending_winner,
            "games_ending_with_truncation": games_ending_truncation,
            "games_with_simulation_errors": games_with_simulation_errors,
            "games_with_no_consistent_winner": games_with_no_consistent_winner,
            "games_where_winner_changes_within_game": games_with_winner_changes,
        },
        "label_validity": {
            "rows_with_usable_binary_label": first.label_validity_counts.get("label_usable_rows", 0),
            "rows_missing_usable_labels": first.label_validity_counts.get("label_missing_rows", 0),
            "terminal_winner_rows_with_empty_winner": first.label_validity_counts.get(
                "terminal_winner_empty_rows", terminal_winner_empty_rows
            ),
            "winner_ended_games_with_backfill_missing": winner_backfill_missing_games,
        },
        "board_state_consistency": {
            "invalid_tile_count_rows": first.snapshot_check_counts.get("snapshot_invalid_tile_count", 0),
            "duplicate_tile_coordinate_rows": first.snapshot_check_counts.get("snapshot_duplicate_tile_coordinates", 0),
            "invalid_robber_count_rows": first.snapshot_check_counts.get("snapshot_invalid_robber_count", 0),
            "robber_mismatch_rows": first.snapshot_check_counts.get("snapshot_robber_mismatch", 0),
            "duplicate_structure_occupancy_rows": first.snapshot_check_counts.get(
                "snapshot_duplicate_structure_occupancy", 0
            ),
            "duplicate_road_occupancy_rows": first.snapshot_check_counts.get("snapshot_duplicate_road_occupancy", 0),
            "owner_id_not_found_rows": first.snapshot_check_counts.get("snapshot_owner_id_not_found", 0),
            "negative_resource_count_rows": first.snapshot_check_counts.get("negative_resource_counts", 0),
            "impossible_remaining_piece_rows": first.snapshot_check_counts.get("impossible_remaining_piece_values", 0),
            "rows_with_malformed_or_corrupt_state": second_pass_summary["rows_excluded_due_to_malformed_schema_or_state"],
        },
        "snapshot_validation": {
            "raw_snapshot_checks": dict(first.snapshot_check_counts),
        },
        "observed_state_safe_validation": {
            "observed_hidden_info_leak_checks": dict(first.leakage_check_counts),
        },
        "observed_state_leakage": {
            "leaked_row_count": second_pass_summary["rows_excluded_due_to_leakage"],
            "leaked_game_count": second_pass_summary["leaked_game_count"],
            "opponent_resource_type_leak_rows": first.leakage_check_counts.get("leak_resource_type_counts", 0),
            "opponent_dev_grouped_leak_rows": first.leakage_check_counts.get("leak_dev_grouped_counts", 0),
            "opponent_dev_subtype_leak_rows": first.leakage_check_counts.get("leak_dev_subtype_counts", 0),
        },
        "training_usable_subset": {
            "rows_usable_for_winner_prediction": second_pass_summary["train_usable_rows"],
            "rows_excluded_due_to_truncation": second_pass_summary["excluded_by_primary_reason"].get(
                "game_truncated", 0
            ),
            "rows_excluded_due_to_non_winner_or_inconsistent_game": second_pass_summary[
                "excluded_by_primary_reason"
            ].get("game_not_winner_ended_or_inconsistent", 0),
            "rows_excluded_due_to_truncation_or_non_winner_game": second_pass_summary["excluded_by_primary_reason"].get("game_truncated", 0)
            + second_pass_summary["excluded_by_primary_reason"].get("game_not_winner_ended_or_inconsistent", 0),
            "rows_excluded_due_to_leakage": second_pass_summary["rows_excluded_due_to_leakage"],
            "rows_excluded_due_to_malformed_schema": second_pass_summary["rows_excluded_due_to_malformed_schema_or_state"],
            "rows_excluded_due_to_missing_label": second_pass_summary["excluded_by_primary_reason"].get(
                "missing_valid_label", 0
            ),
            "final_clean_usable_row_count": second_pass_summary["train_usable_rows"],
            "final_clean_usable_game_count": second_pass_summary["train_usable_games"],
            "train_usable_rule": "winner-ended game + valid label + observed_state present + no leakage + no critical schema/state corruption",
        },
        "distribution_analysis": {
            "phase_distribution": dict(first.phase_counts),
            "action_frequency": dict(first.action_counts),
            "multi_action_per_turn": {
                "rows_with_multiple_actions": multi_action_rows,
                "rows_with_multiple_actions_pct": round((multi_action_rows / first.total_rows) * 100.0, 4)
                if first.total_rows
                else 0.0,
                "histogram_actions_per_turn": {str(k): v for k, v in sorted(first.multi_action_turn_counts.items())},
            },
            "winner_distribution_by_player": dict(first.winner_name_counts),
            "winner_distribution_by_seat": dict(first.winner_seat_counts),
            "turn_index_distribution": turn_dist,
            "duplicate_position_id_rows": first.duplicate_position_id_count,
            "action_total_events": action_total,
        },
    }

    summary["exclusion_breakdown_primary_reason"] = second_pass_summary["excluded_by_primary_reason"]
    return summary


def run_analysis(input_path: Path, output_dir: Path, max_rows: Optional[int]) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    first = first_pass(input_path=input_path, max_rows=max_rows)
    game_quality = build_game_quality(first.game_stats)

    row_flags_csv = output_dir / "row_qa_flags.csv"
    game_json = output_dir / "game_qa_summary.json"
    game_csv = output_dir / "game_qa_summary.csv"
    summary_json = output_dir / "summary.json"

    second = second_pass_and_write_outputs(
        input_path=input_path,
        max_rows=max_rows,
        game_quality=game_quality,
        row_flags_csv_path=row_flags_csv,
    )

    write_game_outputs(
        game_stats=first.game_stats,
        game_quality=game_quality,
        game_json_path=game_json,
        game_csv_path=game_csv,
    )

    summary = build_summary(
        input_path=input_path,
        first=first,
        game_quality=game_quality,
        second_pass_summary=second,
    )

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "summary_json": summary_json,
        "row_flags_csv": row_flags_csv,
        "game_summary_json": game_json,
        "game_summary_csv": game_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze self-play dataset quality and quantify train-usable rows/games."
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write summary JSON and QA flag CSV outputs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for quick smoke analysis",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    outputs = run_analysis(input_path=input_path, output_dir=output_dir, max_rows=args.max_rows)

    print(f"Wrote summary JSON: {outputs['summary_json']}")
    print(f"Wrote row-level QA CSV: {outputs['row_flags_csv']}")
    print(f"Wrote game-level QA JSON: {outputs['game_summary_json']}")
    print(f"Wrote game-level QA CSV: {outputs['game_summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
