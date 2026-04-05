from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


def _run_command(command: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=True)


def _load_snapshot_extractor(repo_root: Path) -> Callable[..., Dict[str, Any]]:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from ai.board_state_extractor import extract_board_state_from_snapshot

    return extract_board_state_from_snapshot


def _enrich_row_with_extracted_state(
    row_text: str,
    extract_board_state_from_snapshot: Callable[..., Dict[str, Any]],
) -> str:
    row = json.loads(row_text)
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, dict):
        return row_text

    turn_metadata = row.get("turn_metadata") if isinstance(row.get("turn_metadata"), dict) else {}
    extracted = extract_board_state_from_snapshot(
        snapshot,
        current_player_name=turn_metadata.get("current_player_name") or snapshot.get("current_player_name"),
        current_player_id=turn_metadata.get("current_player_id") or snapshot.get("current_player_id"),
        reveal_private=False,
        strict_validation=False,
        top_level_mirror="observed",
    )

    row["state_schema_version"] = extracted.get("schema_version")
    row["state_ids"] = extracted.get("state_ids")
    row["position_ids"] = extracted.get("position_ids")
    row["encoding_metadata"] = extracted.get("encoding_metadata")
    row["observed_state"] = extracted.get("observed_state")
    row["omniscient_state"] = extracted.get("omniscient_state")
    row["state_top_level_mirror"] = extracted.get("top_level_mirror")
    return json.dumps(row, separators=(",", ":"))


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _render_progress(
    completed_games: int,
    total_games: int,
    total_rows: int,
    started_at: float,
    bar_width: int,
    enabled: bool,
    final: bool = False,
) -> None:
    if not enabled:
        return

    elapsed = max(0.001, time.time() - started_at)
    progress = (completed_games / total_games) if total_games > 0 else 1.0
    filled = int(progress * bar_width)
    if filled > bar_width:
        filled = bar_width

    eta_seconds = 0.0
    if completed_games > 0 and completed_games < total_games:
        games_per_sec = completed_games / elapsed
        if games_per_sec > 0:
            eta_seconds = (total_games - completed_games) / games_per_sec

    bar = "#" * filled + "-" * (bar_width - filled)
    line = (
        f"[{bar}] {completed_games}/{total_games} "
        f"({progress * 100:5.1f}%) rows={total_rows} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)}"
    )

    if sys.stdout.isatty():
        prefix = "\r"
        suffix = "\n" if final else ""
        sys.stdout.write(prefix + line + suffix)
        sys.stdout.flush()
    else:
        print(line)


def _compile_runner(repo_root: Path) -> None:
    command = [
        "javac",
        "-cp",
        "src",
        "src/game/GameRunner.java",
        "src/game/SelfPlayGameRunner.java",
    ]
    _run_command(command, cwd=repo_root)


def _generate_single_game(
    repo_root: Path,
    java_debug: bool,
    java_no_strict: bool,
    java_max_turns: int,
    java_seed: int | None,
    java_guided: bool,
    java_guided_model_path: str | None,
    java_guided_python_exec: str | None,
) -> tuple[List[str], str]:
    command = ["java", "-cp", "src", "game.SelfPlayGameRunner", "--max-turns", str(java_max_turns)]
    if java_debug:
        command.append("--debug")
    if java_no_strict:
        command.append("--no-strict")
    if java_guided:
        command.append("--guided")
    if java_guided_model_path:
        command.extend(["--guided-model-path", java_guided_model_path])
    if java_guided_python_exec:
        command.extend(["--guided-python-exec", java_guided_python_exec])
    if java_seed is not None:
        command.extend(["--seed", str(java_seed)])
    result = _run_command(command, cwd=repo_root)
    return [line for line in result.stdout.splitlines() if line.strip()], result.stderr


def generate_trajectories(
    repo_root: Path,
    output_path: Path,
    num_games: int,
    compile_java: bool,
    java_debug: bool,
    java_no_strict: bool,
    java_max_turns: int,
    base_seed: int | None,
    java_guided: bool,
    java_guided_model_path: str | None,
    java_guided_python_exec: str | None,
    extract_state: bool,
    progress_enabled: bool,
    progress_bar_width: int,
) -> int:
    extract_board_state_from_snapshot: Callable[..., Dict[str, Any]] | None = None
    if extract_state:
        extract_board_state_from_snapshot = _load_snapshot_extractor(repo_root)

    if compile_java:
        _compile_runner(repo_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    started_at = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for game_index in range(num_games):
            game_seed = None if base_seed is None else base_seed + game_index
            rows, stderr_output = _generate_single_game(
                repo_root=repo_root,
                java_debug=java_debug,
                java_no_strict=java_no_strict,
                java_max_turns=java_max_turns,
                java_seed=game_seed,
                java_guided=java_guided,
                java_guided_model_path=java_guided_model_path,
                java_guided_python_exec=java_guided_python_exec,
            )
            for row in rows:
                out_row = row
                if extract_board_state_from_snapshot is not None:
                    out_row = _enrich_row_with_extracted_state(
                        row_text=row,
                        extract_board_state_from_snapshot=extract_board_state_from_snapshot,
                    )
                handle.write(out_row)
                handle.write("\n")
            total_rows += len(rows)
            _render_progress(
                completed_games=game_index + 1,
                total_games=num_games,
                total_rows=total_rows,
                started_at=started_at,
                bar_width=progress_bar_width,
                enabled=progress_enabled,
                final=(game_index + 1 == num_games),
            )
            if java_debug and stderr_output.strip():
                print(stderr_output, end="")
    return total_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate legal Catan self-play trajectories via Java runner")
    parser.add_argument("--num-games", type=int, default=5, help="Number of full games to simulate")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root containing src/",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip javac step and run with existing compiled classes",
    )
    parser.add_argument("--java-max-turns", type=int, default=1200, help="Forwarded to Java runner --max-turns")
    parser.add_argument("--java-debug", action="store_true", help="Forwarded to Java runner --debug")
    parser.add_argument("--java-no-strict", action="store_true", help="Forwarded to Java runner --no-strict")
    parser.add_argument("--java-guided", action="store_true", help="Forwarded to Java runner --guided")
    parser.add_argument("--java-guided-model-path", default=None, help="Forwarded to Java runner --guided-model-path")
    parser.add_argument("--java-guided-python-exec", default=None, help="Forwarded to Java runner --guided-python-exec")
    parser.add_argument(
        "--no-extract-state",
        action="store_true",
        help="Skip Python board-state extraction enrichment and emit raw Java rows only",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
    parser.add_argument("--progress-bar-width", type=int, default=30, help="Progress bar width in characters")
    parser.add_argument("--seed", type=int, default=None, help="Base seed; game i uses seed+i")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    total_rows = generate_trajectories(
        repo_root=repo_root,
        output_path=output_path,
        num_games=args.num_games,
        compile_java=not args.skip_compile,
        java_debug=args.java_debug,
        java_no_strict=args.java_no_strict,
        java_max_turns=args.java_max_turns,
        base_seed=args.seed,
        java_guided=args.java_guided,
        java_guided_model_path=args.java_guided_model_path,
        java_guided_python_exec=args.java_guided_python_exec,
        extract_state=not args.no_extract_state,
        progress_enabled=not args.no_progress,
        progress_bar_width=max(10, args.progress_bar_width),
    )
    print(f"Wrote {total_rows} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
