from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SEED_STRIDE_DEFAULT = 1_000_003
POLL_INTERVAL_SECONDS = 0.4


@dataclass
class WorkerShard:
    worker_id: int
    games: int
    seed: int
    output_path: Path
    log_path: Path


@dataclass
class RunResult:
    elapsed_seconds: float
    workers_used: int
    games_requested: int
    shard_line_counts: Dict[int, int]
    merged_line_count: int
    merged_output: Path
    run_dir: Path


@dataclass
class CommandPlan:
    worker_id: int
    command: List[str]
    cwd: Path
    log_path: Path
    label: str


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex_quote(part) for part in parts)


def shlex_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:=")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _recommended_workers(cpu_count: Optional[int] = None) -> int:
    cpu = cpu_count or (os.cpu_count() or 8)
    if cpu <= 6:
        return 2
    if cpu <= 10:
        return 4
    if cpu <= 14:
        return 6
    return min(8, max(2, cpu - 4))


def _split_games(total_games: int, workers: int) -> List[int]:
    if total_games <= 0:
        raise ValueError("--num-games must be >= 1")
    if workers <= 0:
        raise ValueError("--workers must be >= 1")
    base = total_games // workers
    remainder = total_games % workers
    counts = [base + (1 if index < remainder else 0) for index in range(workers)]
    return counts


def _build_shards(
    counts: Sequence[int],
    base_seed: int,
    seed_stride: int,
    shard_dir: Path,
    log_dir: Path,
    prefix: str,
) -> List[WorkerShard]:
    shards: List[WorkerShard] = []
    for worker_id, games in enumerate(counts):
        if games <= 0:
            continue
        output_path = shard_dir / f"{prefix}_worker_{worker_id:02d}.jsonl"
        log_path = log_dir / f"{prefix}_worker_{worker_id:02d}.log"
        shards.append(
            WorkerShard(
                worker_id=worker_id,
                games=games,
                seed=base_seed + worker_id * seed_stride,
                output_path=output_path,
                log_path=log_path,
            )
        )
    return shards


def _ensure_parent_dirs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _compile_java_once(repo_root: Path) -> None:
    command = [
        "javac",
        "-cp",
        "src",
        "src/game/GameRunner.java",
        "src/game/SelfPlayGameRunner.java",
    ]
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Java precompile failed\n"
            f"Command: {_shell_join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _launch_parallel_commands(
    plans: Sequence[CommandPlan],
    dry_run: bool,
    max_parallel: Optional[int] = None,
) -> None:
    if not plans:
        return

    if dry_run:
        print("[dry-run] Commands:")
        for plan in plans:
            print(f"  - [{plan.label}#{plan.worker_id}] {_shell_join(plan.command)}")
        return

    running: Dict[int, Dict[str, Any]] = {}
    queue: List[CommandPlan] = list(plans)
    parallel_limit = max_parallel if max_parallel is not None else len(plans)
    parallel_limit = max(1, min(parallel_limit, len(plans)))
    failed: List[Tuple[int, int]] = []

    def start_more() -> None:
        while queue and len(running) < parallel_limit:
            plan = queue.pop(0)
            plan.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = plan.log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                plan.command,
                cwd=str(plan.cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[plan.worker_id] = {
                "process": process,
                "log_handle": log_handle,
                "started_at": time.time(),
                "plan": plan,
            }
            print(
                f"[start] {plan.label} worker={plan.worker_id} pid={process.pid} "
                f"log={plan.log_path}"
            )

    start_more()

    try:
        while running:
            completed_ids: List[int] = []
            for worker_id, record in running.items():
                process: subprocess.Popen[str] = record["process"]
                exit_code = process.poll()
                if exit_code is None:
                    continue
                elapsed = time.time() - float(record["started_at"])
                plan: CommandPlan = record["plan"]
                status = "ok" if exit_code == 0 else "fail"
                print(
                    f"[done] {plan.label} worker={worker_id} status={status} "
                    f"exit={exit_code} elapsed={elapsed:.1f}s"
                )
                record["log_handle"].close()
                if exit_code != 0:
                    failed.append((worker_id, exit_code))
                completed_ids.append(worker_id)

            for worker_id in completed_ids:
                running.pop(worker_id, None)

            if not failed:
                start_more()

            if failed and running:
                print("[abort] A worker failed; terminating remaining workers...")
                for record in running.values():
                    process = record["process"]
                    if process.poll() is None:
                        process.terminate()
                time.sleep(0.5)
                for record in running.values():
                    process = record["process"]
                    if process.poll() is None:
                        process.kill()
                    record["log_handle"].close()
                running.clear()
                break

            if running:
                time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("[abort] Keyboard interrupt; terminating workers...")
        for record in running.values():
            process = record["process"]
            if process.poll() is None:
                process.terminate()
            record["log_handle"].close()
        raise

    if failed:
        details = []
        for worker_id, exit_code in failed:
            matching = [p for p in plans if p.worker_id == worker_id]
            log_path = matching[0].log_path if matching else None
            details.append(f"worker={worker_id} exit={exit_code} log={log_path}")
        raise RuntimeError("One or more workers failed: " + "; ".join(details))


def _validate_shards(shards: Sequence[WorkerShard], require_non_empty: bool = True) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for shard in shards:
        if not shard.output_path.exists():
            raise RuntimeError(f"Shard output missing: {shard.output_path}")
        line_count = _line_count(shard.output_path)
        if require_non_empty and line_count <= 0:
            raise RuntimeError(f"Shard output empty: {shard.output_path}")
        counts[shard.worker_id] = line_count
    return counts


def _merge_jsonl(shard_paths: Sequence[Path], merged_output: Path) -> int:
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with merged_output.open("w", encoding="utf-8") as out_handle:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)
                    total += 1
    return total


def _enrich_raw_file(input_path: Path, output_path: Path, repo_root: Path) -> int:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from ai.board_state_extractor import extract_board_state_from_snapshot

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as in_handle, output_path.open(
        "w", encoding="utf-8"
    ) as out_handle:
        for line_number, line in enumerate(in_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            snapshot = row.get("snapshot")
            if isinstance(snapshot, dict):
                turn_meta = row.get("turn_metadata") if isinstance(row.get("turn_metadata"), dict) else {}
                extracted = extract_board_state_from_snapshot(
                    snapshot,
                    current_player_name=turn_meta.get("current_player_name") or snapshot.get("current_player_name"),
                    current_player_id=turn_meta.get("current_player_id") or snapshot.get("current_player_id"),
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
            else:
                raise RuntimeError(
                    f"Missing snapshot object in {input_path} line {line_number}; cannot run extraction stage"
                )

            out_handle.write(json.dumps(row, separators=(",", ":")))
            out_handle.write("\n")
            count += 1
    return count


def _run_pipeline(
    repo_root: Path,
    generator_path: Path,
    output_path: Path,
    num_games: int,
    workers: int,
    seed: int,
    seed_stride: int,
    java_max_turns: int,
    java_debug: bool,
    java_no_strict: bool,
    java_guided: bool,
    java_guided_model_path: Optional[str],
    java_guided_python_exec: Optional[str],
    no_precompile: bool,
    dry_run: bool,
    generation_mode: str,
    extract_after: bool,
    extract_workers: Optional[int],
    work_dir: Path,
    keep_shards: bool,
    python_exec: str,
) -> RunResult:
    if num_games < 1:
        raise ValueError("--num-games must be >= 1")

    workers = max(1, workers)
    workers = min(workers, num_games)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = work_dir / f"run_{run_tag}_{num_games}g_{workers}w"
    raw_dir = run_dir / "raw_shards"
    extracted_dir = run_dir / "extracted_shards"
    logs_dir = run_dir / "logs"

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cpu = os.cpu_count() or 8
    print(
        f"[config] cpu_count={cpu} workers={workers} recommended_workers={_recommended_workers(cpu)} "
        f"mode={generation_mode} extract_after={extract_after}"
    )

    counts = _split_games(num_games, workers)
    raw_shards = _build_shards(
        counts=counts,
        base_seed=seed,
        seed_stride=seed_stride,
        shard_dir=raw_dir,
        log_dir=logs_dir,
        prefix="raw",
    )

    if not no_precompile:
        if dry_run:
            print("[dry-run] Would precompile Java classes once via javac")
        else:
            print("[step] Precompiling Java classes once...")
            _compile_java_once(repo_root)

    generation_plans: List[CommandPlan] = []
    for shard in raw_shards:
        cmd = [
            python_exec,
            str(generator_path),
            "--num-games",
            str(shard.games),
            "--output",
            str(shard.output_path),
            "--seed",
            str(shard.seed),
            "--java-max-turns",
            str(java_max_turns),
            "--no-progress",
        ]

        if not no_precompile:
            cmd.append("--skip-compile")
        if java_debug:
            cmd.append("--java-debug")
        if java_no_strict:
            cmd.append("--java-no-strict")
        if java_guided:
            cmd.append("--java-guided")
        if java_guided_model_path:
            cmd.extend(["--java-guided-model-path", java_guided_model_path])
        if java_guided_python_exec:
            cmd.extend(["--java-guided-python-exec", java_guided_python_exec])

        if generation_mode == "raw" or extract_after:
            cmd.append("--no-extract-state")

        generation_plans.append(
            CommandPlan(
                worker_id=shard.worker_id,
                command=cmd,
                cwd=repo_root,
                log_path=shard.log_path,
                label="generate",
            )
        )

    started = time.time()
    print(f"[step] Launching generation workers ({len(generation_plans)})...")
    _launch_parallel_commands(generation_plans, dry_run=dry_run)

    if dry_run:
        return RunResult(
            elapsed_seconds=time.time() - started,
            workers_used=len(generation_plans),
            games_requested=num_games,
            shard_line_counts={},
            merged_line_count=0,
            merged_output=output_path,
            run_dir=run_dir,
        )

    shard_line_counts = _validate_shards(raw_shards, require_non_empty=True)

    raw_merged_output = output_path if not extract_after else run_dir / "merged_raw.jsonl"
    merged_raw_lines = _merge_jsonl(
        shard_paths=[s.output_path for s in sorted(raw_shards, key=lambda s: s.worker_id)],
        merged_output=raw_merged_output,
    )
    expected_raw_lines = sum(shard_line_counts.values())
    if merged_raw_lines != expected_raw_lines:
        raise RuntimeError(
            f"Raw merge count mismatch: merged={merged_raw_lines} shard_sum={expected_raw_lines}"
        )
    print(
        f"[merge] raw merged lines={merged_raw_lines} output={raw_merged_output}"
    )

    final_line_count = merged_raw_lines
    final_output = raw_merged_output

    if extract_after:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        extract_worker_count = min(
            extract_workers or workers,
            len(raw_shards),
        )
        extract_worker_count = max(1, extract_worker_count)

        extract_shards = [
            WorkerShard(
                worker_id=shard.worker_id,
                games=shard.games,
                seed=shard.seed,
                output_path=extracted_dir / f"extracted_worker_{shard.worker_id:02d}.jsonl",
                log_path=logs_dir / f"extract_worker_{shard.worker_id:02d}.log",
            )
            for shard in raw_shards
        ]

        extract_plans: List[CommandPlan] = []
        script_path = Path(__file__).resolve()
        for shard in sorted(raw_shards, key=lambda s: s.worker_id):
            target = [s for s in extract_shards if s.worker_id == shard.worker_id][0]
            cmd = [
                python_exec,
                str(script_path),
                "extract-shard",
                "--repo-root",
                str(repo_root),
                "--input",
                str(shard.output_path),
                "--output",
                str(target.output_path),
            ]
            extract_plans.append(
                CommandPlan(
                    worker_id=target.worker_id,
                    command=cmd,
                    cwd=repo_root,
                    log_path=target.log_path,
                    label="extract",
                )
            )

        print(f"[step] Launching extraction workers ({len(extract_plans)})...")
        _launch_parallel_commands(
            extract_plans,
            dry_run=False,
            max_parallel=extract_worker_count,
        )

        extracted_line_counts = _validate_shards(extract_shards, require_non_empty=True)
        merged_extracted_lines = _merge_jsonl(
            shard_paths=[s.output_path for s in sorted(extract_shards, key=lambda s: s.worker_id)],
            merged_output=output_path,
        )
        expected_extracted_lines = sum(extracted_line_counts.values())
        if merged_extracted_lines != expected_extracted_lines:
            raise RuntimeError(
                "Extracted merge count mismatch: "
                f"merged={merged_extracted_lines} shard_sum={expected_extracted_lines}"
            )

        final_line_count = merged_extracted_lines
        final_output = output_path
        print(
            f"[merge] extracted merged lines={merged_extracted_lines} output={output_path}"
        )

    elapsed = time.time() - started
    games_per_min = (num_games / elapsed) * 60.0 if elapsed > 0 else 0.0
    print(
        f"[done] games={num_games} workers={workers} elapsed={elapsed:.1f}s "
        f"throughput={games_per_min:.2f} games/min final_output={final_output}"
    )

    if not keep_shards:
        try:
            shutil.rmtree(raw_dir)
            if extract_after and extracted_dir.exists():
                shutil.rmtree(extracted_dir)
        except Exception:
            print("[warn] Could not clean shard directories; leaving run artifacts in place")

    return RunResult(
        elapsed_seconds=elapsed,
        workers_used=workers,
        games_requested=num_games,
        shard_line_counts=shard_line_counts,
        merged_line_count=final_line_count,
        merged_output=final_output,
        run_dir=run_dir,
    )


def _parse_workers_csv(value: str) -> List[int]:
    out: List[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        workers = int(stripped)
        if workers < 1:
            raise ValueError("worker list values must be >= 1")
        out.append(workers)
    if not out:
        raise ValueError("worker list must not be empty")
    return out


def _add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--generator-path", default="ai/selfplay/selfplay_trajectory_generator.py")
    parser.add_argument("--output", required=True, help="Final merged JSONL output path")
    parser.add_argument("--work-dir", default="ai/selfplay/data/parallel_runs")
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seed-stride", type=int, default=SEED_STRIDE_DEFAULT)
    parser.add_argument("--java-max-turns", type=int, default=1200)
    parser.add_argument("--java-debug", action="store_true")
    parser.add_argument("--java-no-strict", action="store_true")
    parser.add_argument("--java-guided", action="store_true")
    parser.add_argument("--java-guided-model-path", default=None)
    parser.add_argument("--java-guided-python-exec", default=None)
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--no-precompile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generation-mode",
        choices=["raw", "inline"],
        default="raw",
        help="raw=fast generation without extraction; inline=attach extracted state during generation",
    )
    parser.add_argument(
        "--extract-after",
        action="store_true",
        help="Run extraction as a second stage after raw shard generation",
    )
    parser.add_argument("--extract-workers", type=int, default=None)
    parser.add_argument("--keep-shards", action="store_true")


def _resolve_path(repo_root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _cmd_extract_shard(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    start = time.time()
    count = _enrich_raw_file(input_path=input_path, output_path=output_path, repo_root=repo_root)
    elapsed = time.time() - start
    print(f"extracted_rows={count} elapsed={elapsed:.2f}s input={input_path} output={output_path}")
    return 0


def _execute_run_like(args: argparse.Namespace) -> RunResult:
    repo_root = Path(args.repo_root).resolve()
    generator_path = _resolve_path(repo_root, args.generator_path)
    output_path = _resolve_path(repo_root, args.output)
    work_dir = _resolve_path(repo_root, args.work_dir)

    workers = args.workers if args.workers is not None else _recommended_workers()

    if args.extract_after and args.generation_mode != "raw":
        raise ValueError("--extract-after requires --generation-mode raw")

    return _run_pipeline(
        repo_root=repo_root,
        generator_path=generator_path,
        output_path=output_path,
        num_games=args.num_games,
        workers=workers,
        seed=args.seed,
        seed_stride=args.seed_stride,
        java_max_turns=args.java_max_turns,
        java_debug=bool(args.java_debug),
        java_no_strict=bool(args.java_no_strict),
        java_guided=bool(args.java_guided),
        java_guided_model_path=args.java_guided_model_path,
        java_guided_python_exec=args.java_guided_python_exec,
        no_precompile=bool(args.no_precompile),
        dry_run=bool(args.dry_run),
        generation_mode=str(args.generation_mode),
        extract_after=bool(args.extract_after),
        extract_workers=args.extract_workers,
        work_dir=work_dir,
        keep_shards=bool(args.keep_shards),
        python_exec=str(args.python_exec),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    _ = _execute_run_like(args)
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    if args.output is None:
        args.output = str(repo_root / "ai/selfplay/data/parallel_smoke_merged.jsonl")
    if args.workers is None:
        args.workers = 2
    if args.num_games is None:
        args.num_games = 2
    if args.java_max_turns is None:
        args.java_max_turns = 120
    _ = _execute_run_like(args)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    worker_options = _parse_workers_csv(args.worker_options)
    base_output_dir = _resolve_path(repo_root, args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    print(
        f"[benchmark] games={args.num_games} worker_options={worker_options} "
        f"repetitions={args.repetitions}"
    )

    for workers in worker_options:
        for rep in range(args.repetitions):
            out = base_output_dir / f"bench_{workers}w_rep{rep + 1}.jsonl"
            run_args = argparse.Namespace(**vars(args))
            run_args.output = str(out)
            run_args.workers = workers
            run_args.generation_mode = "raw"
            run_args.extract_after = False
            run_args.extract_workers = None
            run_args.java_max_turns = args.java_max_turns
            run_args.no_precompile = bool(args.no_precompile)
            run_args.dry_run = bool(args.dry_run)

            started = time.time()
            result = _execute_run_like(run_args)
            elapsed = time.time() - started if args.dry_run else result.elapsed_seconds
            gpm = (args.num_games / elapsed) * 60.0 if elapsed > 0 else 0.0
            row = {
                "workers": workers,
                "rep": rep + 1,
                "games": args.num_games,
                "elapsed_seconds": round(elapsed, 3),
                "games_per_minute": round(gpm, 3),
                "output": str(out),
            }
            rows.append(row)
            print(
                f"[benchmark] workers={workers} rep={rep + 1} elapsed={elapsed:.1f}s "
                f"throughput={gpm:.2f} games/min"
            )

    if not args.dry_run:
        summary_path = base_output_dir / "benchmark_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"[benchmark] wrote summary to {summary_path}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel local self-play runner with shard/merge validation and optional two-stage extraction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one parallel generation job")
    _add_shared_run_args(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    smoke_parser = subparsers.add_parser("smoke", help="Tiny smoke run (defaults: 2 games, 2 workers)")
    smoke_parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    smoke_parser.add_argument("--generator-path", default="ai/selfplay/selfplay_trajectory_generator.py")
    smoke_parser.add_argument("--output", default=None)
    smoke_parser.add_argument("--work-dir", default="ai/selfplay/data/parallel_runs")
    smoke_parser.add_argument("--num-games", type=int, default=None)
    smoke_parser.add_argument("--workers", type=int, default=None)
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--seed-stride", type=int, default=SEED_STRIDE_DEFAULT)
    smoke_parser.add_argument("--java-max-turns", type=int, default=None)
    smoke_parser.add_argument("--java-debug", action="store_true")
    smoke_parser.add_argument("--java-no-strict", action="store_true")
    smoke_parser.add_argument("--java-guided", action="store_true")
    smoke_parser.add_argument("--java-guided-model-path", default=None)
    smoke_parser.add_argument("--java-guided-python-exec", default=None)
    smoke_parser.add_argument("--python-exec", default=sys.executable)
    smoke_parser.add_argument("--no-precompile", action="store_true")
    smoke_parser.add_argument("--dry-run", action="store_true")
    smoke_parser.add_argument("--generation-mode", choices=["raw", "inline"], default="raw")
    smoke_parser.add_argument("--extract-after", action="store_true")
    smoke_parser.add_argument("--extract-workers", type=int, default=None)
    smoke_parser.add_argument("--keep-shards", action="store_true")
    smoke_parser.set_defaults(func=_cmd_smoke)

    bench_parser = subparsers.add_parser("benchmark", help="Run throughput benchmark for worker-count options")
    bench_parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    bench_parser.add_argument("--generator-path", default="ai/selfplay/selfplay_trajectory_generator.py")
    bench_parser.add_argument("--output-dir", default="ai/selfplay/data/parallel_benchmark")
    bench_parser.add_argument("--work-dir", default="ai/selfplay/data/parallel_runs")
    bench_parser.add_argument("--num-games", type=int, default=12)
    bench_parser.add_argument("--worker-options", default="2,4,6")
    bench_parser.add_argument("--repetitions", type=int, default=1)
    bench_parser.add_argument("--seed", type=int, default=200)
    bench_parser.add_argument("--seed-stride", type=int, default=SEED_STRIDE_DEFAULT)
    bench_parser.add_argument("--java-max-turns", type=int, default=300)
    bench_parser.add_argument("--java-debug", action="store_true")
    bench_parser.add_argument("--java-no-strict", action="store_true")
    bench_parser.add_argument("--java-guided", action="store_true")
    bench_parser.add_argument("--java-guided-model-path", default=None)
    bench_parser.add_argument("--java-guided-python-exec", default=None)
    bench_parser.add_argument("--python-exec", default=sys.executable)
    bench_parser.add_argument("--no-precompile", action="store_true")
    bench_parser.add_argument("--dry-run", action="store_true")
    bench_parser.add_argument("--keep-shards", action="store_true")
    bench_parser.add_argument("--extract-workers", type=int, default=None)
    bench_parser.add_argument("--generation-mode", choices=["raw", "inline"], default="raw")
    bench_parser.add_argument("--extract-after", action="store_true")
    bench_parser.set_defaults(func=_cmd_benchmark)

    extract_parser = subparsers.add_parser("extract-shard", help=argparse.SUPPRESS)
    extract_parser.add_argument("--repo-root", required=True)
    extract_parser.add_argument("--input", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.set_defaults(func=_cmd_extract_shard)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
