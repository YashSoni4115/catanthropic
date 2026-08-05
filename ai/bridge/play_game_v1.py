"""Play one full Catan game with the decision layer as the AI player.

Usage::

    python -m ai.bridge.play_game_v1 \\
        --ai-player 0 --seed 42 --max-turns 400

Options:
    --ai-player    Java player index (0-3) the decision layer controls.
    --seed         Optional deterministic seed for the Java engine.
    --games N      Play N games sequentially; prints win-rate summary.
    --quiet        Suppress per-turn trace (still prints final summary).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

from ai.bridge.game_bridge_v1 import (
    GameBridge, GameResult, TurnDecision, build_default_scorer,
)
from ai.decision.opening_value_evaluator_v1 import ValueModelOracle


def _print_decision(td: TurnDecision) -> None:
    hs = ",".join(td.fired_heuristics) or "-"
    alts = " | alts: " + ", ".join(
        f"{a['action']}({a['delta']:+.3f})" for a in td.top_alternatives
    ) if td.top_alternatives else ""
    sys.stdout.write(
        f"[turn {td.turn:>3} {td.phase:<6}] {td.player:>3} "
        f"→ {td.chosen.short():<32} "
        f"p(win) {td.win_prob_before:.3f}→{td.win_prob_after:.3f} "
        f"Δ={td.delta:+.4f}  heur:{hs}{alts}\n"
    )
    sys.stdout.flush()


def _print_result(result: GameResult) -> None:
    print("-" * 72)
    print(f"winner:            {result.winner!r}")
    print(f"turns:             {result.turns}")
    print(f"AI player:         {result.ai_player}")
    print(f"AI won:            {result.ai_won}")
    print(f"termination:       {result.termination_reason}")
    print(f"fallbacks:         {result.fallbacks} (engine-rejected or malformed)")
    print(f"decisions logged:  {len(result.decisions)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Play interactive Catan game(s) with the decision layer.")
    ap.add_argument("--ai-player", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--model-dir", default="ai/models/value_model_economy_v1")
    ap.add_argument("--model-name", default="hgb")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"[setup] loading value model from {args.model_dir}/model_{args.model_name}.pkl ...")
    scorer = build_default_scorer(args.model_dir, args.model_name)

    results: List[GameResult] = []
    t0 = time.time()
    for g in range(args.games):
        print(f"\n{'='*72}\n[GAME {g+1}/{args.games}] ai-slot=index{args.ai_player} seed={args.seed}")
        bridge = GameBridge(
            scorer=scorer,
            ai_player_index=args.ai_player,
            max_turns=args.max_turns,
            seed=None if args.seed is None else args.seed + g,
            trace_callback=None if args.quiet else _print_decision,
        )
        result = bridge.play()
        _print_result(result)
        results.append(result)

    elapsed = time.time() - t0
    wins = sum(1 for r in results if r.ai_won)
    total = len(results)
    print("\n" + "=" * 72)
    print(f"[SUMMARY] {wins}/{total} wins  ({wins*100.0/total if total else 0.0:.1f}%)  "
          f"| avg turns: {sum(r.turns for r in results)/max(total,1):.1f}  "
          f"| avg fallbacks/game: {sum(r.fallbacks for r in results)/max(total,1):.2f}  "
          f"| elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
