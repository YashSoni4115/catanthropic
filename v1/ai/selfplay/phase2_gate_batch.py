"""Phase 2 — GATE batch runner.

Runs N games with the decision-layer AI (with the SCAFFOLDING
build_vp_prior heuristic ENABLED) vs 3 guided-random opponents.
Reports win rate, avg turns, truncations, avg fallbacks.

If win rate >= --gate-threshold (default 25%), prints
"GATE PASSED (win rate X%) — proceeding to Phase 3".
Otherwise prints "GATE FAILED (win rate X%) — stopping"
and dumps sample loss traces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from ai.bridge.game_bridge_v1 import (
    GameBridge, GameResult, TurnDecision, build_default_scorer,
)
from ai.decision.heuristics import HeuristicConfig


def _summarize_result(r: GameResult) -> dict:
    from collections import Counter
    ai = r.ai_player
    ai_main = [d for d in r.decisions if d.player == ai and d.phase == "main"]
    action_mix = dict(Counter(d.chosen.type.value for d in ai_main))
    return {
        "winner": r.winner,
        "turns": r.turns,
        "ai_player": r.ai_player,
        "ai_won": r.ai_won,
        "termination_reason": r.termination_reason,
        "fallbacks": r.fallbacks,
        "num_decisions": len(r.decisions),
        "ai_action_mix": action_mix,
    }


def _sample_trace(r: GameResult, n_head: int = 6, n_tail: int = 6) -> List[str]:
    ai = r.ai_player
    ai_decs: List[TurnDecision] = [d for d in r.decisions if d.player == ai and d.phase == "main"]
    lines = []
    for d in ai_decs[:n_head]:
        lines.append(
            f"  t{d.turn:>3} {d.chosen.short():<28} "
            f"Δ={d.delta:+.4f} heur:{','.join(d.fired_heuristics) or '-'}"
        )
    if len(ai_decs) > n_head + n_tail:
        lines.append(f"  ... ({len(ai_decs) - n_head - n_tail} more) ...")
    for d in ai_decs[-n_tail:]:
        lines.append(
            f"  t{d.turn:>3} {d.chosen.short():<28} "
            f"Δ={d.delta:+.4f} heur:{','.join(d.fired_heuristics) or '-'}"
        )
    # Build counts summary
    from collections import Counter
    kinds = Counter(d.chosen.type.value for d in ai_decs)
    lines.append(f"  [action_counts] {dict(kinds)}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--ai-player", type=int, default=0)
    ap.add_argument("--gate-threshold", type=float, default=0.25)
    ap.add_argument("--vp-prior-weight", type=float, default=0.03)
    ap.add_argument("--model-dir", default="ai/models/value_model_economy_v1")
    ap.add_argument("--model-name", default="hgb")
    ap.add_argument("--disable-scaffold", action="store_true",
                    help="Baseline mode: run WITHOUT the scaffolding heuristic.")
    ap.add_argument("--out", default=None, help="Optional JSON dump path.")
    args = ap.parse_args()

    scorer = build_default_scorer(args.model_dir, args.model_name)
    hcfg = HeuristicConfig(
        enable_build_vp_prior=not args.disable_scaffold,
        weight_build_vp_prior=args.vp_prior_weight,
    )
    print(f"[setup] games={args.games} max_turns={args.max_turns} "
          f"scaffold={'ON' if hcfg.enable_build_vp_prior else 'OFF'} "
          f"vp_prior_weight={hcfg.weight_build_vp_prior} "
          f"gate={args.gate_threshold:.0%}")

    results: List[GameResult] = []
    t0 = time.time()
    for g in range(args.games):
        seed = args.seed_base + g
        bridge = GameBridge(
            scorer=scorer,
            ai_player_index=args.ai_player,
            heuristic_config=hcfg,
            max_turns=args.max_turns,
            seed=seed,
            trace_callback=None,
        )
        r = bridge.play()
        results.append(r)
        elapsed = time.time() - t0
        wins = sum(1 for x in results if x.ai_won)
        print(f"[game {g+1:>3}/{args.games}] seed={seed} "
              f"winner={r.winner} turns={r.turns:>3} "
              f"term={r.termination_reason:<12} "
              f"ai_won={r.ai_won} fallbacks={r.fallbacks} "
              f"| cum {wins}/{g+1} ({wins*100.0/(g+1):.1f}%) "
              f"elapsed={elapsed:.0f}s")

    total = len(results)
    wins = sum(1 for r in results if r.ai_won)
    truncations = sum(1 for r in results if r.termination_reason == "max_turns")
    avg_turns = sum(r.turns for r in results) / max(total, 1)
    avg_fallbacks = sum(r.fallbacks for r in results) / max(total, 1)
    win_rate = wins / max(total, 1)

    print("\n" + "=" * 72)
    print(f"[BATCH SUMMARY] games={total} wins={wins} win_rate={win_rate:.1%} "
          f"avg_turns={avg_turns:.1f} truncations={truncations} "
          f"avg_fallbacks={avg_fallbacks:.2f} elapsed={time.time()-t0:.0f}s")

    passed = win_rate >= args.gate_threshold
    if passed:
        print(f"\n>>> GATE PASSED (win rate {win_rate:.1%}) — proceeding to Phase 3 <<<")
    else:
        print(f"\n>>> GATE FAILED (win rate {win_rate:.1%}) — stopping, see loss traces below <<<")
        losses = [r for r in results if not r.ai_won]
        for i, r in enumerate(losses[:5]):
            print(f"\n--- LOSS TRACE {i+1} (winner={r.winner} turns={r.turns} "
                  f"term={r.termination_reason} fallbacks={r.fallbacks}) ---")
            for line in _sample_trace(r):
                print(line)

    # A couple of sample traces regardless
    if passed:
        wins_r = [r for r in results if r.ai_won]
        for i, r in enumerate(wins_r[:2]):
            print(f"\n--- SAMPLE WIN TRACE {i+1} (turns={r.turns}) ---")
            for line in _sample_trace(r):
                print(line)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "summary": {
                    "games": total, "wins": wins, "win_rate": win_rate,
                    "avg_turns": avg_turns, "truncations": truncations,
                    "avg_fallbacks": avg_fallbacks,
                    "gate_passed": passed,
                },
                "per_game": [_summarize_result(r) for r in results],
            }, f, indent=2)
        print(f"[out] wrote {args.out}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
