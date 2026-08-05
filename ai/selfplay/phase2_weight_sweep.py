"""Bounded weight sweep for the scaffolding heuristic build_vp_prior.

Runs a FIXED sweep over exactly three weights {0.06, 0.08, 0.10} with a
SHARED seed set (10 games each). Committed decision procedure:

    * If ANY weight >= 25% win rate -> GATE PASSED, recommend Phase 3 at
      the LOWEST passing weight.
    * If ALL THREE < 25% -> GATE FAILED conclusively, Path A exhausted.

Does not modify any production code; only orchestrates
phase2_gate_batch.py invocations and aggregates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (0.06, 0.08, 0.10)


def run_batch(weight: float, games: int, seed_base: int, max_turns: int,
              out_path: Path, log_path: Path) -> dict:
    cmd = [
        sys.executable, "-m", "ai.selfplay.phase2_gate_batch",
        "--games", str(games),
        "--seed-base", str(seed_base),
        "--max-turns", str(max_turns),
        "--vp-prior-weight", f"{weight}",
        "--out", str(out_path),
    ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              stdout=logf, stderr=subprocess.STDOUT)
    with open(out_path) as f:
        return json.load(f)


def _agg_action_mix(per_game: List[dict]) -> Dict[str, float]:
    keys = set()
    for g in per_game:
        keys.update((g.get("ai_action_mix") or {}).keys())
    n = max(len(per_game), 1)
    return {k: sum((g.get("ai_action_mix") or {}).get(k, 0) for g in per_game) / n
            for k in sorted(keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--gate-threshold", type=float, default=0.25)
    ap.add_argument("--out-dir", default="data/phase2_weight_sweep")
    args = ap.parse_args()

    weights = DEFAULT_WEIGHTS
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    est_min = 12 * len(weights)
    print(f"[sweep] weights={weights} games={args.games} seed-base={args.seed_base} "
          f"max-turns={args.max_turns} gate={args.gate_threshold:.0%}")
    print(f"[sweep] estimated runtime ~{est_min} min ({est_min//len(weights)} min/weight)")

    per_weight: Dict[float, dict] = {}
    t0 = time.time()
    for w in weights:
        tag = f"w{w:.2f}"
        out = out_dir / f"sweep_{tag}.json"
        log = out_dir / f"sweep_{tag}.log"
        print(f"\n[sweep] === running weight={w} (log={log.name}) ===")
        ts = time.time()
        data = run_batch(w, args.games, args.seed_base, args.max_turns, out, log)
        elapsed = time.time() - ts
        summ = data["summary"]
        mix = _agg_action_mix(data.get("per_game", []))
        per_weight[w] = {"summary": summ, "action_mix": mix, "log": str(log),
                         "out": str(out), "elapsed_s": elapsed}
        print(f"[sweep] weight={w} win_rate={summ['win_rate']:.1%} "
              f"({summ['wins']}/{summ['games']})  "
              f"avg_turns={summ['avg_turns']:.1f}  "
              f"truncations={summ['truncations']}  "
              f"avg_fallbacks={summ['avg_fallbacks']:.2f}  "
              f"elapsed={elapsed:.0f}s")
        print(f"[sweep] weight={w} avg_action_mix={ {k: round(v, 2) for k, v in mix.items()} }")

    # ------------- Aggregate + decision -------------
    print("\n" + "=" * 78)
    print(f"[SWEEP SUMMARY]  seeds {args.seed_base}..{args.seed_base+args.games-1}  "
          f"gate={args.gate_threshold:.0%}  total elapsed={time.time()-t0:.0f}s")
    print("-" * 78)
    print(f"{'weight':>8} | {'win_rate':>9} | {'truncs':>6} | "
          f"{'avg_set':>7} | {'avg_city':>8} | {'avg_road':>8} | "
          f"{'avg_trade':>9} | {'avg_dev':>7} | {'avg_end':>7}")
    print("-" * 78)
    rows = []
    for w in weights:
        s = per_weight[w]["summary"]
        m = per_weight[w]["action_mix"]
        row = {
            "weight": w,
            "win_rate": s["win_rate"],
            "truncations": s["truncations"],
            "avg_settlement": m.get("build_settlement", 0.0),
            "avg_city": m.get("build_city", 0.0),
            "avg_road": m.get("build_road", 0.0),
            "avg_trade_bank": m.get("trade_bank", 0.0),
            "avg_buy_dev": m.get("buy_dev_card", 0.0),
            "avg_end_turn": m.get("end_turn", 0.0),
        }
        rows.append(row)
        print(f"{w:>8.2f} | {s['win_rate']:>8.1%} | {s['truncations']:>6d} | "
              f"{row['avg_settlement']:>7.2f} | {row['avg_city']:>8.2f} | "
              f"{row['avg_road']:>8.2f} | {row['avg_trade_bank']:>9.2f} | "
              f"{row['avg_buy_dev']:>7.2f} | {row['avg_end_turn']:>7.2f}")
    print("-" * 78)

    passers = [(w, per_weight[w]["summary"]["win_rate"])
               for w in weights
               if per_weight[w]["summary"]["win_rate"] >= args.gate_threshold]
    best_w = max(weights, key=lambda w: per_weight[w]["summary"]["win_rate"])
    best_wr = per_weight[best_w]["summary"]["win_rate"]

    decision = {}
    if passers:
        winning_weight = min(w for w, _ in passers)  # lowest passing weight
        wr = per_weight[winning_weight]["summary"]["win_rate"]
        line = (f"GATE PASSED at weight={winning_weight} (win rate {wr:.1%}) — "
                f"recommend Phase 3 retrain at weight={winning_weight}")
        decision = {"outcome": "PASSED", "weight": winning_weight, "win_rate": wr}
        print(f"\n>>> {line} <<<")
    else:
        line = (f"GATE FAILED — all weights {list(weights)} below {args.gate_threshold:.0%} "
                f"(best was {best_wr:.1%} at weight={best_w}). Path A exhausted. "
                f"Recommend moving to policy-gradient RL (Path B).")
        decision = {"outcome": "FAILED", "best_weight": best_w, "best_win_rate": best_wr}
        print(f"\n>>> {line} <<<")

    # Persist aggregate
    agg = {
        "config": vars(args),
        "weights": list(weights),
        "per_weight": {str(w): per_weight[w] for w in weights},
        "table": rows,
        "decision": decision,
    }
    agg_path = out_dir / "sweep_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"[sweep] wrote {agg_path}")
    return 0 if passers else 2


if __name__ == "__main__":
    raise SystemExit(main())
