"""Final evaluation of PPO checkpoints (argmax, non-peeking).

For each checkpoint ('bc' = the BC policy, else a ppo checkpoint path):
  vs_vf:  checkpoint vs 3 value_function, seeds 5000..5249 x 4 rotations (1,000 games), win rate (seed-clustered
          95% CI) and paired difference vs bc on the same seeds (seed-clustered bootstrap).
  vs_wr:  checkpoint vs 3 weighted_random, 125 seeds (5000..5124) x 4.
  vs_bc:  checkpoint vs 3 bc_policy, 125 seeds x 4 (for 'bc' this is a mirror match, ~0.25 by construction).
All checkpoints use identical seeds. Seeds 5000.. are fresh (training evals used 0..62/0..124).

python -m catan_rl.rl.final_eval --ckpts bc,runs/rl/A_noshape/final.pt --out runs/rl/final_eval
"""
from __future__ import annotations

import argparse
import json
import os
import time

from catan_rl.eval.match import clustered_win_rate, paired_comparison, run_match

SEED_BASE = 5000
LINEUPS = {"vs_vf": ("value_function", 250), "vs_wr": ("weighted_random", 125), "vs_bc": ("bc_policy", 125)}


def agent_name(ckpt: str) -> str:
    if ckpt == "bc":
        return "bc_policy"
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    return f"ppo:{os.path.abspath(ckpt)}"


def evaluate(ckpts, workers=8, seeds=None, seed_base=SEED_BASE, which=("vs_vf", "vs_wr", "vs_bc")) -> dict:
    """seeds: override per-lineup seed counts (tests)."""
    ckpts = list(ckpts)
    if "bc" not in ckpts:
        ckpts = ["bc"] + ckpts  # reference for the paired difference
    out = {}
    for key in which:
        opp, n = LINEUPS[key]
        n = (seeds or {}).get(key, n)
        res = {}
        for c in ckpts:
            t = time.time()
            res[c] = run_match([agent_name(c), opp, opp, opp], n, seed_base, workers)
            m = clustered_win_rate(res[c], [0])
            out.setdefault(c, {})[key] = {"p": m["p"], "ci95": m["ci95"], "games": len(res[c]), "n_seeds": n,
                                          "seeds": [seed_base, seed_base + n - 1], "seconds": time.time() - t}
        for c in ckpts:
            d = paired_comparison(res[c], res["bc"], slot=0)
            out[c][key]["diff_vs_bc"] = {"mean": d["mean_diff"], "ci95": d["ci95"], "n_seeds": d["n_seeds"]}
    return out


def fmt(out) -> str:
    lines = []
    for c, r in out.items():
        for k, m in r.items():
            d = m["diff_vs_bc"]
            lines.append(f"{c:45s} {k:6s} win {m['p']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  "
                         f"diff vs bc {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  ({m['games']} games)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated: 'bc' or checkpoint paths")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/rl/final_eval")
    ap.add_argument("--only", default="vs_vf,vs_wr,vs_bc")
    a = ap.parse_args(argv)
    out = evaluate(a.ckpts.split(","), a.workers, which=tuple(a.only.split(",")))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/results.json", "w"), indent=1)
    print(fmt(out), flush=True)
    return out


if __name__ == "__main__":
    main()
