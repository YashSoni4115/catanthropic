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
B313 = "checkpoints/ppo_B313.pt"
# key -> (opponent agent name, default seeds). Opponents fill seats 1-3; the checkpoint is slot 0.
LINEUPS = {"vs_vf": ("value_function", 250), "vs_ab": ("alpha_beta", 125), "vs_wr": ("weighted_random", 125),
           "vs_bc": ("bc_policy", 125), "vs_b313": (f"ppo:{os.path.abspath(B313)}", 125),
           "vs_pg": ("phi_greedy", 125)}


def agent_name(ckpt: str) -> str:
    if ckpt == "bc":
        return "bc_policy"
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    return f"ppo:{os.path.abspath(ckpt)}"


def evaluate(ckpts, workers=8, seeds=None, seed_base=SEED_BASE, which=("vs_vf", "vs_wr", "vs_bc"), on_set=None) -> dict:
    """seeds: override per-lineup seed counts. on_set(out): called after each opponent set (partial saves)."""
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
        if on_set:
            on_set(out)
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
    ap.add_argument("--only", default="vs_vf,vs_wr,vs_bc", help=f"comma-separated subset of {','.join(LINEUPS)}")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--seeds", default="", help="per-set overrides, e.g. vs_vf=250,vs_ab=125")
    ap.add_argument("--estimate", action="store_true", help="play 1 seed per set/checkpoint and extrapolate wall-clock")
    a = ap.parse_args(argv)
    which = tuple(a.only.split(","))
    ck = a.ckpts.split(",")
    seeds = {k: int(v) for k, v in (s.split("=") for s in a.seeds.split(",") if s)}
    if a.estimate:
        return estimate(ck, which, seeds, a.seed_base, a.workers)
    os.makedirs(a.out, exist_ok=True)
    save = lambda o: json.dump({"seed_base": a.seed_base, "results": o}, open(f"{a.out}/results.json", "w"), indent=1)
    out = evaluate(ck, a.workers, seeds, a.seed_base, which, on_set=lambda o: (save(o), print(fmt(o), flush=True)))
    save(out)
    print("FINAL\n" + fmt(out), flush=True)
    return out


def estimate(ckpts, which, seeds, seed_base, workers) -> dict:
    """Time 1 seed (4 games, serial) per set and checkpoint; extrapolate to the full set with `workers` in parallel."""
    ckpts = (["bc"] if "bc" not in ckpts else []) + list(ckpts)
    est = {}
    for key in which:
        opp, n = LINEUPS[key]
        n = seeds.get(key, n)
        t = time.time()
        for c in ckpts:
            run_match([agent_name(c), opp, opp, opp], 1, seed_base + 10_000, 1)
        per_game = (time.time() - t) / (4 * len(ckpts))
        est[key] = {"s_per_game_serial": per_game, "games": 4 * n * len(ckpts),
                    "est_minutes": per_game * 4 * n * len(ckpts) / workers / 60}
        print(f"{key:8s} {per_game:6.2f} s/game serial -> {est[key]['games']} games ~ {est[key]['est_minutes']:.0f} min "
              f"on {workers} workers", flush=True)
    print(f"total ~ {sum(e['est_minutes'] for e in est.values()):.0f} min", flush=True)
    return est


if __name__ == "__main__":
    main()
