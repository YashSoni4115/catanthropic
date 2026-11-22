"""PPO fine-tuning of bc_policy against a league, optional potential-based shaping with Φ.

python -m catan_rl.rl.train --out runs/rl/A --shaping off --updates 300 --games-per-update 64 --workers 8 --seed 0
Outputs in --out: config.json (incl. Φ version stamp, git hashes), log.jsonl (one line per update),
eval.csv (update, win rate vs 3 value_function, clustered CI), checkpoints ckpt_XXXX.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from multiprocessing import get_context

import numpy as np
import torch

from catan_rl.rl import rollout as R
from catan_rl.rl.net import BC_PATH, ActorCritic
from catan_rl.rl.ppo import Batch, PPOConfig, update

TRAIN_SEED_BASE = 1_000_000  # disjoint from eval (0..), winprob (0..2999), diagnostics (100000..), BC (200000..)


def evaluate(ckpt: str, n_seeds: int, workers: int) -> dict:
    """Checkpoint (argmax) vs 3 value_function on the standard eval seeds 0..n_seeds-1 (x4 rotations)."""
    from catan_rl.eval.match import clustered_win_rate, run_match
    res = run_match([f"ppo:{os.path.abspath(ckpt)}", "value_function", "value_function", "value_function"],
                    n_seeds, 0, workers)
    m = clustered_win_rate(res, [0])
    return {"win_rate": m["p"], "ci_lo": m["ci95"][0], "ci_hi": m["ci95"][1], "games": len(res)}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--games-per-update", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shaping", choices=["on", "off"], default="off")
    ap.add_argument("--shaping-coef", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--ent-coef", type=float, default=0.005, help="annealed linearly to 0")
    ap.add_argument("--kl-bc-coef", type=float, default=0.2)
    ap.add_argument("--kl-bc-decay", type=float, default=0.99, help="coef * decay^update")
    ap.add_argument("--ckpt-every", type=int, default=10, help="add a league checkpoint every K updates")
    ap.add_argument("--eval-every", type=int, default=25, help="0 = no eval")
    ap.add_argument("--eval-seeds", type=int, default=63, help="x4 rotations (63 -> 252 games)")
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--belief-features", action="store_true")
    ap.add_argument("--bc", default=BC_PATH)
    return ap.parse_args(argv)


def _git(path):
    import subprocess
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main(argv=None) -> dict:
    a = parse(argv)
    if a.belief_features:
        raise SystemExit("--belief-features: extraction exists (catan_rl/rl/belief_features.py); training with it is not enabled yet")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net, bc = ActorCritic.from_bc(a.bc, a.seed), ActorCritic.from_bc(a.bc, a.seed)
    for p in bc.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    from catan_rl.winprob.phi import load_default
    phi_version = load_default().version
    cfg = PPOConfig(epochs=a.epochs, minibatch=a.minibatch)
    config = {**vars(a), "phi_version": phi_version, "git": _git("."),
              "fork_git": _git(os.path.expanduser("~/Desktop/catanatron-fork")), "train_seed_base": TRAIN_SEED_BASE}
    json.dump(config, open(f"{a.out}/config.json", "w"), indent=1)
    dims = (net.ds, net.da, net.norm)
    past = [net.params()]  # league checkpoints; index 0 = BC init
    log = open(f"{a.out}/log.jsonl", "a")
    ecsv = f"{a.out}/eval.csv"

    def do_eval(u):
        path = f"{a.out}/ckpt_{u:04d}.pt"
        net.save(path, {"update": u, "phi_version": phi_version, "shaping": a.shaping})
        if a.eval_every:
            e = evaluate(path, a.eval_seeds, a.eval_workers)
            new = not os.path.exists(ecsv)
            with open(ecsv, "a") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["update", "win_rate", "ci_lo", "ci_hi", "games"])
                w.writerow([u, e["win_rate"], e["ci_lo"], e["ci_hi"], e["games"]])
            print(f"eval update {u}: vs 3 value_function {e['win_rate']:.3f} [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}] ({e['games']} games)", flush=True)

    do_eval(0)
    n_games = 0
    with get_context("spawn").Pool(a.workers, initializer=R.init_worker) as pool:
        for u in range(1, a.updates + 1):
            params = net.params()
            jobs = []
            for i in range(a.games_per_update):
                seed = TRAIN_SEED_BASE + a.seed * 1_000_000 + n_games + i
                cat, seats = R.sample_lineup(random.Random(f"league:{a.seed}:{seed}"), len(past))
                pj = {int(s.split(":")[1]): past[int(s.split(":")[1])] for s in seats if s.startswith("past:")}
                jobs.append({"seed": seed, "seats": seats, "category": cat, "params": params, "past": pj, "dims": dims,
                             "shaping": a.shaping == "on", "coef": a.shaping_coef, "gamma": a.gamma,
                             "phi_version": phi_version})
            t = time.perf_counter()
            games = pool.map(R.play_game, jobs, chunksize=1)
            t_roll = time.perf_counter() - t
            n_games += len(games)
            trajs = [tr for g in games for tr in g["trajs"]]
            b = Batch.from_trajs(trajs, a.gamma, a.lam)
            frac = (u - 1) / max(a.updates, 1)
            t = time.perf_counter()
            st = update(net, bc, opt, b, cfg, a.ent_coef * (1 - frac), a.kl_bc_coef * a.kl_bc_decay ** u, rng)
            t_upd = time.perf_counter() - t
            win = {c: [tr["outcome"] for tr in trajs if tr["category"] == c] for c in R.CATEGORIES}
            rec = {"update": u, "games": n_games, "mean_return": float(np.mean([tr["reward"].sum() for tr in trajs])),
                   "win_rate": {c: (float(np.mean(v)) if v else None) for c, v in win.items()},
                   "learner_seats": {c: len(v) for c, v in win.items()}, **st,
                   "games_per_s": len(games) / t_roll, "decisions_per_s": sum(g["decisions"] for g in games) / t_roll,
                   "learner_decisions_per_s": len(b) / t_roll, "learner_steps_per_s": len(b) * cfg.epochs / t_upd,
                   "rollout_s": t_roll, "update_s": t_upd}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            wr = " ".join(f"{c}={v:.3f}" if v is not None else f"{c}=-" for c, v in rec["win_rate"].items())
            print(f"upd {u} ret {rec['mean_return']:.3f} win[{wr}] pl {st['policy_loss']:.4f} vl {st['value_loss']:.4f} "
                  f"ent {st['entropy']:.3f} kl_bc {st['kl_bc']:.4f} kl_prev {st['kl_prev']:.5f} clip {st['clip_frac']:.3f} "
                  f"ev {st['explained_variance']:.3f} | {rec['games_per_s']:.2f} games/s {rec['decisions_per_s']:.0f} dec/s "
                  f"{rec['learner_steps_per_s']:.0f} steps/s (roll {t_roll:.0f}s upd {t_upd:.0f}s)", flush=True)
            if u % a.ckpt_every == 0:
                past.append(net.params())
            if a.eval_every and u % a.eval_every == 0:
                do_eval(u)
    final = f"{a.out}/final.pt"
    net.save(final, {"update": a.updates, "phi_version": phi_version, "shaping": a.shaping})
    digest = __import__("hashlib").sha256(b"".join(v.tobytes() for _, v in sorted(net.params().items()))).hexdigest()
    print(f"final params sha256 {digest}", flush=True)
    return {"digest": digest}


if __name__ == "__main__":
    main()
