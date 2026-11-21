"""Build the win-probability dataset.

    python -m catan_rl.winprob.build_dataset --games 3000 --n 16 --policy weighted_random --workers 8 --out DIR

Writes DIR/dataset.npz, DIR/split_summary.json, DIR/build_summary.json, DIR/skipped.json.
Split is game-level: sha256(seed) bucket 0..99 -> train <80, val <90, test otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from catan_rl.winprob.features import names_a
from catan_rl.winprob.labeler import label_game, lineup, select_states

SPLITS = ("train", "val", "test")


def split_of(seed: int) -> str:
    b = int(hashlib.sha256(f"winprob-split:{seed}".encode()).hexdigest(), 16) % 100
    return "train" if b < 80 else "val" if b < 90 else "test"


def _game_job(job):
    seed, policy, n = job
    return label_game(seed, select_states(seed), policy, n)


def entropy_bits(counts: np.ndarray) -> np.ndarray:
    p = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)


def to_arrays(rows, policy: str, n: int) -> dict:
    sp = [r["spec"] for r in rows]
    return {
        "state_id": np.array([s.id for s in sp]), "seed": np.array([s.seed for s in sp], np.int64),
        "lineup": np.array([",".join(lineup(s.seed)) for s in sp]), "decision": np.array([s.decision for s in sp], np.int32),
        "phase": np.array([s.phase for s in sp]), "viewer": np.array([s.viewer for s in sp]),
        "split": np.array([split_of(s.seed) for s in sp]),
        "xa": np.stack([r["xa"] for r in rows]), "xb": np.stack([r["xb"] for r in rows]),
        "xa_names": np.array(names_a()),
        "wins": np.stack([r["wins"] for r in rows]), "no_winner": np.array([r["no_winner"] for r in rows], np.int32),
        "mean_vp": np.stack([r["mean_vp"] for r in rows]),
        "mean_remaining_turns": np.array([r["mean_remaining_turns"] for r in rows], np.float32),
        "label_s": np.array([r["label_s"] for r in rows], np.float32),
        "policy": np.array(policy), "n_rollouts": np.array(n),
    }


def summarize(d: dict, elapsed: float) -> dict:
    n = int(d["n_rollouts"])
    counts = np.concatenate([d["wins"], d["no_winner"][:, None]], axis=1)
    p = d["wins"][:, 0] / n
    return {
        "states": len(p), "games": int(len(set(d["seed"].tolist()))), "rollouts": int(len(p) * n),
        "elapsed_s": elapsed, "rollouts_per_s": len(p) * n / elapsed, "states_per_s": len(p) / elapsed,
        "mean_label_entropy_bits": float(entropy_bits(counts).mean()),
        "no_winner_fraction": float(d["no_winner"].sum() / (len(p) * n)),
        "mean_viewer_win_prob": float(p.mean()),
        "label_noise_sd_mean": float(np.sqrt(p * (1 - p) / n).mean()),  # binomial SE of p_viewer, per state
        "label_noise_sd_rms": float(np.sqrt((p * (1 - p) / n).mean())),
        "by_phase": {ph: {"states": int((d["phase"] == ph).sum()),
                          "mean_viewer_win_prob": float(p[d["phase"] == ph].mean()),
                          "label_noise_sd_rms": float(np.sqrt((p * (1 - p) / n)[d["phase"] == ph].mean()))}
                     for ph in ("early", "mid", "late")},
    }


def split_summary(d: dict) -> dict:
    out = {}
    for s in SPLITS:
        m = d["split"] == s
        out[s] = {"games": int(len(set(d["seed"][m].tolist()))), "states": int(m.sum()),
                  "seeds_sha256": hashlib.sha256(",".join(map(str, sorted(set(d["seed"][m].tolist())))).encode()).hexdigest()}
    out["rule"] = "sha256('winprob-split:{seed}') % 100: <80 train, <90 val, else test"
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--policy", default="weighted_random")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="runs/winprob/pilot")
    p.add_argument("--keep-awake", action="store_true")
    a = p.parse_args(argv)
    if a.keep_awake:
        from catan_rl.eval.run import keep_awake
        keep_awake()
    t = time.perf_counter()
    jobs = [(s, a.policy, a.n) for s in range(a.seed_base, a.seed_base + a.games)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_game_job, jobs, chunksize=4))
    elapsed = time.perf_counter() - t
    rows = [r for rs, _ in res for r in rs]
    skipped = [x for _, sk in res for x in sk]
    d = to_arrays(rows, a.policy, a.n)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "dataset.npz"), **d)
    s = summarize(d, elapsed)
    s.update(skipped=len(skipped), workers=a.workers, policy=a.policy)
    for name, obj in (("build_summary", s), ("split_summary", split_summary(d)), ("skipped", skipped)):
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(obj, f, indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
