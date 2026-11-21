"""Label-policy study + high-N truth set.

    python -m catan_rl.winprob.label_study --data runs/winprob/pilot --states 1000 --n 16 --workers 8 --out DIR
    python -m catan_rl.winprob.label_study --data runs/winprob/pilot --truth 300 --truth-n 200 --policy weighted_random --out DIR

Held-out states come from the test split (disjoint from the truth set when both are drawn: the
study takes the first ``--states`` test states in a seeded shuffle, the truth set the last ``--truth``).
Both policies use the same determinization seeds per (state id, rollout index), so the comparison is paired.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from catan_rl.winprob.labeler import StateSpec, label_states


def spearman(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float) for x in (a, b))
    # average ranks for ties
    for x, r in ((a, ra), (b, rb)):
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def test_specs(data: str):
    d = np.load(os.path.join(data, "dataset.npz"))
    m = d["split"] == "test"
    specs = [StateSpec(int(s), int(k), str(p), str(v)) for s, k, p, v in
             zip(d["seed"][m], d["decision"][m], d["phase"][m], d["viewer"][m])]
    random.Random("label-study").shuffle(specs)
    return specs


def run_policy(specs, policy, n, workers):
    t = time.perf_counter()
    rows, skipped = label_states(specs, policy, n, workers)
    return rows, skipped, time.perf_counter() - t


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/winprob/pilot")
    p.add_argument("--states", type=int, default=1000)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--truth", type=int, default=0)
    p.add_argument("--truth-n", type=int, default=200)
    p.add_argument("--policy", default="weighted_random", help="policy for the truth set")
    p.add_argument("--early-check", type=int, default=0, help="label this many early truth-pool states (wr N=200, vf N=--early-n)")
    p.add_argument("--early-n", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="runs/winprob/label_study")
    p.add_argument("--keep-awake", action="store_true")
    a = p.parse_args(argv)
    if a.keep_awake:
        from catan_rl.eval.run import keep_awake
        keep_awake()
    os.makedirs(a.out, exist_ok=True)
    specs = test_specs(a.data)

    if a.early_check:
        # early states from the truth pool (the last 1000 shuffled test states, i.e. the extended truth set)
        pool = [s for s in specs[-1000:] if s.phase == "early"][-a.early_check:]
        out = {}
        for pol, n in (("weighted_random", a.truth_n), ("value_function", a.early_n)):
            rows, skipped, el = run_policy(pool, pol, n, a.workers)
            out[pol] = ({r["spec"].id: r["wins"] for r in rows}, el)
        ids = sorted(set(out["weighted_random"][0]) & set(out["value_function"][0]))
        pv = lambda pol: np.array([out[pol][0][i][0] / out[pol][0][i].sum() for i in ids])
        np.savez_compressed(os.path.join(a.out, "early_check.npz"), state_id=np.array(ids),
                            p_wr=pv("weighted_random"), p_vf=pv("value_function"))
        s = {"states": len(ids), "wr_n": a.truth_n, "vf_n": a.early_n,
             "seconds": {pol: el for pol, (_, el) in out.items()}}
        json.dump(s, open(os.path.join(a.out, "early_check_summary.json"), "w"), indent=2)
        print(json.dumps(s, indent=2))
        return s

    if a.truth:
        ts = specs[-a.truth:]
        rows, skipped, el = run_policy(ts, a.policy, a.truth_n, a.workers)
        np.savez_compressed(os.path.join(a.out, "truth_set.npz"),
                            state_id=np.array([r["spec"].id for r in rows]), wins=np.stack([r["wins"] for r in rows]),
                            no_winner=np.array([r["no_winner"] for r in rows]), policy=np.array(a.policy), n=np.array(a.truth_n))
        s = {"truth_states": len(rows), "skipped": skipped, "n": a.truth_n, "policy": a.policy, "elapsed_s": el,
             "s_per_state_wall": el / len(rows)}
        json.dump(s, open(os.path.join(a.out, "truth_summary.json"), "w"), indent=2)
        print(json.dumps(s, indent=2))
        return s

    ss = specs[:a.states]
    res = {pol: run_policy(ss, pol, a.n, a.workers) for pol in ("weighted_random", "value_function")}
    ids = set.intersection(*({r["spec"].id for r in rs} for rs, _, _ in res.values()))
    P = {pol: {r["spec"].id: r["wins"][0] / a.n for r in rs} for pol, (rs, _, _) in res.items()}
    phase = {s.id: s.phase for s in ss}
    order = sorted(ids)
    wr, vf = (np.array([P[pol][i] for i in order]) for pol in ("weighted_random", "value_function"))
    ph = np.array([phase[i] for i in order])
    noise = np.sqrt(np.mean(wr * (1 - wr) / a.n + vf * (1 - vf) / a.n))
    s = {
        "states": len(order), "n": a.n,
        "spearman_viewer_winprob": spearman(wr, vf),
        "pearson_viewer_winprob": float(np.corrcoef(wr, vf)[0, 1]),
        "mean_abs_diff": float(np.abs(wr - vf).mean()),
        "mean_diff_vf_minus_wr": float((vf - wr).mean()),
        "expected_abs_diff_from_noise_only": float(noise * np.sqrt(2 / np.pi)),
        "by_phase": {p_: {"states": int((ph == p_).sum()), "mean_abs_diff": float(np.abs(wr - vf)[ph == p_].mean()),
                          "spearman": spearman(wr[ph == p_], vf[ph == p_]),
                          "mean_wr": float(wr[ph == p_].mean()), "mean_vf": float(vf[ph == p_].mean())}
                     for p_ in ("early", "mid", "late")},
        "time": {pol: {"elapsed_s": el, "s_per_state_wall": el / len(rs), "workers": a.workers,
                       "skipped": len(sk)} for pol, (rs, sk, el) in res.items()},
        "no_winner_fraction": {pol: float(sum(r["no_winner"] for r in rs) / (len(rs) * a.n)) for pol, (rs, _, _) in res.items()},
    }
    json.dump(s, open(os.path.join(a.out, "study_summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.out, "study_labels.npz"), state_id=np.array(order), wr=wr, vf=vf, phase=ph)
    print(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    main()
