"""Evaluate win-probability models on held-out test games, with game-clustered bootstrap CIs.

    python -m catan_rl.winprob.eval --data runs/winprob/pilot --models runs/winprob/models \
        --truth runs/winprob/label_study --out runs/winprob/eval
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from catan_rl.winprob.calibrate import apply_temperature, fit_temperature
from catan_rl.winprob.train import ALLOWED, entropy, load_data, soft_ce

PHASES = ("early", "mid", "late")
BASELINES = ("uniform", "vp_only", "value_function")


# ---------------------------------------------------------------------------- clustered bootstrap
def cluster_mean_ci(x, g, n_boot=2000, seed=0, alpha=0.05):
    """Mean of per-state values x, CI from resampling whole games g (ratio of resampled sums)."""
    _, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, weights=x), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return [float(x.mean()), float(lo), float(hi)]


def cluster_stat_ci(fn, g, n_boot=400, seed=0, alpha=0.05):
    """Any statistic fn(row indices); CI from resampling games."""
    groups = [np.where(g == u)[0] for u in np.unique(g)]
    rng = np.random.default_rng(seed)
    b = [fn(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)]
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return [float(fn(np.arange(len(g)))), float(lo), float(hi)]


def ece(p, q, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == k].mean() - q[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def reliability(p, q, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": k, "n": int((b == k).sum()), "mean_pred": float(p[b == k].mean()), "mean_label": float(q[b == k].mean())}
            for k in range(bins) if (b == k).any()]


# ---------------------------------------------------------------------------- evaluation
def evaluate(P, d, rows, vp_ce=None, truth=None):
    Q, g, ph = d["Q"][rows], d["seed"][rows], d["phase"][rows]
    assert np.allclose(P.sum(1), 1, atol=1e-5)
    ce, H = soft_ce(P, Q), entropy(Q)
    out = {"ce": cluster_mean_ci(ce, g), "excess": cluster_mean_ci(ce - H, g),
           "ece_viewer": cluster_stat_ci(lambda i: ece(P[i, 0], Q[i, 0]), g),
           "by_phase": {p: {"ce": cluster_mean_ci(ce[ph == p], g[ph == p]), "excess": cluster_mean_ci((ce - H)[ph == p], g[ph == p]),
                            "ece_viewer": ece(P[ph == p, 0], Q[ph == p, 0])} for p in PHASES}}
    if vp_ce is not None:
        out["ce_minus_vp_only"] = cluster_mean_ci(ce - vp_ce, g)
        out["by_phase_ce_minus_vp_only"] = {p: cluster_mean_ci((ce - vp_ce)[ph == p], g[ph == p]) for p in PHASES}
    if truth is not None:
        r, pt, tg, tph = truth
        err = np.abs(P[r, 0] - pt)
        out["truth_mae"] = cluster_mean_ci(err, tg)
        out["truth_mae_by_phase"] = {p: cluster_mean_ci(err[tph == p], tg[tph == p]) for p in PHASES if (tph == p).any()}
    return out


def load_truth(path, d, rows):
    t = np.load(os.path.join(path, "truth_set.npz"))
    pos = {s: i for i, s in enumerate(d["state_id"][rows])}
    r = np.array([pos[s] for s in t["state_id"]])
    pt = t["wins"][:, 0] / t["wins"].sum(1)
    return r, pt, d["seed"][rows][r], d["phase"][rows][r]


def vp_lead_table(P, X_vp):
    lead = X_vp[:, 0] - X_vp[:, 1:].max(1)
    out = []
    for v in range(int(lead.min()), int(lead.max()) + 1):
        m = lead == v
        if m.sum() >= 20:
            out.append({"lead": v, "n": int(m.sum()), "mean_p_viewer": float(P[m, 0].mean())})
    mono = all(a["mean_p_viewer"] <= b["mean_p_viewer"] for a, b in zip(out, out[1:]))
    return out, mono


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/winprob/pilot")
    ap.add_argument("--models", default="runs/winprob/models")
    ap.add_argument("--truth", default="runs/winprob/label_study")
    ap.add_argument("--out", default="runs/winprob/eval")
    a = ap.parse_args(argv)
    d = load_data(a.data)
    va, te = (np.where(d["split"] == s)[0] for s in ("val", "test"))
    assert not set(d["seed"][va]) & set(d["seed"][te])
    truth = load_truth(a.truth, d, te)
    names = sorted(f[:-4] for f in os.listdir(os.path.join(a.models, "preds")))
    preds = {n: dict(np.load(os.path.join(a.models, "preds", f"{n}.npz"))) for n in names}
    vp_ce = soft_ce(preds["vp_only"]["test"], d["Q"][te])
    res, cal = {}, {}
    for n in names:
        Pv, Pt = preds[n]["val"], preds[n]["test"]
        T = 1.0 if n == "uniform" else fit_temperature(Pv, d["Q"][va])
        cal[n] = (T, apply_temperature(Pt, T))
        res[n] = {"raw": evaluate(Pt, d, te, vp_ce, truth),
                  "calibrated": {"T": T, **evaluate(cal[n][1], d, te, vp_ce, truth)},
                  "val_ce_calibrated": float(soft_ce(apply_temperature(Pv, T), d["Q"][va]).mean())}
        print(n, "test ce", res[n]["raw"]["ce"][0], flush=True)
    models = [n for n in names if n not in BASELINES]
    best = min(models, key=lambda n: res[n]["val_ce_calibrated"])  # selected on val
    # pairwise feature-set comparisons (test CE differences, clustered)
    pair = {}
    for kind in sorted({n.split("_", 1)[0] for n in models}):
        fam = [n for n in models if n.startswith(kind + "_")]
        for x, y in itertools.combinations(fam, 2):
            pair[f"{x} - {y}"] = cluster_mean_ci(soft_ce(preds[x]["test"], d["Q"][te]) - soft_ce(preds[y]["test"], d["Q"][te]), d["seed"][te])
    from catan_rl.winprob.features import names_a
    ia = names_a().index
    X_vp = d["xa"][te][:, [ia(f"p{i}_vp") for i in range(4)]]
    lead, mono = vp_lead_table(cal[best][1], X_vp)
    import pickle
    m = pickle.load(open(os.path.join(a.models, f"{best}.pkl"), "rb"))
    assert set(m.cols) <= ALLOWED
    r, pt, tg, tph = truth
    label16 = np.abs(d["Q"][te][r, 0] - pt)
    summary = {
        "best_model": best, "results": res, "pairwise_ce_diff": pair,
        "vp_lead_table": lead, "vp_lead_monotone": mono,
        "reliability_best_raw": reliability(preds[best]["test"][:, 0], d["Q"][te][:, 0]),
        "reliability_best_calibrated": reliability(cal[best][1][:, 0], d["Q"][te][:, 0]),
        "truth_reference": {"n_states": len(r), "mae_of_N16_labels_vs_truth": cluster_mean_ci(label16, tg),
                            "truth_label_se_rms": float(np.sqrt((pt * (1 - pt) / 200).mean()))},
        "data": {"test_states": len(te), "test_games": int(len(set(d["seed"][te]))), "val_states": len(va),
                 "no_winner_fraction": d["no_winner_fraction"], "dropped_no_winner_states": d["dropped_no_winner_states"]},
    }
    ec = os.path.join(a.truth, "early_check.npz")
    if os.path.exists(ec):
        e = np.load(ec)
        pos = {s: i for i, s in enumerate(d["state_id"][te])}
        rr = np.array([pos[s] for s in e["state_id"]])
        g = d["seed"][te][rr]
        pb = cal[best][1][rr, 0]
        summary["early_check"] = {"n": len(rr), "model": best,
                                  "mae_vs_wr200": cluster_mean_ci(np.abs(pb - e["p_wr"]), g),
                                  "mae_vs_vf64": cluster_mean_ci(np.abs(pb - e["p_vf"]), g),
                                  "mae_wr200_vs_vf64": cluster_mean_ci(np.abs(e["p_wr"] - e["p_vf"]), g),
                                  "mean_p_wr": float(e["p_wr"].mean()), "mean_p_vf": float(e["p_vf"].mean())}
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for k, lab in (("reliability_best_raw", "raw"), ("reliability_best_calibrated", "calibrated")):
            ax.plot([b["mean_pred"] for b in summary[k]], [b["mean_label"] for b in summary[k]], "o-", label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="predicted P(viewer wins)", ylabel="mean soft label", title=best)
        ax.legend()
        fig.savefig(os.path.join(a.out, "reliability.png"), dpi=120, bbox_inches="tight")
    except ImportError:
        pass
    return summary


if __name__ == "__main__":
    main()
