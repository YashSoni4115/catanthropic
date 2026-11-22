"""Φ: the calibrated win-probability model as a reusable, version-stamped evaluator.

Φ(game, viewer) -> 4 probabilities over relative seats (viewer, then opponents in turn order).
Model: HGB on block A without opponent-belief features (``public_features``), temperature-scaled.
Inputs come only from ``observed_state(viewer)`` + the static public board map: no tracker, no
true opponent hands, so Φ can be applied to the real game or any determinized copy.

Build:  python -m catan_rl.winprob.phi build   [--model hgb_A_nobelief]
Bench:  python -m catan_rl.winprob.phi bench
Recheck (non-overlapping truth rollouts):  python -m catan_rl.winprob.phi recheck
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from typing import Sequence

import numpy as np

from catan_rl.winprob import features as F
from catan_rl.winprob.calibrate import apply_temperature

FORMAT = "phi/1"
DEFAULT_PATH = "runs/winprob/phi/phi_v1_pilot.pkl"
MODELS = "runs/winprob/models"


class FlatForest:
    """sklearn HistGradientBoostingClassifier compiled to flat numpy arrays: all trees are walked
    in lockstep (vectorized over rows x trees). sklearn's per-call overhead is ~60 ms; this is ~100x faster
    for single rows. Exactness vs clf.predict_proba is tested."""

    def __init__(self, clf):
        trees = [(k, t) for it in clf._predictors for k, t in enumerate(it)]
        sizes = np.array([len(t.nodes) for _, t in trees])
        off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        nodes = np.concatenate([t.nodes for _, t in trees])
        assert not nodes["is_categorical"].any()
        tree_of = np.repeat(np.arange(len(trees)), sizes)
        self.roots = off.astype(np.int64)
        self.feat = nodes["feature_idx"].astype(np.int64)
        self.thr = nodes["num_threshold"].astype(np.float64)
        self.mleft = nodes["missing_go_to_left"].astype(bool)
        self.left = (nodes["left"] + off[tree_of]).astype(np.int64)
        self.right = (nodes["right"] + off[tree_of]).astype(np.int64)
        self.leaf = nodes["is_leaf"].astype(bool)
        self.value = nodes["value"].astype(np.float64)
        self.cls = np.array([k for k, _ in trees])
        self.k = clf.n_trees_per_iteration_
        self.base = np.asarray(clf._baseline_prediction, dtype=np.float64).ravel()
        self.max_depth = int(nodes["depth"].max())

    def raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        idx = np.broadcast_to(self.roots, (len(X), len(self.roots))).copy()
        rows = np.arange(len(X))[:, None]
        for _ in range(self.max_depth):
            leaf = self.leaf[idx]
            if leaf.all():
                break
            x = X[rows, self.feat[idx]]
            go_left = np.where(np.isnan(x), self.mleft[idx], x <= self.thr[idx])
            nxt = np.where(go_left, self.left[idx], self.right[idx])
            idx = np.where(leaf, idx, nxt)
        out = np.tile(self.base, (len(X), 1))
        v = self.value[idx]
        for c in range(self.k):
            out[:, c] += v[:, self.cls == c].sum(1)
        return out

    def predict_proba(self, X) -> np.ndarray:
        z = self.raw(X)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class PhiModel:
    def __init__(self, clf, cols: Sequence[str], T: float, meta: dict):
        assert list(cols) == F.names_public(), "Φ columns must be exactly the observation-only features"
        assert not any(c[0] == "o" and c[1].isdigit() for c in cols), "belief feature in Φ"
        self.clf, self.cols, self.T, self.meta = clf, list(cols), float(T), dict(meta)
        self.forest = FlatForest(clf)

    @property
    def version(self) -> str:
        return self.meta["version"]

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"format": FORMAT, "clf": self.clf, "cols": self.cols, "T": self.T, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "PhiModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        assert d.get("format") == FORMAT, f"not a Φ file: {path}"
        return cls(d["clf"], d["cols"], d["T"], d["meta"])

    # ------------------------------------------------------------------ inference
    def predict_x(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        assert X.shape[1] == len(self.cols)
        return apply_temperature(self.forest.predict_proba(X), self.T)

    def features(self, game, viewer) -> np.ndarray:
        return F.public_features(game, viewer)

    def phi(self, game, viewer) -> np.ndarray:
        return self.predict_x(self.features(game, viewer)[None])[0]

    def phi_batch(self, games, viewer) -> np.ndarray:
        if not games:
            return np.zeros((0, 4))
        return self.predict_x(np.stack([self.features(g, viewer) for g in games]))


_CACHE = {}


def load_default(path: str = DEFAULT_PATH) -> PhiModel:
    """Process-wide cached Φ (agents in worker processes share one copy)."""
    if path not in _CACHE:
        _CACHE[path] = PhiModel.load(path)
    return _CACHE[path]


# ---------------------------------------------------------------------------- build
def build(model: str = "hgb_A_nobelief", out: str = DEFAULT_PATH) -> PhiModel:
    import sklearn
    src = os.path.join(MODELS, f"{model}.pkl")
    with open(src, "rb") as f:
        raw = f.read()
    m = pickle.loads(raw)
    T = json.load(open("runs/winprob/eval/metrics.json"))["results"][model]["calibrated"]["T"]
    meta = {"version": f"phi_v1_pilot:{model}:{hashlib.sha256(raw).hexdigest()[:12]}",
            "source_model": model, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset": "runs/winprob/pilot", "label_policy": "weighted_random", "best_iter": getattr(m, "best_iter", None),
            "sklearn": sklearn.__version__}
    phi = PhiModel(m.clf, m.cols, T, meta)
    phi.save(out)
    return phi


def bench(n_states: int = 200, seed: int = 0) -> dict:
    from catan_rl.tests.belief_helpers import random_walk
    phi = PhiModel.load()
    games = []
    while len(games) < n_states:
        games += [env.game.copy() for env in random_walk(seed, viewers=["RED"], every=7)]
        seed += 1
    games = games[:n_states]
    t = time.perf_counter()
    for g in games:
        phi.phi(g, "RED")
    single = len(games) / (time.perf_counter() - t)
    t = time.perf_counter()
    X = np.stack([phi.features(g, "RED") for g in games])
    feat = len(games) / (time.perf_counter() - t)
    t = time.perf_counter()
    phi.predict_x(X)
    batch = len(games) / (time.perf_counter() - t)
    t = time.perf_counter()
    phi.phi_batch(games, "RED")
    full = len(games) / (time.perf_counter() - t)
    return {"states": len(games), "phi_calls_per_s": single, "features_per_s": feat,
            "predict_batched_per_s": batch, "phi_batch_per_s": full}


# ---------------------------------------------------------------------------- same-label recheck
def recheck(n_overlap: int = 16) -> dict:
    """MAE vs truth rollouts that do NOT share seeds with the dataset labels.
    Truth rollouts r = 0..n-1 use determinize seeds f"{id}:{r}" with the same policy as the dataset's
    N=16 labels, so rollouts 0..15 are identical; rollouts 16..199 = truth_wins - dataset_wins."""
    from catan_rl.winprob.eval import cluster_mean_ci
    from catan_rl.winprob.train import load_data, matrix
    t = dict(np.load("runs/winprob/label_study/truth_set.npz", allow_pickle=True))
    d = dict(np.load("runs/winprob/pilot/dataset.npz", allow_pickle=True))
    assert int(d["n_rollouts"]) == n_overlap
    row = {s: i for i, s in enumerate(d["state_id"])}
    idx = np.array([row[s] for s in t["state_id"]])
    n_truth = int(t["n"]) if "n" in t else 200
    rest_w = t["wins"] - d["wins"][idx]
    rest_nw = (t["no_winner"] - d["no_winner"][idx]) if "no_winner" in t else 0
    assert (rest_w >= 0).all(), "truth rollouts 0..15 do not match dataset rollouts (policy/seed mismatch)"
    n_rest = n_truth - n_overlap
    assert np.all(rest_w.sum(1) + rest_nw == n_rest)
    keep = rest_w.sum(1) > 0
    p_rest = rest_w[keep] / rest_w[keep].sum(1, keepdims=True)
    lab = d["wins"][idx][keep]
    labk = lab.sum(1) > 0
    Q16 = lab / np.maximum(lab.sum(1, keepdims=True), 1)
    phi = PhiModel.load()
    X = np.asarray(d["xa"][idx][keep])[:, [F.names_a().index(c) for c in phi.cols]]
    P = phi.predict_x(X)
    games = np.array([int(str(s).split(":")[0]) for s in t["state_id"][keep]])
    mae_model = np.abs(P - p_rest).mean(1)
    mae_lab = np.abs(Q16 - p_rest).mean(1)
    out = {"states": int(keep.sum()), "rollouts_per_state": n_rest,
           "model_mae_vs_rest": cluster_mean_ci(mae_model, games),
           "label16_mae_vs_rest": cluster_mean_ci(mae_lab[labk], games[labk]),
           "model_minus_label16": cluster_mean_ci((mae_model - mae_lab)[labk], games[labk]),
           "expected_mae_floor_of_rest_noise": float(np.mean(np.sqrt(p_rest * (1 - p_rest) / n_rest)) * np.sqrt(2 / np.pi))}
    # in-split note: truth states are test-split states (never trained on)
    if "split" in d:
        out["truth_splits"] = sorted(set(map(str, d["split"][idx])))
    return {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "bench", "recheck"])
    ap.add_argument("--model", default="hgb_A_nobelief")
    a = ap.parse_args(argv)
    r = build(a.model).meta if a.cmd == "build" else bench() if a.cmd == "bench" else recheck()
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
