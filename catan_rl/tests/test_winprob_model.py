import os
import pickle

import numpy as np
import pytest

from catan_rl.winprob.calibrate import apply_temperature, fit_temperature
from catan_rl.winprob.eval import cluster_mean_ci, cluster_stat_ci, ece
from catan_rl.winprob.train import ALLOWED, HGB, MLP, VPOnly, feature_sets, soft_ce

PILOT = "runs/winprob/pilot"


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    logits = np.stack([X[:, 0], X[:, 1], -X[:, 0], np.zeros(n)], 1)
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    wins = np.stack([rng.multinomial(16, p) for p in P])
    return X, wins / 16


def test_cluster_bootstrap_known_values():
    # 400 games x 5 states, value = game effect only: CI half-width ~ 1.96 * sd(game means) / sqrt(G)
    rng = np.random.default_rng(1)
    eff = rng.normal(size=400)
    x, g = np.repeat(eff, 5), np.repeat(np.arange(400), 5)
    m, lo, hi = cluster_mean_ci(x, g, n_boot=4000)
    assert m == pytest.approx(eff.mean())
    assert (hi - lo) / 2 == pytest.approx(1.96 * eff.std() / np.sqrt(400), rel=0.1)
    assert cluster_mean_ci(np.full(20, 0.3), np.repeat(np.arange(4), 5)) == pytest.approx([0.3, 0.3, 0.3])
    s = cluster_stat_ci(lambda i: x[i].mean(), g, n_boot=1000)
    assert s[1] == pytest.approx(lo, abs=0.03) and s[2] == pytest.approx(hi, abs=0.03)


def test_ece_and_temperature():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    assert ece(p, p) == 0 and ece(p, np.array([0.2, 0.2, 0.8, 0.8])) == pytest.approx(0.1)
    X, Q = synthetic()
    logits = np.stack([X[:, 0], X[:, 1], -X[:, 0], np.zeros(len(X))], 1) * 2.0  # overconfident by 2x
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    T = fit_temperature(P, Q)
    assert T == pytest.approx(2.0, rel=0.2)
    assert np.allclose(apply_temperature(P, T).sum(1), 1)


@pytest.mark.parametrize("cls", [HGB, MLP])
def test_models_sum_to_one_and_checkpoint_reproduces(cls, tmp_path):
    X, Q = synthetic()
    cols = ["p0_vp", "p1_vp", "p2_vp", "p3_vp", "turn", "dev_deck"]
    from catan_rl.winprob.features import names_a
    xa = np.zeros((len(X), len(names_a())), np.float32)
    for j, c in enumerate(cols):
        xa[:, names_a().index(c)] = X[:, j]
    d = {"xa": xa, "xb": np.zeros((len(X), 0), np.float32)}
    tr, va = slice(0, 400), slice(400, None)
    sub = lambda s: {"xa": xa[s], "xb": d["xb"][s]}
    kw = {"max_iter": 40} if cls is HGB else {"epochs": 15}
    m = cls(cols).fit(X[tr], Q[tr], X[va], Q[va], **kw)
    P = m.predict(sub(va))
    assert np.allclose(P.sum(1), 1) and soft_ce(P, Q[va]).mean() < np.log(4)
    path = tmp_path / "m.pkl"
    pickle.dump(m, open(path, "wb"))
    m2 = pickle.load(open(path, "rb"))
    assert np.array_equal(m2.predict(sub(va)), P)
    with pytest.raises(AssertionError):
        cls(["P0_TRUE_HAND"])


def test_feature_sets_only_allowed_names():
    fs = feature_sets()
    assert all(set(c) <= ALLOWED for c in fs.values())
    assert not any("indep" in c for c in fs["A_joint"]) and not any("joint" in c for c in fs["A_indep"])
    assert not any(c.startswith(("o1_", "o2_", "o3_")) for c in fs["A_nobelief"])
    assert set(VPOnly().cols) <= ALLOWED


@pytest.mark.skipif(not os.path.exists(f"{PILOT}/dataset.npz"), reason="pilot dataset not built")
def test_pilot_split_no_game_overlap():
    d = np.load(f"{PILOT}/dataset.npz")
    sets = {s: set(d["seed"][d["split"] == s].tolist()) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])


@pytest.mark.skipif(not os.path.exists("runs/winprob/models/train_log.json"), reason="models not trained")
def test_saved_checkpoint_reproduces_predictions():
    from catan_rl.winprob.train import load_data
    d = load_data(PILOT)
    te = np.where(d["split"] == "test")[0]
    D = {k: v[te] for k, v in d.items() if isinstance(v, np.ndarray) and len(v) == len(d["Q"])}
    for name in ("vp_only", "hgb_A_joint"):
        m = pickle.load(open(f"runs/winprob/models/{name}.pkl", "rb"))
        saved = np.load(f"runs/winprob/models/preds/{name}.npz")["test"]
        P = m.predict(D)
        assert np.allclose(P, saved) and np.allclose(P.sum(1), 1)
        assert soft_ce(P, D["Q"]).mean() == pytest.approx(soft_ce(saved, D["Q"]).mean())
