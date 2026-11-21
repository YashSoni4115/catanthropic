"""Train 4-way win-probability models (viewer, next, next, next) and baselines.

    python -m catan_rl.winprob.train --data runs/winprob/pilot --out runs/winprob/models [--workers 8]

Targets: soft labels q = wins / sum(wins) (states with no winner in any rollout are dropped and counted).
Outputs per model: OUT/{name}.pkl (checkpoint) and OUT/preds/{name}.npz (P for val and test rows).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from catanatron.features import get_feature_ordering

from catan_rl.winprob import features as F

B_NAMES = ["B:" + n for n in get_feature_ordering(4)]
EPS = 1e-7


# ---------------------------------------------------------------------------- data
def load_data(path: str) -> dict:
    d = dict(np.load(os.path.join(path, "dataset.npz")))
    assert list(d["xa_names"]) == F.names_a(), "feature A names changed since the dataset was built"
    assert d["xb"].shape[1] == len(B_NAMES)
    tot = d["wins"].sum(1)
    keep = tot > 0
    out = {k: v[keep] for k, v in d.items() if np.ndim(v) and len(v) == len(keep)}
    out["Q"] = out["wins"] / out["wins"].sum(1, keepdims=True)
    out["dropped_no_winner_states"] = int((~keep).sum())
    out["no_winner_fraction"] = float(d["no_winner"].sum() / (len(keep) * int(d["n_rollouts"])))
    return out


def feature_sets() -> dict:
    A = F.names_a()
    belief = [n for n in A if n[0] == "o" and n[1].isdigit()]
    base = [n for n in A if n not in belief]
    dev = [n for n in belief if "_dev_" in n]
    indep = base + [n for n in belief if "_indep_" in n] + dev
    joint = base + [n for n in belief if "_joint_" in n] + dev
    return {"A_nobelief": base, "A_indep": indep, "A_joint": joint, "A_joint+B": joint + B_NAMES}


ALLOWED = set(F.names_a()) | set(B_NAMES)


def matrix(d: dict, cols) -> np.ndarray:
    assert set(cols) <= ALLOWED, f"unknown features: {set(cols) - ALLOWED}"
    ia = {n: i for i, n in enumerate(F.names_a())}
    ib = {n: i for i, n in enumerate(B_NAMES)}
    A = [ia[c] for c in cols if c in ia]
    B = [ib[c] for c in cols if c in ib]
    return np.concatenate([d["xa"][:, A], d["xb"][:, B]], axis=1).astype(np.float32)


def soft_ce(P, Q) -> np.ndarray:
    return -(Q * np.log(np.clip(P, EPS, 1))).sum(1)


def entropy(Q) -> np.ndarray:
    return -(Q * np.log(np.where(Q > 0, Q, 1))).sum(1)


def _expand(X, Q):
    """One row per (state, class) weighted by the soft label (= win counts / N)."""
    w = Q.ravel()
    m = w > 0
    return np.repeat(X, 4, 0)[m], np.tile(np.arange(4), len(X))[m], w[m]


# ---------------------------------------------------------------------------- models
class Model:
    kind = "base"

    def __init__(self, cols):
        self.cols = list(cols)
        assert set(self.cols) <= ALLOWED

    def predict(self, d) -> np.ndarray:
        P = self._predict(matrix(d, self.cols))
        assert np.allclose(P.sum(1), 1, atol=1e-5)
        return P


class Uniform(Model):
    kind = "uniform"

    def fit(self, *a):
        return self

    def _predict(self, X):
        return np.full((len(X), 4), 0.25)


class VPOnly(Model):
    """Multinomial logistic on viewer-first public VP and VP differences to the viewer."""
    kind = "logistic"

    def __init__(self):
        super().__init__([f"p{i}_vp" for i in range(4)])

    @staticmethod
    def _feats(X):
        return np.concatenate([X, X[:, 1:] - X[:, :1]], axis=1)

    def fit(self, X, Q, Xv, Qv):
        from sklearn.linear_model import LogisticRegression
        Xe, y, w = _expand(self._feats(X), Q)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xe, y, sample_weight=w)
        return self

    def _predict(self, X):
        return self.clf.predict_proba(self._feats(X))


class HGB(Model):
    """sklearn HistGradientBoosting (LightGBM not loadable: missing libomp). Early stopping on val
    by staged predictions; refit at the best iteration count (deterministic)."""
    kind = "hgb"

    def fit(self, X, Q, Xv, Qv, max_iter=1000):
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda n: HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_leaf_nodes=31,
                                                      l2_regularization=1.0, early_stopping=False, random_state=0)
        Xe, y, w = _expand(X, Q)
        clf = mk(max_iter).fit(Xe, y, sample_weight=w)
        self.val_curve = [float(soft_ce(P, Qv).mean()) for P in clf.staged_predict_proba(Xv)]
        self.best_iter = int(np.argmin(self.val_curve)) + 1
        self.clf = clf if self.best_iter == max_iter else mk(self.best_iter).fit(Xe, y, sample_weight=w)
        return self

    def _predict(self, X):
        return self.clf.predict_proba(X)


class MLP(Model):
    kind = "mlp"

    def fit(self, X, Q, Xv, Qv, epochs=200, patience=10, bs=512):
        import torch
        torch.manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        Xt, Qt, Xvt, Qvt = t((X - self.mu) / self.sd), t(Q), t((Xv - self.mu) / self.sd), t(Qv)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(128, 4))
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        best, best_state, bad, self.val_curve = np.inf, None, 0, []
        for ep in range(epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=g).split(bs):
                loss = -(Qt[idx] * torch.log_softmax(self.net(Xt[idx]), 1)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(-(Qvt * torch.log_softmax(self.net(Xvt), 1)).sum(1).mean())
            self.val_curve.append(v)
            if v < best - 1e-5:
                best, bad, best_state = v, 0, {k: x.clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.val_curve)) + 1
        return self

    def _predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)), 1).double().numpy()


# ---------------------------------------------------------------------------- value_function baseline
def _vf_job(job):
    from catanatron.players.value import DEFAULT_WEIGHTS, base_fn
    from catan_rl.winprob.labeler import replay
    seed, items = job  # items: [(decision, viewer, state_id)]
    fn, out = base_fn(DEFAULT_WEIGHTS), {}
    by = {dec: (v, sid) for dec, v, sid in items}
    for dec, env in replay(seed, sorted(by), track=True):
        v, sid = by[dec]
        g = env.determinize(v, f"{sid}:vf")  # value_fn reads hidden state: evaluate on a belief sample
        order = [c for c in g.state.colors]
        k = [c.value for c in order].index(v)
        out[sid] = [float(fn(g, c)) for c in order[k:] + order[:k]]
    return out


def vf_values(d: dict, rows: np.ndarray, workers: int) -> np.ndarray:
    jobs = {}
    for i in rows:
        jobs.setdefault(int(d["seed"][i]), []).append((int(d["decision"][i]), str(d["viewer"][i]), str(d["state_id"][i])))
    with ProcessPoolExecutor(workers) as ex:
        vals = {k: v for r in ex.map(_vf_job, sorted(jobs.items())) for k, v in r.items()}
    return np.array([vals[str(d["state_id"][i])] for i in rows])


def vf_softmax(V, T):
    z = (V - V.max(1, keepdims=True)) / T
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_vf_temperature(V, Q) -> float:
    logs = np.linspace(-3, 20, 461)  # T = 10**x; value_fn spans many orders of magnitude
    ces = [soft_ce(vf_softmax(V, 10 ** x), Q).mean() for x in logs]
    return float(10 ** logs[int(np.argmin(ces))])


# ---------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/winprob/pilot")
    p.add_argument("--out", default="runs/winprob/models")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", default="", help="comma list of model names to (re)train")
    a = p.parse_args(argv)
    d = load_data(a.data)
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    tr, va, te = (np.where(d["split"] == s)[0] for s in ("train", "val", "test"))
    log = {"dropped_no_winner_states": d["dropped_no_winner_states"], "no_winner_fraction": d["no_winner_fraction"],
           "n": {"train": len(tr), "val": len(va), "test": len(te)}, "models": {}}

    def save(name, model, P_val, P_test, extra=None):
        with open(os.path.join(a.out, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
        np.savez_compressed(os.path.join(a.out, "preds", f"{name}.npz"), val=P_val, test=P_test)
        log["models"][name] = {"val_ce": float(soft_ce(P_val, d["Q"][va]).mean()), **(extra or {})}
        print(name, log["models"][name], flush=True)

    sub = lambda idx: {k: v[idx] for k, v in d.items() if isinstance(v, np.ndarray) and len(v) == len(d["Q"])}
    D = {s: sub(i) for s, i in (("train", tr), ("val", va), ("test", te))}
    only = set(filter(None, a.only.split(",")))
    want = lambda n: not only or n in only

    if want("uniform"):
        m = Uniform(["p0_vp"])
        save("uniform", m, m.predict(D["val"]), m.predict(D["test"]))
    if want("vp_only"):
        m = VPOnly()
        m.fit(matrix(D["train"], m.cols), D["train"]["Q"], None, None)
        save("vp_only", m, m.predict(D["val"]), m.predict(D["test"]))
    if want("value_function"):
        t = time.perf_counter()
        Vv, Vt = vf_values(d, va, a.workers), vf_values(d, te, a.workers)
        T = fit_vf_temperature(Vv, D["val"]["Q"])
        save("value_function", {"T": T, "kind": "vf_softmax"}, vf_softmax(Vv, T), vf_softmax(Vt, T),
             {"temperature": T, "seconds": time.perf_counter() - t})
    for fs, cols in feature_sets().items():
        for cls in (HGB, MLP):
            name = f"{cls.kind}_{fs}"
            if not want(name):
                continue
            t = time.perf_counter()
            m = cls(cols).fit(matrix(D["train"], cols), D["train"]["Q"], matrix(D["val"], cols), D["val"]["Q"])
            save(name, m, m.predict(D["val"]), m.predict(D["test"]),
                 {"seconds": time.perf_counter() - t, "n_features": len(cols),
                  "best": getattr(m, "best_iter", getattr(m, "best_epoch", None))})
    path = os.path.join(a.out, "train_log.json")
    old = json.load(open(path)) if os.path.exists(path) and only else {"models": {}}
    old["models"].update(log["models"])
    log["models"] = old["models"]
    json.dump(log, open(path, "w"), indent=2)


if __name__ == "__main__":
    from catan_rl.winprob.train import main as _main  # so pickled classes live in catan_rl.winprob.train, not __main__
    _main()
