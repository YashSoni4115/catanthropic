"""Per-action BC scorer: score(s, a) = MLP([state_emb(s), action_features(a)]), softmax over legal actions.

python -m catan_rl.policy.train --data runs/policy/bc_v1 [--fractions 0.25,0.5,1]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from catan_rl.policy import action_features as AF
from catan_rl.policy.teacher import GROUPS
from catan_rl.winprob.features import names_public

VERSION = "bc_v1"


class Scorer(nn.Module):
    def __init__(self, ds: int, da: int, h: int = 128, ha: int = 256):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(ds, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h + da, ha), nn.ReLU(), nn.Linear(ha, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, A, mask):
        e = self.state(x)
        z = torch.cat([e[:, None].expand(-1, A.shape[1], -1), A], -1)
        return self.head(z).squeeze(-1).masked_fill(~mask, float("-inf"))


class Data:
    def __init__(self, path):
        d = np.load(os.path.join(path, "dataset.npz"))
        self.x, self.a, self.n = d["x"], d["a"], d["n_act"].astype(np.int64)
        self.choice, self.group, self.seed, self.split = d["choice"].astype(np.int64), d["group"], d["seed"], d["split"]
        self.off = np.concatenate([[0], np.cumsum(self.n)[:-1]])

    def batch(self, idx, norm):
        n = self.n[idx]
        M = int(n.max())
        rows = np.minimum(self.off[idx, None] + np.arange(M), len(self.a) - 1)
        mask = np.arange(M) < n[:, None]
        A = (self.a[rows].astype(np.float32) - norm["am"]) / norm["as"]
        x = (self.x[idx] - norm["xm"]) / norm["xs"]
        return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy((A * mask[..., None]).astype(np.float32)),
                torch.from_numpy(mask), torch.from_numpy(self.choice[idx]))


def norms(D, idx):
    rows = np.concatenate([np.arange(D.off[i], D.off[i] + D.n[i]) for i in idx[:: max(1, len(idx) // 20000)]])
    A = D.a[rows].astype(np.float32)
    return {"xm": D.x[idx].mean(0), "xs": D.x[idx].std(0) + 1e-3, "am": A.mean(0), "as": A.std(0) + 1e-3}


@torch.no_grad()
def evaluate(model, D, idx, norm, bs=2048):
    model.eval()
    ce, r = [], []
    for s in range(0, len(idx), bs):
        x, A, m, y = D.batch(idx[s:s + bs], norm)
        z = model(x, A, m)
        ce.append(nn.functional.cross_entropy(z, y, reduction="none").numpy())
        r.append((z > z.gather(1, y[:, None])).sum(1).numpy())  # rank of teacher action (0 = top-1)
    ce, r = np.concatenate(ce), np.concatenate(r)
    g = D.group[idx]
    by = {GROUPS[k]: {"n": int((g == k).sum()), "top1": float((r[g == k] == 0).mean()), "top3": float((r[g == k] < 3).mean())}
          for k in np.unique(g)}
    return {"n": len(idx), "ce": float(ce.mean()), "top1": float((r == 0).mean()), "top3": float((r < 3).mean()), "by_group": by}


def fit(D, tr, va, epochs=20, bs=256, lr=1e-3, patience=3, seed=0, log=print):
    torch.manual_seed(seed)
    norm = norms(D, tr)
    model = Scorer(D.x.shape[1], D.a.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        t = time.time()
        perm = rng.permutation(tr)
        for s in range(0, len(tr), bs):
            x, A, m, y = D.batch(perm[s:s + bs], norm)
            loss = nn.functional.cross_entropy(model(x, A, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(model, D, va, norm)
        log(f"epoch {ep} val ce {v['ce']:.4f} top1 {v['top1']:.4f} top3 {v['top3']:.4f} ({time.time() - t:.0f}s)")
        if v["ce"] < best - 1e-4:
            best, best_state, bad = v["ce"], {k: t_.clone() for k, t_ in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, norm


def save(model, norm, path, meta):
    torch.save({"version": VERSION, "state_dict": model.state_dict(), "norm": norm, "ds": model.state[0].in_features,
                "da": model.head[0].in_features - model.state[0].out_features, "state_names": names_public(),
                "action_names": AF.NAMES, "meta": meta}, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/policy/bc_v1")
    ap.add_argument("--fractions", default="0.25,0.5,1")
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    D = Data(a.data)
    va, te = np.flatnonzero(D.split == "val"), np.flatnonzero(D.split == "test")
    tr_all = np.flatnonzero(D.split == "train")
    games = np.unique(D.seed[tr_all])
    order = np.random.default_rng(0).permutation(games)
    res = {}
    for f in map(float, a.fractions.split(",")):
        keep = set(order[: int(round(f * len(games)))].tolist())
        tr = tr_all[np.isin(D.seed[tr_all], list(keep))]
        print(f"== fraction {f}: {len(keep)} games, {len(tr)} decisions", flush=True)
        model, norm = fit(D, tr, va, epochs=a.epochs, log=lambda s: print(s, flush=True))
        res[str(f)] = {"train_games": len(keep), "train_decisions": int(len(tr)), "val": evaluate(model, D, va, norm)}
        if f == 1.0:
            res[str(f)]["test"] = evaluate(model, D, te, norm)
            n_params = sum(p.numel() for p in model.parameters())
            save(model, norm, os.path.join(a.data, "model.pt"), {"params": n_params, "train_decisions": int(len(tr))})
            res["params"] = n_params
    json.dump(res, open(os.path.join(a.data, "train_report.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
