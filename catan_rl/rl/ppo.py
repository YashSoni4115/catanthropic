"""GAE and the PPO update (clipped surrogate + value loss - entropy + KL(pi || frozen BC))."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn

from catan_rl.rl.net import ActorCritic, masked_log_softmax, prep


def gae(r: np.ndarray, v: np.ndarray, gamma: float = 1.0, lam: float = 0.95):
    """Per-trajectory GAE; the episode ends after the last step (bootstrap 0). Returns (adv, returns)."""
    adv, last = np.zeros(len(r)), 0.0
    for t in reversed(range(len(r))):
        nv = v[t + 1] if t + 1 < len(r) else 0.0
        last = r[t] + gamma * nv - v[t] + gamma * lam * last
        adv[t] = last
    return adv, adv + v


@dataclass
class Batch:
    x: np.ndarray
    A: List[np.ndarray]
    choice: np.ndarray
    logp: np.ndarray
    value: np.ndarray
    adv: np.ndarray
    ret: np.ndarray

    @classmethod
    def from_trajs(cls, trajs, gamma=1.0, lam=0.95) -> "Batch":
        ga = [gae(t["reward"], t["value"], gamma, lam) for t in trajs]
        cat = lambda k: np.concatenate([t[k] for t in trajs])
        return cls(cat("x"), [a for t in trajs for a in t["A"]], cat("choice"), cat("logp"), cat("value"),
                   np.concatenate([a for a, _ in ga]), np.concatenate([r for _, r in ga]))

    def __len__(self):
        return len(self.choice)


@dataclass
class PPOConfig:
    epochs: int = 3
    minibatch: int = 512
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_bc_coef: float = 0.1
    max_grad_norm: float = 0.5


def explained_variance(v, ret) -> float:
    var = np.var(ret)
    return float(1 - np.var(ret - v) / var) if var > 0 else float("nan")


def update(net: ActorCritic, bc: ActorCritic, opt, b: Batch, cfg: PPOConfig, ent_coef: float, kl_bc_coef: float,
           rng: np.random.Generator) -> dict:
    adv = (b.adv - b.adv.mean()) / (b.adv.std() + 1e-8)  # per-batch normalization
    acc = {k: [] for k in ("policy_loss", "value_loss", "entropy", "kl_bc", "kl_prev", "clip_frac")}
    net.train()
    for _ in range(cfg.epochs):
        perm = rng.permutation(len(b))
        for s in range(0, len(b), cfg.minibatch):
            i = perm[s:s + cfg.minibatch]
            X, A, M = prep(net.norm, b.x[i], [b.A[j] for j in i])
            y = torch.from_numpy(b.choice[i]).long()
            z, v = net(X, A, M)
            lp_all = masked_log_softmax(z, M)
            lp = lp_all.gather(1, y[:, None]).squeeze(1)
            old = torch.from_numpy(b.logp[i]).float()
            ratio = torch.exp(lp - old)
            a = torch.from_numpy(adv[i]).float()
            pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vl = 0.5 * ((v - torch.from_numpy(b.ret[i]).float()) ** 2).mean()
            p = lp_all.exp() * M
            ent = -(p * lp_all).sum(1).mean()
            with torch.no_grad():
                zb, _ = bc(X, A, M)
            kl_bc = (p * (lp_all - masked_log_softmax(zb, M))).sum(1).mean()
            loss = pl + cfg.vf_coef * vl - ent_coef * ent + kl_bc_coef * kl_bc
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                lr_ = lp - old
                acc["kl_prev"].append(float(((lr_.exp() - 1) - lr_).mean()))  # k3 estimator of KL(old || new)
                acc["clip_frac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
            for k, val in (("policy_loss", pl), ("value_loss", vl), ("entropy", ent), ("kl_bc", kl_bc)):
                acc[k].append(float(val.detach()))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["explained_variance"] = explained_variance(b.value, b.ret)
    out["samples"] = len(b)
    return out
