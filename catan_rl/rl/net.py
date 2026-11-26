"""Actor-critic: the BC masked per-action scorer (shared state encoder + action head) plus a value head
on the state embedding. Inputs are normalized with the BC checkpoint's (fixed) statistics."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from catan_rl.policy.action_features import NAMES
from catan_rl.policy.train import Scorer
from catan_rl.winprob.features import names_public

BC_PATH = "runs/policy/bc_v1/model.pt"


class ActorCritic(nn.Module):
    def __init__(self, ds: int, da: int, norm: Dict[str, np.ndarray], db: int = 0):
        """db: number of belief action features appended after the da BC action features (0 = off).
        They are fed raw (identity normalization), see prep."""
        super().__init__()
        self.pi = Scorer(ds, da + db)
        h = self.pi.state[0].out_features
        self.v = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, 1))
        self.ds, self.da, self.db, self.norm = ds, da, db, {k: np.asarray(v, np.float32) for k, v in norm.items()}

    @property
    def belief_features(self) -> bool:
        return self.db > 0

    def with_belief(self, db: int) -> "ActorCritic":
        """Warm start: a copy with db belief inputs appended to the action head's first layer, their
        weights initialized to ZERO, so logits and values are identical to self."""
        assert self.db == 0 and db > 0
        new = ActorCritic(self.ds, self.da, self.norm, db)
        sd = {k: v.clone() for k, v in self.state_dict().items()}
        w = sd["pi.head.0.weight"]
        sd["pi.head.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], db, dtype=w.dtype)], 1)
        new.load_state_dict(sd)
        new.meta = dict(getattr(self, "meta", {}))
        return new

    def forward(self, x, A, mask):
        e = self.pi.state(x)
        z = torch.cat([e[:, None].expand(-1, A.shape[1], -1), A], -1)
        logits = self.pi.head(z).squeeze(-1).masked_fill(~mask, float("-inf"))
        return logits, self.v(e).squeeze(-1)

    @classmethod
    def from_bc(cls, path: str = BC_PATH, seed: int = 0) -> "ActorCritic":
        ck = torch.load(path, weights_only=False)
        assert ck["state_names"] == names_public() and ck["action_names"] == NAMES, "feature layout changed"
        torch.manual_seed(seed)
        net = cls(ck["ds"], ck["da"], ck["norm"])
        net.pi.load_state_dict(ck["state_dict"])
        return net

    # ------------------------------------------------------------------ io
    def params(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().numpy().copy() for k, v in self.state_dict().items()}

    def load_params(self, p: Dict[str, np.ndarray]) -> None:
        self.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in p.items()})

    def save(self, path: str, meta: dict) -> None:
        ck = {"kind": "ppo_actor_critic", "ds": self.ds, "da": self.da, "norm": self.norm,
              "params": self.params(), "state_names": names_public(), "action_names": NAMES,
              "meta": {**meta, "belief_features": self.belief_features}}
        if self.db:
            from catan_rl.rl.belief_features import NAMES as BNAMES
            ck["db"], ck["belief_names"] = self.db, BNAMES
        torch.save(ck, path)

    @classmethod
    def load(cls, path: str) -> "ActorCritic":
        ck = torch.load(path, weights_only=False)
        assert ck["kind"] == "ppo_actor_critic" and ck["state_names"] == names_public() and ck["action_names"] == NAMES
        db = int(ck.get("db", 0))  # checkpoints from before R3 have no belief inputs
        if db:
            from catan_rl.rl.belief_features import NAMES as BNAMES
            assert ck["belief_names"] == BNAMES, "belief feature layout changed"
        net = cls(ck["ds"], ck["da"], ck["norm"], db)
        net.load_params(ck["params"])
        net.meta = ck["meta"]
        return net


def prep(norm, xs: Sequence[np.ndarray], As: Sequence[np.ndarray]):
    """Normalize + pad. Action features are rounded through float16 exactly as in the BC dataset."""
    n = np.array([len(a) for a in As])
    M = int(n.max())
    X = (np.stack(xs).astype(np.float32) - norm["xm"]) / norm["xs"]
    W = As[0].shape[1]
    am, as_ = norm["am"], norm["as"]
    if W > len(am):  # belief columns: identity normalization
        am = np.concatenate([am, np.zeros(W - len(am), np.float32)])
        as_ = np.concatenate([as_, np.ones(W - len(as_), np.float32)])
    A = np.zeros((len(As), M, W), np.float32)
    for i, a in enumerate(As):
        A[i, : len(a)] = (np.asarray(a).astype(np.float16).astype(np.float32) - am) / as_
    mask = np.arange(M) < n[:, None]
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(A), torch.from_numpy(mask)


def masked_log_softmax(logits, mask):
    lp = torch.log_softmax(logits, -1)
    return torch.where(mask, lp, torch.zeros_like(lp))  # 0 at padding: safe in p * logp sums
