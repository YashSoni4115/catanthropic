"""PolicyAgent: non-peeking BC policy as a CatanEnv seat callback.

Reads only public_features(game, seat) and action_features (public + own hand; see action_features.py).
Inference in numpy (weights from the torch checkpoint). Modes: "argmax" (first max) or "sample"
(softmax, RNG seeded by (agent seed, game seed, number of actions so far)).
"""
from __future__ import annotations

import random
import time

import numpy as np

from catan_rl.env.actions import to_index
from catan_rl.policy.action_features import NAMES, action_features
from catan_rl.policy.dataset import legal_actions
from catan_rl.winprob.features import names_public, public_features

DEFAULT_PATH = "runs/policy/bc_v1/model.pt"
_CACHE = {}


class PolicyNet:
    def __init__(self, path: str = DEFAULT_PATH):
        import torch
        ck = torch.load(path, weights_only=False)
        assert ck["state_names"] == names_public() and ck["action_names"] == NAMES, "feature layout changed"
        self.version, self.norm = ck["version"], ck["norm"]
        sd = {k: v.numpy().astype(np.float64) for k, v in ck["state_dict"].items()}
        self.S = [(sd["state.0.weight"].T, sd["state.0.bias"]), (sd["state.2.weight"].T, sd["state.2.bias"])]
        self.H = [(sd[f"head.{i}.weight"].T, sd[f"head.{i}.bias"]) for i in (0, 2, 4)]

    def logits(self, x: np.ndarray, A: np.ndarray) -> np.ndarray:
        e = (x - self.norm["xm"]) / self.norm["xs"]
        for W, b in self.S:
            e = np.maximum(e @ W + b, 0)
        z = np.concatenate([np.repeat(e[None], len(A), 0), (A.astype(np.float16).astype(np.float32) - self.norm["am"]) / self.norm["as"]], 1)
        for i, (W, b) in enumerate(self.H):
            z = z @ W + b
            if i < 2:
                z = np.maximum(z, 0)
        return z[:, 0]


def load_default(path: str = DEFAULT_PATH) -> PolicyNet:
    if path not in _CACHE:
        _CACHE[path] = PolicyNet(path)
    return _CACHE[path]


def sample_key(game, viewer, salt=0) -> str:
    """Sampling RNG key: (game seed, viewer, decision index = actions so far). Distinct games never share draws."""
    return f"sample:{salt}:{game.seed}:{getattr(viewer, 'value', viewer)}:{len(game.state.action_records)}"


class PolicyAgent:
    def __init__(self, viewer, net: PolicyNet = None, mode: str = "argmax", seed: int = 0, path: str = DEFAULT_PATH):
        assert mode in ("argmax", "sample")
        self.viewer = getattr(viewer, "value", viewer)
        self.net, self.mode, self.seed = net or load_default(path), mode, seed
        self.decisions, self.time, self.last_logits = 0, 0.0, None

    def choose(self, game, color) -> int:
        acts = legal_actions(game)
        if len(acts) == 1:
            return to_index(acts[0])
        z = self.net.logits(public_features(game, color), action_features(game, color, acts))
        self.last_logits = z
        if self.mode == "argmax":
            i = int(np.argmax(z))
        else:
            p = np.exp(z - z.max())
            p /= p.sum()
            rng = random.Random(sample_key(game, self.viewer, self.seed))
            i = int(np.searchsorted(np.cumsum(p), rng.random() * p.sum(), side="right").clip(0, len(p) - 1))
        return to_index(acts[i])

    def __call__(self, env, color) -> int:
        assert color.value == self.viewer
        t = time.perf_counter()
        a = self.choose(env.game, color)
        self.decisions += 1
        self.time += time.perf_counter() - t
        return a
