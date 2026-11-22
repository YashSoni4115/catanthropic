"""Seat callbacks driven by an ActorCritic: the learner (samples, records trajectories) and PPOAgent (eval)."""
from __future__ import annotations

import random

import numpy as np
import torch

from catan_rl.env.actions import to_index
from catan_rl.policy.action_features import action_features
from catan_rl.policy.agent import sample_key
from catan_rl.policy.dataset import legal_actions
from catan_rl.rl.net import ActorCritic, prep
from catan_rl.winprob.features import public_features


class NetSeat:
    """Non-peeking: reads public_features + action_features (public + own hand) of env.game only.
    mode "sample": RNG keyed by (game seed, viewer, decision index); "argmax": first max.
    record: list to append per-decision dicts (learner seats only); phi: PhiModel for shaping (or None)."""

    def __init__(self, net: ActorCritic, color, mode="sample", record=None, phi=None, salt="ppo"):
        self.net, self.viewer, self.mode, self.record, self.phi, self.salt = net, getattr(color, "value", color), mode, record, phi, salt

    def __call__(self, env, color) -> int:
        game = env.game
        acts = legal_actions(game)
        x, A = public_features(game, color), action_features(game, color, acts)
        with torch.no_grad():
            X, AA, M = prep(self.net.norm, [x], [A])
            z, v = self.net(X, AA, M)
            lp = torch.log_softmax(z[0], -1).numpy().astype(np.float64)
        if self.mode == "argmax":
            i = int(np.argmax(lp))
        else:
            c = np.cumsum(np.exp(lp))
            i = int(min(np.searchsorted(c, random.Random(sample_key(game, color, self.salt)).random() * c[-1], side="right"), len(c) - 1))
        if self.record is not None:
            self.record.append({"x": x, "A": A, "choice": i, "logp": float(lp[i]), "value": float(v[0]),
                                "phi": float(self.phi.phi(game, color)[0]) if self.phi is not None else np.nan})
        return to_index(acts[i])


_NETS = {}


class PPOAgent(NetSeat):
    def __init__(self, color, path: str, mode: str = "argmax"):
        torch.set_num_threads(1)
        if path not in _NETS:
            _NETS[path] = ActorCritic.load(path)
        super().__init__(_NETS[path], color, mode=mode)
