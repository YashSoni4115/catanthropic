import os
import random
import subprocess
import sys

import numpy as np
import pytest
from catanatron.players.value import ValueFunctionPlayer

from catan_rl.env import COLORS, CatanEnv
from catan_rl.env.actions import DOMESTIC, NUM_ACTIONS
from catan_rl.policy.action_features import DIM, action_features
from catan_rl.policy.dataset import legal_actions
from catan_rl.policy.teacher import vf_choice, vf_values
from catan_rl.tests.belief_helpers import random_walk
from catan_rl.tests.test_determinize import V, _hidden_twin

HAVE_MODEL = os.path.exists("runs/policy/bc_v1/model.pt")
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="train first: python -m catan_rl.policy.train")


def test_teacher_matches_value_function_player():
    n = 0
    for env in random_walk(4, viewers=[V], every=11):
        if env.done:
            continue
        g, c = env.game, env.current_player()
        acts = g.playable_actions
        if len(acts) < 2:
            continue
        mine = acts[vf_choice(vf_values(g, c, acts))]
        assert mine == ValueFunctionPlayer(c).decide(g, acts)
        n += 1
    assert n > 5


def test_action_features_leak_free():
    rng, n = random.Random(2), 0
    for env in random_walk(5, viewers=[V], every=7):
        if env.done or env.current_player().value != V:
            continue
        g, changed = _hidden_twin(env, rng)
        if not changed:
            continue
        c = env.current_player()
        a, b = legal_actions(env.game), legal_actions(g)
        assert a == b
        X, Y = action_features(env.game, c, a), action_features(g, c, b)
        assert X.shape == (len(a), DIM) and np.array_equal(X, Y)
        n += 1
    assert n >= 3


def _play(seed, mode="argmax"):
    from catan_rl.policy.agent import PolicyAgent
    from catanatron.models.player import RandomPlayer
    ag = PolicyAgent(COLORS[0], mode=mode, seed=1)
    random.seed(seed)
    env = CatanEnv(record_events=False).reset(seed, {COLORS[0]: ag, **{c: RandomPlayer(c) for c in COLORS[1:]}})
    env.play()  # an illegal index raises in env._resolve
    return env, ag


@needs_model
@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_complete_games_valid_indices(mode):
    env, ag = _play(3, mode)
    assert env.done and ag.decisions > 10


@needs_model
def test_masks_illegal_and_valid_index():
    import torch
    from catan_rl.policy.agent import PolicyAgent
    from catan_rl.policy.train import Scorer
    ag = PolicyAgent(V)
    for env in random_walk(6, viewers=[V], every=13):
        if env.done:
            continue
        c = env.current_player()
        idx = ag.choose(env.game, c)
        assert 0 <= idx < NUM_ACTIONS and idx in env.legal_actions()
    m = Scorer(4, 3)
    z = m(torch.zeros(1, 4), torch.zeros(1, 5, 3), torch.tensor([[True, False, True, False, False]]))
    assert torch.isinf(z[0, [1, 3, 4]]).all() and torch.softmax(z, 1)[0, [1, 3, 4]].sum() == 0


@needs_model
def test_policy_leak_free():
    from catan_rl.policy.agent import PolicyAgent
    rng, n = random.Random(3), 0
    for env in random_walk(7, viewers=[V], every=9):
        if env.done or env.current_player().value != V:
            continue
        g, changed = _hidden_twin(env, rng)
        if not changed or len([a for a in g.playable_actions if a.action_type not in DOMESTIC]) < 2:
            continue
        for mode in ("argmax", "sample"):
            a, b = PolicyAgent(V, mode=mode), PolicyAgent(V, mode=mode)
            assert a.choose(env.game, env.current_player()) == b.choose(g, env.current_player())
            assert np.array_equal(a.last_logits, b.last_logits)
        n += 1
    assert n >= 3


SCRIPT = ("from catan_rl.tests.test_policy import _play; e,a=_play(9,'sample'); "
          "print(e.winner, [str(r.action) for r in e.game.state.action_records][-8:])")


@needs_model
def test_reproducible_across_processes():
    outs = [subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout for h in (1, 2)]
    assert outs[0] == outs[1] and outs[0].strip()
