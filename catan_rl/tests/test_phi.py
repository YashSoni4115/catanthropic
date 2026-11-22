import os
import random
import subprocess
import sys

import numpy as np
import pytest

from catan_rl.env import COLORS, CatanEnv
from catan_rl.tests.belief_helpers import random_walk
from catan_rl.tests.test_determinize import V, _hidden_twin
from catan_rl.winprob import features as F
from catan_rl.winprob.phi import DEFAULT_PATH, PhiModel

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="build Φ first: python -m catan_rl.winprob.phi build")


@pytest.fixture(scope="module")
def phi():
    return PhiModel.load()


def test_phi_observation_only_and_sums_to_one(phi):
    assert phi.cols == F.names_public() and len(phi.cols) == 104 and phi.version.startswith("phi_v1")
    for env in random_walk(0, viewers=[V], every=20):
        for c in COLORS:
            p = phi.phi(env.game, c)
            assert p.shape == (4,) and np.all(p >= 0) and p.sum() == pytest.approx(1, abs=1e-9)
        X = np.stack([F.public_features(env.game, V)])
        assert np.allclose(phi.forest.predict_proba(X), phi.clf.predict_proba(X), atol=1e-9)


def test_public_features_match_engineered():
    keep = [i for i, n in enumerate(F.names_a()) if n in set(F.names_public())]
    for env in random_walk(1, viewers=[V], every=40):
        full = F.engineered(env.game, V, env.belief(V), env.dev_belief(V), 0)
        assert np.array_equal(full[keep], F.public_features(env.game, V))


def test_phi_ignores_hidden_info(phi):
    rng, n = random.Random(0), 0
    for env in random_walk(2, viewers=[V], every=15):
        g, changed = _hidden_twin(env, rng)
        if changed:
            n += 1
            assert np.array_equal(phi.phi(g, V), phi.phi(env.game, V))
    assert n > 3


def test_decisions_leak_free(phi):
    """Same viewer information, different hidden opponent cards -> identical decision and scores."""
    from catan_rl.agents.phi_greedy import PhiGreedyAgent
    rng, n = random.Random(1), 0
    for env in random_walk(3, viewers=[V], every=9):
        if env.done or env.current_player().value != V or len(env.legal_actions()) < 2:
            continue
        g, changed = _hidden_twin(env, rng)
        if not changed:
            continue
        twin = CatanEnv(track_viewers=[V], record_events=False)
        twin.__dict__.update(env.__dict__)
        twin.game = g
        a, b = PhiGreedyAgent(V, phi), PhiGreedyAgent(V, phi)
        assert a(env, env.current_player()) == b(twin, twin.current_player())
        assert a.last_scores == b.last_scores
        n += 1
    assert n >= 2


def _play(seed, phi=None):
    from catan_rl.agents.phi_greedy import PhiGreedyAgent
    from catanatron.models.player import RandomPlayer
    ag = PhiGreedyAgent(COLORS[0], phi)
    seats = {COLORS[0]: ag, **{c: RandomPlayer(c) for c in COLORS[1:]}}
    random.seed(seed)
    env = CatanEnv(track_viewers=[COLORS[0]], record_events=False).reset(seed, seats)
    env.play()  # env._resolve raises on any illegal index
    return env, ag


def test_complete_games(phi):
    for seed in (0, 1):
        env, ag = _play(seed, phi)
        assert env.done and ag.decisions > 10


SCRIPT = ("from catan_rl.tests.test_phi import _play; e,a=_play(7); "
          "print(e.winner, len(e.game.state.action_records), [str(r.action) for r in e.game.state.action_records][-5:])")


def test_reproducible_across_processes():
    outs = [subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout for h in (1, 2)]
    assert outs[0] == outs[1] and outs[0].strip()
