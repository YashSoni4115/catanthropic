import numpy as np
import pytest

from catan_rl.tests.belief_helpers import random_walk
from catan_rl.tests.test_determinize import _hidden_twin
from catan_rl.winprob.build_dataset import split_of
from catan_rl.winprob.features import engineered, names_a, sample_vector
from catan_rl.winprob.label_study import spearman
from catan_rl.winprob.labeler import label_states, lineup, select_states


def test_lineup_and_selection_deterministic():
    assert lineup(7) == lineup(7) and len(lineup(7)) == 4
    a, b = select_states(4), select_states(4)
    assert a == b and 6 <= len(a) <= 10 and {s.phase for s in a} == {"early", "mid", "late"}


def test_rollouts_reproducible_and_sum_to_n():
    specs = select_states(11)[::3] + select_states(12)[::4]
    r1, _ = label_states(specs, "weighted_random", 6, workers=1)
    r2, _ = label_states(specs[::-1], "weighted_random", 6, workers=3)
    by2 = {r["spec"].id: r for r in r2}
    assert len(r1) == len(specs)
    for r in r1:
        o = by2[r["spec"].id]
        assert np.array_equal(r["wins"], o["wins"]) and r["no_winner"] == o["no_winner"]
        assert np.array_equal(r["mean_vp"], o["mean_vp"]) and r["mean_remaining_turns"] == o["mean_remaining_turns"]
        assert r["wins"].sum() + r["no_winner"] == 6


def test_features_leak_free():
    import random
    rng, tested = random.Random(3), 0
    for seed in range(3):
        for env in random_walk(seed, viewers=["RED"], every=30):
            real = env.game
            twin, changed = _hidden_twin(env, rng)
            if not changed:
                continue
            args = (env.belief("RED"), env.dev_belief("RED"), "s")
            a, ab = engineered(real, "RED", *args), sample_vector(real, "RED")
            b, bb = engineered(twin, "RED", *args), sample_vector(twin, "RED")
            assert np.array_equal(a, b) and np.array_equal(ab, bb)
            assert len(a) == len(names_a())
            tested += 1
    assert tested > 5


def test_split_deterministic_and_disjoint():
    seeds = range(5000)
    s = {x: split_of(x) for x in seeds}
    assert s == {x: split_of(x) for x in seeds}
    groups = {k: {x for x in seeds if s[x] == k} for k in ("train", "val", "test")}
    assert sum(map(len, groups.values())) == 5000 and not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"])
    assert abs(len(groups["train"]) / 5000 - 0.8) < 0.02 and abs(len(groups["test"]) / 5000 - 0.1) < 0.02


def test_spearman():
    assert spearman(np.array([1, 2, 3, 4.]), np.array([10, 20, 30, 40.])) == pytest.approx(1)
    assert spearman(np.array([1, 2, 3, 4.]), np.array([4, 3, 2, 1.])) == pytest.approx(-1)
