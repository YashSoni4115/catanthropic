import os
import subprocess
import sys
from collections import Counter

import pytest

from catan_rl.env import COLORS, CatanEnv
from catan_rl.eval import agents as A
from catan_rl.eval.match import metrics, paired_comparison, run_match, wins_by_seed
from catan_rl.eval.stats import bootstrap_mean_ci, cluster_bootstrap, wilson

R4 = ["random"] * 4


def strip(rows):
    return [{k: v for k, v in r.items() if k != "wall_time"} for r in rows]


@pytest.mark.parametrize("lineup", ["random", "alpha_beta"])
def test_hash_independent(lineup):
    def run(h):
        return subprocess.run([sys.executable, "-m", "catan_rl.eval.hashcheck", "5", lineup], check=True,
                              capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": h}).stdout
    a, b = run("0"), run("1")
    assert a and a == b


def test_identical_agents_clustered_ci():
    """Identical agents replay the same game in all 4 rotations, so each slot wins exactly 1 of 4 per
    seed: the clustered CI collapses to 0.25 and the design effect is 0 (Wilson would claim n=4*seeds)."""
    res = run_match(R4, 30, workers=4)
    assert all(v == [1, 4] for s in range(4) for v in wins_by_seed(res, s).values())
    for s in metrics(res, R4)["per_slot"]:
        w = s["win_rate"]
        assert w["p"] == 0.25 and w["ci95"] == [0.25, 0.25] and w["design_effect"] == 0
        assert w["n_seeds"] == 30 and w["wilson_ci95_reference"][0] < 0.25


def test_design_effect_fully_correlated_clusters():
    """Clusters of 4 identical outcomes: effective sample size = number of clusters (deff ~ 4)."""
    k = [4 if i % 2 else 0 for i in range(400)]
    c = cluster_bootstrap(k, [4] * 400)
    assert c["p"] == 0.5 and c["design_effect"] == pytest.approx(4, rel=0.15)
    assert c["effective_n"] == pytest.approx(400, rel=0.15)
    ind = cluster_bootstrap([1] * 800 + [0] * 800, [1] * 1600)  # independent games
    assert ind["design_effect"] == pytest.approx(1, rel=0.15)


def test_deterministic():
    assert strip(run_match(R4, 3)) == strip(run_match(R4, 3, workers=2))


def test_rotation_balanced():
    res = run_match(["weighted_random", "random", "random", "random"], 4)
    for seed in range(4):
        rows = [r for r in res if r["seed"] == seed]
        pos = Counter(r["turn_order"].index(next(c for c, s in r["slots"].items() if s == 0)) for r in rows)
        assert pos == Counter(range(4))


def _dice(lineup, seed):
    env = CatanEnv(record_events=False).reset(seed, {c: A.get(n).factory(c) for c, n in zip(COLORS, lineup)})
    env.play()
    return [r.result for r in env.game.state.action_records if r.action.action_type.value == "ROLL"]


def test_same_dice_across_lineups():
    a, b = _dice(R4, 7), _dice(["weighted_random", "random", "random", "weighted_random"], 7)
    n = min(len(a), len(b))
    assert n > 20 and a[:n] == b[:n]


def test_paired_comparison_self_is_zero():
    res = run_match(["weighted_random", "random", "random", "random"], 5)
    c = paired_comparison(res, res)
    assert c["n_seeds"] == 5 and c["mean_diff"] == 0 and c["ci95"] == [0, 0]


def test_wilson_known():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)


def test_bootstrap_known():
    assert bootstrap_mean_ci([1, 1, 1]) == (1, 1, 1)
    m, lo, hi = bootstrap_mean_ci([0, 1] * 50)
    assert m == 0.5 and lo < 0.5 < hi and hi - lo == pytest.approx(0.2, abs=0.04)
