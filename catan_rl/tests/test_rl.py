import hashlib
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from catan_rl.env import COLORS
from catan_rl.rl.ppo import Batch, PPOConfig, gae, update
from catan_rl.rl.rollout import play_game, rewards
from catan_rl.tests.belief_helpers import random_walk
from catan_rl.tests.test_determinize import V, _hidden_twin

pytestmark = pytest.mark.skipif(not os.path.exists("runs/policy/bc_v1/model.pt"), reason="needs the BC checkpoint")


@pytest.fixture(scope="module")
def net():
    from catan_rl.rl.net import ActorCritic
    torch.set_num_threads(1)
    return ActorCritic.from_bc(seed=0)


def _job(net, seats, seed=1_000_123, shaping=False):
    return {"seed": seed, "seats": seats, "category": "test", "params": net.params(), "past": {0: net.params()},
            "dims": (net.ds, net.da, net.norm), "shaping": shaping, "coef": 1.0, "gamma": 1.0,
            "phi_version": __import__("catan_rl.winprob.phi", fromlist=["x"]).load_default().version}


def test_gae_hand_computed():
    adv, ret = gae(np.array([0, 0, 1.0]), np.array([0.5, 0.6, 0.7]), gamma=1.0, lam=0.5)
    # deltas: 0.1, 0.1, 0.3 -> adv2 = .3, adv1 = .1 + .5*.3 = .25, adv0 = .1 + .5*.25 = .225
    assert np.allclose(adv, [0.225, 0.25, 0.3]) and np.allclose(ret, [0.725, 0.85, 1.0])
    adv, _ = gae(np.array([0, 1.0]), np.array([0.2, 0.4]), gamma=1.0, lam=1.0)
    assert np.allclose(adv, [0.8, 0.6])  # lam = 1: Monte Carlo return - value


def test_shaping_telescopes():
    phis = np.array([0.3, 0.1, 0.6, 0.9])
    for out in (0.0, 1.0):
        r = rewards(phis, out, shaping=True, coef=1.0)
        assert r.sum() == pytest.approx(out + (out - phis[0]))
        assert (r - rewards(phis, out, shaping=False)).sum() == pytest.approx(out - phis[0])
        assert rewards(phis, out, True, coef=0.5).sum() == pytest.approx(out + 0.5 * (out - phis[0]))


def test_real_game_shaping_and_learner_only(net):
    seats = ["learner", "vf", "bc", "past:0"]
    g = play_game(_job(net, seats, shaping=True))
    assert g["learner_seats"] == [COLORS[0].value]
    assert [t["viewer"] for t in g["trajs"]] == [COLORS[0].value]
    t = g["trajs"][0]
    assert len(t["choice"]) == len(t["x"]) == len(t["A"]) > 10
    assert all(0 <= c < len(a) for c, a in zip(t["choice"], t["A"]))
    shaping = t["reward"] - rewards(t["phi"], t["outcome"], shaping=False)
    assert shaping.sum() == pytest.approx(t["outcome"] - t["phi"][0])
    assert np.isfinite(t["phi"]).all() and np.all(t["logp"] <= 0)


def test_update_changes_params_masks_valid(net):
    from catan_rl.rl.net import ActorCritic, prep
    g = play_game(_job(net, ["learner"] * 4, seed=1_000_321))
    assert sorted(t["viewer"] for t in g["trajs"]) == sorted(c.value for c in COLORS)
    b = Batch.from_trajs(g["trajs"])
    torch.manual_seed(0)
    learner, bc = ActorCritic.from_bc(seed=0), ActorCritic.from_bc(seed=0)
    before = {k: v.copy() for k, v in learner.params().items()}
    st = update(learner, bc, torch.optim.Adam(learner.parameters(), lr=1e-3), b, PPOConfig(epochs=1, minibatch=128),
                0.01, 0.1, np.random.default_rng(0))
    after = learner.params()
    assert any(not np.array_equal(before[k], after[k]) for k in before)
    assert all(np.isfinite(v) for k, v in st.items() if isinstance(v, float) and k != "explained_variance")
    X, A, M = prep(learner.norm, b.x[:64], b.A[:64])
    with torch.no_grad():
        z, _ = learner(X, A, M)
    assert torch.isinf(z[~M]).all() and torch.isfinite(z[M]).all()
    assert torch.allclose(torch.softmax(z, -1).sum(-1), torch.ones(len(z)))


def test_value_warmup_freezes_policy(net):
    from catan_rl.rl.net import ActorCritic
    g = play_game(_job(net, ["learner"] * 4, seed=1_000_555))
    b = Batch.from_trajs(g["trajs"])
    torch.manual_seed(0)
    learner, bc = ActorCritic.from_bc(seed=0), ActorCritic.from_bc(seed=0)
    opt = torch.optim.Adam([{"params": learner.pi.parameters(), "lr": 1e-3}, {"params": learner.v.parameters(), "lr": 1e-3}])
    cfg, rng = PPOConfig(epochs=1, minibatch=128), np.random.default_rng(0)
    p0 = learner.params()
    st = update(learner, bc, opt, b, cfg, 0.01, 0.1, rng, value_only=True)
    p1 = learner.params()
    assert st["phase"] == "value_warmup"
    assert all(np.array_equal(p0[k], p1[k]) for k in p0 if k.startswith("pi."))
    assert any(not np.array_equal(p0[k], p1[k]) for k in p0 if k.startswith("v."))
    st = update(learner, bc, opt, b, cfg, 0.01, 0.1, rng, value_only=False)
    p2 = learner.params()
    assert st["phase"] == "ppo" and any(not np.array_equal(p1[k], p2[k]) for k in p1 if k.startswith("pi."))


def test_init_from_and_final_eval(tmp_path):
    import json
    from catan_rl.rl import final_eval as FE
    from catan_rl.rl.net import ActorCritic
    from catan_rl.rl.train import main
    _run(str(tmp_path / "a"), updates=1, warmup=0)
    src = tmp_path / "a" / "final.pt"
    init = ActorCritic.load(str(src)).params()
    bc = ActorCritic.from_bc(seed=0).params()
    assert any(not np.array_equal(init[k], bc[k]) for k in init if k.startswith("pi."))
    # --init-from with 0 updates: saved update-0 checkpoint must equal the source exactly
    main(["--out", str(tmp_path / "b"), "--updates", "0", "--workers", "1", "--eval-every", "0", "--init-from", str(src)])
    got = ActorCritic.load(str(tmp_path / "b" / "ckpt_0000.pt")).params()
    assert all(np.array_equal(init[k], got[k]) for k in init)
    assert json.load(open(tmp_path / "b" / "config.json"))["init_from"] == str(src)
    out = FE.evaluate([str(src)], workers=2, seeds={"vs_vf": 1, "vs_wr": 1, "vs_bc": 1})
    assert set(out) == {"bc", str(src)}
    for c in out:
        for k in ("vs_vf", "vs_wr", "vs_bc"):
            m = out[c][k]
            assert m["games"] == 4 and m["seeds"] == [5000, 5000] and 0 <= m["p"] <= 1
    assert out["bc"]["vs_vf"]["diff_vs_bc"]["mean"] == 0


def test_sampling_key_distinct_across_games():
    from catan_rl.policy.agent import sample_key
    from catan_rl.tests.test_policy import _play
    games = [e.game.copy() for s in (1, 2) for e in random_walk(s, viewers=[V], every=50)]  # walk mutates env
    keys = [sample_key(g, V) for g in games]
    assert len(set(keys)) == len(keys) >= 4
    g = games[0]
    assert sample_key(g, V) == sample_key(g.copy(), V) != sample_key(g, "BLUE")
    a, b = _play(21, "sample")[0], _play(22, "sample")[0]  # different games: different trajectories
    assert [str(r.action) for r in a.game.state.action_records] != [str(r.action) for r in b.game.state.action_records]


def test_belief_features_leak_free():
    from catanatron.models.enums import Action, ActionType
    from catan_rl.env.actions import action_array
    from catan_rl.rl.belief_features import DIM, belief_action_features
    kinds = (ActionType.MOVE_ROBBER, ActionType.PLAY_MONOPOLY, ActionType.PLAY_YEAR_OF_PLENTY)
    rng, n, nz = random.Random(4), 0, 0
    for env in random_walk(8, viewers=[V], every=10):
        g, changed = _hidden_twin(env, rng)
        if not changed:
            continue
        c = next(x for x in env.game.state.colors if x.value == V)
        acts = [Action(c, t, v) for t, v in action_array() if t in kinds
                and not (t == ActionType.MOVE_ROBBER and v[1] is not None and v[1].value == V)]
        X, Y = belief_action_features(env.game, c, env.belief(V), acts), belief_action_features(g, c, env.belief(V), acts)
        assert X.shape == (len(acts), DIM) and np.array_equal(X, Y)
        n, nz = n + 1, nz + int(np.abs(X).sum() > 0)
    assert n >= 3 and nz >= 1


def _run(out, workers=2, updates=2, eval_every=0, warmup=1):
    from catan_rl.rl.train import main
    return main(["--out", out, "--updates", str(updates), "--games-per-update", "2", "--workers", str(workers),
                 "--minibatch", "64", "--epochs", "1", "--ckpt-every", "1", "--eval-every", str(eval_every),
                 "--eval-seeds", "1", "--eval-workers", "2", "--shaping", "on", "--value-warmup", str(warmup),
                 "--lr", "2e-4", "--value-lr", "1e-3"])


def test_smoke_two_updates(tmp_path):
    import json
    _run(str(tmp_path), eval_every=1)
    log = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().strip().splitlines()]
    assert [r["phase"] for r in log] == ["value_warmup", "ppo"]
    rows = (tmp_path / "eval.csv").read_text().strip().splitlines()
    assert rows[0] == "update,win_rate,ci_lo,ci_hi,games,diff_vs_0,diff_lo,diff_hi"
    assert [r.split(",")[0] for r in rows[1:]] == ["0", "1", "2"]
    assert [float(x) for x in rows[1].split(",")[5:]] == [0.0, 0.0, 0.0]  # update 0 vs itself
    cfg = json.load(open(tmp_path / "config.json"))
    assert cfg["phi_version"].startswith("phi_v1") and cfg["lr"] == 2e-4 and cfg["value_lr"] == 1e-3
    assert cfg["value_warmup"] == 1


def test_reproducible_across_processes(tmp_path):
    code = "import sys; from catan_rl.tests.test_rl import _run; print(_run(sys.argv[1], updates=1)['digest'])"
    outs = [subprocess.run([sys.executable, "-c", code, str(tmp_path / f"r{h}")], capture_output=True, text=True,
                           check=True, env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip().splitlines()[-1]
            for h in (1, 2)]
    assert outs[0] == outs[1] and len(outs[0].split()[-1]) == 64
