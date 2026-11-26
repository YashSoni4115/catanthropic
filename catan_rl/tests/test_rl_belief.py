"""Spec R3: belief action features in the PPO net, agent, rollout and train (behind --belief-features)."""
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from catanatron.models.enums import ActionType

from catan_rl.policy.dataset import legal_actions
from catan_rl.rl.belief_features import DIM
from catan_rl.rl.net import ActorCritic, prep
from catan_rl.tests.belief_helpers import random_walk
from catan_rl.tests.test_determinize import V, _hidden_twin

pytestmark = pytest.mark.skipif(not os.path.exists("runs/policy/bc_v1/model.pt"), reason="needs the BC checkpoint")
KINDS = (ActionType.MOVE_ROBBER, ActionType.PLAY_MONOPOLY, ActionType.PLAY_YEAR_OF_PLENTY)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """A non-belief PPO checkpoint whose policy differs from BC (perturbed weights)."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    net = ActorCritic.from_bc(seed=0)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01 * torch.randn_like(p))
    path = str(tmp_path_factory.mktemp("src") / "src.pt")
    net.save(path, {"update": 0})
    return path


def _states(seed=8, n=6, belief_only=False):
    """Envs (V tracked) at states where V is to act; belief_only: V has a robber/monopoly/YoP choice."""
    out = []
    for s in range(seed, seed + 40):
        for env in random_walk(s, viewers=[V], every=1):
            st = env.game.state
            if st.current_color().value != V or st.is_initial_build_phase:
                continue
            kinds = {a.action_type for a in env.game.playable_actions}
            if belief_only and not kinds & set(KINDS):
                continue
            out.append(env)
            if len(out) >= n:
                return out
            break  # one state per walk (the env object keeps advancing)
    return out


def _acts(game, c):
    return legal_actions(game)  # action_features simulates each action, so only legal ones


def _probs(seat, env, c, acts):
    x, A = seat.features(env, c, acts)
    with torch.no_grad():
        z, v = seat.net(*prep(seat.net.norm, [x], [A]))
    return torch.softmax(z[0], -1).numpy(), z[0].numpy(), float(v[0])


def test_zero_init_warm_start_equivalence(src):
    from catan_rl.rl.agent import NetSeat
    base = ActorCritic.load(src)
    wide = base.with_belief(DIM)
    assert wide.belief_features and not base.belief_features
    n = 0
    for env in _states(n=4):
        c = env.game.state.current_color()
        acts = _acts(env.game, c)
        _, z0, v0 = _probs(NetSeat(base, c), env, c, acts)
        _, z1, v1 = _probs(NetSeat(wide, c), env, c, acts)
        assert np.allclose(z0, z1, atol=1e-6, rtol=0) and abs(v0 - v1) < 1e-6
        n += 1
    assert n >= 3


def test_agent_path_leak_free(src, tmp_path):
    """Non-zero belief weights; resampling opponents' hidden hands/dev cards must not move the policy."""
    from catan_rl.rl.agent import NetSeat
    net = ActorCritic.load(src).with_belief(DIM)
    torch.manual_seed(3)
    with torch.no_grad():
        net.pi.head[0].weight[:, -DIM:] = torch.randn(net.pi.head[0].weight.shape[0], DIM)
    rng, n, moved = random.Random(4), 0, 0
    for env in _states(seed=8, n=8, belief_only=True):
        g, changed = _hidden_twin(env, rng)
        if not changed:
            continue
        c = env.game.state.current_color()
        acts = _acts(env.game, c)
        seat = NetSeat(net, c)
        twin = SimpleNamespace(game=g, belief=env.belief)  # same public tracker, different hidden truth
        p, _, _ = _probs(seat, env, c, acts)
        q, _, _ = _probs(seat, twin, c, acts)
        assert np.array_equal(p, q)
        n += 1
        # sanity: belief columns actually influence the policy here
        _, A = seat.features(env, c, acts)
        moved += int(np.abs(A[:, -DIM:]).sum() > 0)
    assert n >= 3 and moved >= 1


def test_belief_checkpoint_round_trip(src, tmp_path):
    from catan_rl.rl import agent as AG
    net = ActorCritic.load(src).with_belief(DIM)
    with torch.no_grad():
        net.pi.head[0].weight[:, -DIM:] = 0.1
    path = str(tmp_path / "belief.pt")
    net.save(path, {"update": 1})
    ck = torch.load(path, weights_only=False)
    assert ck["meta"]["belief_features"] is True and ck["db"] == DIM
    assert AG.needs_belief(path)
    from catan_rl.eval import agents as EA
    assert EA.get(f"ppo:{path}").needs_belief
    a = AG.PPOAgent(next(iter(_states(n=1))).game.state.current_color(), path)
    assert a.net.db == DIM
    for env in _states(n=3):
        c = env.game.state.current_color()
        acts = _acts(env.game, c)
        from catan_rl.rl.agent import NetSeat
        p, _, v = _probs(NetSeat(net, c), env, c, acts)
        q, _, w = _probs(AG.PPOAgent(c, path), env, c, acts)
        assert np.array_equal(p, q) and v == w


def test_old_checkpoint_loads_and_plays():
    from catan_rl.eval.match import run_match
    from catan_rl.rl.agent import needs_belief
    path = os.path.abspath("checkpoints/ppo_B313.pt")
    if not os.path.exists(path):
        pytest.skip("no B313 checkpoint")
    net = ActorCritic.load(path)
    assert net.db == 0 and not needs_belief(path)
    res = run_match([f"ppo:{path}", "random", "random", "random"], 1, 300, 1)
    assert len(res) == 4


def test_belief_match_plays(src, tmp_path):
    from catan_rl.eval.match import run_match
    net = ActorCritic.load(src).with_belief(DIM)
    path = str(tmp_path / "b.pt")
    net.save(path, {})
    res = run_match([f"ppo:{path}", "random", "random", "random"], 1, 300, 1)
    assert len(res) == 4


def _run_belief(out, src):
    from catan_rl.rl.train import main
    return main(["--out", out, "--belief-features", "--init-from", src, "--updates", "1", "--games-per-update", "2",
                 "--workers", "2", "--eval-every", "0", "--minibatch", "64", "--epochs", "1", "--value-warmup", "0",
                 "--shaping", "on", "--lr", "2e-4", "--value-lr", "1e-3"])


def test_belief_reproducible_across_processes(src, tmp_path):
    import subprocess
    import sys
    code = ("import sys; from catan_rl.tests.test_rl_belief import _run_belief; "
            "print(_run_belief(sys.argv[1], sys.argv[2])['digest'])")
    outs = [subprocess.run([sys.executable, "-c", code, str(tmp_path / f"r{h}"), src], capture_output=True, text=True,
                           check=True, env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip().splitlines()[-1]
            for h in (1, 2, 3)]
    assert outs[0] == outs[1] == outs[2] and len(outs[0].split()[-1]) == 64


def test_smoke_train_belief_warm_start(src, tmp_path):
    from catan_rl.rl.train import main
    out = tmp_path / "r3"
    main(["--out", str(out), "--belief-features", "--init-from", src, "--updates", "1", "--games-per-update", "4",
          "--workers", "2", "--eval-every", "0", "--minibatch", "64", "--epochs", "1", "--value-warmup", "0",
          "--shaping", "on"])
    c0, base = ActorCritic.load(str(out / "ckpt_0000.pt")), ActorCritic.load(src)
    assert c0.db == DIM and c0.meta["belief_features"] is True
    p0, pb = c0.params(), base.params()
    for k in pb:
        if k == "pi.head.0.weight":
            assert np.array_equal(p0[k][:, :-DIM], pb[k]) and not p0[k][:, -DIM:].any()
        else:
            assert np.array_equal(p0[k], pb[k])
    fin = ActorCritic.load(str(out / "final.pt"))
    assert fin.db == DIM and fin.params()["pi.head.0.weight"][:, -DIM:].any()  # belief weights trained
