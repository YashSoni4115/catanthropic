import random
from collections import Counter, defaultdict

import pytest
from catanatron.models.enums import RESOURCES as R, ActionRecord, ActionType
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import player_key

from catan_rl.belief.devcards import DEV_DECK
from catan_rl.env.determinize import CAT_OF_TRK, conservation_sums, joint_support, own_hand, sample_hands
from catan_rl.tests.belief_helpers import color, random_walk, true_dev, true_hand

V = "RED"
PUBLIC_KEYS = ("VICTORY_POINTS", "ROADS_AVAILABLE", "SETTLEMENTS_AVAILABLE", "CITIES_AVAILABLE", "HAS_ROAD",
               "HAS_ARMY", "HAS_ROLLED", "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", "LONGEST_ROAD_LENGTH") + tuple(
    f"PLAYED_{c}" for c in DEV_DECK)


def snapshot(g):
    st = g.state
    return (sorted(st.player_state.items()), list(st.development_listdeck), list(st.resource_freqdeck),
            [repr(r) for r in st.action_records], [repr(a) for a in g.playable_actions],
            st.dice_rng.getstate(), st.steal_rng.getstate(), g.random.getstate())


def check_consistent(env, g):
    real, st = env.game.state, g.state
    db = env.dev_belief(V)
    S = conservation_sums(real.resource_freqdeck, own_hand(env.belief(V), V))
    opp = [c for c in real.colors if c.value != V]
    assert tuple(sum(true_hand(st, c)[i] for c in opp) for i in range(5)) == S
    assert real.resource_freqdeck == st.resource_freqdeck and real.num_turns == st.num_turns
    assert real.board.robber_coordinate == st.board.robber_coordinate and real.buildings_by_color == st.buildings_by_color
    assert len(st.development_listdeck) == len(real.development_listdeck)
    assert +db.unseen == Counter(st.development_listdeck) + sum((true_dev(st, c) for c in opp), Counter())
    for c in real.colors:
        k = player_key(st, c)
        assert sum(true_hand(st, c)) == sum(true_hand(real, c))
        assert sum(true_dev(st, c).values()) == sum(true_dev(real, c).values())
        for key in PUBLIC_KEYS:
            assert st.player_state[f"{k}_{key}"] == real.player_state[f"{k}_{key}"]
        assert st.player_state[f"{k}_ACTUAL_VICTORY_POINTS"] == (
            st.player_state[f"{k}_VICTORY_POINTS"] + st.player_state[f"{k}_VICTORY_POINT_IN_HAND"])
        assert st.player_state[f"{k}_ACTUAL_VICTORY_POINTS"] < g.vps_to_win
        if c.value == V:
            assert true_hand(st, c) == true_hand(real, c) and true_dev(st, c) == true_dev(real, c)
        else:
            old = db.old[c.value]
            for card in ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING"):
                if st.player_state[f"{k}_{card}_OWNED_AT_START"]:
                    assert old > 0 and st.player_state[f"{k}_{card}_IN_HAND"] > 0


@pytest.mark.parametrize("seed", range(4))
def test_consistency(seed):
    n = 0
    for env in random_walk(seed, viewers=[V], every=7):
        for s in range(3):
            check_consistent(env, env.determinize(V, s))
            n += 1
    assert n > 50


def test_deterministic_given_seed():
    for env in random_walk(0, viewers=[V], every=60):
        assert snapshot(env.determinize(V, 5)) == snapshot(env.determinize(V, 5))
        assert snapshot(env.determinize(V, 5)) != snapshot(env.determinize(V, 6))


def _hidden_twin(env, rng):
    """A copy of the real game differing only in hidden info (same public totals and per-resource sums)."""
    g = env.game.copy()
    st = g.state
    opp = [c for c in st.colors if c.value != V]
    changed = False
    for a in opp:  # swap one x from a with one y from b
        for b in opp:
            ha, hb = true_hand(st, a), true_hand(st, b)
            for x in range(5):
                for y in range(5):
                    if a != b and x != y and ha[x] and hb[y] and not changed:
                        ka, kb, rx, ry = player_key(st, a), player_key(st, b), R[CAT_OF_TRK[x]], R[CAT_OF_TRK[y]]
                        st.player_state[f"{ka}_{rx}_IN_HAND"] -= 1; st.player_state[f"{ka}_{ry}_IN_HAND"] += 1
                        st.player_state[f"{kb}_{ry}_IN_HAND"] -= 1; st.player_state[f"{kb}_{rx}_IN_HAND"] += 1
                        changed = True
    for c in opp:  # swap an opponent's hidden card with a different deck card
        k = player_key(st, c)
        for card, n in true_dev(st, c).items():
            j = next((i for i, d in enumerate(st.development_listdeck) if d != card), None)
            if n and j is not None:
                other = st.development_listdeck[j]
                st.development_listdeck[j] = card
                st.player_state[f"{k}_{card}_IN_HAND"] -= 1
                st.player_state[f"{k}_{other}_IN_HAND"] += 1
                d = (other == "VICTORY_POINT") - (card == "VICTORY_POINT")
                st.player_state[f"{k}_ACTUAL_VICTORY_POINTS"] += d
                st.player_state[f"{k}_KNIGHT_OWNED_AT_START"] ^= True
                changed = True
                break
    rng.shuffle(st.development_listdeck)
    st.action_records = [ActionRecord(action=r.action, result="VICTORY_POINT")
                         if r.action.action_type == ActionType.BUY_DEVELOPMENT_CARD and r.action.color.value != V else r
                         for r in st.action_records]
    return g, changed


def test_no_leakage():
    rng, tested = random.Random(0), 0
    for seed in range(3):
        for env in random_walk(seed, viewers=[V], every=25):
            real = env.game
            twin, changed = _hidden_twin(env, rng)
            if not changed:
                continue
            assert snapshot(twin) != snapshot(real)
            a = env.determinize(V, 11)
            env.game = twin
            try:
                b = env.determinize(V, 11)
            finally:
                env.game = real
            assert snapshot(a) == snapshot(b)
            tested += 1
    assert tested > 10


def _ambiguous_position():
    for seed in range(40):
        for env in random_walk(seed, viewers=[V], every=3):
            trk = env.belief(V)
            opp = [c.value for c in env.game.state.colors if c.value != V]
            pmfs = {o: trk.players[o].pmf for o in opp}
            if sum(len(p) > 1 for p in pmfs.values()) >= 2 and max(len(p) for p in pmfs.values()) <= 30:
                return env, pmfs
    raise AssertionError("no ambiguous position found")


def test_calibration():
    env, pmfs = _ambiguous_position()
    st = env.game.state
    S = conservation_sums(st.resource_freqdeck, own_hand(env.belief(V), V))
    support, Z = joint_support(pmfs, S)
    exact = defaultdict(float)
    for a, w in support:
        for o, h in a.items():
            exact[o, h] += w / Z
    n, rng, freq, tries = 4000, random.Random(1), Counter(), 0
    for _ in range(n):
        hs, info = sample_hands(pmfs, S, rng)
        tries += info["tries"]
        assert tuple(sum(h[i] for h in hs.values()) for i in range(5)) == S
        for o, h in hs.items():
            freq[o, h] += 1
    for o in pmfs:
        keys = {h for (p, h) in list(exact) + list(freq) if p == o}
        tv = 0.5 * sum(abs(freq[o, h] / n - exact[o, h]) for h in keys)
        assert tv < 0.04, (o, tv)
        assert all(pmfs[o].get(h, 0) > 0 for h in keys if freq[o, h])  # never outside tracker support
    assert n / tries > 0.01  # acceptance rate sanity


@pytest.mark.parametrize("seed", range(3))
def test_plays_to_completion(seed):
    played = 0
    for env in random_walk(seed, viewers=[V], every=40):
        g = env.determinize(V, seed)
        g.state.players = [WeightedRandomPlayer(c) for c in g.state.colors]
        g.play()
        assert g.winning_color() is not None or g.state.num_turns >= 1000
        played += 1
    assert played >= 3
