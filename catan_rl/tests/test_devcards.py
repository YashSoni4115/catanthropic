import itertools
from collections import Counter

import pytest

from catan_rl.belief import events as ev
from catan_rl.belief.devcards import DevCardBelief
from catan_rl.env.determinize import conservation_sums, own_hand
from catan_rl.tests.belief_helpers import ALL, color, random_walk, true_dev, true_hand


def brute(unseen: Counter, slots):
    """Enumerate all orderings of the unseen cards as distinct objects; slots = list of sizes."""
    cards = list(unseen.elements())
    out = []
    for perm in itertools.permutations(range(len(cards))):
        hands, i = [], 0
        for s in slots:
            hands.append(Counter(cards[j] for j in perm[i:i + s]))
            i += s
        out.append(hands)
    return out


def test_closed_form_matches_brute_force():
    deck = {"KNIGHT": 3, "VICTORY_POINT": 2, "MONOPOLY": 1, "ROAD_BUILDING": 1}
    b = DevCardBelief("A", "ABC", deck)
    for e in [ev.DevBuyEvent("B"), ev.TurnEndEvent("B"), ev.DevBuyEvent("B"), ev.DevBuyEvent("B"),
              ev.DevBuyEvent("A", "KNIGHT"), ev.DevBuyEvent("C"), ev.TurnEndEvent("C"),
              ev.DevPlayEvent("C", "MONOPOLY"), ev.DevBuyEvent("C")]:
        b.apply(e)
    assert (b.old["B"], b.new["B"], b.hidden("C")) == (1, 2, 1) and b.deck_size == 1
    worlds = brute(b.unseen, [b.old["B"], b.new["B"], b.hidden("C")])
    n = len(worlds)
    B = [w[0] + w[1] for w in worlds]
    C = [w[2] for w in worlds]
    for card in ("VICTORY_POINT", "KNIGHT"):
        assert b.p_at_least_one("B", card) == pytest.approx(sum(h[card] > 0 for h in B) / n)
        assert b.p_at_least_one("C", card) == pytest.approx(sum(h[card] > 0 for h in C) / n)
        assert b.expected_count("B", card) == pytest.approx(sum(h[card] for h in B) / n)
        assert b.count_pmf("B", card) == pytest.approx([sum(h[card] == k for h in B) / n for k in range(4)])
    assert b.p_at_least_one("B", "KNIGHT", old_only=True) == pytest.approx(sum(w[0]["KNIGHT"] > 0 for w in worlds) / n)
    assert b.summary("B", public_vp=3)["expected_total_vp"] == pytest.approx(3 + sum(h["VICTORY_POINT"] for h in B) / n)


def test_contradiction_on_impossible_play():
    b = DevCardBelief("A", "AB")
    with pytest.raises(Exception):
        b.apply(ev.DevPlayEvent("B", "KNIGHT"))  # B owns no playable card


@pytest.mark.parametrize("seed", range(6))
def test_ground_truth_dev_and_conservation(seed):
    """In real games: true hidden dev cards always have nonzero probability, counts match, and the
    per-resource conservation identity holds for every viewer."""
    checks = 0
    for env in random_walk(seed):
        st = env.game.state
        deck = Counter(st.development_listdeck)
        for v in ALL:
            db, trk = env.dev_belief(v), env.belief(v)
            own = own_hand(trk, v)
            assert own == true_hand(st, color(v))
            assert db.own == true_dev(st, color(v))
            S = conservation_sums(st.resource_freqdeck, own)
            opp = [true_hand(st, color(o)) for o in ALL if o != v]
            assert tuple(sum(h[i] for h in opp) for i in range(5)) == S
            assert db.deck_size == len(st.development_listdeck)
            assert +db.unseen == deck + sum((true_dev(st, color(o)) for o in ALL if o != v), Counter())
            for o in db.opponents():
                td = true_dev(st, color(o))
                assert db.hidden(o) == sum(td.values())
                for card in ("VICTORY_POINT", "KNIGHT"):
                    assert db.count_pmf(o, card)[td[card]] > 0
                checks += 1
    assert checks > 500
