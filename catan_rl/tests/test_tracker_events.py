import random
import unittest

from catan_rl.belief.events import (
    BankTradeEvent, DevBuyEvent, DiscardEvent, MonopolyEvent, PlayerTradeEvent,
    ProductionEvent, SpendEvent, StealEvent, TotalReconcileEvent, YearOfPlentyEvent,
)
from catan_rl.belief.tracker import (
    RESOURCES, BeliefContradictionError, BeliefTracker, random_cards_discard,
)


def tracker_with(**hands):
    t = BeliefTracker()
    for p, h in hands.items():
        if isinstance(h, int):
            t.initialize_player(p, observed_total_cards=h)
        else:
            t.initialize_player(p, observed_total_cards=sum(h.values()), known_counts={r: h.get(r, 0) for r in RESOURCES})
    return t


def pmf(t, p):
    return t.players[p].pmf


class TestTypedHandlers(unittest.TestCase):
    def test_production_and_yop_gain(self):
        t = tracker_with(A={})
        t.apply(ProductionEvent("A", {"ORE": 2}))
        t.apply(YearOfPlentyEvent("A", {"WOOL": 1, "ORE": 1}))
        self.assertEqual(pmf(t, "A"), {(0, 1, 3, 0, 0): 1.0})

    def test_spend_and_devbuy(self):
        t = tracker_with(A=4)
        t.apply(DevBuyEvent("A"))
        t.apply(SpendEvent("A", {"BRICK": 1}))
        self.assertEqual(pmf(t, "A"), {(0, 0, 0, 0, 0): 1.0})

    def test_bank_trade(self):
        t = tracker_with(A={"GRAIN": 4})
        t.apply(BankTradeEvent("A", give={"GRAIN": 4}, get={"ORE": 1}))
        self.assertEqual(pmf(t, "A"), {(0, 0, 1, 0, 0): 1.0})

    def test_player_trade(self):
        t = tracker_with(A={"BRICK": 1}, B={"WOOL": 2})
        t.apply(PlayerTradeEvent("A", "B", give={"BRICK": 1}, get={"WOOL": 2}))
        self.assertEqual(pmf(t, "A"), {(0, 2, 0, 0, 0): 1.0})
        self.assertEqual(pmf(t, "B"), {(1, 0, 0, 0, 0): 1.0})

    def test_known_discard(self):
        t = tracker_with(A={"ORE": 3, "WOOL": 5})
        t.apply(DiscardEvent("A", {"WOOL": 4}, 4))
        self.assertEqual(pmf(t, "A"), {(0, 1, 3, 0, 0): 1.0})

    def test_hidden_discard_uniform_sub_multisets(self):
        t = tracker_with(A={"ORE": 2, "WOOL": 2})
        t.apply(DiscardEvent("A", None, 2))
        # sub-multisets of size 2: {OO, OW, WW} -> uniform 1/3 each
        p = pmf(t, "A")
        self.assertEqual(len(p), 3)
        for v in p.values():
            self.assertAlmostEqual(v, 1 / 3)

    def test_hidden_discard_configurable_model(self):
        t = BeliefTracker(discard_model=random_cards_discard)
        t.initialize_player("A", 4, {r: (2 if r in ("ORE", "WOOL") else 0) for r in RESOURCES})
        t.apply(DiscardEvent("A", None, 2))
        # hypergeometric: OO 1/6, OW 4/6, WW 1/6
        self.assertAlmostEqual(pmf(t, "A")[(0, 1, 1, 0, 0)], 4 / 6)

    def test_steal_from_one_card_victim(self):
        t = tracker_with(T={}, V=1)
        t.apply(StealEvent("T", "V"))
        self.assertEqual(pmf(t, "V"), {(0, 0, 0, 0, 0): 1.0})
        for v in pmf(t, "T").values():
            self.assertAlmostEqual(v, 0.2)

    def test_known_steal_is_evidence(self):
        t = tracker_with(T={}, V=2)
        t.apply(StealEvent("T", "V", "ORE"))
        p = pmf(t, "V")
        # prior uniform over 15 hands; P(ORE|hand)=ore/2. (0,0,2,0,0)->1, (.,.,1,.,.)x4 ->0.5 each
        self.assertAlmostEqual(p[(0, 0, 1, 0, 0)], 1 / 3)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertEqual(pmf(t, "T"), {(0, 0, 1, 0, 0): 1.0})

    def test_monopoly_per_victim(self):
        t = tracker_with(A={}, B=2, C=1)
        t.apply(MonopolyEvent("A", "GRAIN", 2, per_victim={"B": 2, "C": 0}))
        self.assertEqual(pmf(t, "B"), {(0, 0, 0, 0, 0): 1.0})
        self.assertTrue(all(h[3] == 0 for h in pmf(t, "C")))
        self.assertEqual(len(pmf(t, "C")), 4)
        self.assertEqual(pmf(t, "A"), {(0, 0, 0, 2, 0): 1.0})

    def test_monopoly_total_only(self):
        t = tracker_with(A={}, B={"GRAIN": 1}, C=1)
        t.apply(MonopolyEvent("A", "GRAIN", 2))  # forces C's card to be GRAIN
        self.assertEqual(pmf(t, "C"), {(0, 0, 0, 0, 0): 1.0})
        self.assertEqual(pmf(t, "B"), {(0, 0, 0, 0, 0): 1.0})
        self.assertEqual(pmf(t, "A"), {(0, 0, 0, 2, 0): 1.0})

    def test_monopoly_total_conflict_strict_raises(self):
        t = tracker_with(A={}, B=1, C=1)
        with self.assertRaises(BeliefContradictionError):
            t.apply(MonopolyEvent("A", "GRAIN", 3))

    def test_reconcile(self):
        t = tracker_with(A=1)
        t.apply(ProductionEvent("A", {"ORE": 1}))
        t.apply(TotalReconcileEvent("A", 2))
        self.assertEqual(t.players["A"].known_total, 2)


class TestPropertyTrueHandSupported(unittest.TestCase):
    """Random event sequences from ground truth: the true hand never gets zero probability."""

    def _run(self, seed, steps=40):
        rng = random.Random(seed)
        players = ["P0", "P1", "P2"]
        truth = {p: [0] * 5 for p in players}
        t = BeliefTracker(strict=True)
        for p in players:
            for _ in range(rng.randint(0, 4)):
                truth[p][rng.randrange(5)] += 1
            t.initialize_player(p, sum(truth[p]))  # only the total is public

        def cards(vec):
            return {RESOURCES[i]: n for i, n in enumerate(vec) if n}

        def take(p, i, n=1):
            truth[p][i] -= n

        for _ in range(steps):
            p = rng.choice(players)
            h = truth[p]
            kind = rng.choice(["prod", "prod", "spend", "bank", "ptrade", "discard", "steal", "mono", "yop", "dev"])
            if kind == "prod":
                g = [0] * 5
                for _ in range(rng.randint(1, 3)):
                    g[rng.randrange(5)] += 1
                for i in range(5):
                    h[i] += g[i]
                t.apply(ProductionEvent(p, cards(g)))
            elif kind == "yop":
                g = [0] * 5
                g[rng.randrange(5)] += 1
                g[rng.randrange(5)] += 1
                for i in range(5):
                    h[i] += g[i]
                t.apply(YearOfPlentyEvent(p, cards(g)))
            elif kind == "spend" and sum(h):
                i = rng.choice([i for i in range(5) if h[i]])
                take(p, i)
                t.apply(SpendEvent(p, {RESOURCES[i]: 1}))
            elif kind == "dev" and h[1] and h[2] and h[3]:
                for i in (1, 2, 3):
                    take(p, i)
                t.apply(DevBuyEvent(p))
            elif kind == "bank" and max(h) >= 4:
                i = h.index(max(h))
                j = rng.randrange(5)
                take(p, i, 4)
                h[j] += 1
                t.apply(BankTradeEvent(p, {RESOURCES[i]: 4}, {RESOURCES[j]: 1}))
            elif kind == "ptrade":
                q = rng.choice([x for x in players if x != p])
                if sum(h) and sum(truth[q]):
                    i = rng.choice([i for i in range(5) if h[i]])
                    j = rng.choice([j for j in range(5) if truth[q][j]])
                    take(p, i); truth[q][i] += 1
                    take(q, j); h[j] += 1
                    t.apply(PlayerTradeEvent(p, q, {RESOURCES[i]: 1}, {RESOURCES[j]: 1}))
            elif kind == "discard" and sum(h) >= 4:
                n = sum(h) // 2
                pool = [i for i in range(5) for _ in range(h[i])]
                d = [0] * 5
                for i in rng.sample(pool, n):
                    d[i] += 1
                for i in range(5):
                    h[i] -= d[i]
                t.apply(DiscardEvent(p, cards(d) if rng.random() < 0.3 else None, n))
            elif kind == "steal":
                q = rng.choice([x for x in players if x != p])
                if sum(truth[q]):
                    i = rng.choice([i for i in range(5) for _ in range(truth[q][i])])
                    take(q, i); h[i] += 1
                    t.apply(StealEvent(p, q, RESOURCES[i] if rng.random() < 0.5 else None))
            elif kind == "mono":
                i = rng.randrange(5)
                per = {q: truth[q][i] for q in players if q != p}
                for q in per:
                    truth[q][i] = 0
                h[i] += sum(per.values())
                t.apply(MonopolyEvent(p, RESOURCES[i], sum(per.values()), per if rng.random() < 0.5 else None))
            for q in players:
                self.assertGreater(t.players[q].pmf.get(tuple(truth[q]), 0.0), 0.0, f"seed={seed} kind={kind}")

    def test_random_sequences(self):
        for seed in range(150):
            self._run(seed)


if __name__ == "__main__":
    unittest.main()
