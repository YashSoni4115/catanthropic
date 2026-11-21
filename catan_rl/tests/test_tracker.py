import unittest

from catan_rl.belief.tracker import (
    BeliefContradictionError,
    ResourceBeliefTrackerV1,
)


class TestResourceBeliefTrackerV1(unittest.TestCase):
    def test_initialize_uniform_total_one(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("P1", observed_total_cards=1)
        summary = tracker.get_player_summary("P1")

        self.assertEqual(summary["support_size"], 5)
        self.assertEqual(summary["known_total"], 1)
        for resource in ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"):
            self.assertAlmostEqual(summary["expected_counts"][resource], 0.2)
            self.assertAlmostEqual(summary["p_at_least_one"][resource], 0.2)

    def test_known_gain_deterministic(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("P1", observed_total_cards=0)
        tracker.apply_event(
            {
                "type": "known_resource_gain",
                "player_id": "P1",
                "resource": "WOOL",
                "amount": 1,
            }
        )
        summary = tracker.get_player_summary("P1")

        self.assertEqual(summary["known_total"], 1)
        self.assertAlmostEqual(summary["expected_counts"]["WOOL"], 1.0)
        self.assertAlmostEqual(summary["p_at_least_one"]["WOOL"], 1.0)
        self.assertAlmostEqual(summary["p_at_least_one"]["BRICK"], 0.0)

    def test_known_spend_filters_inconsistent_states(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("P1", observed_total_cards=3)
        tracker.apply_event({"type": "known_spend", "player_id": "P1", "spend_type": "dev_card"})
        summary = tracker.get_player_summary("P1")

        self.assertEqual(summary["support_size"], 1)
        self.assertEqual(summary["known_total"], 0)
        for resource in ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"):
            self.assertAlmostEqual(summary["expected_counts"][resource], 0.0)

    def test_random_steal_from_deterministic_victim(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("thief", observed_total_cards=0)
        tracker.initialize_player("victim", observed_total_cards=1, known_counts={"WOOL": 1})

        tracker.apply_event({"type": "random_steal", "thief_id": "thief", "victim_id": "victim", "amount": 1})

        thief_summary = tracker.get_player_summary("thief")
        victim_summary = tracker.get_player_summary("victim")

        self.assertAlmostEqual(thief_summary["expected_counts"]["WOOL"], 1.0)
        self.assertAlmostEqual(victim_summary["expected_counts"]["WOOL"], 0.0)
        self.assertEqual(thief_summary["known_total"], 1)
        self.assertEqual(victim_summary["known_total"], 0)

    def test_random_steal_infers_victim_distribution(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("thief", observed_total_cards=0)
        tracker.initialize_player("victim", observed_total_cards=1)
        tracker.apply_event({"type": "random_steal", "thief_id": "thief", "victim_id": "victim", "amount": 1})

        thief_summary = tracker.get_player_summary("thief")
        for resource in ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"):
            self.assertAlmostEqual(thief_summary["expected_counts"][resource], 0.2)
            self.assertAlmostEqual(thief_summary["p_at_least_one"][resource], 0.2)

    def test_reconcile_public_total(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("P1", observed_total_cards=1)
        tracker.apply_event({"type": "known_resource_gain", "player_id": "P1", "resource": "BRICK", "amount": 1})
        tracker.reconcile_public_totals("P1", total_cards=2)

        summary = tracker.get_player_summary("P1")
        self.assertEqual(summary["known_total"], 2)
        self.assertEqual(summary["support_size"], 5)

    def test_affordability_probabilities(self):
        tracker = ResourceBeliefTrackerV1()
        tracker.initialize_player("P1", observed_total_cards=2, known_counts={"BRICK": 1, "LUMBER": 1})
        summary = tracker.get_player_summary("P1")

        self.assertAlmostEqual(summary["p_can_afford"]["road"], 1.0)
        self.assertAlmostEqual(summary["p_can_afford"]["settlement"], 0.0)
        self.assertAlmostEqual(summary["p_can_afford"]["city"], 0.0)
        self.assertAlmostEqual(summary["p_can_afford"]["dev_card"], 0.0)

    def test_strict_contradiction_raises(self):
        tracker = ResourceBeliefTrackerV1(strict=True)
        tracker.initialize_player("P1", observed_total_cards=1, known_counts={"BRICK": 1})
        with self.assertRaises(BeliefContradictionError):
            tracker.apply_event({"type": "known_spend", "player_id": "P1", "spend_type": "city"})


if __name__ == "__main__":
    unittest.main()
