import copy
import json
import random
import unittest

from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, ActionType
from catanatron.players.minimax import AlphaBetaPlayer

from catan_rl.belief import events as ev
from catan_rl.env import COLORS, NUM_ACTIONS, CatanEnv, action_array, random_agent
from catan_rl.env import event_adapter as ea
from catan_rl.env.validate import ValidatingEnv

ALL = [c.value for c in COLORS]


def play(seed, agents=None, **kw):
    env = CatanEnv(**{"track_viewers": ALL, **kw}).reset(seed, agents or [random_agent] * 4)
    env.play()
    return env


class TestActionSpace(unittest.TestCase):
    def test_matches_catanatron_action_array(self):
        from catanatron.gym.envs.action_space import get_action_array  # test-only reference
        self.assertEqual(action_array(), get_action_array(COLORS, "BASE"))
        self.assertEqual(NUM_ACTIONS, 370)

    def test_every_playable_action_maps_and_no_domestic_trades(self):
        seen = set()

        def agent(env, color):
            acts = env.legal_actions()
            self.assertGreater(len(acts), 1)  # forced moves never reach agents
            for a in env.game.playable_actions:
                self.assertNotIn(a.action_type.name, ("OFFER_TRADE", "ACCEPT_TRADE", "CONFIRM_TRADE"))
            seen.update(action_array()[i][0] for i in acts)
            return acts[env.game.state.random.randrange(len(acts))]

        for s in range(5):
            env = play(s, [agent] * 4)
            self.assertEqual(env.forced_decisions + env.agent_decisions, len(env.game.state.action_records))
        self.assertIn(ActionType.MARITIME_TRADE, seen)


class TestExternalStep(unittest.TestCase):
    def test_step_drives_external_seat(self):
        env = CatanEnv().reset(3, [None, random_agent, random_agent, random_agent])
        rng = random.Random(0)
        steps = 0
        while not env.done:
            self.assertEqual(env.current_player(), COLORS[0])
            env.step(rng.choice(env.legal_actions()))
            steps += 1
        self.assertGreater(steps, 10)
        self.assertEqual(set(env.public_vp()), {c.value for c in COLORS})


class TestObservationLeak(unittest.TestCase):
    HIDDEN_KEYS = ("ACTUAL_VICTORY_POINTS", "OWNED_AT_START", "development_listdeck",
                   "random_state", "seed", "dice_rng", "steal_rng", "random")

    def _scan_keys(self, obj, viewer, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                own = path == f"players.{viewer}"
                for bad in self.HIDDEN_KEYS:
                    if own and ks == "actual_victory_points":
                        continue  # the viewer's own actual VP is allowed
                    self.assertNotIn(bad.lower(), ks.lower(), f"hidden key {ks} at {path}")
                if path.startswith("players.") and not path.startswith(f"players.{viewer}"):
                    self.assertNotIn(ks, ("hand", "dev_cards", "actual_victory_points"), path)
                self._scan_keys(v, viewer, f"{path}.{ks}" if path else ks)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                self._scan_keys(x, viewer, path)

    def _perturb_hidden(self, env, viewer):
        """Change everything hidden from ``viewer`` while keeping public totals."""
        g = env.game.copy()
        s, rng = g.state, random.Random(99)
        for c in s.colors:
            if c == viewer:
                continue
            k = f"P{s.color_to_index[c]}_"
            for group in (RESOURCES, DEVELOPMENT_CARDS[:-1]):  # VP cards would change actual VP only
                n = sum(s.player_state[f"{k}{r}_IN_HAND"] for r in group)
                new = [0] * len(group)
                for _ in range(n):
                    new[rng.randrange(len(group))] += 1
                for r, x in zip(group, new):
                    s.player_state[f"{k}{r}_IN_HAND"] = x
            s.player_state[f"{k}ACTUAL_VICTORY_POINTS"] += 7
            for d in DEVELOPMENT_CARDS[:-1]:
                s.player_state[f"{k}{d}_OWNED_AT_START"] = not s.player_state[f"{k}{d}_OWNED_AT_START"]
        s.development_listdeck = list(s.development_listdeck)
        rng.shuffle(s.development_listdeck)
        s.reseed_rule_streams(rng)
        s.random = random.Random(123)
        return g

    def test_no_hidden_field_or_value(self):
        from catan_rl.env.observation import observed_state
        for seed in (987654321, 987654322, 987654323):
            env = CatanEnv().reset(seed, [None] * 4)
            rng = random.Random(seed)
            secrets = [str(seed)] + [str(x) for x in env.game.state.dice_rng.getstate()[1][1:4]]
            for i in range(400):
                if env.done:
                    break
                if i % 20 == 0:
                    for v in COLORS:
                        obs = env.observed_state(v)
                        blob = json.dumps(obs)  # plain data only
                        self._scan_keys(obs, v.value)
                        for secret in secrets:
                            self.assertNotIn(secret, blob)
                        self.assertEqual(obs, observed_state(self._perturb_hidden(env, v).state, v),
                                         "observation depends on hidden info")
                env.step(rng.choice(env.legal_actions()))


class TestEvents(unittest.TestCase):
    def test_steal_card_visible_only_to_parties(self):
        env = play(4)
        steals = {v: [e for e in es if isinstance(e, ev.StealEvent)] for v, es in env.events.items()}
        self.assertTrue(any(steals.values()))
        n = len(next(iter(steals.values())))
        for v, es in steals.items():
            self.assertEqual(len(es), n)  # everyone sees every steal, once
            for e in es:
                self.assertEqual(e.card is not None, v in (e.thief, e.victim))

    def test_event_types_and_no_ground_truth(self):
        types = set()
        for s in range(10):
            env = play(s)
            for es in env.events.values():
                types.update(type(e).__name__ for e in es)
        for need in ("ProductionEvent", "SpendEvent", "BankTradeEvent", "DevBuyEvent", "DiscardEvent",
                     "StealEvent", "TotalReconcileEvent", "YearOfPlentyEvent", "MonopolyEvent"):
            self.assertIn(need, types)

    def test_forced_moves_still_emit_events(self):
        env = play(5)
        rolls = sum(r.action.action_type == ActionType.ROLL for r in env.game.state.action_records)
        prods = [e for e in env.events["RED"] if isinstance(e, ev.ProductionEvent)]
        self.assertGreater(env.forced_decisions, 0)
        self.assertGreater(len(prods), rolls // 2)


class TestLookahead(unittest.TestCase):
    def _future(self, rng, n, f):
        r = copy.deepcopy(rng)
        return [f(r) for _ in range(n)]

    def test_lookahead_cannot_reproduce_real_dice_or_steals(self):
        env = CatanEnv().reset(6, [None] * 4)
        s = env.game.state
        dice = lambda r: (r.randint(1, 6), r.randint(1, 6))
        real_dice, real_steal = self._future(s.dice_rng, 20, dice), self._future(s.steal_rng, 20, lambda r: r.random())
        before = (s.dice_rng.getstate(), s.steal_rng.getstate(), s.random.getstate())
        for _ in range(5):
            g = env.lookahead_game()
            self.assertNotEqual(self._future(g.state.dice_rng, 20, dice), real_dice)
            self.assertNotEqual(self._future(g.state.steal_rng, 20, lambda r: r.random()), real_steal)
            self.assertIsNot(g.state.random, s.random)
            g.play()  # simulate to the end on the copy
        self.assertEqual(before, (s.dice_rng.getstate(), s.steal_rng.getstate(), s.random.getstate()))


class TestDeterminism(unittest.TestCase):
    def _trace(self, seed, ab):
        from catan_rl.env.actions import to_index
        obs = []

        def recording(inner):
            def agent(env, color):
                obs.append(json.dumps([env.observed_state(c) for c in COLORS]))
                return inner(env, color)
            return agent

        agents = [recording(random_agent)] * 4
        if ab:
            bot = AlphaBetaPlayer(COLORS[2])
            agents[2] = recording(lambda env, c: to_index(bot.decide(env.game, env.game.playable_actions)))
        env = play(seed, agents)
        return env.events, obs

    def test_same_seed_same_agents_identical(self):
        for ab in (False, True):
            a, b = self._trace(11, ab), self._trace(11, ab)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[1], b[1])


class TestBeliefAgainstTruth(unittest.TestCase):
    def test_true_hand_always_supported(self):
        for s in range(30):
            ValidatingEnv().reset(1000 + s, [random_agent] * 4).play()
        agents = [random_agent] * 4
        agents[3] = AlphaBetaPlayer(COLORS[3])
        ValidatingEnv().reset(7, agents).play()


if __name__ == "__main__":
    unittest.main()
