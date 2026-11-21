"""Shared helpers for belief/determinizer tests (ground truth lives here, never in agents)."""
import random
from collections import Counter

from catanatron.state_functions import player_key

from catan_rl.belief.devcards import DEV_DECK
from catan_rl.env import COLORS, CatanEnv
from catan_rl.env.determinize import CAT_OF_TRK
from catanatron.models.enums import RESOURCES as CAT_RES

ALL = [c.value for c in COLORS]


def true_hand(state, color):
    """Ground truth resource hand in tracker order."""
    k = player_key(state, color)
    return tuple(state.player_state[f"{k}_{CAT_RES[CAT_OF_TRK[i]]}_IN_HAND"] for i in range(5))


def true_dev(state, color) -> Counter:
    k = player_key(state, color)
    return Counter({c: state.player_state[f"{k}_{c}_IN_HAND"] for c in DEV_DECK if state.player_state[f"{k}_{c}_IN_HAND"]})


def color(name):
    return next(c for c in COLORS if c.value == name)


def random_walk(seed, viewers=ALL, every=1, weights_seed=None):
    """Play a random game step by step (external seats); yields env after every ``every`` steps.
    Buys dev cards more often than uniform so dev beliefs get exercised."""
    env = CatanEnv(track_viewers=viewers, record_events=False).reset(seed, {c: None for c in COLORS})
    rng = random.Random(f"walk:{seed}")
    n = 0
    while not env.done:
        acts = env.legal_actions()
        buys = [a for a in env.game.playable_actions if a.action_type.value == "BUY_DEVELOPMENT_CARD"]
        if buys and rng.random() < 0.5:
            from catan_rl.env.actions import to_index
            env.step(to_index(buys[0]))
        else:
            env.step(rng.choice(acts))
        n += 1
        if n % every == 0 and not env.done:
            yield env
