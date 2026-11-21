from catan_rl.env.actions import COLORS, NUM_ACTIONS, action_array, to_index
from catan_rl.env.env import CatanEnv
from catan_rl.env.observation import observed_state

__all__ = ["CatanEnv", "COLORS", "NUM_ACTIONS", "action_array", "to_index", "observed_state", "random_agent"]


def random_agent(env: "CatanEnv", color) -> int:
    """Uniform over legal indices, using the env's own bot stream (deterministic per seed)."""
    acts = env.legal_actions()
    return acts[env.game.state.random.randrange(len(acts))]
