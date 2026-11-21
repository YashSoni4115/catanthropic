"""Print a seeded game's action_records as JSON lines (run under different PYTHONHASHSEEDs and diff).

    python -m catan_rl.eval.hashcheck SEED AGENT[,AGENT,AGENT,AGENT]
"""
import sys

from catan_rl.env import COLORS, CatanEnv
from catan_rl.eval import agents as A


def records(seed: int, lineup) -> list:
    env = CatanEnv(record_events=False).reset(
        seed, {c: A.get(n).factory(c) for c, n in zip(COLORS, lineup)})
    env.play()
    return [repr((r.action, r.result)) for r in env.game.state.action_records]


if __name__ == "__main__":
    names = sys.argv[2].split(",")
    for line in records(int(sys.argv[1]), names * (4 // len(names))):
        print(line)
