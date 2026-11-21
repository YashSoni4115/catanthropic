"""State selection, deterministic replay and belief-sampled rollout labels.

A state is (seed, decision index); the lineup is a function of the seed. Replay with external seats
reproduces the game exactly (fork RNG streams + hash-independent engine). Rollout r of state s uses
determinize(viewer, seed=(s, r)) and the determinized game's own RNG for the rollout policy, so labels
depend only on (state id, r, policy) - never on worker scheduling.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from catan_rl.env import COLORS, CatanEnv
from catan_rl.env.actions import DOMESTIC, to_index
from catan_rl.env.determinize import DeterminizeError
from catan_rl.eval import agents as A

LINEUP_KINDS = ("4wr", "4vf", "2+2", "1vf+3wr")
PHASES = ("early", "mid", "late")
PER_THIRD = (3, 4, 3)


def lineup(seed: int) -> Tuple[str, ...]:
    rng = random.Random(f"lineup:{seed}")
    kind = rng.choice(LINEUP_KINDS)
    names = {"4wr": ["weighted_random"] * 4, "4vf": ["value_function"] * 4,
             "2+2": ["weighted_random"] * 2 + ["value_function"] * 2,
             "1vf+3wr": ["value_function"] + ["weighted_random"] * 3}[kind]
    rng.shuffle(names)
    return tuple(names)


@dataclass(frozen=True)
class StateSpec:
    seed: int
    decision: int  # index among non-forced decisions
    phase: str
    viewer: str

    @property
    def id(self) -> str:
        return f"{self.seed}:{self.decision}"


def replay(seed: int, stop_at: Sequence[int] = (), track: bool = False) -> Iterator[Tuple[int, CatanEnv]]:
    """Play seed's lineup with external seats; yield (decision index, env) before each listed decision.
    After the last listed index the generator stops (or plays out if stop_at is empty)."""
    names = lineup(seed)
    bots = {c: A.get(n).factory(c) for c, n in zip(COLORS, names)}
    random.seed(seed)  # in case a bot touches the global RNG
    env = CatanEnv(track_viewers=[c.value for c in COLORS] if track else (), record_events=False)
    env.reset(seed, {c: None for c in COLORS})
    want, last, i = set(stop_at), max(stop_at, default=-1), 0
    while not env.done:
        if i in want:
            yield i, env
            if i == last:
                return
        c = env.current_player()
        acts = [a for a in env.game.playable_actions if a.action_type not in DOMESTIC]
        env.step(to_index(bots[c].decide(env.game, acts)))
        i += 1
    if not stop_at:
        yield i, env


def select_states(seed: int, per_third: Sequence[int] = PER_THIRD) -> List[StateSpec]:
    """~10 non-forced decisions stratified by thirds of the game's final action count."""
    marks: List[Tuple[int, int, str]] = []  # (decision, action count before it, actor)
    names = lineup(seed)
    bots = {c: A.get(n).factory(c) for c, n in zip(COLORS, names)}
    random.seed(seed)
    env = CatanEnv(record_events=False).reset(seed, {c: None for c in COLORS})
    while not env.done:
        c = env.current_player()
        marks.append((len(marks), len(env.game.state.action_records), c.value))
        acts = [a for a in env.game.playable_actions if a.action_type not in DOMESTIC]
        env.step(to_index(bots[c].decide(env.game, acts)))
    total = len(env.game.state.action_records)
    rng = random.Random(f"select:{seed}")
    out = []
    for t, k in enumerate(per_third):
        pool = [m for m in marks if min(2, 3 * m[1] // total) == t]
        out += [StateSpec(seed, d, PHASES[t], v) for d, _, v in sorted(rng.sample(pool, min(k, len(pool))))]
    return out


def rollout_labels(env: CatanEnv, spec: StateSpec, policy: str, n: int) -> Optional[dict]:
    """N determinized rollouts from the current env state. None if determinization fails."""
    order = [c.value for c in env.game.state.colors]
    k = order.index(spec.viewer)
    rel = order[k:] + order[:k]  # viewer, next, next, next
    wins, vp, rem, none = np.zeros(4, np.int32), np.zeros(4), 0.0, 0
    t0, turn0 = time.perf_counter(), env.game.state.num_turns
    pol = A.get(policy)
    for r in range(n):
        try:
            g = env.determinize(spec.viewer, f"{spec.id}:{r}")
        except DeterminizeError:
            return None
        g.state.players = [pol.factory(c) for c in g.state.colors]
        g.play()
        st, w = g.state, g.winning_color()
        if w is None:
            none += 1
        else:
            wins[rel.index(w.value)] += 1
        for i, c in enumerate(rel):
            vp[i] += st.player_state[f"P{order.index(c)}_ACTUAL_VICTORY_POINTS"]
        rem += st.num_turns - turn0
    return {"wins": wins, "no_winner": none, "mean_vp": (vp / n).astype(np.float32),
            "mean_remaining_turns": rem / n, "label_s": time.perf_counter() - t0}


def label_game(seed: int, specs: Sequence[StateSpec], policy: str, n: int, with_features: bool = True):
    """One replay pass over a game: features + labels at each requested state. Returns (rows, skipped)."""
    from catan_rl.winprob.features import engineered, sample_vector
    by_dec = {s.decision: s for s in specs}
    rows, skipped = [], []
    for d, env in replay(seed, sorted(by_dec), track=True):
        s = by_dec[d]
        lab = rollout_labels(env, s, policy, n)
        if lab is None:
            skipped.append(s.id)
            continue
        row = {"spec": s, **lab}
        if with_features:
            row["xa"] = engineered(env.game, s.viewer, env.belief(s.viewer), env.dev_belief(s.viewer), s.id)
            row["xb"] = sample_vector(env.game, s.viewer)
        rows.append(row)
    return rows, skipped


def _label_job(job):
    seed, specs, policy, n, feats = job
    return label_game(seed, specs, policy, n, feats)


def label_states(specs: Sequence[StateSpec], policy: str, n: int, workers: int = 1, with_features: bool = False):
    """Label arbitrary states (grouped per game). Output order follows ``specs``; independent of ``workers``."""
    by_seed: Dict[int, List[StateSpec]] = {}
    for s in specs:
        by_seed.setdefault(s.seed, []).append(s)
    jobs = [(seed, ss, policy, n, with_features) for seed, ss in sorted(by_seed.items())]
    if workers == 1:
        res = list(map(_label_job, jobs))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_label_job, jobs))
    rows = {r["spec"].id: r for rs, _ in res for r in rs}
    skipped = [x for _, sk in res for x in sk]
    return [rows[s.id] for s in specs if s.id in rows], skipped
