"""Turn-based multi-seat environment over the patched Catanatron fork.

Every seat (color) is controlled by an agent:
  * a callable ``agent(env, color) -> action_index`` (may use env.legal_actions(),
    env.observed_state(color), env.belief(color), env.lookahead_game()),
  * a catanatron ``Player`` (bot; it decides on the real game, as catanatron bots do), or
  * ``None`` = external: the env pauses and the caller drives it with ``step(index)``.
Decisions with exactly one legal action are auto-applied (still producing events).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Union

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.player import Player, RandomPlayer

from catan_rl.belief.tracker import BeliefTracker
from catan_rl.env import event_adapter as ea
from catan_rl.env.actions import COLORS, DOMESTIC, action_array, to_index
from catan_rl.env.observation import observed_state

Agent = Union[None, Player, Callable[["CatanEnv", object], int]]


class CatanEnv:
    def __init__(self, track_beliefs: bool = True, auto_forced: bool = True, record_events: bool = True):
        self.track_beliefs, self.auto_forced, self.record_events = track_beliefs, auto_forced, record_events
        self.on_event: Optional[Callable[[str, object], None]] = None  # hook(viewer, event)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int, seat_agents: Union[Sequence[Agent], Dict[object, Agent]]) -> "CatanEnv":
        agents = dict(zip(COLORS, seat_agents)) if not isinstance(seat_agents, dict) else dict(seat_agents)
        assert set(agents) == set(COLORS), "need exactly one agent per color"
        players = [a if isinstance(a, Player) else RandomPlayer(c) for c, a in agents.items()]
        assert all(p.color == c for p, c in zip(players, agents)), "bot color must match its seat"
        self.seed, self.agents = seed, agents
        self.game = Game(players, seed=seed)
        self._lookahead_rng = random.Random(f"{seed}:lookahead")
        self.forced_decisions = self.agent_decisions = 0
        self.events: Dict[str, List[object]] = {c.value: [] for c in COLORS}
        self.trackers: Dict[str, BeliefTracker] = {}
        if self.track_beliefs:
            for v in COLORS:
                t = self.trackers[v.value] = BeliefTracker(strict=True)
                for c in COLORS:
                    t.initialize_player(c.value, 0)
        self._advance()
        return self

    @property
    def done(self) -> bool:
        return self.winner is not None or self.game.state.num_turns >= TURNS_LIMIT

    @property
    def winner(self):
        return self.game.winning_color()

    def current_player(self):
        return self.game.state.current_color()

    def public_vp(self) -> Dict[str, int]:
        s = self.game.state
        return {c.value: s.player_state[f"P{s.color_to_index[c]}_VICTORY_POINTS"] for c in s.colors}

    def legal_actions(self) -> List[int]:
        return sorted(to_index(a) for a in self.game.playable_actions if a.action_type not in DOMESTIC)

    def observed_state(self, viewer) -> dict:
        return observed_state(self.game.state, viewer)

    def belief(self, viewer) -> BeliefTracker:
        return self.trackers[getattr(viewer, "value", viewer)]

    # ------------------------------------------------------------------ stepping
    def step(self, action_index: int) -> None:
        """Apply the current (external) player's action, then advance to the next external decision."""
        assert not self.done, "game is over"
        self.agent_decisions += 1
        self._apply(self._resolve(action_index))
        self._advance()

    def play(self) -> Optional[object]:
        assert all(a is not None for a in self.agents.values()), "external seats need step()"
        self._advance()
        return self.winner

    def _resolve(self, index: int):
        key = action_array()[index]
        for a in self.game.playable_actions:
            if (a.action_type, a.value) == key:
                return a
        raise ValueError(f"illegal action index {index}: {key}")

    def _advance(self) -> None:
        while not self.done:
            acts = self.game.playable_actions
            if self.auto_forced and len(acts) == 1:
                self.forced_decisions += 1
                self._apply(acts[0])
                continue
            agent = self.agents[self.current_player()]
            if agent is None:
                return
            self.agent_decisions += 1
            if isinstance(agent, Player):
                self._apply(agent.decide(self.game, acts))
            else:
                self._apply(self._resolve(agent(self, self.current_player())))

    def _apply(self, action) -> None:
        if not (self.track_beliefs or self.record_events or self.on_event):
            self.game.play_tick(decide_fn=lambda *_: action)
            return
        before = ea.hands(self.game.state)
        record = self.game.play_tick(decide_fn=lambda *_: action)
        after = ea.hands(self.game.state)
        pairs = ea.public_events(record, before, after)
        for v in COLORS:
            for e in ea.events_for(v.value, pairs, after):
                if self.record_events:
                    self.events[v.value].append(e)
                if self.track_beliefs:
                    self.trackers[v.value].apply(e)
                if self.on_event:
                    self.on_event(v.value, e)

    # ------------------------------------------------------------------ lookahead (agent-facing)
    def lookahead_game(self) -> Game:
        """A simulation copy for agents. Dice/steal streams, bot RNG and dev deck order are
        re-randomized from a private stream, so it cannot reproduce the real game's future.
        NOTE: the copy still holds true opponent hands (catanatron state); determinize
        from a belief sample before handing it to a hidden-information agent."""
        g = self.game.copy()
        rng = random.Random(self._lookahead_rng.getrandbits(64))
        g.random = g.state.random = random.Random(rng.getrandbits(64))
        g.state.reseed_rule_streams(rng)
        g.state.development_listdeck = list(g.state.development_listdeck)
        rng.shuffle(g.state.development_listdeck)
        return g

    # ------------------------------------------------------------------ debug channel (ground truth)
    def debug_true_hands(self) -> Dict[str, Dict[str, int]]:
        """GROUND TRUTH. For tests/validation only; never pass to agents."""
        return {c: ea.cards(h) for c, h in ea.hands(self.game.state).items()}
