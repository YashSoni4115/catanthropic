"""Python↔Java bridge (v1) for interactive Catan self-play.

Protocol: line-delimited JSON over the Java subprocess's stdout / stdin.
Python owns the subprocess.  Java is the RULES AUTHORITY -- it validates
and applies every action.  Python only *chooses*.

Message types (Java -> Python on stdout):
  * ``decision_request`` -- ask for the AI player's next action
        {"type":"decision_request","request_id":N,"turn":T,"phase":"main"|"robber",
         "player":"P0","legal_action_kinds":["build_road",...],
         "snapshot_payload":{...full snapshot line...}}
  * ``error``            -- last decision was rejected; game falls back
        {"type":"error","request_id":N,"reason":"..."}
  * ``game_over``        -- final marker before the snapshot dump
        {"type":"game_over","winner":"P2","turns":89,"ai_player":"P0",
         "ai_won":false,"termination_reason":"winner","fallbacks":0}
  * *anything else*      -- treated as a trajectory snapshot line
                            (accumulated for post-game inspection)

Message types (Python -> Java on stdin):
  * ``decision_response``
        {"type":"decision_response","request_id":N,"action_kind":"build_road",
         "edge_x":1,"edge_y":2,"edge_o":0}

The ``action_kind`` and its parameter names match what Java's
``applyExternalAction`` parses (see SelfPlayGameRunner.java).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ai.board_state_extractor import extract_board_state_from_snapshot
from ai.bridge.setup_chooser import choose_setup_road, choose_setup_settlement
from ai.decision.actions import Action, ActionType
from ai.decision.decision_layer_v1 import DEFAULT_MODEL_DIR, rank_actions
from ai.decision.heuristics import HeuristicConfig
from ai.decision.opening_value_evaluator_v1 import ValueModelOracle
from ai.decision.scorer import ActionScorer, ScoredAction

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Action <-> protocol JSON
# ---------------------------------------------------------------------------

def action_to_response(action: Action, request_id: int) -> Dict[str, Any]:
    """Serialize an :class:`Action` to the wire-format decision_response."""
    payload: Dict[str, Any] = {
        "type": "decision_response",
        "request_id": request_id,
        "action_kind": action.engine_kind,
    }
    if action.type == ActionType.BUILD_ROAD and action.edge is not None:
        payload["edge_x"], payload["edge_y"], payload["edge_o"] = action.edge
    elif action.type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY) \
            and action.vertex is not None:
        payload["vertex_x"], payload["vertex_y"], payload["vertex_o"] = action.vertex
    elif action.type == ActionType.MOVE_ROBBER and action.tile is not None:
        payload["tile_x"], payload["tile_y"] = action.tile
        if action.steal_target:
            payload["steal_target"] = action.steal_target
    elif action.type == ActionType.TRADE_BANK:
        payload["give_resource"] = action.give_resource
        payload["take_resource"] = action.take_resource
        payload["give_qty"] = action.give_qty
    return payload


def action_from_response(response: Dict[str, Any]) -> Action:
    """Deserialize a decision_response back into an :class:`Action`.

    Round-trip guarantee for the supported action types.
    """
    kind = response.get("action_kind")
    if kind == "build_road":
        return Action(type=ActionType.BUILD_ROAD,
                      edge=(int(response["edge_x"]), int(response["edge_y"]), int(response["edge_o"])))
    if kind == "build_settlement":
        return Action(type=ActionType.BUILD_SETTLEMENT,
                      vertex=(int(response["vertex_x"]), int(response["vertex_y"]), int(response["vertex_o"])))
    if kind == "build_city":
        return Action(type=ActionType.BUILD_CITY,
                      vertex=(int(response["vertex_x"]), int(response["vertex_y"]), int(response["vertex_o"])))
    if kind == "buy_dev_card":
        return Action(type=ActionType.BUY_DEV_CARD)
    if kind == "trade_bank":
        return Action(type=ActionType.TRADE_BANK,
                      give_resource=response.get("give_resource"),
                      take_resource=response.get("take_resource"),
                      give_qty=response.get("give_qty"))
    if kind == "move_robber":
        return Action(type=ActionType.MOVE_ROBBER,
                      tile=(int(response["tile_x"]), int(response["tile_y"])),
                      steal_target=response.get("steal_target"))
    if kind in ("pass", "end_turn"):
        return Action(type=ActionType.END_TURN)
    raise ValueError(f"Unknown action_kind in response: {kind!r}")


# ---------------------------------------------------------------------------
# Request -> observed_state
# ---------------------------------------------------------------------------

def observed_state_from_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the same-shape state payload the decision layer expects."""
    payload = request.get("snapshot_payload") or {}
    snapshot = payload.get("snapshot") or payload
    return extract_board_state_from_snapshot(
        snapshot,
        current_player_name=request.get("player"),
        reveal_private=False,
        top_level_mirror="observed",
    )


# ---------------------------------------------------------------------------
# Turn trace
# ---------------------------------------------------------------------------

@dataclass
class TurnDecision:
    turn: int
    phase: str
    player: str
    chosen: Action
    win_prob_before: float
    win_prob_after: float
    delta: float
    top_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    fired_heuristics: List[str] = field(default_factory=list)


@dataclass
class GameResult:
    winner: Optional[str]
    turns: int
    ai_player: str
    ai_won: bool
    termination_reason: str
    fallbacks: int
    decisions: List[TurnDecision] = field(default_factory=list)
    raw_snapshots: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class GameBridge:
    """Manages a single interactive Catan game.

    Instantiate with an :class:`ActionScorer` (which owns the value model)
    to avoid reloading the pickle for each game.
    """

    def __init__(
        self,
        scorer: ActionScorer,
        ai_player_index: int = 0,
        heuristic_config: Optional[HeuristicConfig] = None,
        java_cp: str = "src",
        max_turns: int = 400,
        seed: Optional[int] = None,
        read_timeout_s: float = 30.0,
        trace_callback: Optional[Callable[[TurnDecision], None]] = None,
        extra_java_args: Optional[List[str]] = None,
    ):
        self.scorer = scorer
        self.ai_player_index = ai_player_index
        self.heuristic_config = heuristic_config
        self.java_cp = java_cp
        self.max_turns = max_turns
        self.seed = seed
        self.read_timeout_s = read_timeout_s
        self.trace_callback = trace_callback
        self.extra_java_args = list(extra_java_args or [])
        # Setup phase: remember the just-placed settlement vertex so the
        # subsequent setup_road request can connect to it.
        self._last_setup_vertex: Optional[tuple] = None

    # -- Public API ---------------------------------------------------------

    def play(self) -> GameResult:
        cmd = [
            "java", "-cp", self.java_cp, "game.SelfPlayGameRunner",
            "--external-policy", "--ai-player", str(self.ai_player_index),
            "--max-turns", str(self.max_turns),
        ]
        if self.seed is not None:
            cmd += ["--seed", str(int(self.seed))]
        cmd += self.extra_java_args

        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert proc.stdin is not None and proc.stdout is not None
        result = GameResult(
            winner=None, turns=0, ai_player=f"P{self.ai_player_index}",
            ai_won=False, termination_reason="pending", fallbacks=0,
        )

        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-protocol line -- assume final trajectory dump; done.
                    result.raw_snapshots.append(line)
                    continue
                mtype = msg.get("type")
                if mtype == "decision_request":
                    self._handle_request(msg, proc.stdin, result)
                elif mtype == "error":
                    result.fallbacks += 1
                    sys.stderr.write(f"[bridge] engine error: {msg}\n")
                elif mtype == "game_over":
                    result.winner = msg.get("winner") or None
                    result.turns = int(msg.get("turns") or 0)
                    result.ai_player = msg.get("ai_player") or result.ai_player
                    result.ai_won = bool(msg.get("ai_won"))
                    result.termination_reason = msg.get("termination_reason") or "unknown"
                    # Prefer Java's fallback tally (authoritative).
                    result.fallbacks = int(msg.get("fallbacks") or result.fallbacks)
                elif mtype is None:
                    # Snapshot line (has no "type" key).
                    result.raw_snapshots.append(line)
                # Other types: ignore.
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        return result

    # -- Internals ----------------------------------------------------------

    def _handle_setup(
        self, msg: Dict[str, Any], stdin, result: GameResult,
        req_id: int, phase: str,
    ) -> None:
        state = observed_state_from_request(msg)
        turn = int(msg.get("turn", -1))
        player = msg.get("player") or ""
        response: Dict[str, Any] = {
            "type": "decision_response", "request_id": req_id,
        }
        trace_action: Action
        note = "-"

        if phase == "setup_settlement":
            pick = choose_setup_settlement(state)
            if pick is None:
                # Signal Java to fall back (unknown action_kind).
                response["action_kind"] = "unavailable"
                stdin.write(json.dumps(response) + "\n"); stdin.flush()
                return
            response["action_kind"] = "setup_settlement"
            response["vertex_x"] = pick["x"]
            response["vertex_y"] = pick["y"]
            response["vertex_o"] = pick["orientation"]
            self._last_setup_vertex = (pick["x"], pick["y"], pick["orientation"])
            note = f"opening_score={pick['opening_score']:.3f}"
            trace_action = Action(
                type=ActionType.BUILD_SETTLEMENT,
                vertex=self._last_setup_vertex,
                tags=("setup",),
            )
        else:  # setup_road
            if self._last_setup_vertex is None:
                response["action_kind"] = "unavailable"
                stdin.write(json.dumps(response) + "\n"); stdin.flush()
                return
            pick = choose_setup_road(state, self._last_setup_vertex)
            if pick is None:
                response["action_kind"] = "unavailable"
                stdin.write(json.dumps(response) + "\n"); stdin.flush()
                return
            response["action_kind"] = "setup_road"
            response["edge_x"] = pick["x"]
            response["edge_y"] = pick["y"]
            response["edge_o"] = pick["orientation"]
            note = f"toward_score={pick['toward_opening_score']:.3f}"
            trace_action = Action(
                type=ActionType.BUILD_ROAD,
                edge=(pick["x"], pick["y"], pick["orientation"]),
                tags=("setup",),
            )

        stdin.write(json.dumps(response) + "\n"); stdin.flush()
        decision = TurnDecision(
            turn=turn, phase=phase, player=player,
            chosen=trace_action,
            win_prob_before=float("nan"), win_prob_after=float("nan"), delta=0.0,
            top_alternatives=[{"action": note, "delta": 0.0}],
            fired_heuristics=[],
        )
        result.decisions.append(decision)
        if self.trace_callback is not None:
            self.trace_callback(decision)

    def _handle_request(self, msg: Dict[str, Any], stdin, result: GameResult) -> None:
        req_id = int(msg.get("request_id", 0))
        phase = msg.get("phase", "main")

        # ---- Setup phases (short-circuit, no value model needed) -----
        if phase in ("setup_settlement", "setup_road"):
            self._handle_setup(msg, stdin, result, req_id, phase)
            return

        state = observed_state_from_request(msg)
        # Rank and choose
        ranked: List[ScoredAction] = rank_actions(
            state, scorer=self.scorer, heuristic_config=self.heuristic_config,
        )

        chosen: Optional[Action] = None
        legal_kinds = set(msg.get("legal_action_kinds") or [])

        # In robber phase, Java only accepts move_robber.
        if phase == "robber":
            for s in ranked:
                if s.action.type == ActionType.MOVE_ROBBER:
                    chosen = s.action
                    break
            if chosen is None:
                # Rank came from main-phase enumerator; fabricate one.
                # Pick any non-desert, non-current-robber tile.
                obs = state.get("observed_state", state)
                cur = obs.get("robber") or {}
                for t in obs.get("tiles") or []:
                    if (t.get("x"), t.get("y")) != (cur.get("x"), cur.get("y")) \
                            and str(t.get("type", "")).upper() != "DESERT":
                        chosen = Action(type=ActionType.MOVE_ROBBER, tile=(int(t["x"]), int(t["y"])))
                        break
        else:
            for s in ranked:
                # Filter to what Java says is legal *right now* (belt and braces).
                if s.action.engine_kind in legal_kinds or s.action.engine_kind == "pass":
                    if s.error is None:
                        chosen = s.action
                        break
            if chosen is None and ranked:
                chosen = ranked[0].action

        if chosen is None:
            chosen = Action(type=ActionType.END_TURN)

        response = action_to_response(chosen, req_id)
        stdin.write(json.dumps(response) + "\n")
        stdin.flush()

        # Record trace for the chosen action
        chosen_scored = next((s for s in ranked if s.action == chosen), None)
        top_alts = [
            {"action": s.action.short(), "delta": round(float(s.model_delta), 6)}
            for s in ranked[:3] if s is not chosen_scored
        ]
        decision = TurnDecision(
            turn=int(msg.get("turn", 0)),
            phase=phase,
            player=msg.get("player") or "",
            chosen=chosen,
            win_prob_before=float(chosen_scored.win_prob_before) if chosen_scored else float("nan"),
            win_prob_after=float(chosen_scored.win_prob_after) if chosen_scored else float("nan"),
            delta=float(chosen_scored.model_delta) if chosen_scored else 0.0,
            top_alternatives=top_alts,
            fired_heuristics=[h["name"] for h in (chosen_scored.heuristic_adjustments if chosen_scored else [])],
        )
        result.decisions.append(decision)
        if self.trace_callback is not None:
            self.trace_callback(decision)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_default_scorer(model_dir: str = DEFAULT_MODEL_DIR,
                         model_name: str = "hgb") -> ActionScorer:
    return ActionScorer(oracle=ValueModelOracle(model_dir, model_name=model_name))
