"""Python-side state approximator for the decision layer.

The Java game engine does NOT expose a ``simulate(action) -> next_state``
capability (see the Step 0 investigation).  This module applies each
action's *obvious* effects to a copy of ``observed_state`` well enough for
the vectorizer and value model to score meaningfully.

APPROXIMATIONS (make sure downstream consumers know these):
-----------------------------------------------------------
- Resources are deducted for costs.  Nothing is *granted* by the
  approximator (no dice roll, no production).
- ``build_settlement``: adds a "settlement" structure, decrements
  ``settlements_remaining``, +1 ``victory_points_visible``.  Does NOT
  reassess longest road, ports gained, or opponent reactions.
- ``build_city``: mutates the target settlement's ``type`` to "city",
  decrements ``cities_remaining`` and increments ``settlements_remaining``,
  +1 ``victory_points_visible`` (settlement already contributed 1, city
  is worth 2, so net +1).
- ``build_road``: adds a road, decrements ``roads_remaining``.  Does NOT
  recompute longest road.
- ``buy_dev_card``: deducts resources.  Does NOT reveal a card; the
  observed development_cards block is left untouched (opponents don't see
  which card was drawn; you drew *something*).
- ``trade_bank``: deducts ``give_qty`` of ``give_resource``, adds 1 of
  ``take_resource``.
- ``move_robber``: sets ``robber = {"x","y"}`` and flips the ``robber``
  boolean on the affected tile entries.  If ``steal_target`` is set, we
  decrement that opponent's ``total_resource_cards`` by 1 (their private
  ``resource_cards`` dict is None to the current viewer, so we cannot
  attribute the stolen resource to a specific type).  The current
  player's own ``resource_cards`` is NOT incremented because we don't
  know which resource was stolen.  This under-counts the actor's hand by
  1 card -- an acceptable approximation because the immediate benefit
  of blocking > 1 stolen card in most positions.
- ``end_turn``: returns a *shallow* copy with no state changes.  We do
  NOT advance ``current_player_id`` because the value model is trained to
  score "P(current active player wins)", so rotating would change the
  *subject* of prediction rather than the *state* -- yielding a mostly-
  neutral baseline (delta ~= 0) that heuristics can adjust.

TODO: replace this module with a true engine dry-run once the Java runner
grows a ``Game.cloneAndApply(action) -> Game`` method (and a JSON bridge).
Every ``_apply_*`` function below should collapse to a single ``bridge.
simulate(state, action)`` call at that point.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from ai.economy.constants import road_id

from .actions import Action, ActionType
from .legal_actions import (
    COST_CITY, COST_DEV_CARD, COST_ROAD, COST_SETTLEMENT,
    RESOURCES, _current_player, _observed,
)


def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy of the payload, deep-ish copy of observed sub-tree."""
    out = dict(state)
    observed = state.get("observed_state")
    if isinstance(observed, dict):
        obs = dict(observed)
        obs["structures"] = [dict(s) for s in (observed.get("structures") or [])]
        obs["roads"] = [dict(r) for r in (observed.get("roads") or [])]
        obs["tiles"] = [dict(t) for t in (observed.get("tiles") or [])]
        obs["robber"] = dict(observed.get("robber") or {})
        obs["turn_metadata"] = dict(observed.get("turn_metadata") or {})
        players_block = observed.get("players")
        if isinstance(players_block, dict):
            pb = dict(players_block)
            pb["all_players"] = [copy.deepcopy(p)
                                 for p in (players_block.get("all_players") or [])]
            obs["players"] = pb
        out["observed_state"] = obs
    return out


def _viewer_player(obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _current_player(obs)


def _deduct(player: Dict[str, Any], cost: Dict[str, int]) -> None:
    rc = player.get("resource_cards")
    if not isinstance(rc, dict):
        return
    for r, q in cost.items():
        rc[r] = int(rc.get(r, 0) or 0) - int(q)
    player["total_resource_cards"] = max(
        0, int(player.get("total_resource_cards") or 0) - sum(cost.values())
    )


def _dec_remaining(player: Dict[str, Any], key: str, delta: int = 1) -> None:
    rp = player.get("remaining_pieces")
    if not isinstance(rp, dict):
        return
    v = rp.get(key)
    if v is None:
        return
    rp[key] = max(0, int(v) - delta)


def _inc_vp_visible(player: Dict[str, Any], delta: int = 1) -> None:
    v = player.get("victory_points_visible")
    if isinstance(v, int):
        player["victory_points_visible"] = v + delta
    v2 = player.get("victory_points")
    if isinstance(v2, int):
        player["victory_points"] = v2 + delta


# ---------------------------------------------------------------------------
# Per-action-type appliers
# ---------------------------------------------------------------------------

def _apply_build_road(obs: Dict[str, Any], player: Dict[str, Any], a: Action) -> None:
    ex, ey, eo = a.edge  # type: ignore[misc]
    _deduct(player, COST_ROAD)
    _dec_remaining(player, "roads_remaining")
    obs["roads"].append({
        "x": ex, "y": ey, "orientation": eo,
        "road_id": road_id(ex, ey, eo),
        "owner_id": player.get("player_id"),
        "owner_name": player.get("player_name"),
    })


def _apply_build_settlement(obs, player, a: Action) -> None:
    vx, vy, vo = a.vertex  # type: ignore[misc]
    _deduct(player, COST_SETTLEMENT)
    _dec_remaining(player, "settlements_remaining")
    obs["structures"].append({
        "x": vx, "y": vy, "orientation": vo,
        "structure_id": f"structure:{vx}:{vy}:{vo}",
        "type": "settlement",
        "owner_id": player.get("player_id"),
        "owner_name": player.get("player_name"),
        "adjacent_tiles": [], "incident_roads": [],
    })
    _inc_vp_visible(player, 1)


def _apply_build_city(obs, player, a: Action) -> None:
    vx, vy, vo = a.vertex  # type: ignore[misc]
    _deduct(player, COST_CITY)
    _dec_remaining(player, "cities_remaining", 1)
    # Restore a settlement to remaining pool (city upgrade returns the piece)
    rp = player.get("remaining_pieces")
    if isinstance(rp, dict) and rp.get("settlements_remaining") is not None:
        rp["settlements_remaining"] = int(rp["settlements_remaining"]) + 1
    for s in obs["structures"]:
        if (s.get("x"), s.get("y"), s.get("orientation")) == (vx, vy, vo):
            s["type"] = "city"
            break
    _inc_vp_visible(player, 1)


def _apply_buy_dev_card(obs, player, a: Action) -> None:
    _deduct(player, COST_DEV_CARD)
    # No visible state change beyond resources.


def _apply_trade_bank(obs, player, a: Action) -> None:
    rc = player.get("resource_cards")
    if not isinstance(rc, dict):
        return
    give, take, qty = a.give_resource, a.take_resource, int(a.give_qty or 4)
    rc[give] = int(rc.get(give, 0) or 0) - qty
    rc[take] = int(rc.get(take, 0) or 0) + 1
    player["total_resource_cards"] = max(
        0, int(player.get("total_resource_cards") or 0) - qty + 1
    )


def _apply_move_robber(obs, player, a: Action) -> None:
    tx, ty = a.tile  # type: ignore[misc]
    obs["robber"] = {"x": tx, "y": ty}
    for t in obs["tiles"]:
        try:
            same = (int(t.get("x")), int(t.get("y"))) == (tx, ty)
        except (TypeError, ValueError):
            continue
        t["robber"] = bool(same)
    # Decrement victim's public total_resource_cards; do NOT add to actor
    # because we don't know the resource type of what was stolen.
    if a.steal_target:
        for p in (obs.get("players") or {}).get("all_players", []):
            if p.get("player_id") == a.steal_target or p.get("player_name") == a.steal_target:
                v = p.get("total_resource_cards")
                if isinstance(v, int) and v > 0:
                    p["total_resource_cards"] = v - 1
                break


def _apply_end_turn(obs, player, a: Action) -> None:
    # See module docstring: intentionally a no-op.
    return


_DISPATCH = {
    ActionType.BUILD_ROAD: _apply_build_road,
    ActionType.BUILD_SETTLEMENT: _apply_build_settlement,
    ActionType.BUILD_CITY: _apply_build_city,
    ActionType.BUY_DEV_CARD: _apply_buy_dev_card,
    ActionType.TRADE_BANK: _apply_trade_bank,
    ActionType.MOVE_ROBBER: _apply_move_robber,
    ActionType.END_TURN: _apply_end_turn,
}


def apply_action(state: Dict[str, Any], action: Action) -> Dict[str, Any]:
    """Return a new state payload with the action's effects applied.

    Raises :class:`ValueError` if the action type is not supported by the
    v1 approximator (all "PLAY_*" / trade-proposal types).
    """
    fn = _DISPATCH.get(action.type)
    if fn is None:
        raise ValueError(
            f"apply_action: action type {action.type.value!r} is not "
            f"supported by simulator_v1 (see actions.py TODO list)"
        )
    new_state = _copy_state(state)
    obs = _observed(new_state)
    player = _viewer_player(obs)
    if player is None:
        raise ValueError("apply_action: current player not resolvable from observed_state")
    fn(obs, player, action)
    return new_state
