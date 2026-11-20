"""Legal action enumerator for the decision layer.

Mirrors the gate logic in ``src/game/SelfPlayGameRunner.legalActionKinds``
but produces *parameterised* :class:`Action` instances (with vertex / edge
/ tile / trade parameters), not just action kinds.

CONSTRAINTS
-----------
- Reads ``observed_state`` only.  Opponents with ``hidden_from_viewer=True``
  are never dereferenced for private fields (``resource_cards is None`` is
  handled).
- Uses ``total_resource_cards`` (public) rather than opponents'
  ``resource_cards`` (private) for robber-victim eligibility.
- Legality checks intentionally err on the *permissive* side when the
  observed extractor stores ``None`` for a field (e.g. ``remaining_pieces``
  may be unknown pre-turn).  This matches the "fail loudly on real
  violations, don't invent constraints" project style.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ai.economy.constants import (
    VALID_ROAD_SLOTS,
    VALID_STRUCTURE_SLOTS,
    adjacent_vertices_for_vertex,
    road_endpoint_vertices,
    vertex_incident_roads,
)

from .actions import Action, ActionType

RESOURCES: Tuple[str, ...] = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER")

# Building costs (canonical Catan)
COST_ROAD = {"BRICK": 1, "LUMBER": 1}
COST_SETTLEMENT = {"BRICK": 1, "LUMBER": 1, "WOOL": 1, "GRAIN": 1}
COST_CITY = {"ORE": 3, "GRAIN": 2}
COST_DEV_CARD = {"ORE": 1, "WOOL": 1, "GRAIN": 1}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _observed(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("observed_state", state)


def _current_player(observed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tm = observed.get("turn_metadata") or {}
    cur_id = tm.get("current_player_id")
    cur_name = tm.get("current_player_name")
    for p in (observed.get("players") or {}).get("all_players", []):
        if cur_id and p.get("player_id") == cur_id:
            return p
        if cur_name and p.get("player_name") == cur_name:
            return p
    return None


def _resources(player: Dict[str, Any]) -> Dict[str, int]:
    rc = player.get("resource_cards")
    if not isinstance(rc, dict):
        return {r: 0 for r in RESOURCES}
    return {r: int(rc.get(r, 0) or 0) for r in RESOURCES}


def _has(resources: Dict[str, int], cost: Dict[str, int]) -> bool:
    return all(resources.get(r, 0) >= q for r, q in cost.items())


def _remaining(player: Dict[str, Any], key: str, default_when_unknown: int = 99) -> int:
    """Fetch remaining piece count; ``None`` -> ``default_when_unknown``."""
    rp = player.get("remaining_pieces") or {}
    v = rp.get(key)
    return default_when_unknown if v is None else int(v)


def _occupied_vertices(observed: Dict[str, Any]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for s in observed.get("structures") or []:
        try:
            key = (int(s["x"]), int(s["y"]), int(s["orientation"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = s
    return out


def _occupied_roads(observed: Dict[str, Any]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in observed.get("roads") or []:
        try:
            key = (int(r["x"]), int(r["y"]), int(r["orientation"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = r
    return out


def _same_owner(entity: Dict[str, Any], owner_id: str, owner_name: str) -> bool:
    return entity.get("owner_id") == owner_id or entity.get("owner_name") == owner_name


def _current_phase(observed: Dict[str, Any]) -> str:
    return str((observed.get("turn_metadata") or {}).get("current_phase") or "").lower()


# ---------------------------------------------------------------------------
# Per-action-type enumerators
# ---------------------------------------------------------------------------

def _enum_build_road(
    observed: Dict[str, Any], player: Dict[str, Any],
    resources: Dict[str, int], occ_struct, occ_road,
    owner_id: str, owner_name: str,
) -> Iterable[Action]:
    if not _has(resources, COST_ROAD):
        return
    if _remaining(player, "roads_remaining") <= 0:
        return
    # Collect endpoints of the player's roads / structures for connectivity
    own_vertices: Set[Tuple[int, int, int]] = set()
    for r in observed.get("roads") or []:
        if not _same_owner(r, owner_id, owner_name):
            continue
        try:
            for v in road_endpoint_vertices(int(r["x"]), int(r["y"]), int(r["orientation"])):
                own_vertices.add((v["x"], v["y"], v["orientation"]))
        except (KeyError, TypeError, ValueError):
            continue
    for s in observed.get("structures") or []:
        if not _same_owner(s, owner_id, owner_name):
            continue
        try:
            own_vertices.add((int(s["x"]), int(s["y"]), int(s["orientation"])))
        except (KeyError, TypeError, ValueError):
            continue

    for (ex, ey, eo) in VALID_ROAD_SLOTS:
        if (ex, ey, eo) in occ_road:
            continue
        endpoints = road_endpoint_vertices(ex, ey, eo)
        touches_own = any(
            (v["x"], v["y"], v["orientation"]) in own_vertices for v in endpoints
        )
        if not touches_own:
            continue
        yield Action(type=ActionType.BUILD_ROAD, edge=(ex, ey, eo))


def _enum_build_settlement(
    observed, player, resources, occ_struct, occ_road,
    owner_id: str, owner_name: str,
) -> Iterable[Action]:
    if not _has(resources, COST_SETTLEMENT):
        return
    if _remaining(player, "settlements_remaining") <= 0:
        return
    own_road_edges: Set[Tuple[int, int, int]] = {
        (int(r["x"]), int(r["y"]), int(r["orientation"]))
        for r in (observed.get("roads") or [])
        if _same_owner(r, owner_id, owner_name)
        and r.get("x") is not None
    }
    for (vx, vy, vo) in VALID_STRUCTURE_SLOTS:
        if (vx, vy, vo) in occ_struct:
            continue
        # Distance rule: no adjacent vertex has a structure
        blocked = False
        for n in adjacent_vertices_for_vertex(vx, vy, vo):
            if (n["x"], n["y"], n["orientation"]) in occ_struct:
                blocked = True
                break
        if blocked:
            continue
        # Must be connected to one of our roads
        connected = False
        for e in vertex_incident_roads(vx, vy, vo):
            if (e["x"], e["y"], e["orientation"]) in own_road_edges:
                connected = True
                break
        if not connected:
            continue
        yield Action(type=ActionType.BUILD_SETTLEMENT, vertex=(vx, vy, vo))


def _enum_build_city(
    observed, player, resources, occ_struct, occ_road,
    owner_id: str, owner_name: str,
) -> Iterable[Action]:
    if not _has(resources, COST_CITY):
        return
    if _remaining(player, "cities_remaining") <= 0:
        return
    for (vx, vy, vo), s in occ_struct.items():
        if not _same_owner(s, owner_id, owner_name):
            continue
        if str(s.get("type", "")).lower() != "settlement":
            continue
        yield Action(type=ActionType.BUILD_CITY, vertex=(vx, vy, vo))


def _enum_buy_dev_card(
    observed, player, resources, *_a, **_kw,
) -> Iterable[Action]:
    if _has(resources, COST_DEV_CARD):
        # Deck size is not surfaced in observed_state.  Assume >0
        # (fail_loudly happens if we later try to apply and deck is empty).
        yield Action(type=ActionType.BUY_DEV_CARD)


def _port_qty(player: Dict[str, Any], resource: str) -> int:
    """Return 4 (bank), 3 (3:1 port), or 2 (2:1 port) for this resource."""
    ports = player.get("ports")
    if isinstance(ports, dict):
        if ports.get(f"{resource.lower()}_two_to_one"):
            return 2
        if ports.get("three_to_one"):
            return 3
        return 4
    if isinstance(ports, list) and len(ports) >= 6:
        # Ordering per snapshot: [BRICK,WOOL,ORE,GRAIN,LUMBER,3:1] (best-effort)
        mapping = {"BRICK": 0, "WOOL": 1, "ORE": 2, "GRAIN": 3, "LUMBER": 4}
        idx = mapping.get(resource)
        if idx is not None and bool(ports[idx]):
            return 2
        if bool(ports[5]):
            return 3
        return 4
    return 4


def _enum_trade_bank(
    observed, player, resources, *_a, **_kw,
) -> Iterable[Action]:
    for give in RESOURCES:
        qty = _port_qty(player, give)
        if resources.get(give, 0) < qty:
            continue
        for take in RESOURCES:
            if take == give:
                continue
            yield Action(
                type=ActionType.TRADE_BANK,
                give_resource=give, take_resource=take, give_qty=qty,
            )


def _enum_move_robber(
    observed, player, resources, occ_struct, occ_road,
    owner_id: str, owner_name: str,
) -> Iterable[Action]:
    """Enumerate move_robber actions when phase indicates a robber-move.

    Phases in which robber-movement is required:
      * ``"robber"`` / ``"move_robber"`` after rolling 7
      * ``"knight"`` after playing a knight (not currently enumerated as an
        action, but the phase may appear in data)
    """
    phase = _current_phase(observed)
    if phase not in {"robber", "move_robber", "knight", "play_knight"}:
        return
    current_robber = observed.get("robber") or {}
    cur_tile = (current_robber.get("x"), current_robber.get("y"))
    # Group structures by adjacent tile for O(N) victim lookup
    from ai.economy.constants import vertex_adjacent_tiles as _vat
    tile_to_owners: Dict[Tuple[int, int], Set[str]] = {}
    for (vx, vy, vo), s in occ_struct.items():
        for t in _vat(vx, vy, vo):
            tile_to_owners.setdefault((t["x"], t["y"]), set()).add(
                s.get("player_id") or s.get("owner_id") or s.get("owner_name") or ""
            )
    # Build a lookup: player_id -> total_resource_cards (public)
    pid_totals: Dict[str, int] = {}
    for p in (observed.get("players") or {}).get("all_players", []):
        pid = p.get("player_id") or ""
        pid_totals[pid] = int(p.get("total_resource_cards") or 0)
        pid_totals[p.get("player_name") or ""] = pid_totals[pid]
    for tile in observed.get("tiles") or []:
        try:
            tx, ty = int(tile["x"]), int(tile["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if (tx, ty) == cur_tile:
            continue  # Must move robber to a different tile
        victims = [
            oid for oid in tile_to_owners.get((tx, ty), set())
            if oid and oid != owner_id and oid != owner_name
            and pid_totals.get(oid, 0) > 0
        ]
        if victims:
            for v in victims:
                yield Action(type=ActionType.MOVE_ROBBER, tile=(tx, ty), steal_target=v)
        else:
            yield Action(type=ActionType.MOVE_ROBBER, tile=(tx, ty), steal_target=None)


def _enum_end_turn(observed, player, *_a, **_kw) -> Iterable[Action]:
    phase = _current_phase(observed)
    if phase in {"main", "play", "roll", "unknown", ""}:
        yield Action(type=ActionType.END_TURN)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def legal_actions(state: Dict[str, Any]) -> List[Action]:
    """Enumerate all legal parameterised actions for the current player.

    Uses ``observed_state`` only.  Returns an empty list if the current
    player cannot be resolved (e.g. malformed payload) -- callers should
    treat that as an explicit signal, not a silent success.
    """
    observed = _observed(state)
    player = _current_player(observed)
    if player is None:
        return []
    resources = _resources(player)
    occ_struct = _occupied_vertices(observed)
    occ_road = _occupied_roads(observed)
    owner_id = player.get("player_id") or ""
    owner_name = player.get("player_name") or ""

    actions: List[Action] = []
    phase = _current_phase(observed)

    # Robber phase pre-empts the normal build/buy/trade menu
    if phase in {"robber", "move_robber", "knight", "play_knight"}:
        actions.extend(_enum_move_robber(
            observed, player, resources, occ_struct, occ_road, owner_id, owner_name,
        ))
        return actions

    enumerators = (
        _enum_build_road, _enum_build_settlement, _enum_build_city,
        _enum_buy_dev_card, _enum_trade_bank, _enum_end_turn,
    )
    for fn in enumerators:
        actions.extend(fn(observed, player, resources, occ_struct, occ_road,
                          owner_id, owner_name))
    return actions
