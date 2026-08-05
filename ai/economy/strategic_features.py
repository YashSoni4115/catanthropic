"""Strategic (win-probability-oriented) per-player feature extractor.

Produces a fixed-length feature dict from ``observed_state`` (schema_version=1)
suitable for appending to the win-probability model vector.

Design conventions
------------------
* **Player padding**: all per-player features emit ``1 + MAX_OPPONENTS_SCHEMA_V1
  = 4`` slots (self, opp0, opp1, opp2). Missing opponents pad to 0.0.
* **Ordering**: self first; opponents sorted alphabetically by ``player_id``.
* **observed_state only** — no omniscient/hidden fields.
* **No history required at inference time** — features that need cross-turn
  history (aggression rates, VP rate of change, trade counts) default to 0.0
  when history isn't provided. The schema slot is always present.

Category structure
------------------
The output dict has one entry per category (dict-of-dicts). Each leaf is a
list of floats. The canonical flattening order is:

  1. sort category names alphabetically
  2. within a category, sort field names alphabetically
  3. within a field, values are already in ordered slot form

Total flat length must equal ``STRATEGIC_TOTAL_DIM``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ai.economy.robber_hotspot_features import MAX_OPPONENTS_SCHEMA_V1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOTS = 1 + MAX_OPPONENTS_SCHEMA_V1  # 4 = self + 3 opps
RESOURCES = ["BRICK", "GRAIN", "LUMBER", "ORE", "WOOL"]

# Pip lookup (dice roll → # of 36 outcomes)
_PIP_TABLE: Dict[int, float] = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
    8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}

# Phase turn boundaries (matching robber_hotspot_features)
_PHASE_EARLY_MAX = 15
_PHASE_MID_MAX = 35

# Category dimensions (see docstring). Sums must match STRATEGIC_TOTAL_DIM.
CATEGORY_DIMS = {
    "board_control":   5 * SLOTS,                                     # 20
    "dev_cards":       7 * SLOTS,                                     # 28
    "game_phase":      1 + 3 + SLOTS + SLOTS + 1,                     # 13
    "largest_army":    3 * SLOTS + (SLOTS + 1) + SLOTS,               # 21
    "longest_road":    3 * SLOTS + (SLOTS + 1) + SLOTS,               # 21
    "robber_threat":   4 * SLOTS + SLOTS,                             # 20
    "trade_ports":     (5 + 1) * SLOTS + 3 * SLOTS + len(RESOURCES),  # 41
    "vp_trajectory":   6 * SLOTS,                                     # 24
}
STRATEGIC_TOTAL_DIM = sum(CATEGORY_DIMS.values())  # 188


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_players(observed_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    players_field = observed_state.get("players") or {}
    if isinstance(players_field, dict):
        raw = players_field.get("all_players") or []
    else:
        raw = players_field
    return raw if isinstance(raw, list) else []


def _active_pid(observed_state: Dict[str, Any]) -> Optional[str]:
    tm = observed_state.get("turn_metadata") or {}
    return tm.get("current_player_id") or tm.get("current_player_name")


def _ordered_player_slots(
    observed_state: Dict[str, Any],
) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
    """Return ``(padded_players, padded_pids)`` with ``self`` first, then up
    to ``MAX_OPPONENTS`` opponents sorted alphabetically by ``player_id``.
    Missing slots are ``None``.
    """
    all_players = _all_players(observed_state)
    active = _active_pid(observed_state)
    self_player: Optional[Dict[str, Any]] = None
    opps: List[Dict[str, Any]] = []
    for p in all_players:
        pid = p.get("player_id") or p.get("player_name")
        if pid == active and self_player is None:
            self_player = p
        else:
            opps.append(p)
    opps.sort(key=lambda p: str(p.get("player_id") or p.get("player_name") or ""))
    opps = opps[:MAX_OPPONENTS_SCHEMA_V1]

    padded: List[Optional[Dict[str, Any]]] = [self_player]
    padded.extend(opps)
    while len(padded) < SLOTS:
        padded.append(None)

    pids: List[Optional[str]] = []
    for p in padded:
        if p is None:
            pids.append(None)
        else:
            pids.append(str(p.get("player_id") or p.get("player_name") or ""))
    return padded, pids


def _tile_lookup(observed_state: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    return {
        (int(t["x"]), int(t["y"])): t
        for t in (observed_state.get("tiles") or [])
        if "x" in t and "y" in t
    }


def _pip(number: Any) -> float:
    try:
        return _PIP_TABLE.get(int(number), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _phase_one_hot(turn_index: int) -> List[float]:
    if turn_index <= _PHASE_EARLY_MAX:
        return [1.0, 0.0, 0.0]
    if turn_index <= _PHASE_MID_MAX:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _pip_ownership_totals(
    observed_state: Dict[str, Any],
    tile_lookup: Dict[Tuple[int, int], Dict[str, Any]],
) -> Tuple[Dict[str, float], float]:
    """For each player id, sum pip × (1 for settlement, 2 for city) across all
    their adjacent tiles; also return board-wide total pips."""
    board_total_pips = sum(_pip(t.get("number")) for t in tile_lookup.values())
    by_owner: Dict[str, float] = {}
    for struct in observed_state.get("structures") or []:
        owner = struct.get("owner_id") or struct.get("owner_name")
        if not owner:
            continue
        weight = 2.0 if str(struct.get("type") or "").lower() == "city" else 1.0
        for adj in struct.get("adjacent_tiles") or []:
            key = (int(adj.get("x", -1)), int(adj.get("y", -1)))
            tile = tile_lookup.get(key)
            if tile is None:
                continue
            by_owner[owner] = by_owner.get(owner, 0.0) + weight * _pip(tile.get("number"))
    return by_owner, board_total_pips


def _resource_production_share(
    observed_state: Dict[str, Any],
    tile_lookup: Dict[Tuple[int, int], Dict[str, Any]],
) -> Dict[str, float]:
    """For each resource, return max share held by any single player."""
    per_resource_totals: Dict[str, float] = {r: 0.0 for r in RESOURCES}
    per_resource_by_owner: Dict[str, Dict[str, float]] = {r: {} for r in RESOURCES}
    for struct in observed_state.get("structures") or []:
        owner = struct.get("owner_id") or struct.get("owner_name")
        if not owner:
            continue
        weight = 2.0 if str(struct.get("type") or "").lower() == "city" else 1.0
        for adj in struct.get("adjacent_tiles") or []:
            key = (int(adj.get("x", -1)), int(adj.get("y", -1)))
            tile = tile_lookup.get(key)
            if tile is None:
                continue
            rt = str(tile.get("type") or "").upper()
            if rt not in per_resource_totals:
                continue
            pip = _pip(tile.get("number"))
            per_resource_totals[rt] += weight * pip
            per_resource_by_owner[rt][owner] = per_resource_by_owner[rt].get(owner, 0.0) + weight * pip
    out: Dict[str, float] = {}
    for r in RESOURCES:
        total = per_resource_totals[r]
        if total <= 0:
            out[r] = 0.0
            continue
        top = max(per_resource_by_owner[r].values()) if per_resource_by_owner[r] else 0.0
        out[r] = top / total
    return out


def _board_center(tile_lookup: Dict[Tuple[int, int], Dict[str, Any]]) -> Tuple[float, float]:
    if not tile_lookup:
        return (0.0, 0.0)
    xs = [k[0] for k in tile_lookup]
    ys = [k[1] for k in tile_lookup]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _centrality_by_owner(
    observed_state: Dict[str, Any],
    tile_lookup: Dict[Tuple[int, int], Dict[str, Any]],
) -> Dict[str, float]:
    cx, cy = _board_center(tile_lookup)
    out: Dict[str, Tuple[float, int]] = {}
    for struct in observed_state.get("structures") or []:
        owner = struct.get("owner_id") or struct.get("owner_name")
        if not owner:
            continue
        dist_sum = 0.0
        count = 0
        for adj in struct.get("adjacent_tiles") or []:
            key = (int(adj.get("x", -1)), int(adj.get("y", -1)))
            if key not in tile_lookup:
                continue
            dist_sum += abs(key[0] - cx) + abs(key[1] - cy)
            count += 1
        if count > 0:
            prev_sum, prev_count = out.get(owner, (0.0, 0))
            out[owner] = (prev_sum + dist_sum, prev_count + count)
    return {o: (s / c) if c else 0.0 for o, (s, c) in out.items()}


# ---------------------------------------------------------------------------
# Category builders
# ---------------------------------------------------------------------------

def _build_robber_threat(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
    tile_lookup: Dict[Tuple[int, int], Dict[str, Any]],
    pip_ownership: Dict[str, float],
    history: Optional[Dict[str, Any]],
) -> Dict[str, List[float]]:
    hist = history or {}
    robber_placements = hist.get("robber_placements_by_player") or {}
    robber_victims = hist.get("robber_victims_by_player") or {}
    last_mover = hist.get("last_robber_mover_pid")

    # steal_ev proxy = expected fraction of an opponent's card = total_resource_cards / max(hand_size, 1) * 1
    steal_ev = []
    aggression = []
    victim = []
    vulnerability = []
    last_mover_one_hot = []
    for player, pid in zip(padded, pids):
        if player is None or pid is None:
            aggression.append(0.0)
            victim.append(0.0)
            vulnerability.append(0.0)
            steal_ev.append(0.0)
            last_mover_one_hot.append(0.0)
            continue
        knights = _safe_float(player.get("knights_played_visible"))
        aggression.append(_safe_float(robber_placements.get(pid), default=knights))
        victim.append(_safe_float(robber_victims.get(pid)))
        vulnerability.append(pip_ownership.get(pid, 0.0))
        cards = _safe_float(player.get("total_resource_cards"))
        # Expected value of one steal ≈ min(1, cards) — capped to 1
        steal_ev.append(min(1.0, cards))
        last_mover_one_hot.append(1.0 if pid == last_mover else 0.0)

    return {
        "aggression": aggression,
        "victim_score": victim,
        "vulnerability": vulnerability,
        "steal_ev": steal_ev,
        "last_mover_one_hot": last_mover_one_hot,
    }


def _build_longest_road(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
) -> Dict[str, List[float]]:
    pgs = observed_state.get("public_game_state") or {}
    lr_owner = ((pgs.get("longest_road_owner") or {}).get("player_id")) or (
        (pgs.get("longest_road_owner") or {}).get("player_name")
    )
    lr_len = _safe_int((pgs.get("longest_road_owner") or {}).get("length"))

    length_norm: List[float] = []
    gap_to_5: List[float] = []
    extensions: List[float] = []
    within_2: List[float] = []
    for player, pid in zip(padded, pids):
        if player is None:
            length_norm.append(0.0)
            gap_to_5.append(0.0)
            extensions.append(0.0)
            within_2.append(0.0)
            continue
        bp = player.get("board_presence") or {}
        # Best available proxy for road length = road_count (segments placed).
        # True longest road requires graph traversal; the model can learn from
        # segment count for now (this is what the Java engine reports too).
        road_count = _safe_int(bp.get("road_count"))
        length_norm.append(road_count / 15.0)  # 15 = max road pieces
        gap = max(0, 5 - road_count)
        gap_to_5.append(float(gap))
        rp = player.get("remaining_pieces") or {}
        extensions.append(_safe_float(rp.get("roads_remaining")))
        within_2.append(1.0 if gap <= 2 else 0.0)

    held_one_hot = _one_hot_owner(pids, lr_owner)
    return {
        "length_norm": length_norm,
        "gap_to_5": gap_to_5,
        "extensions": extensions,
        "held_one_hot": held_one_hot,
        "within_2_stealing": within_2,
    }


def _one_hot_owner(pids: List[Optional[str]], owner: Optional[str]) -> List[float]:
    """Return length-(SLOTS+1) one-hot for owner, last slot = unowned."""
    out = [0.0] * (SLOTS + 1)
    if owner is None:
        out[-1] = 1.0
        return out
    for i, pid in enumerate(pids):
        if pid is not None and pid == owner:
            out[i] = 1.0
            return out
    # Owner not in known slots — treat as unowned
    out[-1] = 1.0
    return out


def _build_largest_army(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
) -> Dict[str, List[float]]:
    pgs = observed_state.get("public_game_state") or {}
    la_owner = ((pgs.get("largest_army_owner") or {}).get("player_id")) or (
        (pgs.get("largest_army_owner") or {}).get("player_name")
    )
    visible_knight_counts = pgs.get("visible_knight_counts") or {}

    knights_played: List[float] = []
    gap_to_3: List[float] = []
    est_unplayed_knights: List[float] = []
    within_1: List[float] = []
    for player, pid in zip(padded, pids):
        if player is None:
            knights_played.append(0.0)
            gap_to_3.append(0.0)
            est_unplayed_knights.append(0.0)
            within_1.append(0.0)
            continue
        kn = _safe_int(player.get("knights_played_visible"))
        if pid is not None and visible_knight_counts.get(pid) is not None:
            kn = max(kn, _safe_int(visible_knight_counts.get(pid)))
        knights_played.append(float(kn))
        gap = max(0, 3 - kn)
        gap_to_3.append(float(gap))
        # Estimated unplayed knights: only visible for self (hidden card counts null'd for opps).
        dev = player.get("development_cards") or {}
        grouped = dev.get("grouped_counts") or {}
        est_unplayed_knights.append(_safe_float(grouped.get("knight")))
        within_1.append(1.0 if gap <= 1 else 0.0)

    held_one_hot = _one_hot_owner(pids, la_owner)
    return {
        "knights_played": knights_played,
        "gap_to_3": gap_to_3,
        "est_unplayed_knights": est_unplayed_knights,
        "held_one_hot": held_one_hot,
        "within_1_stealing": within_1,
    }


def _build_vp_trajectory(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
    history: Optional[Dict[str, Any]],
) -> Dict[str, List[float]]:
    hist = history or {}
    vp_rate = hist.get("vp_rate_last_5_by_player") or {}

    vp_norm: List[float] = []
    hidden_vp: List[float] = []
    vp_gap: List[float] = []
    vp_rate_change: List[float] = []
    at_9_vp: List[float] = []
    city_conv: List[float] = []
    for player, pid in zip(padded, pids):
        if player is None:
            vp_norm.append(0.0)
            hidden_vp.append(0.0)
            vp_gap.append(0.0)
            vp_rate_change.append(0.0)
            at_9_vp.append(0.0)
            city_conv.append(0.0)
            continue
        vp = _safe_float(player.get("victory_points_visible"))
        vp_norm.append(vp / 10.0)
        dev = player.get("development_cards") or {}
        grouped = dev.get("grouped_counts") or {}
        hidden_vp.append(_safe_float(grouped.get("victory_point")))
        vp_gap.append(max(0.0, 10.0 - vp))
        vp_rate_change.append(_safe_float(vp_rate.get(pid)))
        at_9_vp.append(1.0 if vp >= 9.0 else 0.0)
        bp = player.get("board_presence") or {}
        s = _safe_int(bp.get("settlement_count"))
        c = _safe_int(bp.get("city_count"))
        total = s + c
        city_conv.append((c / total) if total > 0 else 0.0)

    return {
        "vp_norm": vp_norm,
        "hidden_vp": hidden_vp,
        "vp_gap_to_10": vp_gap,
        "vp_rate_change": vp_rate_change,
        "at_9_vp": at_9_vp,
        "city_conversion_rate": city_conv,
    }


def _build_dev_cards(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
) -> Dict[str, List[float]]:
    total_bought: List[float] = []
    total_played: List[float] = []
    est_unplayed: List[float] = []
    p_road_building: List[float] = []
    p_yop: List[float] = []
    p_monopoly: List[float] = []
    knight_ratio: List[float] = []

    for player in padded:
        if player is None:
            for lst in (total_bought, total_played, est_unplayed,
                        p_road_building, p_yop, p_monopoly, knight_ratio):
                lst.append(0.0)
            continue
        dev = player.get("development_cards") or {}
        grouped = dev.get("grouped_counts") or {}
        total_visible = _safe_float(dev.get("total_cards"))
        knights_played = _safe_float(player.get("knights_played_visible"))
        # total_bought = knights_played + visible_unplayed (best obs proxy)
        unplayed_total = (
            _safe_float(grouped.get("knight"))
            + _safe_float(grouped.get("progress"))
            + _safe_float(grouped.get("victory_point"))
        )
        total_bought.append(knights_played + unplayed_total)
        total_played.append(knights_played)  # progress plays are ephemeral, not tracked here
        est_unplayed.append(unplayed_total)
        progress_sub = dev.get("progress_subtype_counts") or {}
        prog_total = max(1.0, _safe_float(grouped.get("progress")))
        p_road_building.append(_safe_float(progress_sub.get("road_building")) / prog_total)
        p_yop.append(_safe_float(progress_sub.get("year_of_plenty")) / prog_total)
        p_monopoly.append(_safe_float(progress_sub.get("monopoly")) / prog_total)
        total_played_all = knights_played  # only knights are truly "played" here
        knight_ratio.append((knights_played / total_played_all) if total_played_all > 0 else 0.0)

    return {
        "total_bought": total_bought,
        "total_played": total_played,
        "est_unplayed": est_unplayed,
        "p_road_building": p_road_building,
        "p_year_of_plenty": p_yop,
        "p_monopoly": p_monopoly,
        "knight_to_dev_ratio": knight_ratio,
    }


def _build_trade_ports(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
    tile_lookup: Dict[Tuple[int, int], Dict[str, Any]],
    history: Optional[Dict[str, Any]],
) -> Dict[str, List[float]]:
    hist = history or {}
    trade_counts = hist.get("trade_counts_by_player") or {}
    maritime_counts = hist.get("maritime_counts_by_player") or {}
    turn_index = _safe_int((observed_state.get("turn_metadata") or {}).get("turn_index"))

    # Per-player: 5 two-to-one flags + 1 three-to-one flag = 6 dims per player.
    port_flags: Dict[str, List[float]] = {r: [] for r in RESOURCES}
    three_to_one: List[float] = []
    trade_activity_rate: List[float] = []
    maritime_ratio: List[float] = []
    surplus_score: List[float] = []
    for player, pid in zip(padded, pids):
        if player is None:
            for r in RESOURCES:
                port_flags[r].append(0.0)
            three_to_one.append(0.0)
            trade_activity_rate.append(0.0)
            maritime_ratio.append(0.0)
            surplus_score.append(0.0)
            continue
        ports = player.get("ports") or {}
        for r in RESOURCES:
            key = f"{r.lower()}_two_to_one"
            port_flags[r].append(1.0 if bool(ports.get(key)) else 0.0)
        three_to_one.append(1.0 if bool(ports.get("three_to_one")) else 0.0)
        total_trades = _safe_float(trade_counts.get(pid))
        maritime = _safe_float(maritime_counts.get(pid))
        turn_denom = max(1.0, float(turn_index))
        trade_activity_rate.append(total_trades / turn_denom)
        maritime_ratio.append((maritime / total_trades) if total_trades > 0 else 0.0)
        surplus_score.append(_safe_float(player.get("total_resource_cards")))

    monopoly_risk = _resource_production_share(observed_state, tile_lookup)
    monopoly_risk_vec = [monopoly_risk[r] for r in RESOURCES]

    # port_flags dict-of-lists → flat named entries (one per resource)
    out: Dict[str, List[float]] = {}
    for r in RESOURCES:
        out[f"port_2to1_{r}"] = port_flags[r]
    out["port_3to1"] = three_to_one
    out["trade_activity_rate"] = trade_activity_rate
    out["maritime_ratio"] = maritime_ratio
    out["surplus_score"] = surplus_score
    out["monopoly_risk"] = monopoly_risk_vec
    return out


def _build_board_control(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
    pids: List[Optional[str]],
    tile_lookup: Dict[Tuple[int, int], Dict[str, Any]],
    pip_ownership: Dict[str, float],
    board_pip_total: float,
) -> Dict[str, List[float]]:
    centrality = _centrality_by_owner(observed_state, tile_lookup)

    expansion_space: List[float] = []
    blocked_expansion: List[float] = []
    city_upgrade_candidates: List[float] = []
    pip_share: List[float] = []
    board_centrality: List[float] = []

    for player, pid in zip(padded, pids):
        if player is None:
            expansion_space.append(0.0)
            blocked_expansion.append(0.0)
            city_upgrade_candidates.append(0.0)
            pip_share.append(0.0)
            board_centrality.append(0.0)
            continue
        rp = player.get("remaining_pieces") or {}
        bp = player.get("board_presence") or {}
        # Expansion proxy: remaining settlements × road_count (roads open doors)
        expansion_space.append(
            _safe_float(rp.get("settlements_remaining")) * (1.0 + _safe_float(bp.get("road_count")))
        )
        # Blocked proxy: 1 - (settlements_remaining / 5)  → more built = more blocked
        settlements_placed = 5.0 - _safe_float(rp.get("settlements_remaining"), default=5.0)
        blocked_expansion.append(min(1.0, settlements_placed / 5.0))
        # City upgrade candidates: count own settlements on tiles with pip >= 4/36
        candidates = 0
        for s in bp.get("settlements") or []:
            for adj in s.get("adjacent_tiles") or []:
                key = (int(adj.get("x", -1)), int(adj.get("y", -1)))
                tile = tile_lookup.get(key)
                if tile is None:
                    continue
                if _pip(tile.get("number")) >= (4.0 / 36.0):
                    candidates += 1
                    break
        city_upgrade_candidates.append(float(candidates))
        pip_share.append(
            (pip_ownership.get(pid, 0.0) / board_pip_total) if board_pip_total > 0 and pid else 0.0
        )
        board_centrality.append(centrality.get(pid, 0.0) if pid else 0.0)

    return {
        "expansion_space": expansion_space,
        "blocked_expansion": blocked_expansion,
        "city_upgrade_candidates": city_upgrade_candidates,
        "pip_ownership_share": pip_share,
        "board_centrality": board_centrality,
    }


def _build_game_phase(
    observed_state: Dict[str, Any],
    padded: List[Optional[Dict[str, Any]]],
) -> Dict[str, List[float]]:
    tm = observed_state.get("turn_metadata") or {}
    turn_index = _safe_int(tm.get("turn_index"))
    phase = _phase_one_hot(turn_index)
    settlements_remaining: List[float] = []
    cities_remaining: List[float] = []
    for player in padded:
        if player is None:
            settlements_remaining.append(0.0)
            cities_remaining.append(0.0)
            continue
        rp = player.get("remaining_pieces") or {}
        settlements_remaining.append(_safe_float(rp.get("settlements_remaining")))
        cities_remaining.append(_safe_float(rp.get("cities_remaining")))
    initial_placement_complete = 1.0 if turn_index >= 1 else 0.0
    return {
        "turn_norm": [min(1.0, turn_index / 120.0)],
        "phase_one_hot": phase,
        "settlements_remaining": settlements_remaining,
        "cities_remaining": cities_remaining,
        "initial_placement_complete": [initial_placement_complete],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_strategic_features(
    observed_state: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Return dict-of-dicts of fixed-length lists for every category.

    ``history`` is an optional per-game history bag used by features that need
    cross-turn context (aggression, VP rate, trade rates). All history-derived
    features default to 0.0 when absent, so single-shot vectorization works.
    """
    padded, pids = _ordered_player_slots(observed_state)
    tile_lookup = _tile_lookup(observed_state)
    pip_ownership, board_pip_total = _pip_ownership_totals(observed_state, tile_lookup)

    features = {
        "robber_threat": _build_robber_threat(
            observed_state, padded, pids, tile_lookup, pip_ownership, history
        ),
        "longest_road": _build_longest_road(observed_state, padded, pids),
        "largest_army": _build_largest_army(observed_state, padded, pids),
        "vp_trajectory": _build_vp_trajectory(observed_state, padded, pids, history),
        "dev_cards": _build_dev_cards(observed_state, padded, pids),
        "trade_ports": _build_trade_ports(
            observed_state, padded, pids, tile_lookup, history
        ),
        "board_control": _build_board_control(
            observed_state, padded, pids, tile_lookup, pip_ownership, board_pip_total
        ),
        "game_phase": _build_game_phase(observed_state, padded),
    }
    return features


def flatten_strategic_features(
    features: Dict[str, Dict[str, List[float]]],
) -> Tuple[List[float], List[str]]:
    """Return ``(flat_vector, ordered_feature_names)``.

    Ordering: category alphabetical, then field alphabetical, then slot index.
    """
    flat: List[float] = []
    names: List[str] = []
    for cat_name in sorted(features.keys()):
        cat = features[cat_name]
        for field_name in sorted(cat.keys()):
            values = cat[field_name]
            for i, v in enumerate(values):
                flat.append(float(v))
                names.append(f"strategic.{cat_name}.{field_name}.{i}")
    return flat, names


def validate_strategic_schema(features: Dict[str, Dict[str, List[float]]]) -> None:
    """Raise ``ValueError`` if the strategic feature dict doesn't match the
    declared per-category dimensions or total dim."""
    for cat_name, expected in CATEGORY_DIMS.items():
        if cat_name not in features:
            raise ValueError(f"strategic: missing category '{cat_name}'")
        actual = sum(len(v) for v in features[cat_name].values())
        if actual != expected:
            raise ValueError(
                f"strategic: category '{cat_name}' dim {actual} != expected {expected}"
            )
    total = sum(
        sum(len(v) for v in cat.values()) for cat in features.values()
    )
    if total != STRATEGIC_TOTAL_DIM:
        raise ValueError(
            f"strategic: total dim {total} != expected {STRATEGIC_TOTAL_DIM}"
        )


def strategic_feature_names() -> List[str]:
    """Return the canonical ordered list of ``STRATEGIC_TOTAL_DIM`` feature
    names produced by ``flatten_strategic_features`` (using an all-zero
    reference schema)."""
    # Build an empty scaffold with correct shapes so flatten is deterministic
    scaffold = _empty_scaffold()
    _, names = flatten_strategic_features(scaffold)
    return names


def _empty_scaffold() -> Dict[str, Dict[str, List[float]]]:
    """Return a zero-filled dict matching the strategic schema — useful for
    schema introspection without needing a real observed_state."""
    return {
        "robber_threat": {
            "aggression": [0.0] * SLOTS,
            "victim_score": [0.0] * SLOTS,
            "vulnerability": [0.0] * SLOTS,
            "steal_ev": [0.0] * SLOTS,
            "last_mover_one_hot": [0.0] * SLOTS,
        },
        "longest_road": {
            "length_norm": [0.0] * SLOTS,
            "gap_to_5": [0.0] * SLOTS,
            "extensions": [0.0] * SLOTS,
            "held_one_hot": [0.0] * (SLOTS + 1),
            "within_2_stealing": [0.0] * SLOTS,
        },
        "largest_army": {
            "knights_played": [0.0] * SLOTS,
            "gap_to_3": [0.0] * SLOTS,
            "est_unplayed_knights": [0.0] * SLOTS,
            "held_one_hot": [0.0] * (SLOTS + 1),
            "within_1_stealing": [0.0] * SLOTS,
        },
        "vp_trajectory": {
            "vp_norm": [0.0] * SLOTS,
            "hidden_vp": [0.0] * SLOTS,
            "vp_gap_to_10": [0.0] * SLOTS,
            "vp_rate_change": [0.0] * SLOTS,
            "at_9_vp": [0.0] * SLOTS,
            "city_conversion_rate": [0.0] * SLOTS,
        },
        "dev_cards": {
            "total_bought": [0.0] * SLOTS,
            "total_played": [0.0] * SLOTS,
            "est_unplayed": [0.0] * SLOTS,
            "p_road_building": [0.0] * SLOTS,
            "p_year_of_plenty": [0.0] * SLOTS,
            "p_monopoly": [0.0] * SLOTS,
            "knight_to_dev_ratio": [0.0] * SLOTS,
        },
        "trade_ports": {
            f"port_2to1_{r}": [0.0] * SLOTS for r in RESOURCES
        } | {
            "port_3to1": [0.0] * SLOTS,
            "trade_activity_rate": [0.0] * SLOTS,
            "maritime_ratio": [0.0] * SLOTS,
            "surplus_score": [0.0] * SLOTS,
            "monopoly_risk": [0.0] * len(RESOURCES),
        },
        "board_control": {
            "expansion_space": [0.0] * SLOTS,
            "blocked_expansion": [0.0] * SLOTS,
            "city_upgrade_candidates": [0.0] * SLOTS,
            "pip_ownership_share": [0.0] * SLOTS,
            "board_centrality": [0.0] * SLOTS,
        },
        "game_phase": {
            "turn_norm": [0.0],
            "phase_one_hot": [0.0] * 3,
            "settlements_remaining": [0.0] * SLOTS,
            "cities_remaining": [0.0] * SLOTS,
            "initial_placement_complete": [0.0],
        },
    }
