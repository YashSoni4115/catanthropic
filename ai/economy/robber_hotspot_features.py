"""Robber-hotspot per-tile feature extractor and output adapter.

Predicts the probability distribution over the 19 tiles for the next robber
placement. Uses only ``observed_state`` (no omniscient / hidden card data).

Feature layout
--------------
Per tile (22 features, in the order listed):

  Static (9)
    0  pip_weight              float, dice probability of tile number
    1  is_desert               {0,1}
    2  has_robber              {0,1} (current robber location)
    3-8 resource_one_hot       6 dims over {BRICK, GRAIN, LUMBER, ORE, WOOL, DESERT}

  Structural (5)
    9  settlement_count        int adjacent settlements
   10  city_count              int adjacent cities
   11  distinct_player_count   int distinct owners adjacent
   12  expected_output_per_turn pip_weight × (settlements + 2*cities)
   13  active_player_has_structure {0,1} — active player can't steal from self

  Opponent-relative (4)
   14  vp_weighted_presence    sum of opponents' visible VP on this tile
   15  belief_resource_wealth  proxy: sum of opponents' public total_resource_cards on tile
   16  leader_is_present       {0,1}  (current VP leader adjacent)
   17  threat_player_present   {0,1}  (any player with VP >= 8 adjacent)

  Historical (2)
   18  times_robbed_this_game  int (cumulative, dataset-time only)
   19  active_player_targeted_tile_before {0,1}

  Strategic (2)
   20  would_block_bottleneck  {0,1} proxy — any opponent on tile with ≥3 cards
   21  steal_resolves_active_bottleneck {0,1} proxy

Global context (6, appended once):
   turn_norm, active_vp, active_knight_count, phase_early, phase_mid, phase_late

Total input dim = 19 × 22 + 6 = 424.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOURCE_ORDER = ["BRICK", "GRAIN", "LUMBER", "ORE", "WOOL", "DESERT"]
FEATURES_PER_TILE = 22
NUM_TILES = 19
GLOBAL_CONTEXT_DIM = 6
INPUT_DIM = NUM_TILES * FEATURES_PER_TILE + GLOBAL_CONTEXT_DIM  # 424

# Standard Catan dice-roll probabilities (out of 36)
PIP_TABLE: Dict[int, float] = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
    8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}

PHASE_EARLY_TURN_MAX = 30
PHASE_MID_TURN_MAX = 80
TURN_NORMALIZE = 120.0

# For output adapter (must match economy_flat_v1.2.0 style)
MAX_OPPONENTS_SCHEMA_V1 = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_tile_order(tiles: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return tiles sorted deterministically by (x, y)."""
    return sorted(tiles, key=lambda t: (int(t.get("x", 0)), int(t.get("y", 0))))


def build_tile_index(observed_state: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    """Return mapping ``(x, y) -> tile_index`` in canonical order."""
    tiles = observed_state.get("tiles") or []
    ordered = _canonical_tile_order(tiles)
    return {(int(t["x"]), int(t["y"])): i for i, t in enumerate(ordered)}


def _pip_weight(number: Any) -> float:
    try:
        n = int(number)
    except (TypeError, ValueError):
        return 0.0
    return PIP_TABLE.get(n, 0.0)


def _resource_one_hot(resource_type: str) -> List[float]:
    rt = (resource_type or "DESERT").upper()
    return [1.0 if rt == r else 0.0 for r in RESOURCE_ORDER]


def _get_all_players(observed_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    players_field = observed_state.get("players") or {}
    if isinstance(players_field, dict):
        result = players_field.get("all_players") or []
    else:
        result = players_field
    return result if isinstance(result, list) else []


def _active_player_id(observed_state: Dict[str, Any]) -> Optional[str]:
    tm = observed_state.get("turn_metadata") or {}
    return tm.get("current_player_id") or tm.get("current_player_name")


def _phase_one_hot(turn_index: int) -> List[float]:
    if turn_index < PHASE_EARLY_TURN_MAX:
        return [1.0, 0.0, 0.0]
    if turn_index < PHASE_MID_TURN_MAX:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _structures_by_tile(
    observed_state: Dict[str, Any],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Map ``(x, y) -> list of structures adjacent to that tile``."""
    out: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for structure in observed_state.get("structures") or []:
        for adj in structure.get("adjacent_tiles") or []:
            key = (int(adj.get("x", -1)), int(adj.get("y", -1)))
            out.setdefault(key, []).append(structure)
    return out


# ---------------------------------------------------------------------------
# Per-tile feature extraction
# ---------------------------------------------------------------------------

def _tile_features(
    tile: Dict[str, Any],
    tile_key: Tuple[int, int],
    structures_here: List[Dict[str, Any]],
    all_players: List[Dict[str, Any]],
    active_pid: Optional[str],
    leader_pid: Optional[str],
    times_robbed: int,
    active_targeted_before: bool,
) -> List[float]:
    resource_type = str(tile.get("type") or "DESERT").upper()
    number = tile.get("number")
    is_desert = 1.0 if resource_type == "DESERT" else 0.0
    has_robber = 1.0 if bool(tile.get("robber")) else 0.0
    pip = 0.0 if is_desert else _pip_weight(number)

    settlement_count = 0
    city_count = 0
    owners: set = set()
    active_has_structure = False
    for s in structures_here:
        stype = str(s.get("type") or "").lower()
        if stype == "settlement":
            settlement_count += 1
        elif stype == "city":
            city_count += 1
        owner = s.get("owner_id") or s.get("owner_name")
        if owner:
            owners.add(owner)
            if active_pid is not None and owner == active_pid:
                active_has_structure = True

    distinct_players = len(owners)
    expected_output = pip * (settlement_count + 2 * city_count)

    # ── Opponent-relative aggregations ────────────────────────────────
    vp_weighted = 0.0
    belief_wealth = 0.0
    leader_present = False
    threat_present = False
    opp_on_tile_max_cards = 0
    any_opp_can_win = False
    for player in all_players:
        pid = player.get("player_id") or player.get("player_name")
        if not pid or pid == active_pid:
            continue
        if pid not in owners:
            continue
        vp = float(player.get("victory_points_visible") or 0)
        cards = int(player.get("total_resource_cards") or 0)
        vp_weighted += vp
        belief_wealth += float(cards)
        opp_on_tile_max_cards = max(opp_on_tile_max_cards, cards)
        if leader_pid is not None and pid == leader_pid:
            leader_present = True
        if vp >= 8.0:
            threat_present = True
        if vp >= 9.0:
            any_opp_can_win = True

    # ── Strategic proxies ────────────────────────────────────────────
    would_block = 1.0 if opp_on_tile_max_cards >= 3 else 0.0

    active_cards = 0
    for player in all_players:
        pid = player.get("player_id") or player.get("player_name")
        if pid == active_pid:
            active_cards = int(player.get("total_resource_cards") or 0)
            break
    steal_resolves = 1.0 if (active_cards <= 2 and opp_on_tile_max_cards >= 1) else 0.0

    features: List[float] = []
    features.append(pip)
    features.append(is_desert)
    features.append(has_robber)
    features.extend(_resource_one_hot(resource_type))
    features.append(float(settlement_count))
    features.append(float(city_count))
    features.append(float(distinct_players))
    features.append(float(expected_output))
    features.append(1.0 if active_has_structure else 0.0)
    features.append(float(vp_weighted))
    features.append(float(belief_wealth))
    features.append(1.0 if leader_present else 0.0)
    features.append(1.0 if threat_present else 0.0)
    features.append(float(times_robbed))
    features.append(1.0 if active_targeted_before else 0.0)
    features.append(would_block)
    features.append(steal_resolves)
    # (opponent_can_win_next_turn is folded into threat_present at VP>=8;
    #  we surface the stricter VP>=9 signal in threat_present as well when true.)
    if any_opp_can_win:
        features[-6] = 1.0  # promote threat_present to 1 unconditionally on VP>=9

    assert len(features) == FEATURES_PER_TILE, (
        f"tile feature count mismatch: {len(features)} != {FEATURES_PER_TILE}"
    )
    return features


def extract_tile_features(
    observed_state: Dict[str, Any],
    *,
    times_robbed_by_tile: Optional[Dict[Tuple[int, int], int]] = None,
    active_targeted_tiles: Optional[set] = None,
) -> Tuple[List[List[float]], List[Tuple[int, int]]]:
    """Return ``(per_tile_features, tile_keys)`` in canonical tile order.

    ``times_robbed_by_tile`` and ``active_targeted_tiles`` are optional history
    inputs supplied at dataset-build time.  Both default to empty when omitted
    (e.g. at inference time when no game history is being tracked).
    """
    times_robbed_by_tile = times_robbed_by_tile or {}
    active_targeted_tiles = active_targeted_tiles or set()

    tiles = _canonical_tile_order(observed_state.get("tiles") or [])
    structures_by_tile = _structures_by_tile(observed_state)
    all_players = _get_all_players(observed_state)
    active_pid = _active_player_id(observed_state)

    # Determine current leader (max visible VP, ties → first alphabetically)
    leader_pid: Optional[str] = None
    leader_vp = -1.0
    for player in all_players:
        vp = float(player.get("victory_points_visible") or 0)
        if vp > leader_vp:
            leader_vp = vp
            leader_pid = player.get("player_id") or player.get("player_name")

    per_tile: List[List[float]] = []
    tile_keys: List[Tuple[int, int]] = []
    for tile in tiles:
        key = (int(tile["x"]), int(tile["y"]))
        tile_keys.append(key)
        per_tile.append(
            _tile_features(
                tile=tile,
                tile_key=key,
                structures_here=structures_by_tile.get(key, []),
                all_players=all_players,
                active_pid=active_pid,
                leader_pid=leader_pid,
                times_robbed=times_robbed_by_tile.get(key, 0),
                active_targeted_before=key in active_targeted_tiles,
            )
        )
    return per_tile, tile_keys


def extract_global_context(observed_state: Dict[str, Any]) -> List[float]:
    tm = observed_state.get("turn_metadata") or {}
    turn_index = int(tm.get("turn_index") or 0)
    turn_norm = min(max(turn_index / TURN_NORMALIZE, 0.0), 1.0)

    active_pid = _active_player_id(observed_state)
    active_vp = 0.0
    active_knights = 0.0
    for p in _get_all_players(observed_state):
        pid = p.get("player_id") or p.get("player_name")
        if pid == active_pid:
            active_vp = float(p.get("victory_points_visible") or 0)
            active_knights = float(p.get("knights_played_visible") or 0)
            break

    phase = _phase_one_hot(turn_index)
    return [turn_norm, active_vp, active_knights, *phase]


def extract_input_vector(
    observed_state: Dict[str, Any],
    *,
    times_robbed_by_tile: Optional[Dict[Tuple[int, int], int]] = None,
    active_targeted_tiles: Optional[set] = None,
) -> Tuple[List[float], List[Tuple[int, int]]]:
    """Return the flat 424-dim input vector and its canonical tile-key order."""
    per_tile, tile_keys = extract_tile_features(
        observed_state,
        times_robbed_by_tile=times_robbed_by_tile,
        active_targeted_tiles=active_targeted_tiles,
    )
    flat: List[float] = []
    for row in per_tile:
        flat.extend(row)
    flat.extend(extract_global_context(observed_state))
    assert len(flat) == INPUT_DIM, f"vec dim {len(flat)} != {INPUT_DIM}"
    return flat, tile_keys


def feature_schema() -> List[str]:
    """Return an ordered list of the 424 feature names."""
    per_tile_names = [
        "pip_weight",
        "is_desert",
        "has_robber",
        *[f"resource.{r}" for r in RESOURCE_ORDER],
        "settlement_count",
        "city_count",
        "distinct_player_count",
        "expected_output_per_turn",
        "active_player_has_structure",
        "vp_weighted_presence",
        "belief_resource_wealth",
        "leader_is_present",
        "threat_player_present",
        "times_robbed_this_game",
        "active_player_targeted_tile_before",
        "would_block_bottleneck",
        "steal_resolves_active_bottleneck",
    ]
    assert len(per_tile_names) == FEATURES_PER_TILE
    names: List[str] = []
    for tile_idx in range(NUM_TILES):
        for feat in per_tile_names:
            names.append(f"tile{tile_idx}.{feat}")
    names.extend([
        "global.turn_norm",
        "global.active_vp",
        "global.active_knight_count",
        "global.phase_early",
        "global.phase_mid",
        "global.phase_late",
    ])
    assert len(names) == INPUT_DIM
    return names


# ---------------------------------------------------------------------------
# Output adapter — for downstream consumption by the win-probability model
# ---------------------------------------------------------------------------

ADAPTER_FEATURE_NAMES: List[str] = [
    "robber.top1_tile_id",
    "robber.top1_prob",
    "robber.top3_entropy",
    *[f"robber.vulnerability_score.opp{i}" for i in range(MAX_OPPONENTS_SCHEMA_V1)],
    *[f"robber.expected_block_score.{r}" for r in RESOURCE_ORDER[:5]],  # BRICK..WOOL
]
ADAPTER_DIM = len(ADAPTER_FEATURE_NAMES)  # 3 + 3 + 5 = 11


def _entropy(probs: Sequence[float]) -> float:
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs:
        if p <= 0:
            continue
        q = p / total
        h -= q * math.log(q)
    return h


def get_robber_hotspot_features(
    observed_state: Dict[str, Any],
    model: Any = None,
    *,
    tile_probs: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Return the fixed-length adapter dict for the win-probability model.

    Either ``model`` (with ``predict_proba``) or a pre-computed ``tile_probs``
    sequence of length 19 must be supplied.  All opponents are padded to
    ``MAX_OPPONENTS_SCHEMA_V1 = 3`` slots (matching the existing belief pattern).
    """
    per_tile, tile_keys = extract_tile_features(observed_state)
    global_ctx = extract_global_context(observed_state)
    flat: List[float] = []
    for row in per_tile:
        flat.extend(row)
    flat.extend(global_ctx)

    if tile_probs is None:
        if model is None:
            raise ValueError("get_robber_hotspot_features requires model or tile_probs")
        import numpy as np  # local import
        x = np.asarray(flat, dtype=float).reshape(1, -1)
        proba = model.predict_proba(x)[0]
        # Model outputs one probability per class present in training. Align to
        # the canonical 19-tile order using ``model.classes_``.
        aligned = [0.0] * NUM_TILES
        classes = getattr(model, "classes_", None)
        if classes is not None:
            for cls_idx, cls_label in enumerate(classes):
                idx = int(cls_label)
                if 0 <= idx < NUM_TILES:
                    aligned[idx] = float(proba[cls_idx])
        else:
            for i in range(min(NUM_TILES, len(proba))):
                aligned[i] = float(proba[i])
        tile_probs = aligned
    else:
        tile_probs = [float(p) for p in tile_probs]
        if len(tile_probs) != NUM_TILES:
            raise ValueError(f"tile_probs must have length {NUM_TILES}")

    # ── top-1 / top-3 stats ──────────────────────────────────────────
    indexed = sorted(range(NUM_TILES), key=lambda i: tile_probs[i], reverse=True)
    top1_idx = indexed[0]
    top1_prob = tile_probs[top1_idx]
    top3_entropy = _entropy([tile_probs[i] for i in indexed[:3]])

    # ── Per-opponent vulnerability ──────────────────────────────────
    active_pid = _active_player_id(observed_state)
    structures_by_tile = _structures_by_tile(observed_state)
    all_players = _get_all_players(observed_state)

    opponents = [
        p for p in all_players
        if (p.get("player_id") or p.get("player_name")) != active_pid
    ]
    # Deterministic order: by player_name / player_id
    opponents.sort(key=lambda p: str(p.get("player_id") or p.get("player_name")))

    vulnerability_by_opp: List[float] = []
    for opp in opponents[:MAX_OPPONENTS_SCHEMA_V1]:
        opp_pid = opp.get("player_id") or opp.get("player_name")
        score = 0.0
        for tile_idx, key in enumerate(tile_keys):
            structs = structures_by_tile.get(key, [])
            weight = 0.0
            for s in structs:
                if (s.get("owner_id") or s.get("owner_name")) != opp_pid:
                    continue
                stype = str(s.get("type") or "").lower()
                weight += 2.0 if stype == "city" else 1.0
            if weight > 0:
                score += tile_probs[tile_idx] * weight
        vulnerability_by_opp.append(score)
    while len(vulnerability_by_opp) < MAX_OPPONENTS_SCHEMA_V1:
        vulnerability_by_opp.append(0.0)

    # ── Expected block per resource ─────────────────────────────────
    resource_block: Dict[str, float] = {r: 0.0 for r in RESOURCE_ORDER[:5]}
    tiles = _canonical_tile_order(observed_state.get("tiles") or [])
    for tile_idx, tile in enumerate(tiles):
        res = str(tile.get("type") or "").upper()
        if res not in resource_block:
            continue
        pip = _pip_weight(tile.get("number"))
        struct_weight = 0.0
        for s in structures_by_tile.get(tile_keys[tile_idx], []):
            stype = str(s.get("type") or "").lower()
            struct_weight += 2.0 if stype == "city" else 1.0
        resource_block[res] += tile_probs[tile_idx] * pip * struct_weight

    out: Dict[str, float] = {
        "robber.top1_tile_id": float(top1_idx),
        "robber.top1_prob": float(top1_prob),
        "robber.top3_entropy": float(top3_entropy),
    }
    for i in range(MAX_OPPONENTS_SCHEMA_V1):
        out[f"robber.vulnerability_score.opp{i}"] = float(vulnerability_by_opp[i])
    for r in RESOURCE_ORDER[:5]:
        out[f"robber.expected_block_score.{r}"] = float(resource_block[r])

    assert len(out) == ADAPTER_DIM, f"adapter dict size {len(out)} != {ADAPTER_DIM}"
    return out


def flatten_adapter_features(features: Dict[str, float]) -> List[float]:
    """Flatten adapter dict into fixed-length vector in canonical name order."""
    return [float(features[name]) for name in ADAPTER_FEATURE_NAMES]
