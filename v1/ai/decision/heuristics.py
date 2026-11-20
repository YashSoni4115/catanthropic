"""Named heuristic rules layered on top of value-model scores.

Each rule is a small pure function with signature::

    rule(state, scored_actions, config) -> list[dict]

returning a list of ``{"action_index": int, "name": str, "delta": float,
"reason": str}`` records describing per-action score adjustments to apply.

The orchestrator :func:`apply_heuristics` sums the deltas onto
:attr:`ScoredAction.final_score` and appends the adjustment records to
:attr:`ScoredAction.heuristic_adjustments` for full inspectability.

The only hard override is :func:`rule_can_win_now`, which promotes a
winning action to a final score of ``+inf`` (guaranteed selection).
Every other rule ADJUSTS but does not replace the model score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .actions import Action, ActionType
from .legal_actions import _current_player, _observed, _same_owner
from .scorer import ScoredAction

# Points contributed by each action type to visible VP (approximation).
_VP_GAIN = {
    ActionType.BUILD_SETTLEMENT: 1,
    ActionType.BUILD_CITY: 1,
    # BUY_DEV_CARD may draw a VP card (~5/25 chance in a base deck) but
    # we cannot see it, so we treat it as 0 for the win-now rule.
}


@dataclass
class HeuristicConfig:
    # Toggle individual rules
    enable_can_win_now: bool = True
    enable_block_opponent_at_9: bool = True
    enable_secure_longest_road: bool = True
    enable_secure_largest_army: bool = True   # currently no-op (no play_knight)

    # ------------------------------------------------------------------
    # SCAFFOLDING — temporary calibration patch to bootstrap non-degenerate
    # self-play. The guided-random training corpus underweights the value
    # of settlements/cities (build deltas ~ +0.001 / +0.06 respectively),
    # so the greedy policy passes over settlements. This rule injects a
    # small additive VP prior on build_settlement / build_city so games
    # progress and the model can be retrained on non-degenerate data.
    # Intended to be DISABLED after value model retraining. Do not tune
    # indefinitely. See docs / phase-1 notes.
    # ------------------------------------------------------------------
    enable_build_vp_prior: bool = False
    weight_build_vp_prior: float = 0.03  # per VP conferred by the build

    # Weight of each rule's adjustment (multiplied into per-rule delta)
    weight_block_opponent_at_9: float = 0.20   # add up to +0.20 to blocking actions
    weight_secure_longest_road: float = 0.10
    weight_secure_largest_army: float = 0.10

    # Thresholds
    opponent_threat_vp: int = 9
    longest_road_target_length: int = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _visible_vp(player: Dict[str, Any]) -> int:
    v = player.get("victory_points_visible")
    return int(v) if isinstance(v, int) else 0


def _opponents(obs: Dict[str, Any], me: Dict[str, Any]) -> List[Dict[str, Any]]:
    my_id = me.get("player_id")
    return [
        p for p in (obs.get("players") or {}).get("all_players", [])
        if p.get("player_id") != my_id
    ]


def _tile_owners_map(obs: Dict[str, Any]) -> Dict[Tuple[int, int], List[Tuple[str, str]]]:
    from ai.economy.constants import vertex_adjacent_tiles
    out: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}
    for s in obs.get("structures") or []:
        try:
            vx, vy, vo = int(s["x"]), int(s["y"]), int(s["orientation"])
        except (KeyError, TypeError, ValueError):
            continue
        pid = s.get("owner_id") or ""
        pname = s.get("owner_name") or ""
        for t in vertex_adjacent_tiles(vx, vy, vo):
            out.setdefault((t["x"], t["y"]), []).append((pid, pname))
    return out


def _own_road_count(obs: Dict[str, Any], me: Dict[str, Any]) -> int:
    owner_id = me.get("player_id") or ""
    owner_name = me.get("player_name") or ""
    return sum(
        1 for r in (obs.get("roads") or [])
        if _same_owner(r, owner_id, owner_name)
    )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def rule_can_win_now(
    state: Dict[str, Any],
    scored: List[ScoredAction],
    config: HeuristicConfig,
) -> List[Dict[str, Any]]:
    """Hard override: if any action pushes visible VP to >= 10, force-pick it.

    Returns a single record with ``delta = +inf`` on the first qualifying
    action (deterministic order = input action order).
    """
    obs = _observed(state)
    me = _current_player(obs)
    if me is None:
        return []
    cur_vp = _visible_vp(me)
    for i, sa in enumerate(scored):
        gain = _VP_GAIN.get(sa.action.type, 0)
        if gain and cur_vp + gain >= 10:
            return [{
                "action_index": i,
                "name": "can_win_now",
                "delta": float("inf"),
                "reason": f"VP {cur_vp}+{gain} >= 10 via {sa.action.short()}",
            }]
    return []


def rule_block_opponent_at_9(
    state: Dict[str, Any],
    scored: List[ScoredAction],
    config: HeuristicConfig,
) -> List[Dict[str, Any]]:
    """If any opponent has visible VP >= threshold, upweight move_robber
    actions targeting their most productive tile.

    Approximation: "most productive" == tile whose adjacent structures
    include the threatening opponent, weighted by pip-equivalent numbers.
    We upweight any move_robber whose ``steal_target`` is the threat.
    """
    obs = _observed(state)
    me = _current_player(obs)
    if me is None:
        return []
    threats = [
        p for p in _opponents(obs, me)
        if _visible_vp(p) >= config.opponent_threat_vp
    ]
    if not threats:
        return []
    threat_ids = {p.get("player_id"): p for p in threats}
    threat_names = {p.get("player_name"): p for p in threats}

    records: List[Dict[str, Any]] = []
    for i, sa in enumerate(scored):
        if sa.action.type != ActionType.MOVE_ROBBER:
            continue
        tgt = sa.action.steal_target
        if tgt and (tgt in threat_ids or tgt in threat_names):
            records.append({
                "action_index": i,
                "name": "block_opponent_at_9",
                "delta": config.weight_block_opponent_at_9,
                "reason": f"opponent {tgt} at threshold VP",
            })
    return records


def rule_secure_longest_road(
    state: Dict[str, Any],
    scored: List[ScoredAction],
    config: HeuristicConfig,
) -> List[Dict[str, Any]]:
    """Upweight build_road when we are one road away from >=5 owned roads
    AND no other player already has largest-road credit visible."""
    obs = _observed(state)
    me = _current_player(obs)
    if me is None:
        return []
    my_roads = _own_road_count(obs, me)
    if my_roads + 1 < config.longest_road_target_length:
        return []
    # If any opponent already reports longest road, this rule is moot.
    for p in _opponents(obs, me):
        if bool(p.get("longest_road_flag") or p.get("has_longest_road")):
            return []
    records: List[Dict[str, Any]] = []
    for i, sa in enumerate(scored):
        if sa.action.type == ActionType.BUILD_ROAD:
            records.append({
                "action_index": i,
                "name": "secure_longest_road",
                "delta": config.weight_secure_longest_road,
                "reason": f"own_roads={my_roads}, +1 reaches {config.longest_road_target_length}",
            })
    return records


def rule_build_vp_prior(
    state: Dict[str, Any],
    scored: List[ScoredAction],
    config: HeuristicConfig,
) -> List[Dict[str, Any]]:
    """SCAFFOLDING — temporary calibration patch.

    Adds an additive bonus ``weight_build_vp_prior * vp_delta`` to
    build_settlement (vp_delta=1) and build_city (vp_delta=1) actions.
    Compensates for the value model's undervaluing of builds in the
    guided-random training corpus. Intended to be removed after the
    model is retrained on trajectories generated with this crutch.
    """
    records: List[Dict[str, Any]] = []
    w = float(config.weight_build_vp_prior)
    if w == 0.0:
        return records
    for i, sa in enumerate(scored):
        vp = _VP_GAIN.get(sa.action.type, 0)
        if vp <= 0:
            continue
        records.append({
            "action_index": i,
            "name": "build_vp_prior",
            "delta": w * float(vp),
            "reason": f"scaffolding VP prior: +{w:.3f} x {vp} = {w*vp:+.3f}",
        })
    return records


def rule_secure_largest_army(
    state: Dict[str, Any],
    scored: List[ScoredAction],
    config: HeuristicConfig,
) -> List[Dict[str, Any]]:
    """No-op in v1.

    play_knight is not enumerated by :mod:`legal_actions`, so there is no
    action to boost.  This rule is kept as a named stub so the config
    surface is stable for a future revision.
    """
    return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_heuristics(
    state: Dict[str, Any],
    scored: List[ScoredAction],
    config: Optional[HeuristicConfig] = None,
) -> List[ScoredAction]:
    """Mutate ``scored`` in place, appending adjustment records and
    updating ``final_score``.  Returns ``scored`` for chaining.
    """
    cfg = config or HeuristicConfig()

    # Order matters: can_win_now first so its +inf dominates.
    rule_pipeline: List[Tuple[bool, Any]] = [
        (cfg.enable_can_win_now, rule_can_win_now),
        (cfg.enable_block_opponent_at_9, rule_block_opponent_at_9),
        (cfg.enable_secure_longest_road, rule_secure_longest_road),
        (cfg.enable_secure_largest_army, rule_secure_largest_army),
        (cfg.enable_build_vp_prior, rule_build_vp_prior),
    ]

    for enabled, rule in rule_pipeline:
        if not enabled:
            continue
        for rec in rule(state, scored, cfg):
            idx = rec["action_index"]
            if 0 <= idx < len(scored):
                scored[idx].heuristic_adjustments.append(rec)
                delta = float(rec.get("delta", 0.0))
                if delta == float("inf"):
                    scored[idx].final_score = float("inf")
                else:
                    scored[idx].final_score += delta

    return scored
