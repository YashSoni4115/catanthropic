from __future__ import annotations

from typing import Dict, List, Optional

from ai.economy.constants import MAX_OPPONENTS_SCHEMA_V1, TRACKED_RESOURCES


# CHANGELOG: v1.2.0 — FIX 3 opening features always emitted (no more
# variable-length vector); FIX 4 belief slots padded to MAX_OPPONENTS.
VERSION = "economy_flat_v1.2.0"


def flatten_economy_features_v1(
    economy_state: Dict[str, object],
    viewer_player_id: str,
    opening_economy: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    board_economy = economy_state.get("board_economy", {}) if isinstance(economy_state.get("board_economy"), dict) else {}
    board_resource_environment = (
        board_economy.get("resource_environment", {}).get("by_resource", {})
        if isinstance(board_economy.get("resource_environment"), dict)
        else {}
    )
    for resource in TRACKED_RESOURCES:
        stats = board_resource_environment.get(resource, {}) if isinstance(board_resource_environment.get(resource), dict) else {}
        features[f"board.{resource}.pip_total"] = float(stats.get("pip_total", 0.0))
        features[f"board.{resource}.scarcity_score"] = float(stats.get("scarcity_score", 0.0))
        features[f"board.{resource}.abundance_score"] = float(stats.get("abundance_score", 0.0))
        features[f"board.{resource}.concentration_score"] = float(stats.get("concentration_score", 0.0))

    # Opening features: prefer explicit argument, then economy_state slot,
    # then board_economy (which already contains opening rankings from
    # build_board_profile).  FIX 3: always emit these features with 0.0
    # defaults to keep vector length constant.
    # CHANGELOG (schema_version=1, economy_flat_v1.2.0): board.opening.best_ev
    # and board.opening.best_balance now carry real values from
    # board_economy.opening_rankings instead of defaulting to 0.0.  Vector
    # length and key order are unchanged; only feature values differ.
    resolved_opening = opening_economy
    if resolved_opening is None:
        resolved_opening = economy_state.get("opening_economy") if isinstance(economy_state.get("opening_economy"), dict) else None
    if resolved_opening is None and isinstance(board_economy.get("opening_rankings"), dict):
        resolved_opening = board_economy

    opening_rankings = (
        resolved_opening.get("opening_rankings", {})
        if isinstance(resolved_opening, dict) and isinstance(resolved_opening.get("opening_rankings"), dict)
        else {}
    )
    top_by_ev = opening_rankings.get("top_by_ev", []) if isinstance(opening_rankings.get("top_by_ev"), list) else []
    top_by_balance = opening_rankings.get("top_by_balance", []) if isinstance(opening_rankings.get("top_by_balance"), list) else []
    features["board.opening.best_ev"] = float(top_by_ev[0].get("ev_score", 0.0)) if top_by_ev else 0.0
    features["board.opening.best_balance"] = float(top_by_balance[0].get("evenness_score", 0.0)) if top_by_balance else 0.0

    player_profiles = (
        economy_state.get("player_economy", {}).get("by_player", [])
        if isinstance(economy_state.get("player_economy"), dict)
        else []
    )
    current_profile = next(
        (profile for profile in player_profiles if isinstance(profile, dict) and profile.get("player_id") == viewer_player_id),
        None,
    )
    if current_profile is None:
        return features

    features["self.expected_total"] = float(current_profile.get("expected_production_total", current_profile.get("total_expected_income", 0.0)))
    features["self.blocked_total"] = float(current_profile.get("blocked_production_total", current_profile.get("total_blocked_income", 0.0)))
    features["self.diversity_score"] = float(current_profile.get("diversity_score", 0.0))
    features["self.balance_score"] = float(current_profile.get("balance_score", current_profile.get("evenness_score", 0.0)))
    features["self.port_leverage_score"] = float(current_profile.get("port_leverage_score", 0.0))
    features["self.expansion_score"] = float((current_profile.get("expansion_potential") or {}).get("score", 0.0))
    features["self.robber_burden_score"] = float(current_profile.get("robber_burden_score", 0.0))
    features["self.total_rank"] = float((current_profile.get("relative_table_metrics") or {}).get("rank", 0.0))
    features["self.total_share"] = float((current_profile.get("relative_table_metrics") or {}).get("share", 0.0))
    features["self.total_leader_gap"] = float((current_profile.get("relative_table_metrics") or {}).get("leader_gap", 0.0))

    for resource in TRACKED_RESOURCES:
        expected_by_resource = current_profile.get("expected_production_by_resource", current_profile.get("expected_income", {}))
        rel_by_resource = (current_profile.get("relative_table_metrics") or {}).get("by_resource", {})
        features[f"self.{resource}.expected"] = float((expected_by_resource or {}).get(resource, 0.0))
        features[f"self.{resource}.rank"] = float(((rel_by_resource or {}).get(resource, {}) or {}).get("rank", 0.0))
        features[f"self.{resource}.share"] = float(((rel_by_resource or {}).get(resource, {}) or {}).get("share", 0.0))
        features[f"self.{resource}.leader_gap"] = float(((rel_by_resource or {}).get(resource, {}) or {}).get("leader_gap", 0.0))

    for plan_name, score in sorted((current_profile.get("orientation_scores") or {}).items()):
        features[f"self.plan.{plan_name}"] = float(score)

    # Expansion features
    expansion = current_profile.get("expansion", {}) if isinstance(current_profile.get("expansion"), dict) else {}
    features["self.expansion.frontier_score"] = float(expansion.get("frontier_score", 0.0))
    features["self.expansion.boxed_in_score"] = float(expansion.get("boxed_in_score", 0.0))
    features["self.expansion.city_upside_score"] = float(expansion.get("city_upside_score", 0.0))
    features["self.expansion.reachable_site_count"] = float(expansion.get("reachable_site_count", 0))
    features["self.expansion.top_site_score"] = float(expansion.get("top_reachable_site_score", 0.0))

    # Pressure features
    pressure = current_profile.get("pressure", {}) if isinstance(current_profile.get("pressure"), dict) else {}
    features["self.pressure.contested_frontier"] = float(pressure.get("contested_frontier_score", 0.0))
    features["self.pressure.race"] = float(pressure.get("race_pressure_score", 0.0))
    features["self.pressure.block_threat"] = float(pressure.get("block_threat_score", 0.0))
    features["self.pressure.resource_competition"] = float(pressure.get("resource_competition_score", 0.0))
    features["self.pressure.leader"] = float(pressure.get("leader_pressure_score", 0.0))

    # Trade features
    trade_profile = economy_state.get("trade_profile", {}) if isinstance(economy_state.get("trade_profile"), dict) else {}
    trade_by_player = trade_profile.get("by_player", {}) if isinstance(trade_profile.get("by_player"), dict) else {}
    self_trade = trade_by_player.get(viewer_player_id, {}) if isinstance(trade_by_player, dict) else {}
    for resource in TRACKED_RESOURCES:
        features[f"self.trade_need.{resource}"] = float((self_trade.get("need_scores") or {}).get(resource, 0.0))
        features[f"self.trade_surplus.{resource}"] = float((self_trade.get("surplus_scores") or {}).get(resource, 0.0))

    # Belief features — flatten all opponents in deterministic rank order.
    # FIX 4: Always produce MAX_OPPONENTS_SCHEMA_V1 (3) opponent slots for
    # schema_version=1 fixed-length vector.  Missing opponent slots are
    # padded with 0.0 placeholders rather than shifting indices.
    belief_economy = economy_state.get("belief_economy", {}) if isinstance(economy_state.get("belief_economy"), dict) else {}
    belief_by_player = belief_economy.get("by_player", {}) if isinstance(belief_economy.get("by_player"), dict) else {}
    # Build opponent list ordered by income rank (stable across runs)
    opponent_profiles: List[Dict[str, object]] = sorted(
        [p for p in player_profiles if isinstance(p, dict) and p.get("player_id") != viewer_player_id],
        key=lambda p: (int(p.get("total_income_rank") or 999), str(p.get("player_id") or "")),
    )
    for opponent_index in range(MAX_OPPONENTS_SCHEMA_V1):
        if opponent_index < len(opponent_profiles):
            opponent_id = str(opponent_profiles[opponent_index].get("player_id") or "")
            summary = belief_by_player.get(opponent_id, {})
            if not isinstance(summary, dict):
                summary = {}
        else:
            # Pad missing opponent slot with empty summary
            summary = {}
        for resource in TRACKED_RESOURCES:
            features[f"belief.opp{opponent_index}.{resource}.p_at_least_one"] = float((summary.get("p_at_least_one") or {}).get(resource, 0.0))

    return features
