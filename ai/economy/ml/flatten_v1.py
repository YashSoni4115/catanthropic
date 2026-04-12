from __future__ import annotations

from typing import Dict

from ai.economy.constants import TRACKED_RESOURCES


VERSION = "economy_flat_v1.0.0"


def flatten_economy_features_v1(
    economy_state: Dict[str, object],
    viewer_player_id: str,
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

    opening_rankings = board_economy.get("opening_rankings", {}) if isinstance(board_economy.get("opening_rankings"), dict) else {}
    top_by_ev = opening_rankings.get("top_by_ev", []) if isinstance(opening_rankings.get("top_by_ev"), list) else []
    top_by_balance = opening_rankings.get("top_by_balance", []) if isinstance(opening_rankings.get("top_by_balance"), list) else []
    features["board.opening.best_ev"] = float(top_by_ev[0].get("ev_score", 0.0)) if top_by_ev else 0.0
    features["board.opening.best_balance"] = float(top_by_balance[0].get("balance_score", 0.0)) if top_by_balance else 0.0

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

    features["self.expected_total"] = float(current_profile.get("expected_production_total", 0.0))
    features["self.blocked_total"] = float(current_profile.get("blocked_production_total", 0.0))
    features["self.diversity_score"] = float(current_profile.get("diversity_score", 0.0))
    features["self.balance_score"] = float(current_profile.get("balance_score", 0.0))
    features["self.port_leverage_score"] = float(current_profile.get("port_leverage_score", 0.0))
    features["self.expansion_score"] = float((current_profile.get("expansion_potential") or {}).get("score", 0.0))
    features["self.robber_burden_score"] = float(current_profile.get("robber_burden_score", 0.0))
    features["self.total_rank"] = float((current_profile.get("relative_table_metrics") or {}).get("rank", 0.0))
    features["self.total_share"] = float((current_profile.get("relative_table_metrics") or {}).get("share", 0.0))
    features["self.total_leader_gap"] = float((current_profile.get("relative_table_metrics") or {}).get("leader_gap", 0.0))

    for resource in TRACKED_RESOURCES:
        expected_by_resource = current_profile.get("expected_production_by_resource", {})
        rel_by_resource = (current_profile.get("relative_table_metrics") or {}).get("by_resource", {})
        features[f"self.{resource}.expected"] = float((expected_by_resource or {}).get(resource, 0.0))
        features[f"self.{resource}.rank"] = float(((rel_by_resource or {}).get(resource, {}) or {}).get("rank", 0.0))
        features[f"self.{resource}.share"] = float(((rel_by_resource or {}).get(resource, {}) or {}).get("share", 0.0))
        features[f"self.{resource}.leader_gap"] = float(((rel_by_resource or {}).get(resource, {}) or {}).get("leader_gap", 0.0))

    for plan_name, score in sorted((current_profile.get("orientation_scores") or {}).items()):
        features[f"self.plan.{plan_name}"] = float(score)

    trade_profile = economy_state.get("trade_profile", {}) if isinstance(economy_state.get("trade_profile"), dict) else {}
    trade_by_player = trade_profile.get("by_player", {}) if isinstance(trade_profile.get("by_player"), dict) else {}
    self_trade = trade_by_player.get(viewer_player_id, {}) if isinstance(trade_by_player, dict) else {}
    for resource in TRACKED_RESOURCES:
        features[f"self.trade_need.{resource}"] = float((self_trade.get("need_scores") or {}).get(resource, 0.0))
        features[f"self.trade_surplus.{resource}"] = float((self_trade.get("surplus_scores") or {}).get(resource, 0.0))

    belief_economy = economy_state.get("belief_economy", {}) if isinstance(economy_state.get("belief_economy"), dict) else {}
    belief_by_player = belief_economy.get("by_player", {}) if isinstance(belief_economy.get("by_player"), dict) else {}
    for opponent_id, summary in sorted(belief_by_player.items()):
        if opponent_id == viewer_player_id or not isinstance(summary, dict):
            continue
        for resource in TRACKED_RESOURCES:
            features[f"belief.{opponent_id}.{resource}.p_at_least_one"] = float((summary.get("p_at_least_one") or {}).get(resource, 0.0))
        break

    return features
