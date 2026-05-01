from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ai.economy.constants import (
    ABUNDANT_TAG_THRESHOLD,
    BALANCED_BOARD_THRESHOLD,
    CONCENTRATED_TAG_THRESHOLD,
    POLARIZED_BOARD_THRESHOLD,
    ROBBER_FRAGILE_CONCENTRATION_THRESHOLD,
    SCARCE_TAG_THRESHOLD,
)


def _resource_tag_prefix(resource: str) -> str:
    return resource.lower()


def resource_tags_from_stats(resource: str, stats: Dict[str, object]) -> List[str]:
    tags: List[str] = []
    prefix = _resource_tag_prefix(resource)
    scarcity = float(stats.get("scarcity_score", 0.0))
    abundance = float(stats.get("abundance_score", 0.0))
    concentration = float(stats.get("concentration_score", 0.0))
    if scarcity >= SCARCE_TAG_THRESHOLD:
        tags.append(f"{prefix}_scarce")
    if abundance >= ABUNDANT_TAG_THRESHOLD:
        tags.append(f"{prefix}_abundant")
    if concentration >= CONCENTRATED_TAG_THRESHOLD:
        tags.append(f"{prefix}_concentrated")
    return tags


def resource_environment_tags(resource_environment: Dict[str, object]) -> List[str]:
    """Derive board tags only from stored numeric resource-environment fields."""
    tags: List[str] = []
    by_resource = resource_environment.get("by_resource", resource_environment)
    if not isinstance(by_resource, dict):
        return tags
    if not by_resource:
        return tags
    strongest = max(
        by_resource.items(),
        key=lambda item: (float(item[1].get("pip_total", 0.0)) if isinstance(item[1], dict) else 0.0, item[0]),
    )[0]
    weakest = min(
        by_resource.items(),
        key=lambda item: (float(item[1].get("pip_total", 0.0)) if isinstance(item[1], dict) else 0.0, item[0]),
    )[0]
    tags.extend([f"{_resource_tag_prefix(strongest)}_environment_strongest", f"{_resource_tag_prefix(weakest)}_environment_weakest"])
    for resource, stats in sorted(by_resource.items()):
        if isinstance(stats, dict):
            tags.extend(resource_tags_from_stats(resource, stats))
    balance_score = float(resource_environment.get("board_balance_score", 0.0))
    polarization_score = float(resource_environment.get("board_polarization_score", 0.0))
    if balance_score >= BALANCED_BOARD_THRESHOLD:
        tags.append("balanced_board")
    if polarization_score >= POLARIZED_BOARD_THRESHOLD:
        tags.append("polarized_board")
    return sorted(set(tags))


def desert_robber_tags(desert_robber: Dict[str, object]) -> List[str]:
    """Derive robber-fragility tags from stored desert/robber metrics."""
    tags: List[str] = []
    fragility_by_resource = desert_robber.get("robber_fragility_by_resource", {})
    if isinstance(fragility_by_resource, dict):
        for resource, score in sorted(fragility_by_resource.items()):
            if float(score) >= ROBBER_FRAGILE_CONCENTRATION_THRESHOLD:
                tags.append(f"robber_fragile_{resource.lower()}")
        return sorted(set(tags))

    for resource in desert_robber.get("robber_fragile_resources", []):
        if isinstance(resource, str):
            tags.append(f"robber_fragile_{resource.lower()}")
    return sorted(set(tags))


def board_economy_tags(board_economy: Dict[str, object]) -> List[str]:
    """Combine symbolic board tags from numeric board economy outputs."""
    tags: List[str] = []
    resource_environment = board_economy.get("resource_environment")
    desert_robber = board_economy.get("desert_robber")
    if isinstance(resource_environment, dict):
        tags.extend(resource_environment_tags(resource_environment))
    if isinstance(desert_robber, dict):
        tags.extend(desert_robber_tags(desert_robber))
    return sorted(set(tags))


def opening_candidate_tags(
    total_pips: float,
    balance_score: float,
    port: Optional[str],
    port_alignment_score: float,
    archetype_scores: Dict[str, float],
) -> List[str]:
    tags: List[str] = []
    if total_pips >= 10.0:
        tags.append("high_ev")
    if balance_score >= 0.7:
        tags.append("economically_balanced")
    if port and port_alignment_score >= 0.6:
        tags.append("port_aligned")
        tags.append(f"port_{port}")
    if archetype_scores:
        favored = max(archetype_scores.items(), key=lambda item: (item[1], item[0]))[0]
        if archetype_scores.get(favored, 0.0) >= 0.55:
            tags.append(f"{favored}_favored")
    return sorted(set(tags))


def player_economy_tags(
    favored_plan: str,
    expected_by_resource: Dict[str, float],
    bottlenecks: Iterable[Dict[str, float | str]],
    port_leverage_score: float,
    robber_burden_score: float,
) -> List[str]:
    tags: List[str] = [f"{favored_plan}_favored"]
    if expected_by_resource:
        strongest = max(expected_by_resource.items(), key=lambda item: (item[1], item[0]))[0]
        weakest = min(expected_by_resource.items(), key=lambda item: (item[1], item[0]))[0]
        if expected_by_resource[strongest] > 0.0:
            tags.append(f"{_resource_tag_prefix(strongest)}_strong")
        tags.append(f"{_resource_tag_prefix(weakest)}_weak")
    for item in bottlenecks:
        resource = item.get("resource")
        severity = float(item.get("severity", 0.0))
        if isinstance(resource, str) and severity >= 0.55:
            tags.append(f"{_resource_tag_prefix(resource)}_bottleneck")
    if port_leverage_score >= 0.6:
        tags.append("port_aligned")
    if robber_burden_score >= 0.35:
        tags.append("robber_burden_high")
    return sorted(set(tags))


def belief_tags(p_at_least_one: Dict[str, float], p_can_afford: Dict[str, float]) -> List[str]:
    tags: List[str] = []
    for resource, probability in sorted(p_at_least_one.items()):
        if probability >= 0.7:
            tags.append(f"likely_has_{resource.lower()}")
    for spend_name, probability in sorted(p_can_afford.items()):
        if probability >= 0.7:
            tags.append(f"likely_can_afford_{spend_name}")
    return tags


def trade_tags(need_top2: List[str], offer_top2: List[str], strongest_resource: Optional[str], weakest_resource: Optional[str]) -> List[str]:
    tags: List[str] = []
    for resource in need_top2:
        tags.append(f"needs_{resource.lower()}")
    for resource in offer_top2:
        tags.append(f"offers_{resource.lower()}")
    if strongest_resource:
        tags.append(f"{strongest_resource.lower()}_market_strength")
    if weakest_resource:
        tags.append(f"{weakest_resource.lower()}_market_weakness")
    return sorted(set(tags))
