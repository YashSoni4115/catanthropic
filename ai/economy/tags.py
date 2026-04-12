from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def _resource_tag_prefix(resource: str) -> str:
    return resource.lower()


def resource_environment_tags(by_resource: Dict[str, Dict[str, float]]) -> List[str]:
    tags: List[str] = []
    if not by_resource:
        return tags
    strongest = max(by_resource.items(), key=lambda item: (item[1].get("pip_total", 0.0), item[0]))[0]
    weakest = min(by_resource.items(), key=lambda item: (item[1].get("pip_total", 0.0), item[0]))[0]
    tags.extend([f"{_resource_tag_prefix(strongest)}_environment_strongest", f"{_resource_tag_prefix(weakest)}_environment_weakest"])
    for resource, stats in sorted(by_resource.items()):
        scarcity = float(stats.get("scarcity_score", 0.0))
        abundance = float(stats.get("abundance_score", 0.0))
        concentration = float(stats.get("concentration_score", 0.0))
        if scarcity >= 0.6:
            tags.append(f"{_resource_tag_prefix(resource)}_scarce")
        if abundance >= 0.6:
            tags.append(f"{_resource_tag_prefix(resource)}_abundant")
        if concentration >= 0.65:
            tags.append(f"{_resource_tag_prefix(resource)}_concentrated")
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
