from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ai.economy.constants import (
    MAX_SINGLE_VERTEX_PIPS,
    PORT_NAMES,
    PORT_RESOURCE_MAP,
    PORT_VERTEX_MAP,
)
from ai.economy.normalize import clamp01, diversity_score, evenness_score, score_from_cap


def port_for_vertex(x_coord: int, y_coord: int, orientation: int) -> Optional[str]:
    target = (x_coord, y_coord, orientation)
    for port_name, vertices in PORT_VERTEX_MAP.items():
        if target in vertices:
            return port_name
    return None


def port_alignment_score(port_name: Optional[str], pip_by_resource: Dict[str, float], total_pips: float) -> float:
    if port_name is None:
        return 0.0
    if port_name == "three_to_one":
        return clamp01(
            0.6 * score_from_cap(total_pips, 12.0)
            + 0.2 * diversity_score(pip_by_resource)
            + 0.2 * (1.0 - evenness_score(pip_by_resource.values()))
        )

    target_resource = PORT_RESOURCE_MAP.get(port_name)
    if target_resource is None:
        return 0.0
    target_pips = float(pip_by_resource.get(target_resource, 0.0))
    target_share = 0.0 if total_pips <= 0.0 else target_pips / total_pips
    return clamp01(0.7 * score_from_cap(target_pips, 5.0) + 0.3 * target_share)


def port_fallback_score(port_name: Optional[str], pip_by_resource: Dict[str, float], total_pips: float) -> float:
    if port_name is None:
        return 0.0
    if port_name == "three_to_one":
        return clamp01(0.7 * score_from_cap(total_pips, MAX_SINGLE_VERTEX_PIPS) + 0.3 * diversity_score(pip_by_resource))
    target_resource = PORT_RESOURCE_MAP.get(port_name)
    if target_resource in {"ORE", "GRAIN", "WOOL"}:
        return clamp01(0.65 * port_alignment_score(port_name, pip_by_resource, total_pips) + 0.35)
    return 0.25 * port_alignment_score(port_name, pip_by_resource, total_pips)


def summarize_port_opportunity(opening_candidates: Iterable[Dict[str, object]], limit: int = 10) -> Dict[str, object]:
    best_by_port: Dict[str, Dict[str, object]] = {}
    adjusted_candidates: List[Dict[str, object]] = []

    for candidate in opening_candidates:
        port = candidate.get("port")
        if not isinstance(port, str):
            continue
        score = float(candidate.get("port_alignment_score", 0.0))
        port_adjusted_score = clamp01(
            0.7 * float(candidate.get("ev_score_norm", 0.0))
            + 0.3 * score
        )
        existing = best_by_port.get(port)
        current = {
            "vertex_id": candidate.get("vertex_id"),
            "port_alignment_score": score,
            "port_adjusted_score": port_adjusted_score,
            "ev_score": float(candidate.get("ev_score", 0.0)),
            "tags": list(candidate.get("tags", [])) if isinstance(candidate.get("tags"), list) else [],
        }
        if existing is None or float(existing.get("port_adjusted_score", 0.0)) < port_adjusted_score:
            best_by_port[port] = current
        adjusted_candidates.append(
            {
                "vertex_id": candidate.get("vertex_id"),
                "port": port,
                "port_alignment_score": score,
                "port_adjusted_score": port_adjusted_score,
            }
        )

    adjusted_candidates.sort(
        key=lambda item: (
            -float(item.get("port_adjusted_score", 0.0)),
            str(item.get("vertex_id") or ""),
        )
    )
    return {
        "best_by_port": {port_name: best_by_port.get(port_name) for port_name in PORT_NAMES},
        "best_port_adjusted_candidates": adjusted_candidates[:limit],
        "available_ports": sorted(best_by_port.keys()),
    }
