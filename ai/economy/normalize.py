from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def score_from_cap(value: float, cap: float) -> float:
    if cap <= 0.0:
        return 0.0
    return clamp01(value / cap)


def evenness_score(values: Iterable[float]) -> float:
    cleaned = [float(v) for v in values if float(v) > 0.0]
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return 0.0
    total = sum(cleaned)
    if total <= 0.0:
        return 0.0
    shares = [value / total for value in cleaned]
    hhi = sum(share * share for share in shares)
    n = len(cleaned)
    min_hhi = 1.0 / float(n)
    if min_hhi >= 1.0:
        return 0.0
    return clamp01(1.0 - ((hhi - min_hhi) / (1.0 - min_hhi)))


def diversity_score(values: Dict[str, float]) -> float:
    non_zero = sum(1 for value in values.values() if value > 0.0)
    if non_zero <= 0:
        return 0.0
    max_support = min(len(values), 5)
    return clamp01(non_zero / float(max_support))


def rank_with_gaps(values_by_key: Dict[str, float]) -> Dict[str, Dict[str, float | int]]:
    ordered: List[Tuple[str, float]] = sorted(
        values_by_key.items(),
        key=lambda item: (-item[1], item[0]),
    )
    total = sum(max(0.0, value) for value in values_by_key.values())
    leader_value = ordered[0][1] if ordered else 0.0
    output: Dict[str, Dict[str, float | int]] = {}
    for rank, (key, value) in enumerate(ordered, start=1):
        output[key] = {
            "rank": rank,
            "share": safe_div(max(0.0, value), total),
            "leader_gap": leader_value - value,
            "max_opponent_delta": leader_value - value if rank != 1 else 0.0,
        }
    return output


def top_keys(scores: Dict[str, float], limit: int = 2, min_score: float = 0.0) -> List[str]:
    ordered = sorted(
        ((key, value) for key, value in scores.items() if value > min_score),
        key=lambda item: (-item[1], item[0]),
    )
    return [key for key, _ in ordered[:limit]]


def rounded(value: float, digits: int = 3) -> float:
    return round(float(value), digits)
