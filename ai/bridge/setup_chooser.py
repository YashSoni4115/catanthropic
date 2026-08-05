"""Setup-phase placement chooser for the interactive bridge.

Reuses the existing opening evaluator (:mod:`ai.economy.board.openings`)
to rank legal opening vertices, then picks the top-scoring vertex that
Java has not already occupied and whose neighbours are open (distance
rule).

**Distance-rule invariant (v1).**  The canonical Python-side distance
check is :func:`ai.economy.constants.adjacent_vertices_for_vertex`.
Both this module (via :func:`_occupied_or_blocked_slots`) AND the
main-phase enumerator :func:`ai.decision.legal_actions._enum_build_settlement`
use it.  Do NOT introduce a divergent implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ai.economy.board.openings import (
    _build_single_candidates, _occupied_or_blocked_slots,
)
from ai.economy.constants import (
    road_endpoint_vertices, vertex_incident_roads,
)

VertexKey = Tuple[int, int, int]
EdgeKey = Tuple[int, int, int]


def _observed(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("observed_state", state)


def choose_setup_settlement(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rank open opening vertices and return the top candidate.

    Returns a dict with ``x``, ``y``, ``orientation``, ``opening_score``,
    and ``rank`` — or ``None`` if there are no legal candidates.
    """
    obs = _observed(state)
    encoding = state.get("encoding_metadata", {})
    # _build_single_candidates already filters by _occupied_or_blocked_slots
    # (distance rule).  We ONLY need to pass the current structures.
    candidates = _build_single_candidates(
        tiles=obs.get("tiles", []),
        board_economy=None,
        structures=obs.get("structures", []),
        slot_ordering=encoding.get("slot_ordering"),
    )
    if not candidates:
        return None
    # candidates come pre-ranked by opening_score desc from the openings
    # module; take the first.
    top = candidates[0]
    return {
        "x": int(top["x"]),
        "y": int(top["y"]),
        "orientation": int(top["orientation"]),
        "opening_score": float(top.get("opening_score", 0.0)),
        "rank": 1,
    }


def choose_setup_road(
    state: Dict[str, Any],
    settlement_vertex: VertexKey,
) -> Optional[Dict[str, Any]]:
    """Pick a road incident to ``settlement_vertex`` that expands toward
    the highest-EV *neighbouring* legal open vertex.

    Heuristic (documented, simple, v1):
        For each of the (up to) 3 roads touching this settlement:
          * find the other endpoint vertex
          * score that endpoint via the opening evaluator (opening_score
            of the endpoint if it appears in the candidate list, else 0)
        Pick the road whose *other endpoint* has the highest opening_score.
        Ties broken by (x,y,orientation) for determinism.
    """
    obs = _observed(state)
    encoding = state.get("encoding_metadata", {})
    sx, sy, so = settlement_vertex

    # Build a lookup: vertex -> opening_score (pre distance rule filter).
    # We temporarily REMOVE our just-placed settlement so neighbours of
    # our own settlement are not incorrectly blocked -- but that would
    # only matter for 2-hop scoring; for our own endpoint we just want
    # its raw opening_score which _build_single_candidates already
    # excludes.  So we score endpoints directly against ALL candidates
    # computed BEFORE our placement (i.e. drop the just-placed structure).
    structures_without_new = [
        s for s in (obs.get("structures") or [])
        if (int(s.get("x", -1)), int(s.get("y", -1)), int(s.get("orientation", -1)))
           != (sx, sy, so)
    ]
    candidates = _build_single_candidates(
        tiles=obs.get("tiles", []),
        board_economy=None,
        structures=structures_without_new,
        slot_ordering=encoding.get("slot_ordering"),
    )
    score_by_vertex: Dict[VertexKey, float] = {
        (int(c["x"]), int(c["y"]), int(c["orientation"])): float(c.get("opening_score", 0.0))
        for c in candidates
    }

    # Occupied roads (cannot re-place)
    occupied_roads: set = set()
    for r in obs.get("roads") or []:
        try:
            occupied_roads.add((int(r["x"]), int(r["y"]), int(r["orientation"])))
        except (KeyError, TypeError, ValueError):
            continue

    best: Optional[Tuple[EdgeKey, VertexKey, float]] = None
    for e in vertex_incident_roads(sx, sy, so):
        ekey = (int(e["x"]), int(e["y"]), int(e["orientation"]))
        if ekey in occupied_roads:
            continue
        # Other endpoint
        endpoints = road_endpoint_vertices(*ekey)
        other = None
        for v in endpoints:
            vkey = (int(v["x"]), int(v["y"]), int(v["orientation"]))
            if vkey != (sx, sy, so):
                other = vkey
                break
        if other is None:
            continue
        score = score_by_vertex.get(other, 0.0)
        if (best is None
                or score > best[2]
                or (score == best[2] and ekey < best[0])):
            best = (ekey, other, score)

    if best is None:
        return None
    ekey, other, score = best
    return {
        "x": ekey[0], "y": ekey[1], "orientation": ekey[2],
        "toward_vertex": {"x": other[0], "y": other[1], "orientation": other[2]},
        "toward_opening_score": score,
    }
