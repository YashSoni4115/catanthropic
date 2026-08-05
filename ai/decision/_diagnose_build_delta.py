"""Diagnostic: does the Python approximator actually reflect a build's
direct effects in the observed_state and vectorized representation?

Two hypotheses:
  A) Model calibration -- the value model genuinely undervalues builds.
  B) Approximator bug  -- apply_action(...) doesn't propagate build
     effects (VP, structure change, production) into the vectorized
     features, so the model can't see the change.

This script tests hypothesis B first by field-inspecting BEFORE vs AFTER
states and by counting changed dimensions in the 2904-dim vector.

Run:
    python -m ai.decision._diagnose_build_delta
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai.decision.actions import Action, ActionType
from ai.decision.opening_value_evaluator_v1 import ValueModelOracle
from ai.decision.simulator import apply_action
from ai.tests.economy_test_utils import make_state_payload
from ai.vectorizer_v1 import StateVectorizerV1

MODEL_DIR = "ai/models/value_model_economy_v1"


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def _mid_game_state() -> Dict[str, Any]:
    """Alice: 2 settlements + roads on high-pip tiles, ample resources.
    Enough to test build_city (upgrade), build_settlement, build_road,
    buy_dev_card."""
    state = make_state_payload()
    obs = state["observed_state"]
    alice = obs["players"]["all_players"][0]
    alice["resource_cards"] = {"BRICK": 5, "WOOL": 5, "ORE": 5, "GRAIN": 5, "LUMBER": 5}
    alice["total_resource_cards"] = 25
    alice["victory_points_visible"] = 4
    alice["victory_points"] = 4
    alice["remaining_pieces"] = {
        "roads_remaining": 12, "settlements_remaining": 3, "cities_remaining": 4,
        "roads_placed_reported": 3, "settlements_placed_reported": 2,
        "cities_placed_reported": 0,
    }
    # Ensure Alice has a settlement to upgrade (fixture already has one at (1,1,0))
    for s in obs["structures"]:
        if s.get("owner_name") == "Alice":
            s["type"] = "settlement"
    # Give Alice at least one road so build_road/build_settlement enum works
    obs["roads"] = [
        {"x": 1, "y": 1, "orientation": 0, "owner_id": "name:Alice", "owner_name": "Alice"},
        {"x": 1, "y": 1, "orientation": 1, "owner_id": "name:Alice", "owner_name": "Alice"},
    ]
    return state


# ---------------------------------------------------------------------------
# Field inspection helpers
# ---------------------------------------------------------------------------

def _viewer(state):
    obs = state.get("observed_state", state)
    tm = obs.get("turn_metadata") or {}
    for p in (obs.get("players") or {}).get("all_players", []):
        if p.get("player_id") == tm.get("current_player_id") \
                or p.get("player_name") == tm.get("current_player_name"):
            return p
    return None


def _structure_at(state, x, y, o) -> Optional[Dict[str, Any]]:
    obs = state.get("observed_state", state)
    for s in obs.get("structures") or []:
        if (s.get("x"), s.get("y"), s.get("orientation")) == (x, y, o):
            return s
    return None


def _road_at(state, x, y, o) -> Optional[Dict[str, Any]]:
    obs = state.get("observed_state", state)
    for r in obs.get("roads") or []:
        if (r.get("x"), r.get("y"), r.get("orientation")) == (x, y, o):
            return r
    return None


# ---------------------------------------------------------------------------
# Diff report
# ---------------------------------------------------------------------------

def _vec(vectorizer, oracle, state) -> Tuple[np.ndarray, float, List[str]]:
    v = vectorizer.vectorize(state)
    x = np.asarray(v.x, dtype=np.float32)
    p = float(oracle.score_batch(x.reshape(1, -1))[0])
    return x, p, v.feature_names


def _diff_summary(before, after, names, top=8):
    idx = np.nonzero(np.abs(after - before) > 1e-9)[0]
    changed = [(i, names[i] if i < len(names) else f"#{i}",
                float(before[i]), float(after[i]))
               for i in idx]
    changed.sort(key=lambda t: -abs(t[3] - t[2]))
    return len(idx), changed[:top]


def _report(label, before_state, after_state, action, oracle, vectorizer):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    me_b, me_a = _viewer(before_state), _viewer(after_state)

    # Field-level inspection
    print(f"[FIELDS] action = {action.short()}")
    print(f"  vp_visible:      {me_b['victory_points_visible']:>4} -> {me_a['victory_points_visible']:>4}")
    print(f"  resources:")
    for r in ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"):
        b = me_b['resource_cards'].get(r, 0)
        a = me_a['resource_cards'].get(r, 0)
        mark = "  ← changed" if a != b else ""
        print(f"    {r:<7} {b:>3} -> {a:>3}{mark}")
    rp_b = me_b.get("remaining_pieces") or {}
    rp_a = me_a.get("remaining_pieces") or {}
    for k in ("roads_remaining", "settlements_remaining", "cities_remaining"):
        b, a = rp_b.get(k), rp_a.get(k)
        mark = "  ← changed" if b != a else ""
        print(f"  {k:<24} {b} -> {a}{mark}")
    print(f"  #structures:  {len(before_state['observed_state']['structures']):>3} -> {len(after_state['observed_state']['structures']):>3}")
    print(f"  #roads:       {len(before_state['observed_state']['roads']):>3} -> {len(after_state['observed_state']['roads']):>3}")

    # Type-change check for build_city
    if action.type == ActionType.BUILD_CITY and action.vertex is not None:
        s_b = _structure_at(before_state, *action.vertex)
        s_a = _structure_at(after_state, *action.vertex)
        print(f"  target vertex type: {s_b.get('type') if s_b else None} -> {s_a.get('type') if s_a else None}")

    # Vector diff
    before_vec, p_before, names = _vec(vectorizer, oracle, before_state)
    after_vec, p_after, _ = _vec(vectorizer, oracle, after_state)
    n_changed, top = _diff_summary(before_vec, after_vec, names)

    print(f"[MODEL]")
    print(f"  p(win) before: {p_before:.5f}")
    print(f"  p(win) after:  {p_after:.5f}")
    print(f"  delta:         {p_after - p_before:+.5f}")
    print(f"[VECTOR]")
    print(f"  changed dims:  {n_changed} / {len(before_vec)}")
    for i, name, b, a in top:
        print(f"    [{i:>4}] {name:<48} {b:>8.3f} -> {a:>8.3f}   Δ={a-b:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading value model ...")
    oracle = ValueModelOracle(MODEL_DIR, model_name="hgb")
    vectorizer = StateVectorizerV1(adjacency_features_enabled=True,
                                   economy_features_enabled=oracle.economy_features_enabled)

    base = _mid_game_state()

    cases: List[Tuple[str, Action]] = [
        ("BUILD_CITY (upgrade Alice's settlement at (1,1,0))",
         Action(type=ActionType.BUILD_CITY, vertex=(1, 1, 0))),
        ("BUILD_SETTLEMENT (synthetic vertex (2,5,1))",
         Action(type=ActionType.BUILD_SETTLEMENT, vertex=(2, 5, 1))),
        ("BUILD_ROAD (synthetic edge (1,1,2))",
         Action(type=ActionType.BUILD_ROAD, edge=(1, 1, 2))),
        ("BUY_DEV_CARD",
         Action(type=ActionType.BUY_DEV_CARD)),
        ("END_TURN (baseline; must be ~0 delta)",
         Action(type=ActionType.END_TURN)),
    ]

    for label, action in cases:
        try:
            nxt = apply_action(base, action)
        except Exception as exc:
            print(f"\n{label}\n  ERROR applying action: {exc}")
            continue
        _report(label, base, nxt, action, oracle, vectorizer)

    print("\n\n" + "="*78 + "\nSUMMARY -- see per-case deltas above.\n" + "="*78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
