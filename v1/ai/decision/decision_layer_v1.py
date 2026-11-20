"""Decision layer v1 — public API.

Composes:
  * :mod:`legal_actions`   — enumerator
  * :mod:`simulator`       — Python approximator (no engine dry-run)
  * :mod:`scorer`          — value-model probability + delta
  * :mod:`heuristics`      — named override rules

Entry points:
    choose_action(state, oracle=..., ...) -> Action
    rank_actions(state, oracle=..., ...) -> list[ScoredAction]

Both use ``observed_state`` only.  Callers may pass a pre-loaded
:class:`ValueModelOracle` (recommended for repeated queries).  If none is
given, one is loaded from ``ai/models/value_model_economy_v1`` — this
requires the model artifact to exist on disk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.vectorizer_v1 import StateVectorizerV1

from .actions import Action
from .heuristics import HeuristicConfig, apply_heuristics
from .legal_actions import legal_actions
from .opening_value_evaluator_v1 import ValueModelOracle
from .scorer import ActionScorer, ScoredAction

DEFAULT_MODEL_DIR = "ai/models/value_model_economy_v1"


def _resolve_scorer(
    oracle: Optional[ValueModelOracle],
    scorer: Optional[ActionScorer],
    model_dir: str,
    model_name: str,
) -> ActionScorer:
    if scorer is not None:
        return scorer
    if oracle is None:
        oracle = ValueModelOracle(model_dir, model_name=model_name)
    return ActionScorer(oracle=oracle)


def rank_actions(
    state: Dict[str, Any],
    *,
    oracle: Optional[ValueModelOracle] = None,
    scorer: Optional[ActionScorer] = None,
    heuristic_config: Optional[HeuristicConfig] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_name: str = "hgb",
) -> List[ScoredAction]:
    """Return legal actions ranked by final score (descending).

    Every :class:`ScoredAction` exposes ``win_prob_before / after``,
    ``model_delta``, ``heuristic_adjustments`` and ``final_score`` for
    full inspectability.  Empty list if there are no legal actions.
    """
    actions = legal_actions(state)
    if not actions:
        return []
    sc = _resolve_scorer(oracle, scorer, model_dir, model_name)
    scored = sc.score_actions(state, actions)
    apply_heuristics(state, scored, heuristic_config)
    # Sort by final_score desc; tie-break by action type for determinism
    scored.sort(
        key=lambda s: (-(s.final_score if s.final_score != float("inf") else 1e18),
                       s.action.type.value, s.action.short()),
    )
    return scored


def choose_action(
    state: Dict[str, Any],
    *,
    oracle: Optional[ValueModelOracle] = None,
    scorer: Optional[ActionScorer] = None,
    heuristic_config: Optional[HeuristicConfig] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_name: str = "hgb",
) -> Optional[Action]:
    """Return the top-ranked legal action, or ``None`` if none exist."""
    ranked = rank_actions(
        state,
        oracle=oracle, scorer=scorer,
        heuristic_config=heuristic_config,
        model_dir=model_dir, model_name=model_name,
    )
    return ranked[0].action if ranked else None


__all__ = [
    "Action", "ScoredAction", "HeuristicConfig", "ValueModelOracle",
    "ActionScorer", "StateVectorizerV1",
    "legal_actions", "rank_actions", "choose_action",
]
