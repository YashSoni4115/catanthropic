"""Value-model based scorer for the decision layer.

Given a state and a list of legal actions, produces a ranked list of
:class:`ScoredAction` with win-probability before/after and delta.

Uses :class:`ai.decision.opening_value_evaluator_v1.ValueModelOracle`
under the hood so we don't duplicate model-loading logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ai.vectorizer_v1 import StateVectorizerV1

from .actions import Action
from .opening_value_evaluator_v1 import ValueModelOracle
from .simulator import apply_action


@dataclass
class ScoredAction:
    action: Action
    win_prob_before: float
    win_prob_after: float
    model_delta: float
    heuristic_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    final_score: float = 0.0
    error: Optional[str] = None

    def total_adjustment(self) -> float:
        return sum(float(h.get("delta", 0.0)) for h in self.heuristic_adjustments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.short(),
            "type": self.action.type.value,
            "win_prob_before": round(self.win_prob_before, 6),
            "win_prob_after": round(self.win_prob_after, 6),
            "model_delta": round(self.model_delta, 6),
            "heuristics": self.heuristic_adjustments,
            "final_score": round(self.final_score, 6),
            "error": self.error,
        }


class ActionScorer:
    """Vectorize and score actions with the value model."""

    def __init__(
        self,
        oracle: ValueModelOracle,
        vectorizer: Optional[StateVectorizerV1] = None,
    ):
        self.oracle = oracle
        self.vectorizer = vectorizer or StateVectorizerV1(
            adjacency_features_enabled=True,
            economy_features_enabled=oracle.economy_features_enabled,
        )

    # -- Public API ---------------------------------------------------------

    def score_state(self, state: Dict[str, Any]) -> float:
        vec = self.vectorizer.vectorize(state)
        x = np.asarray(vec.x, dtype=np.float32).reshape(1, -1)
        return float(self.oracle.score_batch(x)[0])

    def score_actions(
        self,
        state: Dict[str, Any],
        actions: List[Action],
    ) -> List[ScoredAction]:
        """Return ScoredAction per action, in the input order."""
        if not actions:
            return []
        p_before = self.score_state(state)

        # Vectorize each resulting state; collect errors for degraded rows.
        results: List[ScoredAction] = []
        vectors: List[np.ndarray] = []
        for a in actions:
            try:
                next_state = apply_action(state, a)
                vec = self.vectorizer.vectorize(next_state)
                x = np.asarray(vec.x, dtype=np.float32)
                if (self.oracle.expected_vector_length > 0
                        and x.shape[0] != self.oracle.expected_vector_length):
                    raise ValueError(
                        f"vector length {x.shape[0]} != expected "
                        f"{self.oracle.expected_vector_length}"
                    )
                vectors.append(x)
                results.append(ScoredAction(
                    action=a,
                    win_prob_before=p_before,
                    win_prob_after=float("nan"),
                    model_delta=float("nan"),
                ))
            except Exception as exc:  # noqa: BLE001
                results.append(ScoredAction(
                    action=a,
                    win_prob_before=p_before,
                    win_prob_after=p_before,
                    model_delta=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                ))

        # Batch-score all valid vectors in a single call
        good_idx = [i for i, r in enumerate(results) if r.error is None]
        if good_idx:
            X = np.vstack([vectors[j] for j in range(len(vectors))])
            probs = self.oracle.score_batch(X)
            for k, i in enumerate(good_idx):
                p_after = float(probs[k])
                results[i].win_prob_after = p_after
                results[i].model_delta = p_after - p_before
                results[i].final_score = results[i].model_delta

        return results
