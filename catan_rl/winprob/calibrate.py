"""Post-hoc temperature scaling: p_T ∝ p^(1/T), T fit on validation soft cross-entropy only."""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(P, EPS, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _ce(P, Q):
    return float(-(Q * np.log(np.clip(P, EPS, 1))).sum(1).mean())


def fit_temperature(P_val: np.ndarray, Q_val: np.ndarray, lo: float = 0.25, hi: float = 4.0) -> float:
    """Golden-section search on log T (CE is unimodal in T for fixed logits)."""
    f = lambda x: _ce(apply_temperature(P_val, np.exp(x)), Q_val)
    a, b, g = np.log(lo), np.log(hi), (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return float(np.exp((a + b) / 2))
