"""Statistics: Wilson (reference only), seed-clustered bootstrap, bootstrap of per-seed values."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _pct(boot: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
    return float(lo), float(hi)


def bootstrap_mean_ci(xs: Sequence[float], n_boot: int = 10000, seed: int = 0,
                      alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``xs`` (one value per seed = one cluster)."""
    x = np.asarray(xs, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    return (float(x.mean()), *_pct(x[idx].mean(axis=1), alpha))


def cluster_bootstrap(wins: Sequence[int], games: Sequence[int], n_boot: int = 10000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Win rate sum(wins)/sum(games), CI from resampling whole clusters (seeds).

    design_effect = clustered bootstrap variance / binomial variance p(1-p)/N (None if p in {0,1});
    effective_n = N / design_effect (None if design_effect is 0 or None).
    """
    k, g = np.asarray(wins, dtype=float), np.asarray(games, dtype=float)
    N, p = g.sum(), k.sum() / g.sum()
    idx = np.random.default_rng(seed).integers(0, len(k), size=(n_boot, len(k)))
    boot = k[idx].sum(axis=1) / g[idx].sum(axis=1)
    var, var_bin = float(boot.var(ddof=1)), float(p * (1 - p) / N)
    deff = var / var_bin if var_bin > 0 else None
    return {"p": float(p), "ci95": list(_pct(boot, alpha)), "n_seeds": len(k), "n_games": int(N),
            "var_clustered": var, "var_binomial": var_bin, "design_effect": deff,
            "effective_n": N / deff if deff else None}
