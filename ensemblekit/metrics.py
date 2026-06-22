"""Ranking metric: AUROC via the Mann-Whitney U statistic, with tie handling.

AUROC is the probability that a uniformly drawn positive scores above a uniformly drawn
negative. Computed from average ranks (so ties contribute 0.5 apiece), it equals
``(sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)``. The combiner scores are
log-odds (any monotone scale), so a threshold-free ranking metric is the natural choice --
calibration of the combined score does not affect it, only the ordering does.
"""

from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve of ``scores`` against binary labels ``y``.

    Returns 0.5 for a degenerate single-class label vector.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to their average (stable for reproducibility)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sv = values[order]
    i = 0
    n = len(sv)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, clipped to avoid overflow warnings."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
