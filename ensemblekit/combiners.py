"""The ensemble combiners: a 2x2 over {weighting} x {aggregation}, plus a single baseline.

Every combiner takes the base-learner log-odds, splits them into a *holdout* half (used to
estimate per-learner competence and to pick the best single learner) and a *test* half
(scored). The two axes are:

* **weighting** -- ``uniform`` gives every learner equal say; ``competence`` weights each
  learner by how well it ranks the holdout labels (``max(AUROC - 0.5, 0)``). Weighting is
  what rescues the *het_competence* regime, where dead learners must be down-weighted.
* **aggregation** -- ``mean`` averages the (weighted) log-odds; ``median`` takes the
  (weighted) median per sample. The median is what rescues the *corrupted* regime, where a
  fixed weight cannot reject a learner that is garbage on only *some* samples -- you need a
  per-sample order statistic.

The four corners:

====================  =================  ==================
combiner              weighting          aggregation
====================  =================  ==================
``average``           uniform            mean
``weighted``          competence         mean   (the "stacking" axis alone)
``robust``            uniform            median (the robust-aggregation axis alone)
``full``              competence         median (both)
====================  =================  ==================

``single`` (best holdout learner) is the no-ensemble reference. The dissociation: each
single-axis combiner collapses on the regime whose failure mode it cannot handle, the
uniform mean collapses on both, and ``full`` stays robust everywhere.
"""

from __future__ import annotations

import numpy as np

from ensemblekit.metrics import auroc
from ensemblekit.types import Ensemble

# Which combiners carry each ingredient (used by docs/tests, not control flow).
WEIGHTED = ("weighted", "full")
ROBUST = ("robust", "full")
COMBINERS = ("single", "average", "weighted", "robust", "full")

# Fraction of samples used to estimate competence / pick the best single learner.
HOLDOUT_FRAC = 0.5


def _split(
    logits: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = logits.shape[1] // 2
    return logits[:, :h], y[:h], logits[:, h:], y[h:]


def competence_weights(holdout_logits: np.ndarray, holdout_y: np.ndarray) -> np.ndarray:
    """Non-negative weights proportional to each learner's holdout ranking skill.

    ``w_k = max(AUROC_k - 0.5, 0)`` normalized to sum to 1. A learner no better than chance
    gets zero weight; if all are at chance the weights fall back to uniform.
    """
    k = holdout_logits.shape[0]
    skill = np.array([max(auroc(holdout_logits[i], holdout_y) - 0.5, 0.0) for i in range(k)])
    total = skill.sum()
    if total <= 0.0:
        return np.full(k, 1.0 / k)
    return skill / total


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-column weighted median of ``values`` (shape ``(k, m)``) with learner ``weights``.

    For each sample the learners are sorted by value and we return the value at which the
    cumulative weight first reaches half the total. With uniform weights this is the
    ordinary (lower) median; with competence weights the strong learners dominate the
    crossing point, so dead learners cannot pull the estimate.
    """
    k, m = values.shape
    order = np.argsort(values, axis=0, kind="mergesort")
    vs = np.take_along_axis(values, order, axis=0)
    ws = weights[order]
    cum = np.cumsum(ws, axis=0)
    half = cum[-1, :] / 2.0
    idx = (cum >= half[None, :]).argmax(axis=0)
    return vs[idx, np.arange(m)]


def combine_scores(ens: Ensemble, combiner: str) -> np.ndarray:
    """Return the combined per-sample test score for ``combiner`` (higher = more positive)."""
    if combiner not in COMBINERS:
        raise ValueError(f"unknown combiner {combiner!r}; choose from {COMBINERS}")
    zh, yh, zt, _ = _split(ens.logits, ens.y)
    k = zt.shape[0]

    if combiner == "single":
        skill = [auroc(zh[i], yh) for i in range(k)]
        return zt[int(np.argmax(skill))]
    if combiner == "average":
        return zt.mean(axis=0)
    if combiner == "robust":
        return np.median(zt, axis=0)

    w = competence_weights(zh, yh)
    if combiner == "weighted":
        return (zt * w[:, None]).sum(axis=0)
    # full = competence-weighted median
    return weighted_median(zt, w)


def combine_auroc(ens: Ensemble, combiner: str) -> float:
    """AUROC of ``combiner`` on the test half of ``ens``."""
    _, _, _, yt = _split(ens.logits, ens.y)
    return auroc(combine_scores(ens, combiner), yt)
