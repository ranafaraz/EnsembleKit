"""The headline result: the 2x2 weighting x aggregation dissociation holds.

These assert the *shape* of the effect on a few seeds (small enough to stay fast, large
enough to be stable) -- the eval gate proves it at full strength over 16 seeds.
"""

from __future__ import annotations

import numpy as np

from ensemblekit.runner import aurocs_by_combiner, diversity_gain

SEEDS = 6
SAMPLES = 1600


def _mean_table(regime: str) -> dict[str, float]:
    rows: dict[str, list[float]] = {}
    for s in range(SEEDS):
        t = aurocs_by_combiner(regime, s, SAMPLES)
        for k, v in t.items():
            rows.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in rows.items()}


def test_homogeneous_control_all_recover():
    t = _mean_table("homogeneous")
    for c in ("average", "weighted", "robust", "full"):
        assert t[c] >= 0.75


def test_het_competence_needs_weighting():
    t = _mean_table("het_competence")
    # weighting recovers; uniform aggregation (average, robust) collapses
    assert t["weighted"] >= 0.75 and t["full"] >= 0.75
    assert t["average"] <= 0.74 and t["robust"] <= 0.74
    # the weighting ingredient is decisive: full >> robust here
    assert t["full"] - t["robust"] >= 0.1


def test_corrupted_needs_robust_aggregation():
    t = _mean_table("corrupted")
    # robust aggregation recovers; mean combiners (average, weighted) collapse
    assert t["robust"] >= 0.75 and t["full"] >= 0.75
    assert t["average"] <= 0.74 and t["weighted"] <= 0.74
    # the aggregation ingredient is decisive: full >> weighted here
    assert t["full"] - t["weighted"] >= 0.05


def test_average_collapses_on_both_regimes():
    assert _mean_table("het_competence")["average"] <= 0.74
    assert _mean_table("corrupted")["average"] <= 0.74


def test_full_robust_everywhere():
    for regime in ("homogeneous", "het_competence", "corrupted"):
        assert _mean_table(regime)["full"] >= 0.75


def test_diversity_gain_positive_then_zero():
    def mean_gain(rho):
        return float(np.mean([diversity_gain("homogeneous", rho, s, SAMPLES)["gain"]
                              for s in range(SEEDS)]))

    gains = [mean_gain(0.0), mean_gain(0.99)]
    assert gains[0] >= 0.02   # diverse learners -> real ensemble benefit
    assert gains[1] <= 0.012  # identical learners -> no benefit
