"""Combiner mechanics: weighting, the weighted median, and the 2x2 registry."""

from __future__ import annotations

import numpy as np
import pytest

from ensemblekit.combiners import (
    COMBINERS,
    ROBUST,
    WEIGHTED,
    combine_auroc,
    combine_scores,
    competence_weights,
    weighted_median,
)
from ensemblekit.config import Settings
from ensemblekit.learners import synth_ensemble


def test_registry_partition():
    assert set(WEIGHTED) == {"weighted", "full"}
    assert set(ROBUST) == {"robust", "full"}
    assert set(COMBINERS) == {"single", "average", "weighted", "robust", "full"}


def test_weighted_median_uniform_equals_lower_median():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(7, 50))  # odd count -> exact middle element
    w = np.ones(7) / 7
    got = weighted_median(vals, w)
    assert np.allclose(got, np.median(vals, axis=0))


def test_weighted_median_concentrated_weight_picks_that_learner():
    vals = np.array([[-5.0], [0.0], [5.0]])
    w = np.array([0.01, 0.98, 0.01])
    assert weighted_median(vals, w)[0] == 0.0


def test_weighted_median_ignores_outlier_with_low_weight():
    # two reliable learners agree near 1.0, one heavily-weighted? no: low weight outlier
    vals = np.array([[1.0], [1.1], [99.0]])
    w = np.array([0.45, 0.45, 0.10])
    assert weighted_median(vals, w)[0] in (1.0, 1.1)


def test_competence_weights_sum_to_one():
    ens = synth_ensemble(Settings(regime="homogeneous", samples=400))
    h = ens.n_samples // 2
    w = competence_weights(ens.logits[:, :h], ens.y[:h])
    assert abs(w.sum() - 1.0) < 1e-12 and (w >= 0).all()


def test_competence_weights_concentrate_on_strong_learner():
    ens = synth_ensemble(Settings(regime="het_competence", samples=2000, seed=0))
    h = ens.n_samples // 2
    w = competence_weights(ens.logits[:, :h], ens.y[:h])
    # the single strong learner (index 0) should carry the most weight by far
    assert w.argmax() == 0
    assert w[0] > 0.5


def test_competence_weights_uniform_fallback_at_chance():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 200))
    y = (rng.random(200) < 0.5).astype(float)
    # labels independent of logits -> all learners ~chance -> some skill may be 0; weights valid
    w = competence_weights(logits, y)
    assert abs(w.sum() - 1.0) < 1e-12


def test_competence_weights_all_zero_skill_is_uniform():
    logits = np.zeros((5, 100))  # constant -> AUROC 0.5 -> skill 0 -> uniform fallback
    y = (np.arange(100) % 2).astype(float)
    w = competence_weights(logits, y)
    assert np.allclose(w, 0.2)


@pytest.mark.parametrize("combiner", COMBINERS)
def test_combine_scores_shape(combiner):
    ens = synth_ensemble(Settings(regime="homogeneous", samples=400))
    scores = combine_scores(ens, combiner)
    assert scores.shape == (ens.n_samples // 2,)


@pytest.mark.parametrize("combiner", COMBINERS)
def test_combine_auroc_in_range(combiner):
    ens = synth_ensemble(Settings(regime="homogeneous", samples=600))
    v = combine_auroc(ens, combiner)
    assert 0.0 <= v <= 1.0


def test_average_is_uniform_mean_of_test_logits():
    ens = synth_ensemble(Settings(regime="homogeneous", samples=400))
    h = ens.n_samples // 2
    assert np.allclose(combine_scores(ens, "average"), ens.logits[:, h:].mean(axis=0))


def test_unknown_combiner_raises():
    ens = synth_ensemble(Settings(regime="homogeneous", samples=100))
    with pytest.raises(ValueError):
        combine_scores(ens, "bogus")
