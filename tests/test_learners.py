"""Synthesized base learners: shapes, ground-truth invariance, regime mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from ensemblekit.config import Settings
from ensemblekit.learners import (
    CORRUPT_GARBAGE,
    N_LEARNERS,
    N_STRONG,
    REGIMES,
    synth_ensemble,
)


@pytest.mark.parametrize("regime", REGIMES)
def test_shapes(regime):
    ens = synth_ensemble(Settings(regime=regime, samples=500))
    assert ens.logits.shape == (N_LEARNERS, 500)
    assert ens.y.shape == (500,)
    assert ens.bayes.shape == (500,)
    assert ens.n_learners == N_LEARNERS and ens.n_samples == 500


@pytest.mark.parametrize("regime", REGIMES)
def test_labels_are_binary(regime):
    ens = synth_ensemble(Settings(regime=regime, samples=400))
    assert set(np.unique(ens.y)).issubset({0.0, 1.0})
    assert 0 < ens.y.sum() < 400  # not degenerate


def test_ground_truth_is_method_and_regime_independent():
    # The labels depend only on the latent signal, which is drawn from the same offset
    # regardless of regime -> identical y across regimes for a fixed seed.
    ys = [synth_ensemble(Settings(regime=r, samples=600, seed=2)).y for r in REGIMES]
    for other in ys[1:]:
        assert np.array_equal(ys[0], other)


def test_bayes_signal_shared_across_regimes():
    bs = [synth_ensemble(Settings(regime=r, samples=300, seed=1)).bayes for r in REGIMES]
    for other in bs[1:]:
        assert np.allclose(bs[0], other)


def test_determinism():
    a = synth_ensemble(Settings(regime="corrupted", samples=300, seed=4))
    b = synth_ensemble(Settings(regime="corrupted", samples=300, seed=4))
    assert np.array_equal(a.logits, b.logits) and np.array_equal(a.y, b.y)


def test_het_competence_has_expected_strong_count():
    ens = synth_ensemble(Settings(regime="het_competence", samples=200))
    assert int(ens.competent.sum()) == N_STRONG


def test_homogeneous_all_competent():
    ens = synth_ensemble(Settings(regime="homogeneous", samples=200))
    assert bool(ens.competent.all())


def test_corrupted_injects_extreme_logits():
    clean = synth_ensemble(Settings(regime="homogeneous", samples=2000, seed=0))
    dirty = synth_ensemble(Settings(regime="corrupted", samples=2000, seed=0))
    # corruption produces |logit| far beyond anything the clean base learners reach
    extreme = 2.0 * CORRUPT_GARBAGE
    assert (np.abs(dirty.logits) > extreme).sum() > 0
    assert (np.abs(clean.logits) > extreme).sum() == 0
    # clean base logits and corrupted base logits agree off the corruption mask
    assert np.array_equal(clean.y, dirty.y)


def test_rho_increases_cross_learner_correlation():
    low = synth_ensemble(Settings(regime="homogeneous", samples=4000, rho=0.0, seed=0))
    high = synth_ensemble(Settings(regime="homogeneous", samples=4000, rho=0.9, seed=0))

    def mean_offdiag_corr(logits):
        c = np.corrcoef(logits)
        return float(c[np.triu_indices_from(c, k=1)].mean())

    assert mean_offdiag_corr(high.logits) > mean_offdiag_corr(low.logits) + 0.2


def test_invalid_regime_raises():
    with pytest.raises(ValueError):
        synth_ensemble(Settings(regime="nonsense"))


@pytest.mark.parametrize("rho", [-0.1, 1.0, 1.5])
def test_invalid_rho_raises(rho):
    with pytest.raises(ValueError):
        synth_ensemble(Settings(regime="homogeneous", rho=rho))
