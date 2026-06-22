"""Synthesize an ensemble of base learners with a *known* Bayes-optimal signal.

There is no trained model and no dataset here. We draw a latent score ``s ~ N(0,1)`` that
*is* the Bayes log-odds: the label is ``y ~ Bernoulli(sigmoid(BETA*s))``, so ``BETA*s`` is
the optimal predictor and the achievable AUROC has a known ceiling below 1. Each base
learner ``k`` reports a noisy log-odds ``z_k = a_k*s + eps_k`` -- a competence gain ``a_k``
times the true signal plus correlated noise. This makes every combiner's job exact to
reason about, and the ground truth (``y``) is identical for every combiner and regime.

Three regimes change *how the learners fail*, never the labels:

* ``homogeneous`` -- every learner equally competent and clean (the control). Diversity
  (controlled by the noise correlation ``rho``) is the only thing that varies here.
* ``het_competence`` -- one strong learner among many dead (zero-gain) learners. A uniform
  average drowns the strong learner in noise; you must *weight* by competence.
* ``corrupted`` -- every learner equally competent, but each emits garbage log-odds on a
  random per-sample fraction (a flaky sensor). A mean (or any fixed-weight combiner) is
  dragged by the per-sample outliers; you must *aggregate robustly* (median).

The competence and corruption are method-independent: the same ``(logits, y)`` are handed
to every combiner, so comparisons are fair.
"""

from __future__ import annotations

import numpy as np

from ensemblekit.config import Settings
from ensemblekit.metrics import sigmoid
from ensemblekit.types import Ensemble

# ---- Locked generative constants (tune the experiment here, never the combiners). ----
N_LEARNERS = 8          # base learners in the ensemble
BASE_GAIN = 1.7         # competence gain a_k of a healthy learner
NOISE_SIGMA = 1.0       # std of each learner's log-odds noise
LABEL_SHARPNESS = 1.7   # actually BETA below; see LABEL_BETA
LABEL_BETA = 1.5        # label sharpness; sets the Bayes ceiling (~0.80 AUROC here)

# het_competence: how many strong learners (the rest have zero gain = pure noise).
N_STRONG = 1
STRONG_MULT = 1.7       # strong learners get BASE_GAIN * STRONG_MULT

# corrupted: per-learner per-sample probability of emitting garbage, and its log-odds std.
CORRUPT_ALPHA = 0.35
CORRUPT_GARBAGE = 8.0

REGIMES = ("homogeneous", "het_competence", "corrupted")

# RNG offset namespaces so independent draws never collide for a given seed. The signal,
# labels, and base learner noise carry NO regime offset, so the underlying task and the
# clean base learners are identical across regimes -- a regime changes only the learner
# *pathology* (competence gains, or overlaid corruption), never the ground truth. This is
# what makes ``homogeneous`` an honest control for the other two regimes.
_OFF = {"signal": 1, "labels": 2, "common": 3, "idio": 4, "mask": 5, "garbage": 6}


def _gains(regime: str) -> np.ndarray:
    """Per-learner competence gain ``a_k`` for the regime."""
    a = np.full(N_LEARNERS, BASE_GAIN, dtype=float)
    if regime == "het_competence":
        strong = np.arange(N_LEARNERS) < N_STRONG
        a = np.where(strong, BASE_GAIN * STRONG_MULT, 0.0)
    return a


def synth_ensemble(settings: Settings) -> Ensemble:
    """Draw one ensemble problem for ``settings`` (regime, samples, rho, seed).

    ``rho`` in [0, 1) is the fraction of each learner's noise that is *shared* across the
    ensemble; ``rho -> 1`` makes the learners identical (no diversity), which is the
    control for "why ensembling helps at all".
    """
    regime = settings.regime
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")
    n = int(settings.samples)
    rho = float(settings.rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")

    # Latent Bayes log-odds and the label it induces (shared across all regimes).
    s = settings.rng(_OFF["signal"]).standard_normal(n)
    p_true = sigmoid(LABEL_BETA * s)
    y = (settings.rng(_OFF["labels"]).random(n) < p_true).astype(float)

    # Correlated learner noise: eps_k = sqrt(rho)*common + sqrt(1-rho)*idiosyncratic.
    common = settings.rng(_OFF["common"]).standard_normal(n)
    idio = settings.rng(_OFF["idio"]).standard_normal((N_LEARNERS, n))
    eps = (np.sqrt(rho) * common[None, :] + np.sqrt(1.0 - rho) * idio) * NOISE_SIGMA

    a = _gains(regime)
    logits = a[:, None] * s[None, :] + eps

    if regime == "corrupted":
        mask = settings.rng(_OFF["mask"]).random((N_LEARNERS, n)) < CORRUPT_ALPHA
        garbage = settings.rng(_OFF["garbage"]).standard_normal((N_LEARNERS, n)) * CORRUPT_GARBAGE
        logits = np.where(mask, garbage, logits)

    competent = a > 0.0
    return Ensemble(logits=logits, y=y, bayes=LABEL_BETA * s, competent=competent, regime=regime)
