"""Plain dataclasses shared across EnsembleKit.

These are deliberately dependency-free value objects: a :class:`Config` describing one run,
an :class:`Ensemble` holding synthesized base-learner predictions plus the ground truth,
and a :class:`CombineResult`/:class:`Aggregate` carrying scores out of the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Config:
    """One benchmark run: a combiner applied to a regime at a seed."""

    combiner: str = "full"
    regime: str = "homogeneous"
    labels: str = "real"
    samples: int = 1600
    rho: float = 0.0
    seed: int = 0
    backend: str = "numpy"


@dataclass(frozen=True)
class Ensemble:
    """A synthesized ensemble problem.

    ``logits`` are the base-learner log-odds, shape ``(n_learners, n_samples)``. ``y`` is
    the binary ground-truth label. ``bayes`` is the Bayes-optimal log-odds (the latent
    signal the learners noisily observe), kept for reference / the ceiling. ``competent``
    marks which learners carry real signal (used only for sanity checks, never by a
    combiner -- combiners must estimate competence from a holdout).
    """

    logits: np.ndarray
    y: np.ndarray
    bayes: np.ndarray
    competent: np.ndarray
    regime: str

    @property
    def n_learners(self) -> int:
        return int(self.logits.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.logits.shape[1])


@dataclass(frozen=True)
class CombineResult:
    """The outcome of one (combiner, regime, seed) run."""

    combiner: str
    regime: str
    labels: str
    seed: int
    auroc: float


@dataclass
class Aggregate:
    """Mean/std of a metric pooled over seeds, with the raw values kept."""

    mean: float = 0.0
    std: float = 0.0
    values: list[float] = field(default_factory=list)
