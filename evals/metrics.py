"""Aggregate combiner AUROC across seeds.

One number carries a (combiner, regime) cell: the mean test AUROC of recovering the known
Bayes label over many seeds. A combiner with the right ingredient for a regime stays near
the Bayes ceiling (~0.80 here); one missing it drops toward the single-learner level or
below on the regime whose failure mode it cannot handle, and recovers on the
``homogeneous`` control.
"""

from __future__ import annotations

import numpy as np

from ensemblekit.runner import diversity_gain, run
from ensemblekit.types import Config


def aurocs(combiner: str, regime: str, labels: str, samples: int, seeds: int) -> np.ndarray:
    return np.array(
        [run(Config(combiner, regime, labels, samples, 0.0, s, "numpy")).auroc
         for s in range(seeds)]
    )


def mean_auroc(combiner: str, regime: str, samples: int, seeds: int, labels: str = "real") -> float:
    return float(aurocs(combiner, regime, labels, samples, seeds).mean())


def std_auroc(combiner: str, regime: str, samples: int, seeds: int, labels: str = "real") -> float:
    return float(aurocs(combiner, regime, labels, samples, seeds).std())


def mean_gain(regime: str, rho: float, samples: int, seeds: int) -> float:
    """Mean ensemble gain (uniform average - best single) over seeds at correlation ``rho``."""
    return float(np.mean([diversity_gain(regime, rho, s, samples)["gain"] for s in range(seeds)]))
