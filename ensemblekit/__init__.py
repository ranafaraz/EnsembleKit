"""EnsembleKit: an offline ensemble-combiner benchmark.

When does combining base learners help, and *which* combining trick buys *which* kind of
robustness? EnsembleKit answers this without a single trained model or downloaded dataset.
Base-learner predictions are synthesized as log-odds of a known Bayes label, so the ground
truth is exact, and combiners are scored by the AUROC of recovering that label. The result
is a clean 2x2 dissociation:

* **competence weighting** (the stacking axis) buys robustness to *heterogeneous
  competence* -- a uniform average drowns a strong learner among dead ones;
* **robust aggregation** (a per-sample median) buys robustness to *intermittent
  corruption* -- a fixed weight cannot reject a learner that is garbage on only some
  samples;

and you need both -- each single-axis combiner collapses on the regime whose failure mode
it cannot handle, the uniform mean collapses on both, and the weighted median stays robust
everywhere, proven by a ``homogeneous`` control. A diversity sweep adds the foundational
result: the ensemble's gain over the best single learner vanishes as the learners' errors
become correlated (identical learners carry no ensemble benefit).
"""

from __future__ import annotations

# Pin BLAS to a single thread *before* numpy is imported. The benchmark is many tiny
# matrix ops (per-learner log-odds over a few thousand samples, small competence fits,
# repeated across seeds and regimes); with multi-threaded BLAS the per-call thread-pool
# overhead dominates and contention is non-deterministic, which would make CI flaky and
# slow. One thread is both faster here and fully reproducible. setdefault so an explicit
# env wins.
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from ensemblekit.combiners import COMBINERS, ROBUST, WEIGHTED, combine_auroc, combine_scores
from ensemblekit.config import Settings
from ensemblekit.learners import REGIMES, synth_ensemble
from ensemblekit.metrics import auroc, sigmoid
from ensemblekit.runner import (
    aurocs_by_combiner,
    diversity_gain,
    run,
    run_from_env,
    run_settings,
)
from ensemblekit.types import Aggregate, CombineResult, Config, Ensemble

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "Config",
    "Ensemble",
    "CombineResult",
    "Aggregate",
    "REGIMES",
    "synth_ensemble",
    "COMBINERS",
    "WEIGHTED",
    "ROBUST",
    "combine_scores",
    "combine_auroc",
    "auroc",
    "sigmoid",
    "run",
    "run_settings",
    "run_from_env",
    "diversity_gain",
    "aurocs_by_combiner",
]
