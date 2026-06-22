"""Optional cross-check backend (needs the ``[sklearn]`` extra).

The benchmark runs entirely on the hand-rolled numpy AUROC. This module lets a curious
user confirm that metric against ``sklearn.metrics.roc_auc_score`` on the exact combined
scores the benchmark produces. It is never on the default path; the tests ``importorskip``
scikit-learn so CI stays green without it.
"""

from __future__ import annotations

import numpy as np

from ensemblekit.combiners import combine_scores
from ensemblekit.config import Settings
from ensemblekit.learners import synth_ensemble
from ensemblekit.metrics import auroc


def sklearn_auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUROC via scikit-learn (imported lazily so the core stays numpy-only)."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores))


def cross_check(
    combiner: str = "full", regime: str = "homogeneous", seed: int = 0
) -> dict[str, float]:
    """Compare the numpy AUROC and the sklearn AUROC for one combined run.

    Returns both values and their absolute difference; they should agree to floating-point
    tolerance because both score the identical combined log-odds against identical labels.
    """
    ens = synth_ensemble(Settings(combiner=combiner, regime=regime, seed=seed))
    scores = combine_scores(ens, combiner)
    h = ens.y.shape[0] // 2
    yt = ens.y[h:]
    ours = auroc(scores, yt)
    theirs = sklearn_auroc(scores, yt)
    return {"numpy": ours, "sklearn": theirs, "abs_diff": abs(ours - theirs)}
