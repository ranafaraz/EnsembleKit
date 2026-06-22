"""Optional sklearn cross-check: our numpy AUROC matches scikit-learn's.

Skipped entirely when scikit-learn is not installed, so the default CI (numpy-only) stays
green. Run `pip install -e ".[sklearn]"` to exercise it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from ensemblekit.combiners import COMBINERS  # noqa: E402
from ensemblekit.metrics import auroc  # noqa: E402
from ensemblekit.sklearn_check import cross_check, sklearn_auroc  # noqa: E402


def test_sklearn_auroc_matches_numpy_random():
    rng = np.random.default_rng(0)
    for _ in range(10):
        scores = rng.normal(size=80)
        y = (rng.random(80) < 0.5).astype(float)
        if 0 < y.sum() < 80:
            assert abs(auroc(scores, y) - sklearn_auroc(scores, y)) < 1e-9


@pytest.mark.parametrize("combiner", COMBINERS)
def test_cross_check_agrees_per_combiner(combiner):
    r = cross_check(combiner=combiner, regime="homogeneous", seed=0)
    assert r["abs_diff"] < 1e-9
