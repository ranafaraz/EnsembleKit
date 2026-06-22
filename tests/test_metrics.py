"""AUROC implementation: edge cases and agreement with a brute-force U statistic."""

from __future__ import annotations

import numpy as np

from ensemblekit.metrics import auroc, sigmoid


def test_perfect_separation():
    y = np.array([0, 0, 1, 1.0])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0


def test_reversed_separation():
    y = np.array([0, 0, 1, 1.0])
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0


def test_constant_scores_is_chance():
    y = np.array([0, 1, 0, 1.0])
    assert auroc(np.zeros(4), y) == 0.5


def test_single_class_returns_half():
    assert auroc(np.array([0.1, 0.9]), np.array([1.0, 1.0])) == 0.5
    assert auroc(np.array([0.1, 0.9]), np.array([0.0, 0.0])) == 0.5


def test_ties_count_half():
    # one positive tied with one negative, rest separable
    y = np.array([0, 1.0])
    assert auroc(np.array([0.5, 0.5]), y) == 0.5


def test_matches_bruteforce_u():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(5, 40)
        scores = rng.normal(size=n)
        y = (rng.random(n) < 0.5).astype(float)
        if y.sum() == 0 or y.sum() == n:
            continue
        pos = scores[y == 1][:, None]
        neg = scores[y == 0][None, :]
        wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
        brute = wins / (len(pos) * neg.shape[1])
        assert abs(auroc(scores, y) - brute) < 1e-9


def test_auroc_invariant_to_monotone_transform():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=50)
    y = (rng.random(50) < 0.5).astype(float)
    if 0 < y.sum() < 50:
        assert abs(auroc(scores, y) - auroc(2.0 * scores + 3.0, y)) < 1e-12


def test_sigmoid_bounds_and_symmetry():
    assert sigmoid(np.array([0.0]))[0] == 0.5
    assert 0.0 < sigmoid(np.array([-50.0]))[0] < 1e-10
    assert 1.0 - sigmoid(np.array([50.0]))[0] < 1e-10
