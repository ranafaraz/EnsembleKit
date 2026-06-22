"""Runner glue: results, the scrambled-label null, and helper queries."""

from __future__ import annotations

import pytest

from ensemblekit.combiners import COMBINERS
from ensemblekit.config import Settings
from ensemblekit.runner import (
    aurocs_by_combiner,
    diversity_gain,
    run,
    run_from_env,
    run_settings,
)
from ensemblekit.types import Config


def test_run_returns_result():
    r = run(Config(combiner="full", regime="homogeneous", samples=600, seed=0))
    assert r.combiner == "full" and r.regime == "homogeneous"
    assert 0.0 <= r.auroc <= 1.0


def test_run_is_deterministic():
    a = run(Config(combiner="full", regime="corrupted", samples=600, seed=3))
    b = run(Config(combiner="full", regime="corrupted", samples=600, seed=3))
    assert a.auroc == b.auroc


def test_scrambled_labels_collapse_to_chance():
    real = run(Config(combiner="full", regime="het_competence", samples=2000, seed=0)).auroc
    null = run(Config("full", "het_competence", "scrambled", 2000, 0.0, 0)).auroc
    assert real > 0.7
    assert abs(null - 0.5) < 0.08


def test_invalid_labels_raises():
    with pytest.raises(ValueError):
        run(Config(combiner="full", regime="homogeneous", labels="weird", samples=100))


def test_diversity_gain_structure():
    d = diversity_gain("homogeneous", 0.0, 0, samples=1000)
    assert set(d) == {"single", "average", "gain"}
    assert abs(d["gain"] - (d["average"] - d["single"])) < 1e-12


def test_diversity_gain_shrinks_with_correlation():
    g0 = diversity_gain("homogeneous", 0.0, 0, samples=2000)["gain"]
    g_hi = diversity_gain("homogeneous", 0.99, 0, samples=2000)["gain"]
    assert g0 > g_hi


def test_aurocs_by_combiner_keys():
    table = aurocs_by_combiner("homogeneous", 0, samples=600)
    assert set(table) == set(COMBINERS)


def test_run_from_env(monkeypatch):
    monkeypatch.setenv("ENSEMBLEKIT_COMBINER", "robust")
    monkeypatch.setenv("ENSEMBLEKIT_REGIME", "corrupted")
    monkeypatch.setenv("ENSEMBLEKIT_SAMPLES", "600")
    r = run_from_env()
    assert r.combiner == "robust" and r.regime == "corrupted"


def test_run_settings_matches_run():
    s = Settings(combiner="weighted", regime="het_competence", samples=600, seed=1)
    assert run_settings(s).auroc == run(s.to_config()).auroc
