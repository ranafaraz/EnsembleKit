"""Settings, the salted RNG, and env round-tripping."""

from __future__ import annotations

import numpy as np

from ensemblekit.config import SALT, Settings, rng_from_seed


def test_salt_is_fixed():
    assert SALT == 0x45_4E


def test_rng_deterministic_for_same_seed_and_offsets():
    a = Settings(seed=3).rng(1, 2).standard_normal(5)
    b = Settings(seed=3).rng(1, 2).standard_normal(5)
    assert np.array_equal(a, b)


def test_rng_differs_across_seeds():
    a = Settings(seed=1).rng(0).standard_normal(10)
    b = Settings(seed=2).rng(0).standard_normal(10)
    assert not np.array_equal(a, b)


def test_rng_differs_across_offsets():
    s = Settings(seed=0)
    assert not np.array_equal(s.rng(1).standard_normal(10), s.rng(2).standard_normal(10))


def test_from_env_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("ENSEMBLEKIT_"):
            monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.combiner == "full" and s.regime == "homogeneous" and s.labels == "real"
    assert s.samples == 1600 and s.rho == 0.0 and s.backend == "numpy"


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("ENSEMBLEKIT_COMBINER", "ROBUST")
    monkeypatch.setenv("ENSEMBLEKIT_REGIME", "Corrupted")
    monkeypatch.setenv("ENSEMBLEKIT_RHO", "0.5")
    monkeypatch.setenv("ENSEMBLEKIT_SAMPLES", "800")
    s = Settings.from_env()
    assert s.combiner == "robust" and s.regime == "corrupted"
    assert s.rho == 0.5 and s.samples == 800


def test_to_config_round_trip():
    s = Settings(combiner="weighted", regime="corrupted", rho=0.3, seed=7)
    c = s.to_config()
    assert c.combiner == "weighted" and c.regime == "corrupted" and c.rho == 0.3 and c.seed == 7


def test_standalone_rng_matches_settings():
    a = rng_from_seed(5, 1).standard_normal(4)
    b = Settings(seed=5).rng(1).standard_normal(4)
    assert np.array_equal(a, b)
