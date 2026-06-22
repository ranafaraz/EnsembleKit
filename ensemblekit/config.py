"""Environment-driven settings and a single deterministic RNG factory.

Every random draw in EnsembleKit (the latent Bayes signal, each base learner's noise, the
per-sample corruption mask, the scrambled-label null) comes from :meth:`Settings.rng`,
seeded from a fixed salt plus explicit integer offsets -- never from ``hash()`` or
wall-clock time -- so a given (combiner, regime, seed) reproduces bit-for-bit on every
machine and Python version, which is what lets the eval gate use tight thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ensemblekit.types import Config

# Fixed salt so seeds are stable across machines and Python versions.
SALT = 0x45_4E  # "EN" -- arbitrary constant, never change once published.

DEFAULT_SAMPLES = 1600  # samples drawn per (combiner, regime, seed)
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Settings:
    combiner: str = "full"
    regime: str = "homogeneous"
    labels: str = "real"
    samples: int = DEFAULT_SAMPLES
    rho: float = 0.0
    seed: int = DEFAULT_SEED
    backend: str = "numpy"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            combiner=os.environ.get("ENSEMBLEKIT_COMBINER", "full").strip().lower(),
            regime=os.environ.get("ENSEMBLEKIT_REGIME", "homogeneous").strip().lower(),
            labels=os.environ.get("ENSEMBLEKIT_LABELS", "real").strip().lower(),
            samples=int(os.environ.get("ENSEMBLEKIT_SAMPLES", DEFAULT_SAMPLES)),
            rho=float(os.environ.get("ENSEMBLEKIT_RHO", 0.0)),
            seed=int(os.environ.get("ENSEMBLEKIT_SEED", DEFAULT_SEED)),
            backend=os.environ.get("ENSEMBLEKIT_BACKEND", "numpy").strip().lower(),
        )

    def to_config(self) -> Config:
        return Config(
            combiner=self.combiner,
            regime=self.regime,
            labels=self.labels,
            samples=self.samples,
            rho=self.rho,
            seed=self.seed,
            backend=self.backend,
        )

    def rng(self, *offsets: int) -> np.random.Generator:
        """A fresh Generator seeded from SALT, the run seed, and explicit offsets."""
        state = (SALT * 0x9E3779B1) ^ (int(self.seed) & 0xFFFFFFFF)
        for off in offsets:
            state = (state * 0x100000001B3) ^ (int(off) & 0xFFFFFFFFFFFFFFFF)
            state &= 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(state & 0xFFFFFFFF)


def rng_from_seed(seed: int, *offsets: int) -> np.random.Generator:
    """Stand-alone RNG factory for code paths that don't hold a Settings."""
    return Settings(seed=seed).rng(*offsets)
