"""Glue: turn a :class:`Settings`/:class:`Config` into a :class:`CombineResult`.

The runner synthesizes the ensemble for the regime, optionally scrambles the labels (the
chance-level null), applies the requested combiner, and reports the test AUROC. The same
synthesized ensemble is used for every combiner at a given (regime, seed), so a comparison
across combiners is apples-to-apples.
"""

from __future__ import annotations

import dataclasses

from ensemblekit.combiners import combine_auroc
from ensemblekit.config import Settings
from ensemblekit.learners import synth_ensemble
from ensemblekit.types import CombineResult, Config, Ensemble

_SCRAMBLE_OFFSET = 9_001


def _maybe_scramble(ens: Ensemble, settings: Settings) -> Ensemble:
    if settings.labels == "real":
        return ens
    if settings.labels != "scrambled":
        raise ValueError(f"unknown labels {settings.labels!r}; choose real|scrambled")
    perm = settings.rng(_SCRAMBLE_OFFSET).permutation(ens.y)
    return dataclasses.replace(ens, y=perm)


def run_settings(settings: Settings) -> CombineResult:
    ens = synth_ensemble(settings)
    ens = _maybe_scramble(ens, settings)
    score = combine_auroc(ens, settings.combiner)
    return CombineResult(
        combiner=settings.combiner,
        regime=settings.regime,
        labels=settings.labels,
        seed=settings.seed,
        auroc=float(score),
    )


def run(config: Config) -> CombineResult:
    """Run from a plain :class:`Config` (used by tests and the eval harness)."""
    settings = Settings(
        combiner=config.combiner,
        regime=config.regime,
        labels=config.labels,
        samples=config.samples,
        rho=config.rho,
        seed=config.seed,
        backend=config.backend,
    )
    return run_settings(settings)


def run_from_env() -> CombineResult:
    return run_settings(Settings.from_env())


def diversity_gain(regime: str, rho: float, seed: int, samples: int = 1600) -> dict[str, float]:
    """Best-single vs uniform-average AUROC at a given noise correlation ``rho``.

    The ensemble *gain* (average - single) is the canonical "why ensembling helps" signal;
    it shrinks to zero as ``rho -> 1`` (identical learners), the diversity control.
    """
    base = Settings(regime=regime, rho=rho, seed=seed, samples=samples)
    ens = synth_ensemble(base)
    single = combine_auroc(ens, "single")
    average = combine_auroc(ens, "average")
    return {"single": single, "average": average, "gain": float(average - single)}


def aurocs_by_combiner(regime: str, seed: int, samples: int = 1600) -> dict[str, float]:
    """AUROC of every combiner on one synthesized (regime, seed) ensemble."""
    from ensemblekit.combiners import COMBINERS

    ens = synth_ensemble(Settings(regime=regime, seed=seed, samples=samples))
    return {c: combine_auroc(ens, c) for c in COMBINERS}
