# EnsembleKit — offline benchmark results

Seeds: **16** · samples/cell: **1600** · each cell is the mean test AUROC of recovering the known Bayes label (chance = 0.5; the Bayes ceiling here is ~0.80). No trained models, no datasets, no API keys.

## AUROC by combiner and regime

| combiner | ingredients | homogeneous (control) | het_competence | corrupted |
|---|---|--:|--:|--:|
| single | baseline | 0.759 | 0.789 | 0.635 |
| average | uniform + mean | 0.801 | 0.713 | 0.668 |
| weighted | competence + mean | 0.801 | 0.789 | 0.668 |
| robust | uniform + median | 0.797 | 0.586 | 0.770 |
| full | competence + median | 0.794 | 0.789 | 0.766 |

## Effect 1 — competence weighting beats heterogeneous competence

On `het_competence` a single strong learner sits among dead (zero-gain) ones. A uniform average drowns it in noise (`average` **0.713**) and a uniform median is even worse — most learners are noise, so the middle one is noise too (`robust` **0.586**). Weighting each learner by its holdout skill recovers the signal (`weighted` **0.789**, `full` **0.789**).

## Effect 2 — robust aggregation beats intermittent corruption

On `corrupted` every learner is equally competent but emits garbage log-odds on a random per-sample fraction. A fixed per-learner weight cannot reject a learner that is only *sometimes* broken, so the mean combiners are dragged off (`average` **0.668**, `weighted` **0.668**). A per-sample median rejects the outliers (`robust` **0.770**, `full` **0.766**).

Only **full** — competence-weighted median — is robust to both regimes; each single-ingredient combiner collapses on the regime it cannot handle, and `average` (neither ingredient) collapses on both.

## Why ensemble at all? Diversity sweep

In the `homogeneous` regime every learner is equally good, so the only thing that makes the ensemble beat the best single learner is **diversity**. As the learners' errors become correlated (`rho` -> 1, identical learners), the gain collapses to zero — the canonical control for why ensembling helps.

| error correlation rho | ensemble gain (average - best single) |
|--:|--:|
| 0.00 | +0.041 |
| 0.30 | +0.029 |
| 0.60 | +0.015 |
| 0.90 | +0.004 |
| 0.99 | +0.001 |

## Scrambled-label null (sanity)

Shuffle the ground-truth labels and every combiner collapses to chance, so the AUROC above is not an artefact of the metric.

| combiner | null AUROC |
|---|--:|
| single | 0.498 |
| average | 0.494 |
| weighted | 0.499 |
| robust | 0.492 |
| full | 0.497 |

> Reproduce: `python -m evals.harness` (writes this file), `python -m evals.gate` (asserts the dissociation).
