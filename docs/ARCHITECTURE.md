# Architecture

EnsembleKit is a small, fully offline benchmark. Data flows in one direction: a synthesizer
produces base-learner predictions with a known Bayes signal, a combiner fuses them into one
score, and a metric grades that score against the ground-truth label.

```
learners.py                 combiners.py                     metrics.py
Ensemble            -->      combined per-sample score   -->  AUROC vs the
(logits, y, bayes,          (weighting x aggregation,         known Bayes
 competent, regime)          + single baseline)               label y
        |                                                          |
        +------------------- runner.run(Config) -------------------+
                                      |
                       evals/ (metrics -> harness -> gate)
```

## Modules

- **`ensemblekit/learners.py`** — synthesizes one `Ensemble`. A latent `s ~ N(0,1)` *is* the
  Bayes log-odds; the label is `y ~ Bernoulli(sigmoid(BETA*s))`. Each of `N_LEARNERS` base
  learners reports `z_k = a_k*s + eps_k` — a competence gain times the true signal plus
  correlated noise (`rho` is the shared fraction). Three regimes change only the learner
  *pathology*, never the labels: `homogeneous` (control), `het_competence` (one strong learner
  among dead ones), `corrupted` (each learner emits garbage on a random per-sample fraction).
- **`ensemblekit/combiners.py`** — the 2×2 factorial plus a `single` baseline. A combiner =
  (weighting ∈ {uniform, competence}) × (aggregation ∈ {mean, median}), applied to the base
  learners' log-odds: `average`, `weighted`, `robust`, `full` are the corners. Competence
  weights come from each learner's holdout ranking skill; the weighted median is the order
  statistic where cumulative competence weight crosses one half.
- **`ensemblekit/metrics.py`** — threshold-free AUROC via the Mann-Whitney U statistic with
  average ranks for ties (no scipy/sklearn on the default path).
- **`ensemblekit/runner.py`** — `run(Config)` synthesizes the ensemble (method-independent, so
  every combiner sees identical learners), optionally scrambles the labels (the null), applies
  one combiner, and returns the test AUROC. Also exposes `diversity_gain` (average vs single
  at a given `rho`) and `aurocs_by_combiner`.
- **`ensemblekit/config.py`** — `Settings` + the salted `np.random.default_rng` factory. Every
  random draw is reproducible from `(SALT, seed, offsets)`.
- **`ensemblekit/sklearn_check.py`** — optional cross-check of the numpy AUROC against
  scikit-learn's `roc_auc_score`.
- **`evals/`** — `metrics` (seed aggregation) → `harness` (writes `RESULTS.md`) → `gate`
  (asserts the dissociation as a battery of inequalities; the CI quality gate).

## The 2×2 and why the combiners behave

Two orthogonal failure modes, two orthogonal fixes:

- **Heterogeneous competence** (`het_competence`). A uniform average spreads its trust evenly,
  so one strong learner is drowned by the dead ones; a uniform median is worse still, since the
  middle learner *is* noise. **Competence weighting** (`weighted`, `full`) concentrates trust on
  the learner that actually ranks the holdout labels. A median cannot fix this — it ignores
  weights — so `robust` collapses here.
- **Intermittent corruption** (`corrupted`). Every learner is equally competent, but each is
  garbage on a random *subset* of samples. A fixed per-learner weight cannot reject a learner
  that is only *sometimes* broken, so any mean (`average`, `weighted`) is dragged off by the
  per-sample outliers. **A per-sample median** (`robust`, `full`) rejects them, because on each
  sample the clean majority outvotes the corrupted minority. Weighting cannot fix this — the
  corruption is per-sample, not per-learner — so `weighted` collapses here.

So weighting fixes `het_competence`, robust aggregation fixes `corrupted`, and only `full`
(both) is robust to both. `average` (neither) collapses on both, and `homogeneous` is the
control where neither pathology is present and every combiner recovers.

## Why ensemble at all? The diversity sweep

In `homogeneous` every learner is equally good, so the only thing that makes the ensemble beat
the best single learner is **diversity** — independent errors that cancel under aggregation.
Sweeping the noise correlation `rho` from 0 to ~1 turns the learners from independent to
identical, and the ensemble gain (`average − single`) shrinks monotonically to zero. That is the
foundational control: no diversity, no ensemble benefit.

## Determinism

`numpy` is the only runtime dependency. BLAS is pinned to one thread before numpy imports
(`ensemblekit/__init__.py`, `tests/conftest.py`, CI `env:`). All randomness is salted
`np.random.default_rng`, which is stable across platforms and Python versions, so the eval
table — and the gate thresholds built around it — reproduce bit-for-bit on 3.10–3.12.
