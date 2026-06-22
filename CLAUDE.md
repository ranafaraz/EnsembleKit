# EnsembleKit — agent guide

Offline ensemble-combiner benchmark. Base-learner predictions are **synthesized** as log-odds
of a **known** Bayes label; ensemble combiners are graded by AUROC of recovering it. Thesis:
**competence weighting** buys robustness to heterogeneous competence (dead learners) and
**robust aggregation** (a per-sample median) buys robustness to intermittent corruption — a 2×2
ablation (weighting × aggregation) where each single-ingredient combiner collapses on the regime
it can't handle, proven by a `homogeneous` control. A diversity sweep adds the foundational
result: the ensemble's gain over the best single learner vanishes as the learners' errors
become correlated.

> `AGENTS.md` mirrors this file for non-Claude tools — **edit both together**.

## Commit policy (hard rule)
Author = **Rana Faraz only**. **Never** add a `Co-Authored-By: Claude` trailer or any
AI/assistant branding to commit messages. Keep history tidy and incremental.

## What must stay true (don't regress)
- **Offline & deterministic.** numpy is the only runtime dep. Every draw comes from
  `Settings.rng` (salted `np.random.default_rng`, `SALT=0x454E`), never `hash()`/clock — so CI
  reproduces `evals/RESULTS.md` bit-for-bit on Python 3.10–3.12.
- **Synthesized, exact ground truth.** Learners + label are method-independent (same ensemble
  for every combiner). The signal, labels, and base noise carry **no** regime offset, so a
  regime changes only the learner *pathology* (gains / corruption), never the labels — that's
  what makes `homogeneous` an honest control. If you touch `learners.py`, keep this invariant.
- **Aggregate in log-odds space.** In probability space a sigmoid bounds each learner's
  contribution to ±0.5 and the mean quietly absorbs outliers → nothing collapses. Log-odds make
  corruption an unbounded outlier the mean feels and the median rejects. Don't change this.
- **BLAS pinned to 1 thread** before numpy imports (`ensemblekit/__init__.py`, CI `env:`,
  `tests/conftest.py`). Never remove.
- **Tune the experiment, not the combiner.** The combiners are textbook (uniform/weighted mean,
  median, weighted median); only regime strengths in `learners.py` (`STRONG_MULT`, `N_STRONG`,
  `CORRUPT_ALPHA`, `CORRUPT_GARBAGE`, `BETA`, `NOISE_SIGMA`) are tuned. `CORRUPT_ALPHA` must keep
  corruption a per-sample *minority* (median needs a clean majority). Don't tweak a combiner to
  clear the gate.
- **The dissociation must hold:** `het_competence` collapses the uniform combiners (average,
  robust) only; `corrupted` collapses the mean combiners (average, weighted) only; `full` stays
  robust everywhere; the diversity gain shrinks to ~0 as `rho→1`; the scrambled-label null sits
  at chance. `python -m evals.gate` asserts all of it (30 checks).

## Layout
`ensemblekit/` — `learners` (synth `Ensemble` + regimes homogeneous/het_competence/corrupted),
`combiners` (2×2: average · weighted · robust · full + single; competence weights + weighted
median), `metrics` (AUROC via Mann-Whitney), `runner` (`run` · `diversity_gain` ·
`aurocs_by_combiner`), `config` (Settings + SALT rng), `cli`, `sklearn_check` (optional).
`evals/` (metrics · harness · gate). `tests/` (76 + 6 sklearn skipped). `examples/run_combine.py`.

## Run (offline)
```
pip install -e ".[dev]"
pytest -q                  # 76 pass (+6 skipped without the [sklearn] extra)
ruff check .
python -m evals.harness    # writes evals/RESULTS.md (~6s)
python -m evals.gate       # asserts the dissociation (CI gate, 30/30)
ensemblekit compare --regime het_competence
```
Backend matrix (env): `ENSEMBLEKIT_COMBINER` single|average|weighted|robust|full ·
`ENSEMBLEKIT_REGIME` homogeneous|het_competence|corrupted · `ENSEMBLEKIT_LABELS` real|scrambled ·
`ENSEMBLEKIT_SAMPLES` (1600) · `ENSEMBLEKIT_RHO` (0.0) · `ENSEMBLEKIT_SEED` (0) ·
`ENSEMBLEKIT_BACKEND` numpy|sklearn (optional `[sklearn]` cross-check of AUROC vs
`roc_auc_score`; importorskip in tests, never on the default path).

## Env notes
Windows 11, PowerShell + Bash. venv at `.venv/Scripts/python.exe` (`python -m venv .venv`).
Windows console is cp1252 — CLI prints ASCII; the harness writes UTF-8 `RESULTS.md`.
`gh` authed as `ranafaraz`.
