# Design decisions

What I built and why — the choices that make EnsembleKit a real, reproducible result rather
than a demo.

## Synthesize the base learners, don't train them
An ensemble benchmark needs **ground truth**: the Bayes-optimal predictor, so you can say how
close a combiner gets to the ceiling. With trained models that ceiling is unknown and entangled
with the dataset. Instead I draw a latent signal `s` that *is* the Bayes log-odds, generate the
label from it, and synthesize each base learner as a noisy, gained view of `s`. The ground truth
is exact, the achievable AUROC has a known ceiling (~0.80 at `BETA = 1.5`), and the whole
benchmark runs offline with no models or datasets. The cost is realism — these are synthetic
log-odds, not trained classifiers — but the **mechanism** under test (how a combiner handles
weak learners and corrupted learners) is exactly the one a real stacking/bagging pipeline faces.

## A 2×2 factorial, not a leaderboard
The interesting claim is not "combiner X wins" but *which ingredient buys robustness to which
failure*. So the combiners are the four corners of {weighting: uniform | competence} ×
{aggregation: mean | median}, and there are two regimes that each isolate one failure mode.
That gives a **dissociation**: `weighted` survives `het_competence` but not `corrupted`;
`robust` is the mirror image; `full` (both) survives both; `average` (neither) fails on both.
Each effect is proven by the `homogeneous` control where neither pathology is present and the
"failing" combiner recovers — so the collapse is the missing ingredient, not the combiner.

## The two axes are genuinely orthogonal (and neither subsumes the other)
This was the crux of the design. Competence weighting and a learned linear stack can fix any
failure that is a *fixed per-learner* defect — so an early calibration-based second axis was a
dead end (stacking just absorbed it). The honest orthogonal axis is **per-sample** robustness:
intermittent corruption is a defect that varies *within* a learner across samples, which no
fixed weight can address but a per-sample order statistic (the median) can. Symmetrically, a
median cannot rescue a consistently weak learner — it has no notion of competence. The two
ingredients therefore cover failure modes neither can reach alone, which is what makes the 2×2 a
true dissociation rather than two views of the same fix.

## Aggregate in log-odds space
Combining is done on the learners' log-odds, not their probabilities. In probability space a
sigmoid bounds every learner's contribution to ±0.5, so a uniform mean quietly absorbs both weak
learners and outliers and *nothing* visibly collapses (an early probability-space prototype
showed exactly this). In log-odds space a corrupted learner is an unbounded outlier that drags
the mean but not the median — which is the whole point — and the ranking metric is unaffected by
the monotone change of variable.

## The weighted median, so "both" is one coherent combiner
The `full` corner needs to weight *and* aggregate robustly at once. A naive "trim then weight"
fails on `het_competence`, because the strong learner looks like an outlier against a noisy
median and gets trimmed. The weighted median solves it cleanly: it is the value at which the
cumulative *competence* weight crosses one half, so the strong learners determine the estimate
while per-sample outliers (a minority of weight) are still rejected. With uniform weights it
reduces exactly to the ordinary median — unit-tested — so the 2×2 axes stay clean.

## Diversity is the foundational control
Before "how to combine" comes "why combine at all". The `homogeneous` regime with a swept noise
correlation `rho` answers it: the ensemble's gain over the best single learner is real when
errors are independent and collapses to zero when they are perfectly correlated (identical
learners). That is the textbook bias–variance/ambiguity story made measurable, and it doubles as
a sanity control — if correlated learners still "helped", something would be wrong.

## Tune the experiment, never the combiner
The combiners are fixed textbook implementations. The only knobs are the *regime strengths* in
`learners.py` (`STRONG_MULT` and `N_STRONG` for competence imbalance; `CORRUPT_ALPHA`,
`CORRUPT_GARBAGE` for corruption; `BETA`, `NOISE_SIGMA` for the signal) — chosen by grid search
so each collapse is clearly separated from each recovery, with headroom to the gate thresholds.
Corruption stays a *minority* per sample (`ALPHA = 0.35` over 8 learners) so the median's clean
majority survives; if a majority were corrupted, nothing could recover and there would be
nothing to measure.

## Offline, deterministic, gated
numpy-only, salted `np.random.default_rng`, BLAS pinned to one thread → the eval table
reproduces bit-for-bit across machines and Python 3.10–3.12. That is what lets `evals/gate.py`
assert the dissociation with tight numeric margins and fail CI if it ever regresses, rather than
eyeballing a plot. scikit-learn is an optional cross-check only, never on the default path.

## Why these numbers
`SAMPLES = 1600` (split into a 800-sample holdout for competence/selection and an 800-sample
test) × `SEEDS = 16` gives stable AUROC while keeping the whole benchmark — harness *and* gate —
to a few seconds, so it fits a CI job comfortably alongside the test suite.
