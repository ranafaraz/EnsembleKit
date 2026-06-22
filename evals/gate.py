"""Quality gate: fail CI unless the 2x2 dissociation actually holds.

The gate asserts the *shape* of the result, not a single lucky number:

* every ensemble combiner recovers on the ``homogeneous`` control;
* on ``het_competence`` the uniform-weight combiners (``average``, ``robust``) collapse and
  the competence-weighted combiners (``weighted``, ``full``) do not -- weighting is the
  ingredient that matters when learners differ in competence;
* on ``corrupted`` the mean combiners (``average``, ``weighted``) collapse and the median
  combiners (``robust``, ``full``) do not -- robust aggregation is the ingredient that
  matters under per-sample corruption;
* ``full`` (both ingredients) stays robust on every regime and beats each single-ingredient
  combiner on the regime that combiner fails;
* the ensemble gain over the best single learner is positive when learners are diverse and
  collapses to zero when their errors are fully correlated;
* the scrambled-label null sits at chance for every combiner.

Thresholds sit inside the measured margins (collapses land near 0.59-0.72, robust cells
near 0.77-0.81); ``np.random.default_rng`` is stable across platforms and Python versions,
so the numbers are identical on CI.
"""

from __future__ import annotations

import sys

from ensemblekit.combiners import COMBINERS, ROBUST, WEIGHTED
from ensemblekit.learners import REGIMES
from evals.harness import RHOS, compute

ROBUST_MIN = 0.75      # a combiner with the right ingredient recovers
COLLAPSE_MAX = 0.74    # a combiner missing the ingredient falls clearly below recovery
DISSOC_MARGIN = 0.05   # homogeneous control exceeds the collapsed cell by this much
GAP_MARGIN = 0.05      # full beats the fooled combiner by this much at its failure regime
GAIN_MIN = 0.02        # diverse ensemble beats the best single learner
GAIN_FLAT = 0.012      # identical learners (rho->1) carry no ensemble gain
GAIN_DECLINE = 0.02    # gain shrinks as correlation rises
NULL_BAND = 0.06       # scrambled labels sit within this of chance (0.5)

ENSEMBLE = [c for c in COMBINERS if c != "single"]
# Combiners lacking each ingredient -> they collapse on the matching regime.
NO_WEIGHTING = [c for c in ENSEMBLE if c not in WEIGHTED]   # average, robust
NO_ROBUST = [c for c in ENSEMBLE if c not in ROBUST]        # average, weighted


def _check(checks: list[tuple[bool, str]], ok: bool, msg: str) -> None:
    checks.append((bool(ok), msg))


def run_checks() -> list[tuple[bool, str]]:
    r = compute()
    mat = r["matrix"]
    null = r["null"]
    div = r["diversity"]
    checks: list[tuple[bool, str]] = []

    def cell(c: str, regime: str) -> float:
        return mat[c][regime]["mean"]

    # ---- control: every ensemble combiner recovers on the homogeneous regime ----
    for c in ENSEMBLE:
        v = cell(c, "homogeneous")
        _check(checks, v >= ROBUST_MIN, f"homogeneous: {c} recovers ({v:.3f} >= {ROBUST_MIN})")

    # ---- Effect 1: competence weighting survives heterogeneous competence ----
    for c in NO_WEIGHTING:  # average, robust
        ho, he = cell(c, "homogeneous"), cell(c, "het_competence")
        _check(checks, he <= COLLAPSE_MAX,
               f"het_competence: {c} (uniform) collapses ({he:.3f} <= {COLLAPSE_MAX})")
        _check(checks, ho - he >= DISSOC_MARGIN,
               f"{c}: homogeneous {ho:.3f} - het_competence {he:.3f} >= {DISSOC_MARGIN}")
    for c in sorted(WEIGHTED):  # weighted, full
        v = cell(c, "het_competence")
        _check(checks, v >= ROBUST_MIN,
               f"het_competence: {c} (weighted) holds ({v:.3f} >= {ROBUST_MIN})")

    # ---- Effect 2: robust aggregation survives intermittent corruption ----
    for c in NO_ROBUST:  # average, weighted
        ho, co = cell(c, "homogeneous"), cell(c, "corrupted")
        _check(checks, co <= COLLAPSE_MAX,
               f"corrupted: {c} (mean) collapses ({co:.3f} <= {COLLAPSE_MAX})")
        _check(checks, ho - co >= DISSOC_MARGIN,
               f"{c}: homogeneous {ho:.3f} - corrupted {co:.3f} >= {DISSOC_MARGIN}")
    for c in sorted(ROBUST):  # robust, full
        v = cell(c, "corrupted")
        _check(checks, v >= ROBUST_MIN,
               f"corrupted: {c} (median) holds ({v:.3f} >= {ROBUST_MIN})")

    # ---- full has both ingredients: robust everywhere, beats each ablation ----
    for regime in REGIMES:
        v = cell("full", regime)
        _check(checks, v >= ROBUST_MIN, f"{regime}: full robust ({v:.3f} >= {ROBUST_MIN})")
    fh, rh = cell("full", "het_competence"), cell("robust", "het_competence")
    _check(checks, fh - rh >= GAP_MARGIN,
           f"full beats robust on het_competence ({fh:.3f} - {rh:.3f} >= {GAP_MARGIN})")
    fc, wc = cell("full", "corrupted"), cell("weighted", "corrupted")
    _check(checks, fc - wc >= GAP_MARGIN,
           f"full beats weighted on corrupted ({fc:.3f} - {wc:.3f} >= {GAP_MARGIN})")

    # ---- diversity control: gain positive when diverse, zero when correlated ----
    g0, g99 = div[RHOS[0]], div[RHOS[-1]]
    _check(checks, g0 >= GAIN_MIN, f"diversity: gain at rho=0 ({g0:+.3f} >= {GAIN_MIN})")
    _check(checks, g99 <= GAIN_FLAT,
           f"diversity: gain at rho={RHOS[-1]} ({g99:+.3f} <= {GAIN_FLAT})")
    _check(checks, g0 - g99 >= GAIN_DECLINE,
           f"diversity: gain shrinks ({g0:+.3f} - {g99:+.3f} >= {GAIN_DECLINE})")
    seq = [round(div[rho], 3) for rho in RHOS]
    mono = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
    _check(checks, mono, f"diversity: gain monotone non-increasing in rho ({seq})")

    # ---- scrambled-label null sits at chance ----
    for c in COMBINERS:
        v = null[c]
        _check(checks, abs(v - 0.5) <= NULL_BAND,
               f"scrambled null: {c} at chance (|{v:.3f} - 0.5| <= {NULL_BAND})")

    return checks


def main() -> int:
    checks = run_checks()
    passed = 0
    for ok, msg in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
        passed += ok
    total = len(checks)
    print(f"\nGate: {passed}/{total} checks passed.")
    if passed == total:
        print("PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
