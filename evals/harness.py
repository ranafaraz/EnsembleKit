"""Run the full offline benchmark and write ``evals/RESULTS.md``.

Every combiner is graded on every regime by the test AUROC of recovering the known Bayes
label, averaged over seeds. The result is a 2x2 dissociation:

1. **Heterogeneous competence** -- on ``het_competence`` one strong learner sits among
   dead (zero-gain) ones, so a uniform average (`average`) and a uniform median (`robust`)
   drown it, while competence weighting (`weighted`, `full`) recovers it. The
   ``homogeneous`` control (equal competence) shows the uniform combiners recover.
2. **Intermittent corruption** -- on ``corrupted`` every learner is equally competent but
   emits garbage log-odds on a random per-sample fraction, so the mean combiners
   (`average`, `weighted`) are dragged off by the per-sample outliers, while the median
   combiners (`robust`, `full`) reject them. The ``homogeneous`` control shows the mean
   combiners recover.

`full` (competence-weighted median) is the only combiner robust to both; each
single-ingredient combiner collapses on the regime it cannot handle, and `average`
(neither) collapses on both. A diversity sweep shows the ensemble's gain over the best
single learner vanishing as the learners' errors become correlated, and a scrambled-label
run drops every combiner to chance.
"""

from __future__ import annotations

import io
from pathlib import Path

from ensemblekit.combiners import COMBINERS, ROBUST, WEIGHTED
from ensemblekit.learners import REGIMES
from evals.metrics import mean_auroc, mean_gain, std_auroc

SEEDS = 16
SAMPLES = 1600
RHOS = (0.0, 0.3, 0.6, 0.9, 0.99)

RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def compute() -> dict:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for combiner in COMBINERS:
        matrix[combiner] = {}
        for regime in REGIMES:
            matrix[combiner][regime] = {
                "mean": mean_auroc(combiner, regime, SAMPLES, SEEDS),
                "std": std_auroc(combiner, regime, SAMPLES, SEEDS),
            }
    null = {
        combiner: sum(
            mean_auroc(combiner, regime, SAMPLES, SEEDS, labels="scrambled") for regime in REGIMES
        ) / len(REGIMES)
        for combiner in COMBINERS
    }
    diversity = {rho: mean_gain("homogeneous", rho, SAMPLES, SEEDS) for rho in RHOS}
    return {
        "matrix": matrix,
        "null": null,
        "diversity": diversity,
        "meta": {"seeds": SEEDS, "samples": SAMPLES},
    }


def _ingredients(combiner: str) -> str:
    if combiner == "single":
        return "baseline"
    w = "competence" if combiner in WEIGHTED else "uniform"
    a = "median" if combiner in ROBUST else "mean"
    return f"{w} + {a}"


def _render(results: dict) -> str:
    out = io.StringIO()
    w = out.write
    m = results["meta"]
    mat = results["matrix"]
    w("# EnsembleKit — offline benchmark results\n\n")
    w(f"Seeds: **{m['seeds']}** · samples/cell: **{m['samples']}** · each cell is the mean "
      "test AUROC of recovering the known Bayes label (chance = 0.5; the Bayes ceiling here "
      "is ~0.80). No trained models, no datasets, no API keys.\n\n")

    w("## AUROC by combiner and regime\n\n")
    w("| combiner | ingredients | homogeneous (control) | het_competence | corrupted |\n")
    w("|---|---|--:|--:|--:|\n")
    for c in COMBINERS:
        r = mat[c]
        w(f"| {c} | {_ingredients(c)} | {r['homogeneous']['mean']:.3f} | "
          f"{r['het_competence']['mean']:.3f} | {r['corrupted']['mean']:.3f} |\n")
    w("\n")

    avg_h = mat["average"]["het_competence"]["mean"]
    rob_h = mat["robust"]["het_competence"]["mean"]
    wgt_h = mat["weighted"]["het_competence"]["mean"]
    full_h = mat["full"]["het_competence"]["mean"]
    avg_c = mat["average"]["corrupted"]["mean"]
    wgt_c = mat["weighted"]["corrupted"]["mean"]
    rob_c = mat["robust"]["corrupted"]["mean"]
    full_c = mat["full"]["corrupted"]["mean"]

    w("## Effect 1 — competence weighting beats heterogeneous competence\n\n")
    w(f"On `het_competence` a single strong learner sits among dead (zero-gain) ones. A "
      f"uniform average drowns it in noise (`average` **{avg_h:.3f}**) and a uniform median "
      f"is even worse — most learners are noise, so the middle one is noise too (`robust` "
      f"**{rob_h:.3f}**). Weighting each learner by its holdout skill recovers the signal "
      f"(`weighted` **{wgt_h:.3f}**, `full` **{full_h:.3f}**).\n\n")

    w("## Effect 2 — robust aggregation beats intermittent corruption\n\n")
    w(f"On `corrupted` every learner is equally competent but emits garbage log-odds on a "
      f"random per-sample fraction. A fixed per-learner weight cannot reject a learner that "
      f"is only *sometimes* broken, so the mean combiners are dragged off (`average` "
      f"**{avg_c:.3f}**, `weighted` **{wgt_c:.3f}**). A per-sample median rejects the "
      f"outliers (`robust` **{rob_c:.3f}**, `full` **{full_c:.3f}**).\n\n")

    w("Only **full** — competence-weighted median — is robust to both regimes; each "
      "single-ingredient combiner collapses on the regime it cannot handle, and `average` "
      "(neither ingredient) collapses on both.\n\n")

    w("## Why ensemble at all? Diversity sweep\n\n")
    w("In the `homogeneous` regime every learner is equally good, so the only thing that "
      "makes the ensemble beat the best single learner is **diversity**. As the learners' "
      "errors become correlated (`rho` -> 1, identical learners), the gain collapses to "
      "zero — the canonical control for why ensembling helps.\n\n")
    w("| error correlation rho | ensemble gain (average - best single) |\n|--:|--:|\n")
    for rho, gain in results["diversity"].items():
        w(f"| {rho:.2f} | {gain:+.3f} |\n")
    w("\n")

    w("## Scrambled-label null (sanity)\n\n")
    w("Shuffle the ground-truth labels and every combiner collapses to chance, so the AUROC "
      "above is not an artefact of the metric.\n\n")
    w("| combiner | null AUROC |\n|---|--:|\n")
    for c in COMBINERS:
        w(f"| {c} | {results['null'][c]:.3f} |\n")
    w("\n> Reproduce: `python -m evals.harness` (writes this file), "
      "`python -m evals.gate` (asserts the dissociation).\n")
    return out.getvalue()


def main() -> None:
    results = compute()
    RESULTS_PATH.write_text(_render(results), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    mat = results["matrix"]
    print(f"{'combiner':12s}{'homogen.':>10s}{'het_comp':>10s}{'corrupt':>10s}")
    for c in COMBINERS:
        r = mat[c]
        print(f"{c:12s}{r['homogeneous']['mean']:10.3f}"
              f"{r['het_competence']['mean']:10.3f}{r['corrupted']['mean']:10.3f}")


if __name__ == "__main__":
    main()
