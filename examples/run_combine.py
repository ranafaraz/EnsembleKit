"""Minimal example: watch the 2x2 dissociation across three regimes.

    python examples/run_combine.py

Prints, for each regime, the AUROC of every combiner at recovering the known Bayes label --
so you can see the uniform-weight combiners collapse on `het_competence`, the mean combiners
collapse on `corrupted`, and `full` (competence-weighted median) stay robust on both. The
last block shows the ensemble's gain over the best single learner vanishing as the learners'
errors become correlated.
"""

from __future__ import annotations

from ensemblekit.combiners import COMBINERS
from ensemblekit.learners import REGIMES
from ensemblekit.runner import aurocs_by_combiner, diversity_gain


def main() -> None:
    print(f"{'combiner':10s}" + "".join(f"{r:>16s}" for r in REGIMES))
    print("-" * (10 + 16 * len(REGIMES)))
    tables = {r: aurocs_by_combiner(r, seed=0) for r in REGIMES}
    for c in COMBINERS:
        cells = "".join(f"{tables[r][c]:16.3f}" for r in REGIMES)
        print(f"{c:10s}{cells}")

    print("\n* uniform combiners (average, robust) drop on 'het_competence';")
    print("  mean combiners (average, weighted) drop on 'corrupted'; full stays robust on both.\n")

    print("Diversity: ensemble gain (average - best single) vs error correlation rho")
    for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
        d = diversity_gain("homogeneous", rho, seed=0)
        print(f"  rho={rho:4.2f}  gain={d['gain']:+.3f}")
    print("  -> gain collapses to ~0 as learners become identical (no diversity).")


if __name__ == "__main__":
    main()
