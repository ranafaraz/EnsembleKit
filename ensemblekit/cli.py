"""Command-line interface for EnsembleKit.

Subcommands (all offline, all ASCII output so a Windows cp1252 console is safe):

* ``combine``  -- run one combiner on one regime and print its AUROC.
* ``compare``  -- run every combiner on one regime side by side.
* ``regimes``  -- the full combiner x regime table (the 2x2 dissociation).
* ``diversity``-- the ensemble gain (average - single) as noise correlation rho rises.
* ``eval``     -- run the eval harness (writes evals/RESULTS.md) and print a summary.
"""

from __future__ import annotations

import argparse

from ensemblekit.combiners import COMBINERS
from ensemblekit.config import Settings
from ensemblekit.learners import REGIMES
from ensemblekit.runner import aurocs_by_combiner, diversity_gain, run_settings


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--regime", default="homogeneous", choices=REGIMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=Settings().samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ensemblekit", description="Offline ensemble-combiner benchmark."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("combine", help="run one combiner on one regime")
    pc.add_argument("--combiner", default="full", choices=COMBINERS)
    pc.add_argument("--labels", default="real", choices=["real", "scrambled"])
    pc.add_argument("--rho", type=float, default=0.0)
    _add_common(pc)

    pcmp = sub.add_parser("compare", help="all combiners on one regime")
    _add_common(pcmp)

    sub.add_parser("regimes", help="combiner x regime table (the 2x2 dissociation)")

    pd = sub.add_parser("diversity", help="ensemble gain vs noise correlation rho")
    pd.add_argument("--regime", default="homogeneous", choices=REGIMES)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--samples", type=int, default=Settings().samples)

    sub.add_parser("eval", help="run the eval harness and write evals/RESULTS.md")

    args = parser.parse_args(argv)

    if args.cmd == "combine":
        s = Settings(
            combiner=args.combiner, regime=args.regime, labels=args.labels,
            samples=args.samples, rho=args.rho, seed=args.seed,
        )
        r = run_settings(s)
        print(f"combiner={r.combiner} regime={r.regime} labels={r.labels} "
              f"seed={r.seed} rho={args.rho:g}  AUROC={r.auroc:.3f}")
        return 0

    if args.cmd == "compare":
        table = aurocs_by_combiner(args.regime, args.seed, args.samples)
        print(f"regime={args.regime} seed={args.seed}")
        for c in COMBINERS:
            print(f"  {c:9s} AUROC={table[c]:.3f}")
        return 0

    if args.cmd == "regimes":
        print(f"{'regime':16s} " + " ".join(f"{c:>9s}" for c in COMBINERS))
        for regime in REGIMES:
            row = aurocs_by_combiner(regime, seed=0)
            print(f"{regime:16s} " + " ".join(f"{row[c]:9.3f}" for c in COMBINERS))
        return 0

    if args.cmd == "diversity":
        print(f"regime={args.regime} seed={args.seed}  (gain = average - single)")
        print(f"  {'rho':>5s} {'single':>8s} {'average':>8s} {'gain':>8s}")
        for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
            d = diversity_gain(args.regime, rho, args.seed, args.samples)
            print(f"  {rho:5.2f} {d['single']:8.3f} {d['average']:8.3f} {d['gain']:+8.3f}")
        return 0

    if args.cmd == "eval":
        from evals.harness import main as harness_main

        harness_main()
        return 0

    parser.error("unknown command")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
