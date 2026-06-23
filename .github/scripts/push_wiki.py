"""Initialize the EnsembleKit GitHub wiki with 5 pages."""
import os
import subprocess
import tempfile
import textwrap

PAGES = {
    "Home.md": textwrap.dedent("""\
        # EnsembleKit

        [![CI](https://github.com/ranafaraz/EnsembleKit/actions/workflows/ci.yml/badge.svg)](https://github.com/ranafaraz/EnsembleKit/actions/workflows/ci.yml)
        [![Live demo](https://img.shields.io/badge/live%20demo-ensemblekit.dexdevs.com-brightgreen?logo=rocket)](https://ensemblekit.dexdevs.com)
        [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ranafaraz/EnsembleKit/blob/main/LICENSE)

        **A benchmark for when ensembling helps and which combining trick buys which kind of robustness.**

        EnsembleKit synthesizes base-learner predictions as log-odds of a *known* Bayes label, then
        scores four combiners on how well they recover it. The result is a clean 2x2 dissociation:
        **competence weighting and robust aggregation each buy robustness to a different failure mode,
        and you need both** to be robust everywhere.

        - Heterogeneous learner competence (dead learners) breaks a uniform average; competence weighting fixes it.
        - Intermittent per-sample corruption breaks any fixed-weight combiner; robust aggregation (median) fixes it.
        - A diversity sweep shows ensemble gain collapses to zero as learner correlation rho approaches 1.
        - A scrambled-label null confirms every AUROC signal is real.
        - No models to train, no datasets, no API keys -- numpy only.

        ## Quick start

        ```bash
        pip install -e ".[dev]"
        ensemblekit compare --regime het_competence
        ensemblekit compare --regime corrupted
        ensemblekit diversity
        python -m evals.harness
        pytest -q
        ```

        ## Wiki pages

        - [[Architecture]] -- log-odds formulation, base learner synthesis, combiner implementations, 2x2 design
        - [[Evaluation]] -- benchmark setup, results table, diversity sweep
        - [[Configuration]] -- env vars, .env.example
        - [[Development]] -- setup, tests, how to add a combiner or learner regime
    """),

    "Architecture.md": textwrap.dedent("""\
        # Architecture

        ## Overview

        EnsembleKit is structured around three concerns: **base learner synthesis** (produce predictions
        with a known Bayes answer), **combining** (fuse predictions into one score), and **evaluation**
        (recover the Bayes label and assert the dissociation). The combiners are textbook; what is
        designed is the experiment.

        ## Log-odds formulation

        A latent score `s ~ N(0, 1)` is the Bayes log-odds. The label is `y ~ Bernoulli(sigmoid(BETA * s))`.
        Each base learner reports: `z_k = a_k * s + noise_k` where `a_k` is competence.

        **Why log-odds?**
        - Combination is exact in log-odds space (not approximate).
        - The Bayes optimal combiner under conditional independence is the sum of individual log-odds.
        - A uniform average is optimal when all learners have equal competence.
        - Competence weighting re-weights the log-likelihood contributions.
        - The Bayes ceiling is exact and computable from `BETA` and `s`.

        ## Base learner synthesis

        - `a_k = 1`: learner perfectly tracks `s` (maximum competence).
        - `a_k = 0`: "dead learner" -- pure noise, zero signal.
        - A regime changes the competence vector `{a_k}` and noise structure, never the Bayes label.

        ## Regimes

        | Regime | Learner construction | What fails |
        |---|---|---|
        | homogeneous | All `a_k` equal; iid noise | Control -- tests diversity gain |
        | het_competence | One strong `a_k`; rest `a_k = 0` | Uniform combiners dilute the strong learner |
        | corrupted | All `a_k` equal; per-sample garbage on random fraction | Fixed-weight combiners can't reject intermittent corruption |

        ## Combiner implementations

        2x2: **weighting** x **aggregation**

        **Uniform weighting.** Each learner weighted equally. Optimal under equal competence;
        dilutes signal when competences differ.

        **Competence weighting.** Weights proportional to holdout ranking skill. Static per-learner
        weight -- cannot reject a learner that is garbage on only some samples.

        **Mean aggregation.** Average (weighted or not) log-odds across learners. Cannot reject
        intermittent per-sample corruption with fixed weights.

        **Median aggregation (robust).** Per-sample median across learners. Rejects outlier learners
        on each individual sample regardless of their average quality.

        | Combiner | Weighting | Aggregation | Dead learners | Per-sample corruption |
        |---|---|---|---|---|
        | average | uniform | mean | no | no |
        | weighted | competence | mean | yes | no |
        | robust | uniform | median | no | yes |
        | full | competence | median | yes | yes |

        ## 2x2 dissociation design

        The axes are genuinely orthogonal:
        - Competence weighting is a static per-learner weight -- cannot reject a learner that is garbage on only some samples.
        - A per-sample median has no concept of learner quality.
        - Neither can do the other's job.

        ## Diversity sweep

        Vary inter-learner error correlation `rho` from 0 to 1. As `rho -> 1` (identical learners),
        ensemble gain (ensemble AUROC minus best-single AUROC) collapses to zero. This is the
        canonical demonstration that diversity is the prerequisite for ensembling.

        ## Module layout

        ```
        ensemblekit/
          synthesis/    -- Bayes label generator, base learner factory
          combiners/    -- average, weighted, robust, full, single
          regimes/      -- homogeneous, het_competence, corrupted
          eval/         -- auroc, gate, diversity
          cli.py        -- CLI entry point
        evals/          -- harness.py, gate.py
        tests/          -- 76 pytest tests
        ```
    """),

    "Evaluation.md": textwrap.dedent("""\
        # Evaluation

        ## Benchmark setup

        - **Samples:** 1600 per cell (combiner x regime), 16 random seeds; results are mean AUROC.
        - **Ground truth:** Bayes label `y ~ Bernoulli(sigmoid(BETA * s))` -- exact, no ambiguity.
        - **Bayes ceiling:** ~0.80 AUROC (set by `BETA`).
        - **Metric:** AUROC. 1.0 = perfect recovery, 0.5 = chance.
        - **Null:** scrambled Bayes labels -- every combiner must fall to ~0.50.

        ## Regimes

        | Regime | Description |
        |---|---|
        | homogeneous | All learners equally competent, iid noise (control) |
        | het_competence | One strong learner, rest are dead (a_k = 0) |
        | corrupted | All competent on average; each learner has per-sample garbage on a random fraction |

        ## Results

        Mean AUROC over 16 seeds, 1600 samples/cell.

        | combiner | ingredients | homogeneous | het_competence | corrupted |
        |---|---|--:|--:|--:|
        | single | baseline (best individual) | 0.759 | 0.789 | 0.635 |
        | average | uniform + mean | 0.801 | 0.713 | 0.668 |
        | weighted | competence + mean | 0.801 | 0.789 | 0.668 |
        | robust | uniform + median | 0.797 | 0.586 | 0.770 |
        | **full** | **competence + median** | **0.794** | **0.789** | **0.766** |

        ## Key findings

        **Effect 1 -- competence weighting beats heterogeneous competence.**
        Uniform combiners drop on `het_competence` (average 0.801->0.713, robust 0.797->0.586).
        Competence-weighted combiners stay at 0.789.

        **Effect 2 -- robust aggregation beats intermittent corruption.**
        Mean combiners drop on `corrupted` (average 0.801->0.668, weighted 0.801->0.668).
        Median combiners stay at ~0.77.

        Only `full` (both ingredients) is robust everywhere.

        ## Diversity sweep

        | error correlation rho | 0.00 | 0.30 | 0.60 | 0.90 | 0.99 |
        |---|--:|--:|--:|--:|--:|
        | gain (average - best single) | +0.041 | +0.029 | +0.015 | +0.004 | +0.001 |

        ## Reproducing the results

        ```bash
        python -m evals.harness     # full table (writes evals/RESULTS.md)
        python -m evals.gate        # CI dissociation gate
        ensemblekit compare --regime het_competence
        ensemblekit compare --regime corrupted
        ensemblekit diversity
        ensemblekit regimes
        ```
    """),

    "Configuration.md": textwrap.dedent("""\
        # Configuration

        All configuration via environment variables. Defaults reproduce the published benchmark.

        ## Environment variables

        | Variable | Default | Description |
        |---|---|---|
        | `ENSEMBLEKIT_COMBINER` | `full` | Combiner: `average`, `weighted`, `robust`, `full`, `single` |
        | `ENSEMBLEKIT_REGIME` | `homogeneous` | Regime: `homogeneous`, `het_competence`, `corrupted` |
        | `ENSEMBLEKIT_LABELS` | *(auto)* | Override Bayes label file path (optional) |
        | `ENSEMBLEKIT_SAMPLES` | `1600` | Number of samples per run |
        | `ENSEMBLEKIT_RHO` | `0.0` | Inter-learner error correlation for diversity sweep (0.0 to 1.0) |
        | `ENSEMBLEKIT_SEED` | `42` | Base random seed |
        | `ENSEMBLEKIT_BACKEND` | `numpy` | Compute backend (numpy = offline) |

        ## .env.example

        ```dotenv
        ENSEMBLEKIT_COMBINER=full
        ENSEMBLEKIT_REGIME=homogeneous
        ENSEMBLEKIT_SAMPLES=1600
        ENSEMBLEKIT_RHO=0.0
        ENSEMBLEKIT_SEED=42
        ENSEMBLEKIT_BACKEND=numpy
        ```

        ## Usage

        ```bash
        # Compare all combiners on one regime
        ensemblekit compare --regime het_competence

        # Single combiner on a regime
        ensemblekit compare --combiner weighted --regime corrupted

        # Diversity sweep
        ensemblekit diversity

        # Full combiner x regime table
        ensemblekit regimes
        ```

        ## Reproducibility

        Fix `ENSEMBLEKIT_SEED` and `ENSEMBLEKIT_SAMPLES` to reproduce exact numbers. The published
        table uses `SEED=42`, `SAMPLES=1600`, iterated over 16 seeds (42 through 57). The harness
        handles this automatically.
    """),

    "Development.md": textwrap.dedent("""\
        # Development

        ## Setup

        ```bash
        git clone https://github.com/ranafaraz/EnsembleKit.git
        cd EnsembleKit
        python -m venv .venv
        source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
        pip install -e ".[dev]"
        pytest -q   # 76 tests
        ```

        Optional sklearn cross-check: `pip install -e ".[sklearn]"`

        ## Running the benchmark

        ```bash
        python -m evals.harness   # full table + RESULTS.md
        python -m evals.gate      # dissociation gate (used in CI)
        ensemblekit compare --regime het_competence
        ensemblekit diversity
        ensemblekit regimes
        ```

        ## Repository structure

        ```
        EnsembleKit/
          ensemblekit/
            synthesis/    -- Bayes label generator, base learner factory (z_k = a_k * s + noise)
            combiners/    -- average.py, weighted.py, robust.py, full.py, single.py
            regimes/      -- homogeneous.py, het_competence.py, corrupted.py
            eval/         -- auroc.py, gate.py, diversity.py
            cli.py        -- ensemblekit CLI entry point
          evals/          -- harness.py, gate.py
          tests/          -- 76 pytest tests
          docs/           -- ARCHITECTURE.md, DECISIONS.md, demo.gif
        ```

        ## Adding a combiner

        1. Create `ensemblekit/combiners/my_combiner.py` with `combine(log_odds, labels, rng) -> np.ndarray`.
           The function takes a `(K, N)` array of per-learner log-odds and returns a `(N,)` combined score.
        2. Register in `ensemblekit/combiners/__init__.py` under a string key.
        3. Add tests in `tests/test_combiners.py`: output shape, AUROC ~1.0 on a perfect learner, deterministic.
        4. Verify with `ensemblekit regimes`.
        5. Add key to `ENSEMBLEKIT_COMBINER` accepted values in Configuration.

        ## Adding a regime

        1. Create `ensemblekit/regimes/my_regime.py` with `build_learners(s, y, rng) -> (log_odds, competence_hint)`.
        2. Register in `ensemblekit/regimes/__init__.py` and add description.
        3. Tests: regime must not change the Bayes label `y`.

        ## CI

        Runs `pytest -q` + `python -m evals.gate` on Python 3.10, 3.11, 3.12. No secrets needed.

        ## Code style

        `black` + `ruff`. All random state via `np.random.default_rng(seed)`. Combiners are stateless:
        they receive all data as arguments and return a score array.
    """),
}

import json
import urllib.request

TOKEN = os.environ["GH_WIKI_TOKEN"]
REPO = os.environ.get("GITHUB_REPOSITORY", "ranafaraz/EnsembleKit")


def gh_api(method, path, body=None):
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# Step 1: try to create the Home page via REST API (initializes the wiki repo)
print("Creating Home page via REST API to bootstrap wiki repo...")
status, resp = gh_api("POST", f"/repos/{REPO}/wiki/pages",
                       {"title": "Home", "content": PAGES["Home.md"]})
print(f"  REST API status: {status} -> {resp.get('message', 'ok')}")

if status in (200, 201):
    # Home created via REST; create remaining pages
    for filename, content in PAGES.items():
        if filename == "Home.md":
            continue
        title = filename.removesuffix(".md")
        status2, resp2 = gh_api("POST", f"/repos/{REPO}/wiki/pages",
                                 {"title": title, "content": content})
        print(f"  Created {title}: {status2} -> {resp2.get('message', 'ok')}")
    print("Done via REST API.")
else:
    # REST API failed; fall back to git push
    print("REST API unavailable, falling back to git push...")
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_url = f"https://x-access-token:{TOKEN}@github.com/{REPO}.wiki.git"
        wiki_dir = os.path.join(tmpdir, "wiki")

        result = subprocess.run(["git", "clone", wiki_url, wiki_dir],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("Wiki repo doesn't exist yet, initializing fresh...")
            os.makedirs(wiki_dir)
            subprocess.run(["git", "init"], cwd=wiki_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", wiki_url],
                           cwd=wiki_dir, check=True)

        for filename, content in PAGES.items():
            path = os.path.join(wiki_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Wrote {filename}")

        subprocess.run(["git", "config", "user.name", "Rana Faraz"],
                       cwd=wiki_dir, check=True)
        subprocess.run(["git", "config", "user.email", "faraz.ahmed@iub.edu.pk"],
                       cwd=wiki_dir, check=True)
        subprocess.run(["git", "add", "."], cwd=wiki_dir, check=True)

        result = subprocess.run(
            ["git", "-c", "user.name=Rana Faraz",
             "-c", "user.email=faraz.ahmed@iub.edu.pk",
             "commit", "-m", "docs: initialize project wiki"],
            cwd=wiki_dir, capture_output=True, text=True
        )
        print(result.stdout)
        for branch in ["master", "main"]:
            push = subprocess.run(
                ["git", "push", "-u", "origin", f"HEAD:{branch}"],
                cwd=wiki_dir, capture_output=True, text=True
            )
            if push.returncode == 0:
                print(f"Pushed to {branch} successfully!")
                break
            else:
                print(f"Push to {branch} failed: {push.stderr}")
    print("Done.")
