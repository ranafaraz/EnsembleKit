"""CLI smoke tests: each subcommand runs offline and prints ASCII."""

from __future__ import annotations

import pytest

from ensemblekit.cli import main


def _run(capsys, argv):
    rc = main(argv)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.isascii()
    return out


def test_combine(capsys):
    argv = ["combine", "--combiner", "full", "--regime", "corrupted", "--samples", "800"]
    out = _run(capsys, argv)
    assert "AUROC=" in out and "full" in out


def test_combine_scrambled(capsys):
    argv = ["combine", "--combiner", "full", "--labels", "scrambled", "--samples", "800"]
    out = _run(capsys, argv)
    assert "AUROC=" in out


def test_compare(capsys):
    out = _run(capsys, ["compare", "--regime", "het_competence", "--samples", "800"])
    for c in ("single", "average", "weighted", "robust", "full"):
        assert c in out


def test_regimes(capsys):
    out = _run(capsys, ["regimes"])
    for regime in ("homogeneous", "het_competence", "corrupted"):
        assert regime in out


def test_diversity(capsys):
    out = _run(capsys, ["diversity", "--samples", "800"])
    assert "gain" in out and "0.99" in out


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
