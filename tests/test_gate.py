"""The eval gate passes end to end (the published dissociation is reproducible)."""

from __future__ import annotations

from evals.gate import run_checks


def test_gate_all_checks_pass():
    checks = run_checks()
    failures = [msg for ok, msg in checks if not ok]
    assert not failures, "gate failures:\n" + "\n".join(failures)


def test_gate_has_expected_check_count():
    # 4 control + 6 effect-1 + 6 effect-2 + 3 full + 2 gaps + 4 diversity + 5 null
    assert len(run_checks()) == 30
