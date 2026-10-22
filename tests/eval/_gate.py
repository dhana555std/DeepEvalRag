"""Gate assertion helpers for the live evaluation suite.

``assert_metric_gate(report, key)`` looks up the aggregate of one metric in a ``RunReport``
and asserts it against its configured threshold. In single mode it checks each run's
aggregate; in multi mode it checks the mean across runs per judge (see
``ragqa.evaluation.aggregate``). Errored cases always fail the test (a broken judge or API
must never read as a pass); metrics with no data or no threshold are skipped, not passed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragqa import config as cfgmod
from ragqa.evaluation.aggregate import error_rate_exceeded
from ragqa.models import RunReport


@dataclass
class GateRow:
    label: str
    value: float | None
    threshold: float | None
    direction: str
    passed: bool | None
    errors: int
    total: int = 0  # scored + errored cases behind this row


def gate_rows(report: RunReport, key: str) -> list[GateRow]:
    """One row per run (single mode) or per judge across runs (multi mode)."""
    if report.mode == "single":
        rows = []
        for run in report.runs:
            for a in run.aggregates:
                if a.key == key:
                    rows.append(GateRow(run.judge, a.value, a.threshold, a.direction, a.passed, a.errors, a.n + a.errors))
        return rows
    errors = {(r.judge, a.key): 0 for r in report.runs for a in r.aggregates}
    totals = dict(errors)
    for r in report.runs:
        for a in r.aggregates:
            errors[(r.judge, a.key)] += a.errors
            totals[(r.judge, a.key)] += a.n + a.errors
    return [GateRow(f"{m.judge} (mean of {m.runs} runs)", m.mean, m.threshold, m.direction, m.passed,
                    errors.get((m.judge, m.key), 0), totals.get((m.judge, m.key), 0))
            for m in report.multi if m.key == key]


def assert_metric_gate(report: RunReport, key: str) -> None:
    max_error_rate = cfgmod.load_thresholds().max_error_rate
    rows = gate_rows(report, key)
    if not rows:
        pytest.skip(f"{key}: no applicable cases in the dataset")
    for row in rows:
        assert not error_rate_exceeded(row.errors, row.total, max_error_rate), (
            f"{key} [{row.label}] {row.errors}/{row.total} cases errored "
            f"(> max_error_rate {max_error_rate:.0%}) - see report"
        )
        if row.passed is None:
            if row.threshold is None:
                pytest.skip(f"{key}: informational metric (no threshold) value={row.value}")
            pytest.skip(f"{key}: no scored cases")
        op = ">=" if row.direction == "min" else "<="
        assert row.passed, f"{key} [{row.label}] {row.value:.4f} not {op} {row.threshold}"
