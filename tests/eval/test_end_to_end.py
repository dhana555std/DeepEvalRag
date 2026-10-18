"""Whole-application gate: the evaluation as a single pass/fail verdict.

Passes only when every gated metric of every layer meets its threshold and nothing errored
(single run), or when the cross-run mean of every metric does (multi run). This is the test
CI relies on for the overall quality gate; the per-layer files pinpoint what failed.
"""

import pytest

pytestmark = pytest.mark.eval


def test_no_stage_or_judge_errors(report):
    errored = sorted({a.key for r in report.runs for a in r.aggregates if a.errors})
    assert not errored, f"metrics with errors (RAG API or judge failures): {errored}"


def test_overall_quality_gate(report):
    if report.mode == "single":
        gate = report.runs[0].gate
        assert gate.passed, f"failed: {gate.failed_metrics}; errored: {gate.errored_metrics}"
    else:
        failed = [f"{m.judge}:{m.key}" for m in report.multi if m.passed is False]
        assert report.passed, f"failed: {failed}"
