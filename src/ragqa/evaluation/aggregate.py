"""Aggregation and quality-gate logic (pure functions, no I/O).

Single run
----------
``aggregate_run`` groups ``MetricResult`` by key over all cases and produces an
``Aggregate`` with mean / p95 / min / max. The value compared with the
threshold is the mean by default or p95 when the metric is configured with
``aggregate: p95`` (use p95 for latency). ``direction`` decides the comparison:
``min`` passes when value >= threshold, ``max`` when value <= threshold.
Metrics without a threshold are informational and never gate.

``evaluate_gate`` fails the run if any gated aggregate fails, or - when
``fail_on_error`` is set - if any metric produced errors (an erroring judge or
API call must not look like a pass).

Multi run
---------
``aggregate_multi`` takes the per-run aggregate values of a metric and reports
mean / standard deviation / min / max across runs per judge; it is the judge
variance view, and the multi-run gate compares the cross-run *mean*.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from ragqa.config import ThresholdsCfg
from ragqa.models import Aggregate, CaseResult, GateResult, MultiAggregate, RunResult

LAYER_ORDER = ["retriever", "generator", "triad", "application"]


def layer_sort_key(layer: str) -> int:
    return LAYER_ORDER.index(layer) if layer in LAYER_ORDER else len(LAYER_ORDER)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


_EPS = 1e-9  # a mean of 0.7, 0.9, 0.8 is 0.7999999999999999: do not fail a 0.8 gate on float noise


def compare(value: float, threshold: float, direction: str) -> bool:
    return value >= threshold - _EPS if direction == "min" else value <= threshold + _EPS


def aggregate_run(cases: list[CaseResult], cfg: ThresholdsCfg) -> list[Aggregate]:
    groups: dict[str, list] = defaultdict(list)
    for case in cases:
        for m in case.metrics:
            groups[m.key].append(m)

    out: list[Aggregate] = []
    for key, results in groups.items():
        first = results[0]
        mcfg = cfg.for_key(key)
        scored = [r for r in results if r.score is not None and r.error is None and not r.skipped]
        scores = [r.score for r in scored]  # type: ignore[misc]
        agg = Aggregate(
            key=key, layer=first.layer, name=first.name, n=len(scored),
            errors=sum(1 for r in results if r.error), threshold=mcfg.threshold,
            direction=first.direction, unit=first.unit,
            aggregate_used="p95" if mcfg.aggregate == "p95" else "mean",
        )
        if scores:
            agg.mean = statistics.fmean(scores)
            agg.p95 = percentile(scores, 95)
            agg.min, agg.max = min(scores), max(scores)
            agg.value = agg.p95 if agg.aggregate_used == "p95" else agg.mean
            judged = [r for r in scored if r.passed is not None]
            if judged:
                agg.case_pass_rate = sum(1 for r in judged if r.passed) / len(judged)
            if mcfg.threshold is not None:
                agg.passed = compare(agg.value, mcfg.threshold, agg.direction)
        out.append(agg)
    return sorted(out, key=lambda a: (layer_sort_key(a.layer), a.key))


def evaluate_gate(aggregates: list[Aggregate], cfg: ThresholdsCfg) -> GateResult:
    failed = [a.key for a in aggregates if a.passed is False]
    errored = [a.key for a in aggregates if a.errors] if cfg.fail_on_error else []
    return GateResult(passed=not failed and not errored, failed_metrics=failed, errored_metrics=errored)


def aggregate_multi(runs: list[RunResult], cfg: ThresholdsCfg) -> list[MultiAggregate]:
    groups: dict[tuple[str, str], list[Aggregate]] = defaultdict(list)
    for run in runs:
        for a in run.aggregates:
            groups[(run.judge, a.key)].append(a)

    out: list[MultiAggregate] = []
    for (judge, key), aggs in groups.items():
        values = [a.value for a in aggs if a.value is not None]
        first = aggs[0]
        m = MultiAggregate(
            key=key, layer=first.layer, name=first.name, judge=judge, runs=len(aggs),
            threshold=first.threshold, direction=first.direction, unit=first.unit,
        )
        if values:
            m.mean = statistics.fmean(values)
            m.std = statistics.pstdev(values) if len(values) > 1 else 0.0
            m.min, m.max = min(values), max(values)
            if m.threshold is not None:
                m.passed = compare(m.mean, m.threshold, m.direction)
        out.append(m)
    return sorted(out, key=lambda a: (a.judge, layer_sort_key(a.layer), a.key))
