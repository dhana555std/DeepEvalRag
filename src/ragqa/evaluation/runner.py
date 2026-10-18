"""Evaluation orchestration: single run and multi run.

Entry point
-----------
``run_evaluation(RunOptions) -> RunReport`` (async). The CLI, the pytest suite
and ``deepeval test run`` all call it, so the three entry points cannot drift.

Phases
------
1. *collect*  - call the RAG API for every golden case (``Pipeline.collect``)
2. *score*    - evaluate every enabled metric on every trace using the judge
3. *aggregate*- per-metric aggregates, quality gate, and (multi mode) cross-run statistics

Modes
-----
* single: ``runs == 1`` and one judge  -> one ``RunResult``.
* multi : ``runs > 1`` and/or several judges. The RAG API is called **once** and the
  same traces are re-scored ``runs`` times per judge, which isolates *judge* variance
  cheaply. ``recollect=True`` calls the RAG API again for every run, which measures
  *application* variance as well.

Failure handling: an API/stage failure or a judge error never aborts the run; it is
recorded as a metric ``error`` (counted by the gate when ``fail_on_error`` is set).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import deepeval

from ragqa import config as cfgmod
from ragqa.config import PipelineCfg, ThresholdsCfg
from ragqa.dataset import load_dataset
from ragqa.evaluation.aggregate import aggregate_multi, aggregate_run, compare, evaluate_gate
from ragqa.judges import create_judge
from ragqa.metrics import MetricSpec, enabled_specs
from ragqa.metrics.base import EvalContext, Skip, build_context
from ragqa.models import CaseResult, CaseTrace, MetricResult, RunReport, RunResult, StageSummary
from ragqa.stages import Pipeline


@dataclass
class RunOptions:
    dataset: str = "datasets/golden.jsonl"
    config_dir: Path | None = None
    runs: int = 1
    judges: list[str] | None = None  # None -> default judge
    recollect: bool = False
    only: list[str] | None = None  # fnmatch patterns on "<layer>.<name>"
    limit: int | None = None
    concurrency: int = 4  # concurrent metric evaluations (judge calls)
    title: str | None = None
    extra: dict = field(default_factory=dict)


def _git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:10]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _case_result(trace: CaseTrace, ctx: EvalContext, metrics: list[MetricResult]) -> CaseResult:
    return CaseResult(
        case_id=trace.case.id, question=trace.case.question, category=trace.case.category,
        expected_answer=trace.case.expected_answer, answer=ctx.pipeline.answer or None,
        citations=ctx.pipeline.citations, contexts=[c.text for c in ctx.retrieval.chunks],
        stages=[
            StageSummary(stage=o.stage, latency_ms=o.latency_ms, ok=o.ok, error=o.error,
                         tokens=(o.prompt_tokens or 0) + (o.completion_tokens or 0) or None,
                         cost_usd=o.cost_usd)
            for o in trace.outputs.values()
        ],
        metrics=metrics,
    )


async def _score_metric(spec: MetricSpec, ctx: EvalContext, cfg: ThresholdsCfg,
                        sem: asyncio.Semaphore) -> MetricResult:
    mcfg = cfg.for_key(spec.key)
    res = MetricResult(key=spec.key, layer=spec.layer, name=spec.name, direction=spec.direction,
                       unit=spec.unit, threshold=spec.case_threshold(ctx, mcfg.threshold))
    async with sem:
        try:
            spec.guard(ctx)
            outcome = await spec.evaluate(ctx)
        except Skip as skip:
            res.skipped, res.reason = True, str(skip)
            return res
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"[:500]
            res.passed = False
            return res
    res.score, res.reason = outcome.score, outcome.reason
    if res.threshold is not None:
        res.passed = compare(outcome.score, res.threshold, spec.direction)
    return res


async def score_traces(traces: list[CaseTrace], specs: list[MetricSpec], judge, pipeline: PipelineCfg,
                       thresholds: ThresholdsCfg, concurrency: int) -> list[CaseResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    contexts = [build_context(t, pipeline, judge) for t in traces]
    tasks = [[_score_metric(s, ctx, thresholds, sem) for s in specs] for ctx in contexts]
    results = await asyncio.gather(*(asyncio.gather(*row) for row in tasks))
    return [_case_result(t, ctx, list(r)) for t, ctx, r in zip(traces, contexts, results)]


async def run_evaluation(opts: RunOptions) -> RunReport:
    cdir = opts.config_dir
    rag_cfg = cfgmod.load_rag_api(cdir)
    thresholds = cfgmod.load_thresholds(cdir)
    judges_cfg = cfgmod.load_judges(cdir)
    reporting = cfgmod.load_reporting(cdir)

    cases, _warnings = load_dataset(opts.dataset)
    if opts.limit:
        cases = cases[: opts.limit]

    specs = enabled_specs(thresholds, opts.only)
    if not specs:
        raise ValueError("no metrics selected: check thresholds.yaml / --only")

    judge_names = opts.judges or [os.environ.get("JUDGE_PROVIDER") or judges_cfg.default]
    pipeline = Pipeline(rag_cfg)
    traces = await pipeline.collect(cases)

    runs: list[RunResult] = []
    first = True
    for jname in judge_names:
        judge = create_judge(jname, judges_cfg) if any(s.needs_judge for s in specs) else None
        label = judge.get_model_name() if judge else "none (no LLM metrics selected)"
        for i in range(opts.runs):
            if opts.recollect and not first:
                traces = await pipeline.collect(cases)
            first = False
            started = time.perf_counter()
            results = await score_traces(traces, specs, judge, rag_cfg.pipeline, thresholds,
                                         opts.concurrency)
            aggregates = aggregate_run(results, thresholds)
            runs.append(RunResult(
                run_id=f"{jname}-{i + 1}", judge=label, cases=results, aggregates=aggregates,
                duration_s=time.perf_counter() - started, gate=evaluate_gate(aggregates, thresholds),
            ))

    multi_mode = opts.runs > 1 or len(judge_names) > 1
    report = RunReport(
        title=opts.title or reporting.title, mode="multi" if multi_mode else "single",
        dataset=str(opts.dataset), dataset_size=len(cases), judges=sorted({r.judge for r in runs}),
        runs=runs, deepeval_version=deepeval.__version__, git_sha=_git_sha(),
    )
    if multi_mode:
        report.multi = aggregate_multi(runs, thresholds)
        errored = any(r.gate.errored_metrics for r in runs)
        report.passed = not errored and all(m.passed is not False for m in report.multi)
    else:
        report.passed = runs[0].gate.passed
    return report


def run_sync(opts: RunOptions) -> RunReport:
    """Blocking wrapper for scripts and pytest fixtures."""
    return asyncio.run(run_evaluation(opts))
