"""Report summary: errors are grouped by root cause, breaches are listed readably, HTML is aligned.

The scenario mirrors a real failure: the generation endpoint returns 404, so every metric that needs
the answer errors. The report must show ONE cause (with the affected metrics folded in), not N symptoms.
"""

from __future__ import annotations

from ragqa import config as cfgmod
from ragqa.config import MetricCfg, ThresholdsCfg
from ragqa.evaluation.aggregate import aggregate_run, evaluate_gate
from ragqa.models import CaseResult, MetricResult, RunReport, RunResult, StageSummary
from ragqa.reporting import write_reports
from ragqa.reporting.summary import summarize

ERR_404 = ("POST http://localhost:8080/v1/generate failed: Client error '404 Not Found' for url "
           "'http://localhost:8080/v1/generate'\nFor more information check: https://developer.mozilla.org/x")
KEYS = ["generator.faithfulness", "triad.answer_relevance", "application.correctness", "application.latency_ms"]


def _metric(key, **kw):
    layer, name = key.split(".")
    return MetricResult(key=key, layer=layer, name=name, **kw)


def _report(*, judge_error: bool = False) -> RunReport:
    cases = []
    for cid in ("Q-01", "Q-02"):
        metrics = [_metric("retriever.recall", score=0.2, threshold=0.7, passed=False)]
        metrics += [_metric(k, error=f"StageFailure: stage 'generation' failed: {ERR_404}", passed=False,
                            threshold=0.7) for k in KEYS]
        if judge_error:
            metrics.append(_metric("generator.relevance", error="RateLimitError: 429 Too Many Requests",
                                   passed=False, threshold=0.7))
        cases.append(CaseResult(
            case_id=cid, question="q", category="c", expected_answer="a",
            stages=[StageSummary(stage="retrieval", latency_ms=5, ok=True),
                    StageSummary(stage="generation", latency_ms=0, ok=False, error=ERR_404),
                    StageSummary(stage="generation_golden", latency_ms=0, ok=False,
                                 error="skipped: case has no expected_context")],
            metrics=metrics))
    cfg = ThresholdsCfg(metrics={"retriever.recall": MetricCfg(threshold=0.7),
                                 **{k: MetricCfg(threshold=0.7) for k in KEYS}})
    aggs = aggregate_run(cases, cfg)
    run = RunResult(run_id="groq-1", judge="groq:x", cases=cases, aggregates=aggs,
                    gate=evaluate_gate(aggs, cfg))
    return RunReport(title="T", mode="single", dataset_size=2, judges=["groq:x"], runs=[run], passed=False)


def test_stage_failure_symptoms_fold_into_one_root_cause():
    s = summarize(_report())
    assert len(s.causes) == 1
    cause = s.causes[0]
    assert cause.kind == "api" and cause.title == "generation stage failed: HTTP 404 Not Found"
    assert cause.detail == "POST http://localhost:8080/v1/generate"
    assert cause.cases == 2 and cause.metrics == sorted(KEYS, key=lambda k: (
        ["retriever", "generator", "triad", "application"].index(k.split(".")[0]), k))
    assert "URL and path" in cause.hint
    assert not any("skipped" in c.title for c in s.causes)  # skipped optional stage is not a problem


def test_counters_and_breaches():
    s = summarize(_report())
    assert (s.metrics_total, s.metrics_errored, s.metrics_failed, s.metrics_passed) == (5, 4, 1, 0)
    assert s.cases_failed == 2
    [b] = s.breaches
    assert (b.key, b.value, b.threshold, b.gap) == ("retriever.recall", "0.200", "≥ 0.700", "0.500")


def test_judge_errors_are_a_separate_cause():
    s = summarize(_report(judge_error=True))
    kinds = sorted(c.kind for c in s.causes)
    assert kinds == ["api", "judge"]
    judge = next(c for c in s.causes if c.kind == "judge")
    assert judge.metrics == ["generator.relevance"] and "Lower --concurrency" in judge.hint


def test_html_is_readable_and_aligned(tmp_path):
    write_reports(_report(), cfgmod.ReportingCfg(), tmp_path, ["html"])
    html = (tmp_path / "report.html").read_text()
    assert "What needs attention" in html and "generation stage failed: HTTP 404 Not Found" in html
    assert "Errors in:" not in html and "Below threshold:" not in html  # old run-on sentence is gone
    assert html.count("HTTP 404 Not Found") == 1  # one cause card, not one line per metric
    assert '<th class="num">Value</th>' in html and '<th class="center">Status</th>' in html
    assert "<colgroup>" in html and 'class="badge fail">ERROR' in html
    assert "Not scored: the generation stage failed" in html  # per-case reason is short, not the full URL error
    css = (tmp_path / "report.css").read_text()
    assert "table-layout: fixed" in css and "th.num, td.num" in css
