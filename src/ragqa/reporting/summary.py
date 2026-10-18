"""Report summary view-model: turns a ``RunReport`` into a readable "what happened" digest.

Why it exists
-------------
One broken dependency (say the generation endpoint returning 404) makes every downstream metric
error. Listing those metrics one by one hides the single cause. This module groups the problems by
**root cause** and separates them from genuine quality misses:

* ``Cause``   - a pipeline/API or judge problem, with how many cases / metrics it affected and a
                hint on the likely fix. Metric errors that merely say "stage X failed" are folded
                into the stage's own cause instead of being listed again.
* ``Breach``  - a gated metric whose value missed its threshold (value, threshold, gap).
* ``Summary`` - verdict, counters (metrics pass / fail / error / informational, cases failed)
                and the two lists above.

It is pure data (no HTML), so any reporter - HTML today, Allure or a dashboard feed later - can use it.
Scope: one evaluation. It deliberately has no notion of history or trends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragqa.evaluation.aggregate import layer_sort_key
from ragqa.models import Aggregate, MultiAggregate, RunReport

LAYER_TITLES = {"retriever": "Retriever", "generator": "Generator", "triad": "RAG Triad",
                "application": "Application"}

_STAGE_FAILED = re.compile(r"stage '([^']+)' failed")
_ENDPOINT = re.compile(r"\b(GET|POST|PUT)\s+(\S+)\s+failed:")
_HTTP_REASON = re.compile(r"(?:Client|Server) error '(\d{3}[^']*)'")

_API_HINTS: list[tuple[str, str]] = [
    (r"url is empty", "Set the endpoint URL in .env (see .env.example)."),
    (r"\b404\b", "Endpoint not found. Check the URL and path in .env / config/rag_api.yaml and that the service serves it."),
    (r"\b40[13]\b", "Authentication failed. Check the *_AUTH_TYPE and *_API_TOKEN settings."),
    (r"\b429\b", "The service is rate limiting. Lower pipeline.concurrency in config/rag_api.yaml."),
    (r"\b5\d\d\b", "Server error inside the RAG service. Check its logs."),
    (r"(?i)connect|refused|timed? ?out|timeout|name or service", "Service unreachable or too slow. Check it is running, the URL/port, and RAG_TIMEOUT_S."),
]
_JUDGE_HINTS: list[tuple[str, str]] = [
    (r"(?i)api[_ ]?key|authenticat|\b401\b|credentials", "Check the judge's API key / credentials."),
    (r"(?i)\b429\b|rate.?limit|throttl", "The judge is rate limiting. Lower --concurrency."),
    (r"(?i)invalid json", "The judge returned invalid JSON. Use a stronger judge model or raise max_tokens."),
]


@dataclass
class Cause:
    kind: str  # "api" | "judge"
    title: str
    detail: str
    hint: str
    cases: int = 0
    metrics: list[str] = field(default_factory=list)


@dataclass
class Breach:
    key: str
    layer: str
    judge: str | None
    value: str
    threshold: str
    gap: str


@dataclass
class Summary:
    passed: bool
    metrics_total: int = 0
    metrics_passed: int = 0
    metrics_failed: int = 0
    metrics_errored: int = 0
    metrics_info: int = 0
    cases_total: int = 0
    cases_failed: int = 0
    causes: list[Cause] = field(default_factory=list)
    breaches: list[Breach] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.causes or self.breaches)


def fmt_value(value: float | None, unit: str) -> str:
    if value is None:
        return "–"
    if unit == "ms":
        return f"{value:,.0f} ms"
    if unit == "tokens":
        return f"{value:,.0f}"
    if unit == "usd":
        return f"${value:.5f}"
    return f"{value:.3f}"


def _hint(patterns: list[tuple[str, str]], text: str, default: str) -> str:
    return next((hint for pat, hint in patterns if re.search(pat, text)), default)


def _first_line(text: str, limit: int = 220) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _api_cause(stage: str, error: str) -> Cause:
    line = _first_line(error, 400)
    endpoint = _ENDPOINT.search(line)
    reason = _HTTP_REASON.search(line)
    if reason:
        title = f"{stage} stage failed: HTTP {reason.group(1)}"
    else:
        tail = line.split("failed:", 1)[-1].strip() if "failed:" in line else line
        title = f"{stage} stage failed: {_first_line(tail, 120)}"
    detail = f"{endpoint.group(1)} {endpoint.group(2)}" if endpoint else _first_line(error)
    return Cause("api", title, detail, _hint(_API_HINTS, error, "Check the RAG service and config/rag_api.yaml."))


def _judge_cause(error: str) -> Cause:
    line = _first_line(error)
    return Cause("judge", f"Judge call failed: {line}", "",
                 _hint(_JUDGE_HINTS, error, "Check the judge model name, credentials and quota."))


def _collect_causes(report: RunReport) -> list[Cause]:
    """Group stage failures and judge errors by root cause across all runs of the report."""
    api: dict[tuple[str, str], tuple[Cause, set[str], set[str]]] = {}
    judge: dict[str, tuple[Cause, set[str], set[str]]] = {}

    for run in report.runs:
        for case in run.cases:
            for st in case.stages:
                if not st.ok and st.error and not st.error.startswith("skipped:"):
                    key = (st.stage, _first_line(st.error, 400))
                    cause, cases, metrics = api.setdefault(key, (_api_cause(st.stage, st.error), set(), set()))
                    cases.add(case.case_id)
            for m in case.metrics:
                if not m.error:
                    continue
                folded = _STAGE_FAILED.search(m.error)
                if folded:  # symptom of a stage failure: attribute it to that stage's cause
                    for (stage, _), (cause, cases, metrics) in api.items():
                        if stage == folded.group(1) and case.case_id in cases:
                            metrics.add(m.key)
                            break
                else:
                    msg = re.sub(r"^\w+(?:\.\w+)*Error: |^\w+Error: ", "", m.error)
                    cause, cases, metrics = judge.setdefault(_first_line(m.error), (_judge_cause(msg), set(), set()))
                    cases.add(case.case_id)
                    metrics.add(m.key)

    causes: list[Cause] = []
    for cause, cases, metrics in [*api.values(), *judge.values()]:
        cause.cases, cause.metrics = len(cases), sorted(metrics, key=lambda k: (layer_sort_key(k.split(".")[0]), k))
        causes.append(cause)
    return causes


def _breach(a: Aggregate | MultiAggregate, judge: str | None) -> Breach:
    value = a.value if isinstance(a, Aggregate) else a.mean
    op = "≥" if a.direction == "min" else "≤"
    gap = abs(value - a.threshold) if value is not None and a.threshold is not None else None
    return Breach(a.key, a.layer, judge, fmt_value(value, a.unit), f"{op} {fmt_value(a.threshold, a.unit)}",
                  fmt_value(gap, a.unit) if gap is not None else "–")


def summarize(report: RunReport) -> Summary:
    """Build the digest for a single-run or multi-run report."""
    errored_keys = {a.key for run in report.runs for a in run.aggregates if a.errors}
    s = Summary(passed=report.passed, cases_total=report.dataset_size)
    s.causes = _collect_causes(report)

    if report.mode == "single":
        rows = [(a, None) for a in report.runs[0].aggregates]
        s.cases_failed = sum(1 for c in report.runs[0].cases if not c.passed)
        for a, _ in rows:
            if a.errors:
                s.metrics_errored += 1
            elif a.passed is True:
                s.metrics_passed += 1
            elif a.passed is False:
                s.metrics_failed += 1
            else:
                s.metrics_info += 1
        s.metrics_total = len(rows)
        s.breaches = [_breach(a, None) for a, _ in rows if a.passed is False and not a.errors]
    else:
        failed_cases = {c.case_id for run in report.runs for c in run.cases if not c.passed}
        s.cases_failed = len(failed_cases)
        for m in report.multi:
            if m.key in errored_keys:
                s.metrics_errored += 1
            elif m.passed is True:
                s.metrics_passed += 1
            elif m.passed is False:
                s.metrics_failed += 1
            else:
                s.metrics_info += 1
        s.metrics_total = len(report.multi)
        s.breaches = [_breach(m, m.judge) for m in report.multi if m.passed is False and m.key not in errored_keys]

    s.breaches.sort(key=lambda b: (layer_sort_key(b.layer), b.key))
    return s
