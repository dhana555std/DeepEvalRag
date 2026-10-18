"""Core data models shared by every layer of the harness.

Flow of data
------------
``GoldenCase`` (dataset row)
    -> stages call the RAG API -> ``StageOutput`` per stage, collected in a ``CaseTrace``
    -> metrics score the trace -> ``MetricResult`` per metric, collected in a ``CaseResult``
    -> a ``RunResult`` holds all cases, per-metric ``Aggregate`` values and the gate verdict
    -> a ``RunReport`` wraps one run (single mode) or many (multi mode) for reporters.

Everything is a pydantic model so results serialise to JSON unchanged; the
JSON reporter, the HTML reporter and any future Allure reporter all consume
the same ``RunReport``.

Extensibility note
------------------
``CaseTrace.outputs`` is keyed by *stage name*, so adding a stage (reranker,
LangGraph node, ...) adds a key; no model changes are needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """One row of the golden dataset."""

    id: str
    question: str
    expected_answer: str
    expected_context: list[str] = Field(default_factory=list)
    expected_context_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    category: str = "general"
    tags: list[str] = Field(default_factory=list)
    safety_probe: bool = False
    # The application should decline (out-of-scope / not in the documents). There is no factual
    # ground truth, so retrieval/generation/triad metrics are skipped; correctness is still judged
    # against ``expected_answer``, which then describes the expected behaviour.
    expect_refusal: bool = False
    # Human-readable origin of the answer (document, section, page). Not used for scoring.
    source_reference: str | None = None
    max_latency_ms: float | None = None


class Chunk(BaseModel):
    """A retrieved passage."""

    id: str | None = None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageOutput(BaseModel):
    """Result of one pipeline stage for one case."""

    stage: str
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    # Stage-specific payload. A stage fills only what it produces.
    chunks: list[Chunk] | None = None
    answer: str | None = None
    citations: list[str] | None = None
    # Passages the stage was *given* as context (used for faithfulness scoring).
    context_used: list[str] | None = None


class CaseTrace(BaseModel):
    """All stage outputs for one case, in execution order."""

    case: GoldenCase
    outputs: dict[str, StageOutput] = Field(default_factory=dict)

    def latest_chunks(self) -> list[Chunk]:
        """Chunks produced by the last stage that produced any (reranker friendly)."""
        for out in reversed(list(self.outputs.values())):
            if out.chunks is not None:
                return out.chunks
        return []


MetricDirection = Literal["min", "max"]


class MetricResult(BaseModel):
    """Score of one metric on one case."""

    key: str  # "<layer>.<name>"
    layer: str
    name: str
    score: float | None = None
    threshold: float | None = None
    direction: MetricDirection = "min"  # "min": score >= threshold passes; "max": score <= threshold
    unit: str = "score"
    passed: bool | None = None  # None = informational (no threshold) or skipped
    skipped: bool = False
    reason: str | None = None
    error: str | None = None


class StageSummary(BaseModel):
    """Compact per-stage info shown in reports."""

    stage: str
    latency_ms: float
    ok: bool
    error: str | None = None
    tokens: int | None = None
    cost_usd: float | None = None


class CaseResult(BaseModel):
    case_id: str
    question: str
    category: str
    expected_answer: str
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    stages: list[StageSummary] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.passed is not False for m in self.metrics)


class Aggregate(BaseModel):
    """A metric aggregated over all cases of one run."""

    key: str
    layer: str
    name: str
    n: int
    errors: int = 0
    mean: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None
    aggregate_used: Literal["mean", "p95"] = "mean"
    value: float | None = None  # the number compared with the threshold
    threshold: float | None = None
    direction: MetricDirection = "min"
    unit: str = "score"
    passed: bool | None = None
    case_pass_rate: float | None = None


class GateResult(BaseModel):
    passed: bool
    failed_metrics: list[str] = Field(default_factory=list)
    errored_metrics: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    judge: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_s: float = 0.0
    cases: list[CaseResult] = Field(default_factory=list)
    aggregates: list[Aggregate] = Field(default_factory=list)
    gate: GateResult = Field(default_factory=lambda: GateResult(passed=True))


class MultiAggregate(BaseModel):
    """A metric aggregated across several runs (judge-variance view)."""

    key: str
    layer: str
    name: str
    judge: str
    runs: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    threshold: float | None = None
    direction: MetricDirection = "min"
    unit: str = "score"
    passed: bool | None = None


class RunReport(BaseModel):
    """Top-level object handed to reporters."""

    title: str
    mode: Literal["single", "multi"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset: str = ""
    dataset_size: int = 0
    judges: list[str] = Field(default_factory=list)
    runs: list[RunResult] = Field(default_factory=list)
    multi: list[MultiAggregate] = Field(default_factory=list)
    passed: bool = True
    deepeval_version: str = ""
    git_sha: str | None = None
