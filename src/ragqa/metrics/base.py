"""Metric abstractions: ``MetricSpec``, ``DeepEvalSpec`` and the evaluation context.

Concepts
--------
* ``MetricSpec``  - one named score in one layer (key ``"<layer>.<name>"``). It
  knows which *view* of the trace it reads, how to decide it is not applicable
  to a case, and how to compute a score.
* ``DeepEvalSpec``- a ``MetricSpec`` backed by a DeepEval metric (LLM judged).
* Views           - ``retrieval`` (what the retriever returned), ``pipeline`` (the
  final answer of the chained pipeline) and ``generator`` (the isolated
  generator output when ``pipeline.generator_isolated_stage`` is configured,
  otherwise the same as ``pipeline``).
* Caching         - identical DeepEval computations (e.g. faithfulness, which is
  both a Generator and a Triad metric) are evaluated once per case per stage and
  reused, so shared metrics do not cost twice.

How to add a metric (open-closed)
---------------------------------
Create a class in ``ragqa/metrics/<layer>.py`` decorated with
``@metric_specs.register("<layer>.<name>")``, then enable it in
``config/thresholds.yaml``. ``ragqa.metrics.__init__`` auto-imports every
module, so no existing file changes.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ragqa.config import PipelineCfg
from ragqa.models import CaseTrace, Chunk, GoldenCase, StageOutput
from ragqa.registry import Registry

View = Literal["retrieval", "pipeline", "generator", "none"]


def has_no_ground_truth(case: GoldenCase) -> bool:
    """Safety probes and expected refusals have no factual answer to score retrieval/generation against."""
    return case.safety_probe or case.expect_refusal


class Skip(Exception):
    """Raised by a spec when the metric does not apply to the case (not a failure)."""


class StageFailure(Exception):
    """Raised when a stage the metric depends on failed (counted as a metric error)."""


@dataclass
class Outcome:
    score: float
    reason: str | None = None


@dataclass
class RetrievalView:
    stage: str
    output: StageOutput | None
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class AnswerView:
    stage: str
    output: StageOutput | None

    @property
    def answer(self) -> str:
        return (self.output.answer if self.output else None) or ""

    @property
    def citations(self) -> list[str]:
        return (self.output.citations if self.output else None) or []

    @property
    def contexts(self) -> list[str]:
        return (self.output.context_used if self.output else None) or []


@dataclass
class EvalContext:
    """Everything a spec needs to score one case."""

    trace: CaseTrace
    retrieval: RetrievalView
    pipeline: AnswerView
    generator: AnswerView
    ops: list[StageOutput]  # stages of the real application (excludes the isolated generator)
    judge: Any | None
    _cache: dict[Any, asyncio.Future[Outcome]] = field(default_factory=dict)

    @property
    def case(self) -> GoldenCase:
        return self.trace.case

    async def cached(self, key: Any, factory: Callable[[], Awaitable[Outcome]]) -> Outcome:
        if key not in self._cache:
            self._cache[key] = asyncio.ensure_future(factory())
        return await self._cache[key]


def build_context(trace: CaseTrace, cfg: PipelineCfg, judge: Any | None) -> EvalContext:
    outs = trace.outputs
    r_out = outs.get(cfg.retriever_stage)
    retrieval = RetrievalView(cfg.retriever_stage, r_out, (r_out.chunks or []) if r_out else [])
    pipeline = AnswerView(cfg.final_stage, outs.get(cfg.final_stage))
    iso = cfg.generator_isolated_stage
    iso_out = outs.get(iso) if iso else None
    # An isolated generator that was skipped for this case falls back to the pipeline view.
    generator = AnswerView(iso, iso_out) if iso_out is not None and iso_out.ok else pipeline
    ops = [o for name, o in outs.items() if name != iso]
    return EvalContext(trace, retrieval, pipeline, generator, ops, judge)


class MetricSpec(ABC):
    layer: str = ""
    name: str = ""
    direction: Literal["min", "max"] = "min"
    unit: str = "score"
    needs_judge: bool = True
    view: View = "none"

    @property
    def key(self) -> str:
        return f"{self.layer}.{self.name}"

    def stage_output(self, ctx: EvalContext) -> StageOutput | None:
        """Output of the stage this metric reads; ``None`` when the view is ``none``."""
        if self.view == "retrieval":
            return ctx.retrieval.output
        if self.view == "pipeline":
            return ctx.pipeline.output
        if self.view == "generator":
            return ctx.generator.output
        return None

    def guard(self, ctx: EvalContext) -> None:
        """Raise ``Skip`` / ``StageFailure`` if the metric cannot be computed."""
        if self.view == "none":
            return
        out = self.stage_output(ctx)
        if out is None:
            raise Skip("stage not part of the configured pipeline")
        if not out.ok:
            raise StageFailure(f"stage '{out.stage}' failed: {out.error}")
        self.extra_guard(ctx)

    def extra_guard(self, ctx: EvalContext) -> None:
        """Hook for case-data checks (e.g. golden context present)."""

    def case_threshold(self, ctx: EvalContext, default: float | None) -> float | None:
        """Per-case threshold override hook (used by latency)."""
        return default

    @abstractmethod
    async def evaluate(self, ctx: EvalContext) -> Outcome:
        """Compute the score for one case."""


class DeepEvalSpec(MetricSpec):
    """A metric computed by a DeepEval metric class using the configured judge."""

    cache_name: str = ""  # same cache_name + same stage => computed once per case

    @abstractmethod
    def build(self, judge: Any) -> Any:
        """Return a configured DeepEval metric instance."""

    @abstractmethod
    def test_case(self, ctx: EvalContext) -> Any:
        """Return the ``LLMTestCase`` for this case/view."""

    async def evaluate(self, ctx: EvalContext) -> Outcome:
        if ctx.judge is None:
            raise RuntimeError(f"metric {self.key} needs a judge but none was configured")
        out = self.stage_output(ctx)
        cache_key = (self.cache_name or self.key, out.stage if out else None)

        async def run() -> Outcome:
            metric = self.build(ctx.judge)
            await metric.a_measure(self.test_case(ctx), _show_indicator=False)
            if metric.score is None:
                raise RuntimeError(metric.error or "metric returned no score")
            return Outcome(float(metric.score), getattr(metric, "reason", None))

        return await ctx.cached(cache_key, run)


metric_specs: Registry[type[MetricSpec]] = Registry("metric")


def register_metric(cls: type[MetricSpec]) -> type[MetricSpec]:
    """Class decorator: register under ``"<layer>.<name>"`` taken from the class."""
    return metric_specs.register(f"{cls.layer}.{cls.name}")(cls)
