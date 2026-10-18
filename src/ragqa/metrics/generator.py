"""Generator layer: faithfulness, relevance and citation quality of the answer.

* ``generator.faithfulness`` - DeepEval ``FaithfulnessMetric``: is every claim in the
  answer supported by the context the generator was given?
* ``generator.relevance``    - DeepEval ``AnswerRelevancyMetric``: does the answer address
  the question?
* ``generator.citation``     - deterministic (no LLM): share of the golden
  ``expected_citations`` present in the answer's citations. An expected citation counts as
  present when it equals, or is contained in, one of the returned citations (case-insensitive),
  so a short stable id such as a document number matches "<id> Rev A - <title>".
  Skipped for cases without ``expected_citations``.

These read the ``generator`` view: the isolated generator run on golden context
when ``pipeline.generator_isolated_stage`` is configured, else the chained
pipeline output. Faithfulness and relevance share cache entries with the Triad
layer, so they are not paid for twice when both views coincide.
"""

from __future__ import annotations

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from ragqa.metrics.base import (
    DeepEvalSpec, EvalContext, MetricSpec, Outcome, Skip, has_no_ground_truth, register_metric,
)


def answer_case(ctx: EvalContext, view_name: str = "generator") -> LLMTestCase:
    """Build the LLMTestCase for answer-oriented metrics from a chosen view."""
    view = ctx.generator if view_name == "generator" else ctx.pipeline
    return LLMTestCase(
        input=ctx.case.question,
        actual_output=view.answer,
        expected_output=ctx.case.expected_answer,
        retrieval_context=view.contexts or [""],
    )


class _GeneratorSpec(DeepEvalSpec):
    layer = "generator"
    view = "generator"

    def extra_guard(self, ctx: EvalContext) -> None:
        if has_no_ground_truth(ctx.case):
            raise Skip("safety probe / expected refusal: no factual ground truth")

    def test_case(self, ctx: EvalContext) -> LLMTestCase:
        return answer_case(ctx, "generator")


@register_metric
class Faithfulness(_GeneratorSpec):
    name = "faithfulness"
    cache_name = "faithfulness"

    def build(self, judge):
        return FaithfulnessMetric(model=judge, include_reason=True)


@register_metric
class Relevance(_GeneratorSpec):
    name = "relevance"
    cache_name = "answer_relevancy"

    def build(self, judge):
        return AnswerRelevancyMetric(model=judge, include_reason=True)


@register_metric
class Citation(MetricSpec):
    layer = "generator"
    name = "citation"
    view = "generator"
    needs_judge = False

    def extra_guard(self, ctx: EvalContext) -> None:
        if not ctx.case.expected_citations:
            raise Skip("case has no expected_citations")

    async def evaluate(self, ctx: EvalContext) -> Outcome:
        expected = {c.strip().lower() for c in ctx.case.expected_citations}
        actual = [c.strip().lower() for c in ctx.generator.citations]
        hit = {e for e in expected if any(e in a for a in actual)}
        missing = sorted(expected - hit)
        reason = f"{len(hit)}/{len(expected)} expected citations present"
        if missing:
            reason += f"; missing: {', '.join(missing)}"
        return Outcome(len(hit) / len(expected), reason)
