"""Retriever layer: recall and precision of the retrieved passages.

* ``retriever.recall``    - DeepEval ``ContextualRecallMetric``: how much of the
  golden answer is supported by what was retrieved (did we fetch everything needed?).
* ``retriever.precision`` - DeepEval ``ContextualPrecisionMetric``: are the relevant
  passages ranked ahead of the irrelevant ones (is the retrieved set clean/ordered?).

Both compare the *question*, the retrieved chunks and the golden answer, so a
case needs ``expected_answer`` (always present). Safety probes are skipped
and expected refusals because they intentionally have no supporting context.
"""

from __future__ import annotations

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from ragqa.metrics.base import (
    DeepEvalSpec,
    EvalContext,
    Skip,
    has_no_ground_truth,
    register_metric,
)


class _RetrieverSpec(DeepEvalSpec):
    layer = "retriever"
    view = "retrieval"

    def extra_guard(self, ctx: EvalContext) -> None:
        if has_no_ground_truth(ctx.case):
            raise Skip("safety probe / expected refusal has no golden context")

    def test_case(self, ctx: EvalContext) -> LLMTestCase:
        return LLMTestCase(
            input=ctx.case.question,
            actual_output=ctx.pipeline.answer or "",
            expected_output=ctx.case.expected_answer,
            retrieval_context=[c.text for c in ctx.retrieval.chunks] or [""],
        )


@register_metric
class Recall(_RetrieverSpec):
    name = "recall"
    cache_name = "contextual_recall"

    def build(self, judge):
        return ContextualRecallMetric(model=judge, include_reason=True)


@register_metric
class Precision(_RetrieverSpec):
    name = "precision"
    cache_name = "contextual_precision"

    def build(self, judge):
        return ContextualPrecisionMetric(model=judge, include_reason=True)
