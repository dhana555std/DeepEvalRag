"""RAG Triad layer: the three pairwise relationships between question, context, answer.

* ``triad.context_relevance`` (Question <-> Context) - DeepEval ``ContextualRelevancyMetric``:
  how much of the retrieved context is relevant to the question.
* ``triad.answer_relevance``  (Question <-> Answer)  - DeepEval ``AnswerRelevancyMetric``.
* ``triad.faithfulness``      (Answer <-> Context)   - DeepEval ``FaithfulnessMetric``
  (groundedness: the answer makes no claims beyond the context).

The Triad always scores the real chained pipeline (``pipeline`` view) - what a user
actually receives. Answer relevance and faithfulness reuse the computation that
the Generator layer already did when both layers look at the same stage.
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from ragqa.metrics.base import (
    DeepEvalSpec,
    EvalContext,
    Skip,
    has_no_ground_truth,
    register_metric,
)
from ragqa.metrics.generator import answer_case


class _TriadSpec(DeepEvalSpec):
    layer = "triad"
    view = "pipeline"

    def extra_guard(self, ctx: EvalContext) -> None:
        if has_no_ground_truth(ctx.case):
            raise Skip("safety probe / expected refusal: no factual ground truth")

    def test_case(self, ctx: EvalContext) -> LLMTestCase:
        return answer_case(ctx, "pipeline")


@register_metric
class ContextRelevance(_TriadSpec):
    name = "context_relevance"
    cache_name = "contextual_relevancy"

    def build(self, judge):
        return ContextualRelevancyMetric(model=judge, include_reason=True)


@register_metric
class AnswerRelevance(_TriadSpec):
    name = "answer_relevance"
    cache_name = "answer_relevancy"

    def build(self, judge):
        return AnswerRelevancyMetric(model=judge, include_reason=True)


@register_metric
class TriadFaithfulness(_TriadSpec):
    name = "faithfulness"
    cache_name = "faithfulness"

    def build(self, judge):
        return FaithfulnessMetric(model=judge, include_reason=True)
