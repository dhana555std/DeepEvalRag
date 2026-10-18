"""Generator layer gates: faithfulness, relevance and citation of the generated answer.

One test per enabled metric of this layer, generated from config/thresholds.yaml
(see conftest.pytest_generate_tests). Run with pytest or `deepeval test run`:

    uv run pytest tests/eval/test_generator.py
    uv run deepeval test run tests/eval/test_generator.py
"""

import pytest
from _gate import assert_metric_gate

LAYER = "generator"
pytestmark = pytest.mark.eval


def test_generator_metric_meets_threshold(report, metric_key):
    assert_metric_gate(report, metric_key)
