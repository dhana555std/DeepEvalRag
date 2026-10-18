"""RAG Triad gates: context relevance (Q-C), answer relevance (Q-A), faithfulness (A-C).

One test per enabled metric of this layer, generated from config/thresholds.yaml
(see conftest.pytest_generate_tests). Run with pytest or `deepeval test run`:

    uv run pytest tests/eval/test_triad.py
    uv run deepeval test run tests/eval/test_triad.py
"""

import pytest
from _gate import assert_metric_gate

LAYER = "triad"
pytestmark = pytest.mark.eval


def test_triad_metric_meets_threshold(report, metric_key):
    assert_metric_gate(report, metric_key)
