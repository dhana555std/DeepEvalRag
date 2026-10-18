"""Live evaluation suite: shared fixtures (runs under pytest AND ``deepeval test run``).

How it works
------------
One session-scoped fixture, ``report``, calls the same ``run_evaluation`` used by the CLI:
the RAG API is called once, every enabled metric is scored, and the HTML/JSON reports are
written to ``$RAGQA_REPORT_DIR`` (default ``reports/``) when the session ends - even when
tests fail, so CI can always upload them. Each ``test_<layer>.py`` then asserts the gate of
each metric of its layer (one test per metric, generated from ``config/thresholds.yaml``).

Environment
-----------
RAGQA_DATASET      golden dataset (default datasets/golden.jsonl)
RAGQA_RUNS         1 (default) = single run; >1 = multi run (judge variance, gate on the mean)
RAGQA_JUDGES       comma separated judge providers (default: JUDGE_PROVIDER / groq)
RAGQA_RECOLLECT    "1" to call the RAG API again for every run
RAGQA_LIMIT        evaluate only the first N cases (default: all)
RAGQA_ONLY         comma separated metric patterns, e.g. 'retriever.*' (default: all enabled)
RAGQA_CONCURRENCY  concurrent judge calls (default 4)
RAGQA_REPORT_DIR   report output directory
"""

from __future__ import annotations

import os

import pytest

from ragqa import config as cfgmod
from ragqa.evaluation import RunOptions, run_sync
from ragqa.models import RunReport
from ragqa.reporting import write_reports


def _options() -> RunOptions:
    judges = os.environ.get("RAGQA_JUDGES")
    return RunOptions(
        dataset=os.environ.get("RAGQA_DATASET", "datasets/golden.jsonl"),
        runs=int(os.environ.get("RAGQA_RUNS", "1")),
        judges=[j.strip() for j in judges.split(",")] if judges else None,
        recollect=os.environ.get("RAGQA_RECOLLECT") == "1",
        concurrency=int(os.environ.get("RAGQA_CONCURRENCY", "4")),
        limit=int(os.environ["RAGQA_LIMIT"]) if os.environ.get("RAGQA_LIMIT") else None,
        only=[o.strip() for o in os.environ["RAGQA_ONLY"].split(",")] if os.environ.get("RAGQA_ONLY") else None,
    )


@pytest.fixture(scope="session")
def report():
    rep: RunReport = run_sync(_options())
    yield rep
    write_reports(rep, cfgmod.load_reporting(), os.environ.get("RAGQA_REPORT_DIR"))


def pytest_generate_tests(metafunc):
    """Parametrise ``metric_key`` with the enabled metrics of the module's ``LAYER``."""
    if "metric_key" in metafunc.fixturenames:
        layer = getattr(metafunc.module, "LAYER")
        cfg = cfgmod.load_thresholds()
        keys = [k for k, m in cfg.metrics.items() if m.enabled and k.startswith(layer + ".")]
        metafunc.parametrize("metric_key", keys, ids=keys)
