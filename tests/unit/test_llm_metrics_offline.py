"""Runs every DeepEval-backed metric through the real judge plumbing with a fake LLM.

``SchemaJudge`` is a registered judge whose "LLM" fabricates a schema-valid JSON answer
from the pydantic schema DeepEval passes in. Scores are therefore meaningless, but the
test proves, without any API key, that:

* each metric (retriever / generator / triad / application) builds, calls the judge via
  ``a_generate(prompt, schema=...)`` and parses the reply with DeepEval 4.x;
* skip rules work (jailbreak only on safety probes, quality metrics not on probes);
* shared metrics (faithfulness, answer relevancy) are computed once, not once per layer.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from ragqa import config as cfgmod
from ragqa.evaluation import RunOptions, run_sync
from ragqa.judges import judges
from ragqa.judges.base import BaseJudge


def synth(schema: dict, defs: dict) -> Any:
    """Build a value satisfying a JSON schema (first enum member, one list item, etc.)."""
    if "$ref" in schema:
        return synth(defs[schema["$ref"].split("/")[-1]], defs)
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return synth(next(s for s in schema["anyOf"] if s.get("type") != "null"), defs)
    t = schema.get("type")
    if t == "object":
        return {k: synth(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [synth(schema.get("items", {}), defs)]
    if t == "string":
        return "ok"
    if t == "integer":
        return 8
    if t == "number":
        return 0.8
    if t == "boolean":
        return True
    return "ok"


@judges.register("schema-fake")
class SchemaJudge(BaseJudge):
    provider = "schema-fake"
    calls = itertools.count()

    def _build_client(self):
        return object()

    def _reply(self, schema) -> str:
        import json

        next(SchemaJudge.calls)
        if schema is None:
            return json.dumps({"score": 8, "reason": "ok"})
        js = schema.model_json_schema()
        return json.dumps(synth(js, js.get("$defs", {})))

    def generate(self, prompt, schema=None, **kw):
        return self._reply(schema)

    async def a_generate(self, prompt, schema=None, **kw):
        return self._reply(schema)

    def _complete(self, prompt):  # pragma: no cover - generate() is overridden
        raise NotImplementedError

    async def _acomplete(self, prompt):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def fake_cfg(tmp_path, mock_env, monkeypatch):
    cdir = tmp_path / "cfg"
    cdir.mkdir()
    src = cfgmod.config_dir()
    for f in ("rag_api.yaml", "reporting.yaml"):
        (cdir / f).write_text((src / f).read_text())
    (cdir / "thresholds.yaml").write_text((src / "thresholds.yaml").read_text()
                                          .replace("application.style:        { enabled: false",
                                                   "application.style:        { enabled: true"))
    (cdir / "judges.yaml").write_text(
        "default: schema-fake\nproviders:\n  schema-fake: {model: fake, api_key_env: FAKE_KEY}\n")
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    return cdir


def test_every_metric_runs_through_deepeval_without_errors(fake_cfg, dataset_path):
    report = run_sync(RunOptions(dataset=dataset_path, config_dir=fake_cfg))
    run = report.runs[0]
    errors = [(c.case_id, m.key, m.error) for c in run.cases for m in c.metrics if m.error]
    assert not errors, errors

    scored = {a.key for a in run.aggregates if a.n > 0}
    expected = {
        "retriever.recall", "retriever.precision",
        "generator.faithfulness", "generator.relevance", "generator.citation",
        "triad.context_relevance", "triad.answer_relevance", "triad.faithfulness",
        "application.correctness", "application.completeness", "application.style",
        "application.toxicity", "application.pii", "application.jailbreak",
        "application.latency_ms", "application.tokens",
    }
    assert expected <= scored, expected - scored
    for c in run.cases:
        for m in c.metrics:
            if m.score is not None and m.unit == "score":
                assert 0.0 <= m.score <= 1.0, (c.case_id, m.key, m.score)


def test_skip_rules(fake_cfg, dataset_path):
    run = run_sync(RunOptions(dataset=dataset_path, config_dir=fake_cfg)).runs[0]
    by_case = {c.case_id: {m.key: m for m in c.metrics} for c in run.cases}
    assert by_case["q001"]["application.jailbreak"].skipped
    assert not by_case["s001"]["application.jailbreak"].skipped
    assert by_case["s001"]["application.correctness"].skipped
    assert by_case["s001"]["retriever.recall"].skipped
    assert by_case["s001"]["application.toxicity"].score is not None  # safety runs on every case


def test_shared_metrics_are_computed_once(fake_cfg, dataset_path):
    def calls(only):
        before = next(SchemaJudge.calls)
        run_sync(RunOptions(dataset=dataset_path, config_dir=fake_cfg, only=only, limit=1))
        return next(SchemaJudge.calls) - before - 1

    alone = calls(["generator.faithfulness"])
    both = calls(["generator.faithfulness", "triad.faithfulness"])
    assert alone > 0 and both == alone  # the Triad reuses the Generator's computation


def test_judge_variance_across_providers(fake_cfg, dataset_path):
    report = run_sync(RunOptions(dataset=dataset_path, config_dir=fake_cfg, only=["triad.*"],
                                 judges=["schema-fake"], runs=2))
    assert report.mode == "multi" and {m.judge for m in report.multi} == {"schema-fake:fake"}
