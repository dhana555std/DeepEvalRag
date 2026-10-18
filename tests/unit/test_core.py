"""Unit tests for the small building blocks: json path, registry, dataset, judges, aggregation."""

from __future__ import annotations

import json

import pytest

from ragqa.config import JudgeProviderCfg, JudgesCfg, MetricCfg, ThresholdsCfg
from ragqa.dataset import DatasetError, load_dataset
from ragqa.evaluation.aggregate import aggregate_multi, aggregate_run, evaluate_gate, percentile
from ragqa.jsonpath import extract
from ragqa.judges import JudgeConfigError, create_judge, judges
from ragqa.models import CaseResult, MetricResult, RunResult
from ragqa.registry import Registry


# ----------------------------------------------------------------------------- jsonpath
def test_jsonpath_nested_index_wildcard():
    doc = {"data": {"results": [{"t": "a", "id": 1}, {"t": "b"}]}}
    assert extract(doc, "data.results[0].t") == "a"
    assert extract(doc, "data.results[*].t") == ["a", "b"]
    assert extract(doc, "data.results[*].id") == [1]
    assert extract(doc, "data.missing", "d") == "d"
    assert extract(doc, "") == doc


# ----------------------------------------------------------------------------- registry
def test_registry_register_and_unknown():
    reg: Registry[type] = Registry("thing")

    @reg.register("a")
    class A: ...

    assert reg.get("a") is A
    with pytest.raises(KeyError, match="Registered: a"):
        reg.get("b")
    with pytest.raises(ValueError):
        reg.register("a")(type("Other", (), {}))


# ----------------------------------------------------------------------------- dataset
def _write(tmp_path, *rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows))
    return p


def test_dataset_valid_and_warns_on_missing_context(tmp_path):
    p = _write(tmp_path, "# comment", {"id": "a", "question": "q", "expected_answer": "x"})
    cases, warnings = load_dataset(p)
    assert len(cases) == 1 and "no expected_context" in warnings[0]


def test_dataset_reports_all_errors(tmp_path):
    p = _write(tmp_path, {"id": "a", "question": "q", "expected_answer": "x"},
               {"id": "a", "question": "q", "expected_answer": "x"}, "{not json",
               {"id": "c", "question": " ", "expected_answer": "x"})
    with pytest.raises(DatasetError) as err:
        load_dataset(p)
    text = " ".join(err.value.errors)
    assert "duplicate id" in text and "invalid JSON" in text and "empty question" in text


def test_shipped_golden_dataset_is_valid():
    from pathlib import Path

    cases, _ = load_dataset(Path(__file__).resolve().parents[2] / "datasets" / "golden.jsonl")
    assert len(cases) >= 40
    assert any(c.expect_refusal for c in cases) and any(c.expected_citations for c in cases)
    assert all(c.expected_citations for c in cases if not (c.expect_refusal or c.safety_probe))


def test_missing_expected_context_is_one_summary_warning(tmp_path):
    rows = [{"id": f"c{i}", "question": "q", "expected_answer": "a"} for i in range(3)]
    rows.append({"id": "r", "question": "q", "expected_answer": "decline", "expect_refusal": True})
    _, warnings = load_dataset(_write(tmp_path, *rows))
    assert len(warnings) == 1 and warnings[0].startswith("3 factual case(s)")


# ----------------------------------------------------------------------------- judges
def test_all_providers_registered_and_groq_default():
    assert {"groq", "anthropic", "openai", "google", "bedrock"} <= set(judges.names())
    from ragqa.config import load_judges

    assert load_judges().default == "groq"


def test_judge_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    with pytest.raises(JudgeConfigError, match="GROQ_API_KEY"):
        create_judge()


def test_judge_provider_env_selects_provider(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(JudgeConfigError, match="ANTHROPIC_API_KEY"):
        create_judge()


def test_new_judge_can_be_added_without_editing_existing_code(monkeypatch):
    """Open-closed: a third-party judge registers itself and is created from config."""
    from ragqa.judges.base import BaseJudge

    @judges.register("echo-test")
    class EchoJudge(BaseJudge):
        provider = "echo-test"

        def _build_client(self):
            return object()

        def _complete(self, prompt):
            return '{"echo": true}'

        async def _acomplete(self, prompt):
            return self._complete(prompt)

    monkeypatch.setenv("ECHO_KEY", "x")
    cfg = JudgesCfg(default="echo-test", providers={
        "echo-test": JudgeProviderCfg(model="m", api_key_env="ECHO_KEY")})
    judge = create_judge(cfg=cfg)
    assert judge.generate("hi") == '{"echo": true}'
    assert judge.get_model_name() == "echo-test:m"


# ----------------------------------------------------------------------------- aggregation
def _case(cid, *metrics):
    return CaseResult(case_id=cid, question="q", category="c", expected_answer="a", metrics=list(metrics))


def _m(key, score, threshold, direction="min", **kw):
    layer, name = key.split(".")
    return MetricResult(key=key, layer=layer, name=name, score=score, threshold=threshold,
                        direction=direction, passed=None if score is None or threshold is None else
                        (score >= threshold if direction == "min" else score <= threshold), **kw)


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5], 95) == 5


def test_gate_min_and_max_directions_and_p95():
    cfg = ThresholdsCfg(metrics={
        "retriever.recall": MetricCfg(threshold=0.7),
        "application.latency_ms": MetricCfg(threshold=1000, aggregate="p95"),
    })
    cases = [_case("1", _m("retriever.recall", 0.9, 0.7), _m("application.latency_ms", 100, 1000, "max")),
             _case("2", _m("retriever.recall", 0.6, 0.7), _m("application.latency_ms", 4000, 1000, "max"))]
    aggs = {a.key: a for a in aggregate_run(cases, cfg)}
    assert aggs["retriever.recall"].passed is True  # mean 0.75 >= 0.7
    assert aggs["application.latency_ms"].passed is False  # p95 4000 > 1000
    gate = evaluate_gate(list(aggs.values()), cfg)
    assert not gate.passed and gate.failed_metrics == ["application.latency_ms"]


def test_errors_fail_gate_and_skips_are_ignored():
    cfg = ThresholdsCfg(metrics={"generator.faithfulness": MetricCfg(threshold=0.5)})
    err = MetricResult(key="generator.faithfulness", layer="generator", name="faithfulness",
                       error="boom", passed=False)
    skip = MetricResult(key="generator.faithfulness", layer="generator", name="faithfulness", skipped=True)
    aggs = aggregate_run([_case("1", _m("generator.faithfulness", 0.9, 0.5)), _case("2", err),
                          _case("3", skip)], cfg)
    assert aggs[0].n == 1 and aggs[0].errors == 1
    assert not evaluate_gate(aggs, cfg).passed
    assert evaluate_gate(aggs, ThresholdsCfg(metrics=cfg.metrics, fail_on_error=False)).passed


def test_informational_metric_never_gates():
    cfg = ThresholdsCfg(metrics={"application.tokens": MetricCfg()})
    aggs = aggregate_run([_case("1", _m("application.tokens", 99999, None, "max"))], cfg)
    assert aggs[0].passed is None and evaluate_gate(aggs, cfg).passed


def test_multi_aggregate_mean_std():
    cfg = ThresholdsCfg(metrics={"triad.faithfulness": MetricCfg(threshold=0.8)})
    runs = []
    for i, score in enumerate([0.7, 0.9, 0.8]):
        aggs = aggregate_run([_case("1", _m("triad.faithfulness", score, 0.8))], cfg)
        runs.append(RunResult(run_id=f"r{i}", judge="groq:x", cases=[], aggregates=aggs))
    multi = aggregate_multi(runs, cfg)[0]
    assert multi.runs == 3 and multi.mean == pytest.approx(0.8)
    assert multi.std == pytest.approx(0.0816, abs=1e-3) and multi.passed is True


# ----------------------------------------------------------------------------- env interpolation / bedrock
def test_env_default_applies_when_variable_unset_or_empty(monkeypatch):
    from ragqa.config import _interpolate

    monkeypatch.delenv("X_UNSET", raising=False)
    monkeypatch.setenv("X_EMPTY", "")
    monkeypatch.setenv("X_SET", "v")
    assert _interpolate("${X_UNSET:-d}|${X_EMPTY:-d}|${X_SET:-d}|${X_UNSET}") == "d|d|v|"


def _bedrock_cfg(monkeypatch, tmp_path):
    from ragqa.config import load_judges

    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)
    return load_judges()


def test_bedrock_judge_uses_converse_and_needs_no_api_key(monkeypatch, tmp_path):
    import asyncio

    cfg = _bedrock_cfg(monkeypatch, tmp_path)
    assert cfg.providers["bedrock"].api_key_env is None and cfg.providers["bedrock"].extra["region"] == "eu-west-1"
    judge = create_judge("bedrock", cfg)
    assert judge.get_model_name().startswith("bedrock:")

    seen = {}

    class FakeRuntime:
        def converse(self, **kw):
            seen.update(kw)
            return {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": '{"ok": '}, {"text": "true}"}]}}}

    judge._client = FakeRuntime()
    assert judge.generate("hello") == '{"ok": true}'
    assert asyncio.run(judge.a_generate("hello")) == '{"ok": true}'
    assert seen["modelId"] == cfg.providers["bedrock"].model
    assert seen["inferenceConfig"] == {"maxTokens": 2048, "temperature": 0.0}
    assert seen["messages"][0]["content"][0]["text"].startswith("hello")


def test_bedrock_judge_without_credentials_fails_clearly(monkeypatch, tmp_path):
    from ragqa.config import load_judges

    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with pytest.raises(JudgeConfigError, match="no AWS credentials"):
        create_judge("bedrock", load_judges())
