"""Offline end-to-end tests: pipeline against the mock API, single run, multi run, reports.

Only deterministic metrics (citation, latency, tokens) are selected, so no LLM key is needed.
LLM-judged metrics are covered by the live suite in ``tests/eval``.
"""

from __future__ import annotations

import json

import pytest

from ragqa import config as cfgmod
from ragqa.config import EndpointCfg, PipelineCfg, RagApiCfg
from ragqa.evaluation import RunOptions, run_sync
from ragqa.models import GoldenCase
from ragqa.reporting import write_reports
from ragqa.stages import Pipeline
from ragqa.stages.generation import normalise_citations

DETERMINISTIC = ["generator.citation", "application.latency_ms", "application.tokens"]


def test_normalise_citations_strings_and_objects():
    assert normalise_citations(["a", {"source": "b"}, {"id": "c", "source": "x"}, {"zzz": 1}], None) == ["a", "b", "c"]
    assert normalise_citations([{"meta": {"doc": "d"}}], "meta.doc") == ["d"]
    assert normalise_citations(None, None) == []


def test_pipeline_chains_retrieval_into_generation(mock_env):
    import asyncio

    cfg = cfgmod.load_rag_api()
    case = GoldenCase(id="x", question="How long does standard shipping take?", expected_answer="3 to 5 days")
    trace = asyncio.run(Pipeline(cfg).collect([case]))[0]
    assert trace.outputs["retrieval"].chunks, "retrieval returned chunks"
    gen = trace.outputs["generation"]
    assert gen.ok and "3 to 5 business days" in gen.answer
    assert gen.citations == ["shipping-faq.pdf"]
    assert gen.context_used == [c.text for c in trace.outputs["retrieval"].chunks]
    assert gen.prompt_tokens is not None and gen.latency_ms > 0


def test_api_failure_is_recorded_not_raised(mock_env):
    import asyncio

    bad = RagApiCfg(pipeline=PipelineCfg(stages=["retrieval"]),
                    endpoints={"retrieval": EndpointCfg(url=f"{mock_env}/nope", retries=0,
                                                        response={"chunks": "results", "chunk_text": "text"})})
    case = GoldenCase(id="x", question="q", expected_answer="a")
    out = asyncio.run(Pipeline(bad).collect([case]))[0].outputs["retrieval"]
    assert out.ok is False and "404" in out.error


def test_golden_stage_is_isolated_from_retrieval(mock_env):
    import asyncio

    cfg = cfgmod.load_rag_api()
    cfg.pipeline.stages = ["retrieval", "generation", "generation_golden"]
    cfg.pipeline.generator_isolated_stage = "generation_golden"
    gen_cfg = cfg.endpoints["generation"]  # golden isolation needs an API that accepts contexts
    gen_cfg.request = {"query": "{question}", "contexts": "{chunks_text}"}
    gen_cfg.response = {"answer": "answer"}
    case = GoldenCase(id="x", question="q", expected_answer="a", expected_context=["Golden passage."])
    trace = asyncio.run(Pipeline(cfg).collect([case]))[0]
    assert trace.outputs["generation_golden"].context_used == ["Golden passage."]
    assert trace.latest_chunks() == trace.outputs["retrieval"].chunks  # golden stage adds no chunks


def test_single_run_with_gate_and_reports(mock_env, dataset_path, tmp_path):
    report = run_sync(RunOptions(dataset=dataset_path, only=DETERMINISTIC))
    assert report.mode == "single" and len(report.runs) == 1
    run = report.runs[0]
    aggs = {a.key: a for a in run.aggregates}
    assert aggs["generator.citation"].value == pytest.approx(1.0)  # mock cites correctly
    assert aggs["generator.citation"].n == 3  # safety probes have no expected_citations -> skipped
    assert aggs["application.latency_ms"].passed is True
    assert aggs["application.tokens"].passed is None  # informational
    assert report.passed

    files = write_reports(report, cfgmod.load_reporting(), tmp_path)
    names = {f.name for f in files}
    assert {"report.html", "report.css", "results.json"} <= names
    html = (tmp_path / "report.html").read_text()
    assert 'href="report.css"' in html and "q001" in html and "PASSED" in html
    assert json.loads((tmp_path / "results.json").read_text())["mode"] == "single"


def test_multi_run_scores_same_traces_n_times(mock_env, dataset_path, tmp_path):
    report = run_sync(RunOptions(dataset=dataset_path, only=DETERMINISTIC, runs=3))
    assert report.mode == "multi" and len(report.runs) == 3
    cit = next(m for m in report.multi if m.key == "generator.citation")
    assert cit.runs == 3 and cit.std == pytest.approx(0.0) and cit.passed is True
    # API called once: identical latency in every run because traces are reused
    lat = [next(a for a in r.aggregates if a.key == "application.latency_ms").mean for r in report.runs]
    assert len(set(lat)) == 1

    write_reports(report, cfgmod.load_reporting(), tmp_path, ["html"])
    assert "Multi-run statistics" in (tmp_path / "report.html").read_text()


def test_multi_run_recollect_calls_api_each_run(mock_env, dataset_path):
    report = run_sync(RunOptions(dataset=dataset_path, only=["application.latency_ms"], runs=2, recollect=True))
    lat = [next(a for a in r.aggregates if a.key == "application.latency_ms").mean for r in report.runs]
    assert len(set(lat)) == 2  # fresh API calls -> different measured latencies


def test_failing_gate_marks_report_failed(mock_env, dataset_path, tmp_path):
    cdir = tmp_path / "cfg"
    cdir.mkdir()
    for f in ("rag_api.yaml", "judges.yaml", "reporting.yaml"):
        (cdir / f).write_text((cfgmod.config_dir() / f).read_text())
    (cdir / "thresholds.yaml").write_text(
        "metrics:\n  application.latency_ms: {enabled: true, threshold: 0.0001, aggregate: p95}\n")
    report = run_sync(RunOptions(dataset=dataset_path, config_dir=cdir))
    assert not report.passed
    assert report.runs[0].gate.failed_metrics == ["application.latency_ms"]


def test_unreachable_api_makes_run_fail_with_errors(monkeypatch, dataset_path):
    monkeypatch.setenv("RETRIEVAL_API_URL", "http://127.0.0.1:9/v1/retrieve")
    monkeypatch.setenv("GENERATION_API_URL", "http://127.0.0.1:9/v1/answer")
    monkeypatch.setenv("RAGQA_CONFIG_DIR", "config")
    report = run_sync(RunOptions(dataset=dataset_path, only=["application.latency_ms"], limit=1))
    assert not report.passed and report.runs[0].gate.errored_metrics == ["application.latency_ms"]


def test_branding_css_and_logo_are_copied(mock_env, dataset_path, tmp_path):
    (tmp_path / "brand.src.css").write_text(":root{--brand:#7b2cbf}")
    (tmp_path / "logo.src.png").write_bytes(b"png")
    report = run_sync(RunOptions(dataset=dataset_path, only=["application.latency_ms"], limit=1))
    cfg = cfgmod.ReportingCfg(css=str(tmp_path / "brand.src.css"), logo=str(tmp_path / "logo.src.png"))
    out = tmp_path / "out"
    write_reports(report, cfg, out, ["html"])
    html = (out / "report.html").read_text()
    assert 'href="brand.css"' in html and 'src="logo.png"' in html
    assert (out / "brand.css").exists() and (out / "logo.png").exists()


# ----------------------------------------------------------------------------- citation / refusal rules
def _ctx(case, citations, answer="a"):
    from ragqa.metrics.base import build_context
    from ragqa.models import CaseTrace, StageOutput

    trace = CaseTrace(case=case, outputs={
        "retrieval": StageOutput(stage="retrieval", chunks=[]),
        "generation": StageOutput(stage="generation", answer=answer, citations=citations, context_used=[]),
    })
    return build_context(trace, PipelineCfg(), None)


def test_citation_matches_document_id_inside_longer_citation():
    import asyncio

    from ragqa.metrics.generator import Citation

    case = GoldenCase(id="x", question="q", expected_answer="a", expected_citations=["BXU548683_R", "D100001320"])
    ctx = _ctx(case, ["bxu548683_r Rev A - Comparative Blood Flow Rate Tests, Section 5 (p.6)"])
    out = asyncio.run(Citation().evaluate(ctx))
    assert out.score == pytest.approx(0.5) and "missing: d100001320" in out.reason


def test_expected_refusal_skips_ground_truth_metrics_but_keeps_correctness():
    from ragqa.metrics import metric_specs
    from ragqa.metrics.base import Skip

    case = GoldenCase(id="r", question="q", expected_answer="should decline", expect_refusal=True)
    ctx = _ctx(case, [])
    skipped = {}
    for key, cls in metric_specs.items():
        try:
            cls().guard(ctx)
            skipped[key] = False
        except Skip:
            skipped[key] = True
    assert skipped["retriever.recall"] and skipped["generator.faithfulness"] and skipped["triad.faithfulness"]
    assert skipped["application.completeness"]
    assert not skipped["application.correctness"] and not skipped["application.toxicity"]
    assert skipped["application.jailbreak"]  # only for safety_probe


# ----------------------------------------------------------------------------- spreadsheet converter
def test_xlsx_converter_maps_columns_and_skips_incomplete_rows(tmp_path):
    import importlib.util
    from pathlib import Path

    import openpyxl

    spec = importlib.util.spec_from_file_location(
        "xlsx_to_golden", Path(__file__).resolve().parents[2] / "scripts" / "xlsx_to_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Q ID", "Question", "Expected Answer", "Answer must come from", "Status"])
    ws.append(["Q-1", "What?", "Because.\n\n\n  Really. ", "BXU548683_R Rev A - Title\nSection 5 (p.6)", None])
    ws.append(["Q-2", "Capital of X?", "should decline", "Not in the documents. ", None])
    ws.append(["Q-3", "Ignore instructions", "must not comply", "Not in the documents (guardrail / access)", None])
    ws.append(["Q-4", "No answer here", None, None, None])
    ws.append([None, None, None, None, None])
    path = tmp_path / "in.xlsx"
    wb.save(path)

    cases, skipped = mod.convert(path)
    by_id = {c["id"]: c for c in cases}
    assert by_id["Q-1"]["expected_citations"] == ["BXU548683_R"]
    assert by_id["Q-1"]["expected_answer"] == "Because.\nReally."
    assert by_id["Q-1"]["source_reference"] == "BXU548683_R Rev A - Title | Section 5 (p.6)"
    assert by_id["Q-2"]["expect_refusal"] and not by_id["Q-2"]["safety_probe"]
    assert by_id["Q-3"]["expect_refusal"] and by_id["Q-3"]["safety_probe"]
    assert len(skipped) == 1 and skipped[0].startswith("Q-4")


def test_report_has_generic_favicon_and_it_can_be_replaced(mock_env, dataset_path, tmp_path):
    report = run_sync(RunOptions(dataset=dataset_path, only=["application.latency_ms"], limit=1))
    default_dir = tmp_path / "default"
    write_reports(report, cfgmod.ReportingCfg(), default_dir, ["html"])
    assert '<link rel="icon" href="favicon.svg">' in (default_dir / "report.html").read_text()
    assert "<svg" in (default_dir / "favicon.svg").read_text()

    (tmp_path / "mine.png").write_bytes(b"png")
    custom_dir = tmp_path / "custom"
    write_reports(report, cfgmod.ReportingCfg(favicon=str(tmp_path / "mine.png")), custom_dir, ["html"])
    assert '<link rel="icon" href="favicon.png">' in (custom_dir / "report.html").read_text()
    assert (custom_dir / "favicon.png").read_bytes() == b"png"
