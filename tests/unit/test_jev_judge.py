"""The Jev judge (TypeSafe System One) as an alternative to LLM judges.

A fake ``/v1/systemone`` server answers every Noul question with a fixed probability and records the
requests, so the tests prove - without an API key - that ``JUDGE_PROVIDER=jev`` is read from the
environment, authenticates with the bearer key, scores every LLM-judged metric through Jev (no LLM
involved), keeps the skip rules, and computes shared metrics once.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ragqa.evaluation import RunOptions, run_sync
from ragqa.judges.base import JudgeConfigError
from ragqa.metrics.jev import statements

REQUESTS: list[dict] = []
PROBABILITY = 0.9


class FakeJev(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"auth": self.headers.get("Authorization"), "body": body})
        answers = {k: {"type": "noul", "noul": PROBABILITY} for k in body["questions"]}
        payload = json.dumps({"model": body["model"], "answers": answers,
                              "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def jev_env(mock_env, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), FakeJev)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    REQUESTS.clear()
    monkeypatch.setenv("JUDGE_PROVIDER", "jev")
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_API_URL", f"http://127.0.0.1:{port}/v1/systemone")
    monkeypatch.setenv("JEV_MODEL", "jev-test")
    yield
    server.shutdown()


def test_statements_split_and_clean():
    text = "**Yes.** The window is 30 days [1][2]. Items must be unused!\n- Refunds take five days.\n- ok"
    assert statements(text) == ["The window is 30 days.", "Items must be unused!", "Refunds take five days."]
    assert statements("") == [] and statements(None) == []
    assert len(statements(". ".join(["This has four words"] * 50))) <= 20


def test_every_llm_metric_runs_on_jev_without_a_llm(jev_env, dataset_path):
    report = run_sync(RunOptions(dataset=dataset_path))
    run = report.runs[0]
    errors = [(c.case_id, m.key, m.error) for c in run.cases for m in c.metrics if m.error]
    assert not errors, errors
    assert report.judges == ["jev:jev-test"]

    scored = {a.key for a in run.aggregates if a.n > 0}
    assert {"retriever.recall", "retriever.precision", "generator.faithfulness", "generator.relevance",
            "triad.context_relevance", "triad.answer_relevance", "triad.faithfulness",
            "application.correctness", "application.completeness", "application.toxicity",
            "application.pii", "application.jailbreak"} <= scored
    for c in run.cases:
        for m in c.metrics:
            if m.score is not None and m.unit == "score":
                assert 0.0 <= m.score <= 1.0, (c.case_id, m.key, m.score)

    assert REQUESTS and all(r["auth"] == "Bearer test-key" for r in REQUESTS)
    assert all(r["body"]["model"] == "jev-test" for r in REQUESTS)
    assert all(q["type"] == "noul" for r in REQUESTS for q in r["body"]["questions"].values())


def test_jev_scores_follow_the_probabilities(jev_env, dataset_path):
    run = run_sync(RunOptions(dataset=dataset_path, only=["application.toxicity", "application.correctness"],
                              limit=1)).runs[0]
    by_key = {m.key: m for m in run.cases[0].metrics}
    assert by_key["application.correctness"].score == pytest.approx(PROBABILITY)
    assert by_key["application.toxicity"].score == pytest.approx(1 - PROBABILITY)  # 1.0 = clean


def test_skip_rules_still_apply_with_jev(jev_env, dataset_path):
    run = run_sync(RunOptions(dataset=dataset_path)).runs[0]
    by_case = {c.case_id: {m.key: m for m in c.metrics} for c in run.cases}
    assert by_case["q001"]["application.jailbreak"].skipped
    assert not by_case["s001"]["application.jailbreak"].skipped
    assert by_case["s001"]["retriever.recall"].skipped


def test_shared_metrics_are_asked_once(jev_env, dataset_path):
    def n_requests(only):
        REQUESTS.clear()
        run_sync(RunOptions(dataset=dataset_path, only=only, limit=1))
        return len(REQUESTS)

    alone = n_requests(["generator.faithfulness"])
    both = n_requests(["generator.faithfulness", "triad.faithfulness"])
    assert alone > 0 and both == alone


def test_missing_key_is_a_clear_config_error(jev_env, dataset_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "")
    with pytest.raises(JudgeConfigError, match="JEV_API_KEY"):
        run_sync(RunOptions(dataset=dataset_path, limit=1))
