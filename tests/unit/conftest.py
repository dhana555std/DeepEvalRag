"""Fixtures for the offline unit tests.

``mock_api`` starts ``scripts/mock_rag_api.py`` on a free port in a background
thread and points the harness at it through environment variables, so runner and
pipeline tests exercise real HTTP without any external service or LLM key.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_mock_module():
    spec = importlib.util.spec_from_file_location("mock_rag_api", ROOT / "scripts" / "mock_rag_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def isolate_from_local_env(monkeypatch):
    """Unit tests must not depend on a developer's local ``.env`` (e.g. JUDGE_PROVIDER=groq).

    ``.env`` loading is disabled and judge-selection variables are cleared; each test sets
    exactly the environment it needs.
    """
    monkeypatch.setattr("ragqa.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("ragqa.cli.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)


@pytest.fixture(scope="session")
def mock_api_url():
    mod = _load_mock_module()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), mod.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def mock_env(monkeypatch, mock_api_url):
    monkeypatch.setenv("RETRIEVAL_API_URL", f"{mock_api_url}/v1/retrieve")
    monkeypatch.setenv("GENERATION_API_URL", f"{mock_api_url}/v1/answer")
    monkeypatch.setenv("RAGQA_CONFIG_DIR", str(ROOT / "config"))
    return mock_api_url


@pytest.fixture
def dataset_path():
    """Small sample dataset that matches scripts/mock_rag_api.py (the real one is datasets/golden.jsonl)."""
    return str(ROOT / "tests" / "fixtures" / "sample_golden.jsonl")
