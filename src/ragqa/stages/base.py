"""Stage abstraction: one step of the RAG pipeline under test.

A ``Stage`` receives the golden case plus the outputs of earlier stages and
returns a ``StageOutput``. Retrieval and generation ship with the harness. To
evaluate another component later (a reranker, a LangGraph node, a query
rewriter...):

1. add ``src/ragqa/stages/<name>.py`` with a ``@stages.register("<name>")`` class;
2. add its endpoint block to ``config/rag_api.yaml`` and list it in
   ``pipeline.stages`` at the desired position;
3. add metric specs for it in ``ragqa.metrics`` if it needs its own scores.

Convention for chaining: a stage that produces passages sets
``StageOutput.chunks``. Downstream stages read ``trace.latest_chunks()``, so a
reranker placed between retrieval and generation transparently changes what the
generator receives, while the retriever metrics keep reading the retrieval
stage's own output.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod

import httpx

from ragqa.config import EndpointCfg
from ragqa.models import CaseTrace, GoldenCase, StageOutput
from ragqa.registry import Registry

stages: Registry[type["Stage"]] = Registry("stage")


class Stage(ABC):
    """One pipeline step. ``name`` is the key used in config and in the trace."""

    name: str = "stage"

    def __init__(self, cfg: EndpointCfg, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client

    @abstractmethod
    async def run(self, case: GoldenCase, trace: CaseTrace) -> StageOutput:
        """Execute the stage. Must not raise for API failures: return ``ok=False``."""


def load_stage_modules() -> None:
    """Import every module of ``ragqa.stages`` so their decorators run."""
    import ragqa.stages as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"ragqa.stages.{m.name}")
