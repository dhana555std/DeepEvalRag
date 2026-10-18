"""Pipeline stages and the pipeline runner.

``Pipeline`` instantiates the stages listed in ``pipeline.stages`` (from
``config/rag_api.yaml``), runs them in order for each golden case with bounded
concurrency, and returns one ``CaseTrace`` per case. A stage that needs an
endpoint block but does not have one fails fast with a clear message.

``generation_golden`` shares the ``generation`` endpoint block unless it has
its own, so isolating the generator needs no extra configuration.
"""

from __future__ import annotations

import asyncio

import httpx

from ragqa.config import RagApiCfg
from ragqa.models import CaseTrace, GoldenCase
from ragqa.stages.base import Stage, load_stage_modules, stages

load_stage_modules()

_ENDPOINT_FALLBACK = {"generation_golden": "generation"}


class Pipeline:
    def __init__(self, cfg: RagApiCfg, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self.stages: list[Stage] = []
        for name in cfg.pipeline.stages:
            endpoint = cfg.endpoints.get(name) or cfg.endpoints.get(_ENDPOINT_FALLBACK.get(name, ""))
            if endpoint is None:
                raise KeyError(f"stage '{name}' has no endpoint block in rag_api.yaml")
            self.stages.append(stages.get(name)(endpoint, client))

    async def trace_case(self, case: GoldenCase) -> CaseTrace:
        trace = CaseTrace(case=case)
        for stage in self.stages:
            trace.outputs[stage.name] = await stage.run(case, trace)
        return trace

    async def collect(self, cases: list[GoldenCase]) -> list[CaseTrace]:
        """Run every case through the pipeline (order preserved)."""
        sem = asyncio.Semaphore(max(1, self.cfg.pipeline.concurrency))

        async def one(case: GoldenCase) -> CaseTrace:
            async with sem:
                return await self.trace_case(case)

        owns_client = self._client is None
        if owns_client:  # share one connection pool across the whole run
            self._client = httpx.AsyncClient()
            for s in self.stages:
                s._client = self._client
        try:
            return list(await asyncio.gather(*(one(c) for c in cases)))
        finally:
            if owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None
                for s in self.stages:
                    s._client = None


__all__ = ["Pipeline", "Stage", "stages"]
