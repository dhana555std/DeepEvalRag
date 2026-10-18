"""Retrieval stage: calls the retrieval REST endpoint.

Mapping (``config/rag_api.yaml`` -> ``retrieval.response``)
----------------------------------------------------------
* ``chunks``      path to the list of retrieved items
* ``chunk_text``  path of the passage text *inside each item* (``""`` if items are strings)
* ``chunk_id``    optional path of the item id inside each item
* ``chunk_score`` optional path of the relevance score inside each item
"""

from __future__ import annotations

import httpx

from ragqa.jsonpath import extract
from ragqa.models import CaseTrace, Chunk, GoldenCase, StageOutput
from ragqa.stages import http
from ragqa.stages.base import Stage, stages


def parse_chunks(body: object, mapping: dict) -> list[Chunk]:
    """Turn a response body into ``Chunk`` objects using the configured paths."""
    items = extract(body, mapping.get("chunks", "chunks"), []) or []
    out: list[Chunk] = []
    for item in items:
        text = extract(item, mapping.get("chunk_text", "text"))
        if text is None:
            continue
        cid = extract(item, mapping["chunk_id"]) if mapping.get("chunk_id") else None
        score = extract(item, mapping["chunk_score"]) if mapping.get("chunk_score") else None
        meta = item if isinstance(item, dict) else {}
        out.append(Chunk(id=str(cid) if cid is not None else None, text=str(text),
                         score=float(score) if score is not None else None, metadata=meta))
    return out


@stages.register("retrieval")
class RetrievalStage(Stage):
    name = "retrieval"

    async def run(self, case: GoldenCase, trace: CaseTrace) -> StageOutput:
        client = self._client or httpx.AsyncClient()
        try:
            res = await http.call(self.cfg, {"question": case.question}, client)
            chunks = parse_chunks(res.body, self.cfg.response)
            return StageOutput(
                stage=self.name, latency_ms=res.latency_ms, chunks=chunks,
                prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                cost_usd=res.cost_usd,
            )
        except Exception as exc:  # API failures become data, not crashes
            return StageOutput(stage=self.name, ok=False, error=str(exc))
        finally:
            if self._client is None:
                await client.aclose()
