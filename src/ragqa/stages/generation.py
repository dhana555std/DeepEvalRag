"""Generation stage: calls the generation REST endpoint.

Two registered variants share one implementation:

* ``generation``        - receives the chunks produced by the earlier stages
                          (``trace.latest_chunks()``): the realistic, chained path.
* ``generation_golden`` - receives the dataset's ``expected_context`` instead, so the
                          generator can be judged in isolation from retrieval quality.
                          Enable it via ``pipeline.generator_isolated_stage``.

Mapping (``config/rag_api.yaml`` -> ``generation.response``)
-----------------------------------------------------------
* ``answer``       path of the answer text
* ``citations``    path of the citations list
* ``citation_id``  optional path *inside each citation* naming it (default: the first of
                   id / source / title / url / doc_id when citations are objects)

Optional, for APIs that retrieve internally and return ``cited_ranks`` + ``sources``:

* ``sources``      path of the list of sources the answer was generated from. Their text
                   becomes ``context_used`` (instead of the chunks sent in the request).
* ``source_text``  path of the passage text inside each source (default ``text``)
* ``source_rank``  path of the rank inside each source (default ``rank``); integer citations
                   are resolved to the source with that rank, then named via ``citation_id``
                   applied to the source.
"""

from __future__ import annotations

import httpx

from ragqa.jsonpath import extract
from ragqa.models import CaseTrace, Chunk, GoldenCase, StageOutput
from ragqa.stages import http
from ragqa.stages.base import Stage, stages

_CITATION_KEYS = ("id", "source", "title", "url", "doc_id")


def normalise_citations(raw: object, id_path: str | None) -> list[str]:
    """Reduce citations (strings or objects) to a list of identifying strings."""
    out: list[str] = []
    for c in raw or []:  # type: ignore[union-attr]
        if isinstance(c, dict):
            value = extract(c, id_path) if id_path else next(
                (c[k] for k in _CITATION_KEYS if c.get(k) is not None), None
            )
        else:
            value = c
        if value is not None:
            out.append(str(value))
    return out


def resolve_rank_citations(raw: object, sources: list, rank_path: str, id_path: str | None) -> list[str]:
    """Resolve integer citations (ranks) to the identifying string of the matching source."""
    by_rank = {extract(src, rank_path): src for src in sources}
    return normalise_citations([by_rank[r] for r in raw or [] if r in by_rank], id_path)  # type: ignore[union-attr]


class _GenerationBase(Stage):
    use_golden_context = False

    def _chunks(self, case: GoldenCase, trace: CaseTrace) -> list[Chunk]:
        if self.use_golden_context:
            return [Chunk(id=None, text=t) for t in case.expected_context]
        return trace.latest_chunks()

    async def run(self, case: GoldenCase, trace: CaseTrace) -> StageOutput:
        if self.use_golden_context and not case.expected_context:
            return StageOutput(stage=self.name, ok=False, error="skipped: case has no expected_context")
        chunks = self._chunks(case, trace)
        values = {
            "question": case.question,
            "chunks": [{"id": c.id, "text": c.text, "score": c.score, **c.metadata} for c in chunks],
            "chunks_text": [c.text for c in chunks],
        }
        client = self._client or httpx.AsyncClient()
        try:
            res = await http.call(self.cfg, values, client)
            m = self.cfg.response
            answer = extract(res.body, m.get("answer", "answer"))
            raw_citations = extract(res.body, m.get("citations", "citations"), []) or []
            sources = extract(res.body, m["sources"], []) or [] if m.get("sources") else None
            context = values["chunks_text"]
            if sources is not None:
                context = [str(t) for t in (extract(src, m.get("source_text", "text")) for src in sources)
                           if t is not None]
            if sources and raw_citations and all(isinstance(c, int) for c in raw_citations):
                citations = resolve_rank_citations(raw_citations, sources, m.get("source_rank", "rank"),
                                                   m.get("citation_id"))
            else:
                citations = normalise_citations(raw_citations, m.get("citation_id"))
            return StageOutput(
                stage=self.name, latency_ms=res.latency_ms,
                answer=str(answer) if answer is not None else "", citations=citations,
                context_used=context, prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens, cost_usd=res.cost_usd,
            )
        except Exception as exc:
            return StageOutput(stage=self.name, ok=False, error=str(exc))
        finally:
            if self._client is None:
                await client.aclose()


@stages.register("generation")
class GenerationStage(_GenerationBase):
    name = "generation"


@stages.register("generation_golden")
class GoldenGenerationStage(_GenerationBase):
    name = "generation_golden"
    use_golden_context = True
