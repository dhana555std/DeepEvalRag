"""Jev judge (TypeSafe System One): an alternative to the LLM judges.

Jev does not generate text. It answers typed questions about a ``state`` (here: yes/no "Noul"
questions that return a probability), so it cannot sit behind DeepEval's prompt-completion
interface like Groq / Anthropic / OpenAI do. Instead the metrics have a Jev implementation
(``ragqa/metrics/jev.py``) that asks narrow Noul questions through ``JevJudge.noul`` and turns the
probabilities into the same 0..1 scores. Select it like any judge:

    JUDGE_PROVIDER=jev            (env / .env)        or        ragqa run --judge jev

Configuration comes from ``.env`` (see ``config/judges.yaml`` -> ``providers.jev``):

    JEV_API_KEY   bearer token for the TypeSafe API (required when Jev is selected)
    JEV_API_URL   endpoint, default https://api.typesafe.ai/v1/systemone
    JEV_MODEL     model name, default jev-latest

HTTP contract (https://docs.typesafe.ai/api.md): ``POST`` ``{"state", "model", "questions"}`` with
``Authorization: Bearer <key>``; the reply holds ``answers[<question id>].noul`` in 0..1.
429 / 529 / 5xx and transport errors are retried with exponential backoff.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ragqa.judges.base import BaseJudge, judges

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 529) or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@judges.register("jev")
class JevJudge(BaseJudge):
    provider = "jev"
    is_jev = True  # metrics check this flag to use their Jev implementation instead of DeepEval

    def _build_client(self) -> Any:
        return None  # a short-lived httpx client is created per request (safe across event loops)

    @property
    def url(self) -> str:
        return self.cfg.base_url or DEFAULT_URL

    async def noul(self, state: Any, questions: dict[str, Any]) -> dict[str, float]:
        """Ask yes/no questions about ``state``; returns probability-of-yes per question id.

        ``questions`` maps an id to the question's ``instructions`` (a string, or an object holding
        the question plus the data it references by backticked field name).
        """
        body = {
            "state": state,
            "model": self.cfg.model,
            "questions": {k: {"type": "noul", "instructions": v} for k, v in questions.items()},
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception(_retryable),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(self.url, json=body, headers=headers)
                    resp.raise_for_status()
                answers = resp.json()["answers"]
        return {k: float(answers[k]["noul"]) for k in questions}

    # -- not a text generator ---------------------------------------------------
    def _unsupported(self) -> RuntimeError:
        return RuntimeError("the Jev judge does not generate text; this metric has no Jev implementation")

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        raise self._unsupported()

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        raise self._unsupported()

    def _complete(self, prompt: str) -> str:  # pragma: no cover - generate() is overridden
        raise self._unsupported()

    async def _acomplete(self, prompt: str) -> str:  # pragma: no cover
        raise self._unsupported()
