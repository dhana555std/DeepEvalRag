"""Groq judge (the default provider).

Uses the official ``groq`` SDK. JSON mode is requested so DeepEval metrics that
expect structured output parse reliably. Model, temperature and key variable
come from ``config/judges.yaml`` / ``.env`` (``GROQ_API_KEY``).
"""

from __future__ import annotations

from typing import Any

from groq import AsyncGroq, Groq

from ragqa.judges.base import BaseJudge, judges


@judges.register("groq")
class GroqJudge(BaseJudge):
    provider = "groq"

    def _build_client(self) -> Any:
        self._aclient = AsyncGroq(api_key=self._api_key, base_url=self.cfg.base_url)
        return Groq(api_key=self._api_key, base_url=self.cfg.base_url)

    def _kwargs(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _complete(self, prompt: str) -> str:
        res = self._client.chat.completions.create(**self._kwargs(prompt))
        return res.choices[0].message.content or ""

    async def _acomplete(self, prompt: str) -> str:
        res = await self._aclient.chat.completions.create(**self._kwargs(prompt))
        return res.choices[0].message.content or ""
