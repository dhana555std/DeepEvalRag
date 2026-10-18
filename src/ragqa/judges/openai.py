"""OpenAI judge.

Uses the official ``openai`` SDK with JSON mode. Requires ``OPENAI_API_KEY``.
``base_url`` in ``config/judges.yaml`` lets the same class target any
OpenAI-compatible gateway.
"""

from __future__ import annotations

from typing import Any

import openai

from ragqa.judges.base import BaseJudge, judges


@judges.register("openai")
class OpenAIJudge(BaseJudge):
    provider = "openai"

    def _build_client(self) -> Any:
        self._aclient = openai.AsyncOpenAI(api_key=self._api_key, base_url=self.cfg.base_url)
        return openai.OpenAI(api_key=self._api_key, base_url=self.cfg.base_url)

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
