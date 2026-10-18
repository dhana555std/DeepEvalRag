"""Anthropic (Claude) judge.

Uses the official ``anthropic`` SDK. Requires ``ANTHROPIC_API_KEY``. Claude has
no JSON mode, so the prompt is suffixed with a reminder to answer with JSON
only; DeepEval's parser tolerates surrounding prose.
"""

from __future__ import annotations

from typing import Any

import anthropic

from ragqa.judges.base import BaseJudge, judges

_JSON_HINT = "\n\nRespond with a single valid JSON object only, no extra text."


@judges.register("anthropic")
class AnthropicJudge(BaseJudge):
    provider = "anthropic"

    def _build_client(self) -> Any:
        self._aclient = anthropic.AsyncAnthropic(api_key=self._api_key, base_url=self.cfg.base_url)
        return anthropic.Anthropic(api_key=self._api_key, base_url=self.cfg.base_url)

    def _kwargs(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "user", "content": prompt + _JSON_HINT}],
        }

    @staticmethod
    def _text(res: Any) -> str:
        return "".join(b.text for b in res.content if getattr(b, "type", "") == "text")

    def _complete(self, prompt: str) -> str:
        return self._text(self._client.messages.create(**self._kwargs(prompt)))

    async def _acomplete(self, prompt: str) -> str:
        return self._text(await self._aclient.messages.create(**self._kwargs(prompt)))
