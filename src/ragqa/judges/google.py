"""Google (Gemini) judge.

Uses the ``google-genai`` SDK with ``response_mime_type=application/json``.
Requires ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from ragqa.judges.base import BaseJudge, judges


@judges.register("google")
class GoogleJudge(BaseJudge):
    provider = "google"

    def _build_client(self) -> Any:
        return genai.Client(api_key=self._api_key)

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
            response_mime_type="application/json",
        )

    def _complete(self, prompt: str) -> str:
        res = self._client.models.generate_content(model=self.cfg.model, contents=prompt, config=self._config())
        return res.text or ""

    async def _acomplete(self, prompt: str) -> str:
        res = await self._client.aio.models.generate_content(
            model=self.cfg.model, contents=prompt, config=self._config()
        )
        return res.text or ""
