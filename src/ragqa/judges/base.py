"""Judge base class and registry (the LLM-as-judge extension point).

How to add a new judge provider (open-closed principle)
-------------------------------------------------------
1. Create ``src/ragqa/judges/<provider>.py``.
2. Subclass ``BaseJudge`` and decorate it with ``@judges.register("<provider>")``.
3. Implement ``_build_client`` plus ``_complete`` / ``_acomplete`` (prompt -> text).
   Set ``requires_api_key = False`` if the provider authenticates some other way (see bedrock.py).
4. Add a ``<provider>`` block to ``config/judges.yaml``.

``ragqa.judges.__init__`` auto-imports every module in this package, so no
existing file needs editing.

Contract with DeepEval
----------------------
DeepEval metrics call ``generate(prompt, schema=...)`` / ``a_generate(...)`` on
any ``DeepEvalBaseLLM``. We return the model's raw text; DeepEval extracts the
JSON itself. Calls are retried with exponential backoff because judge APIs
(especially free-tier Groq) rate-limit under concurrent evaluation.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from tenacity import AsyncRetrying, Retrying, stop_after_attempt, wait_random_exponential

from ragqa.config import JudgeProviderCfg
from ragqa.registry import Registry

judges: Registry[type["BaseJudge"]] = Registry("judge")


class JudgeConfigError(RuntimeError):
    """Raised for missing credentials or unknown providers."""


class BaseJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge. Subclasses only talk to their provider SDK."""

    provider: str = "base"
    # False for providers that authenticate through the environment instead of one API key
    # (e.g. Bedrock uses the AWS credential chain); such judges validate access in _build_client.
    requires_api_key: bool = True

    def __init__(self, cfg: JudgeProviderCfg, max_retries: int = 5) -> None:
        api_key = os.environ.get(cfg.api_key_env or "") if cfg.api_key_env else None
        if self.requires_api_key and not api_key:
            raise JudgeConfigError(
                f"{self.provider} judge needs the {cfg.api_key_env or '<api_key_env>'} environment variable"
            )
        self.cfg = cfg
        self.max_retries = max_retries
        self._api_key = api_key
        super().__init__(model=cfg.model)  # calls load_model()

    # -- provider hooks -------------------------------------------------------
    @abstractmethod
    def _build_client(self) -> Any:
        """Return a sync client (and may set ``self._aclient`` for async use)."""

    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Blocking completion: prompt -> text."""

    @abstractmethod
    async def _acomplete(self, prompt: str) -> str:
        """Async completion: prompt -> text."""

    # -- DeepEvalBaseLLM API --------------------------------------------------
    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        self._client = self._build_client()
        return self._client

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            reraise=True,
        ):
            with attempt:
                return self._complete(prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            reraise=True,
        ):
            with attempt:
                return await self._acomplete(prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.provider}:{self.cfg.model}"
