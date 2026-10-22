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

import asyncio
import json
import os
import re
import threading
from abc import abstractmethod
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from tenacity import (
    AsyncRetrying,
    Retrying,
    stop_after_attempt,
    wait_random_exponential,
)

from ragqa.config import JudgeProviderCfg
from ragqa.registry import Registry

judges: Registry[type[BaseJudge]] = Registry("judge")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> str | None:
    """Return the first valid JSON object/array in ``text`` (fences/prose stripped), else None."""
    text = _FENCE.sub("", text.strip())
    for start, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, end = json.JSONDecoder().raw_decode(text[start:])
            except ValueError:
                continue
            if isinstance(obj, dict | list):
                return text[start : start + end]
    return None


class _Limiter:
    """Process-wide cap on in-flight judge calls, shared by every metric and judge instance."""

    _sync: threading.BoundedSemaphore | None = None
    _async: dict[int, asyncio.Semaphore] = {}
    _size = 0

    @classmethod
    def configure(cls, size: int) -> None:
        if size != cls._size:
            cls._size, cls._sync, cls._async = size, threading.BoundedSemaphore(size), {}

    @classmethod
    def sync(cls) -> threading.BoundedSemaphore:
        assert cls._sync is not None
        return cls._sync

    @classmethod
    def aio(cls) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())  # asyncio semaphores are bound to one loop
        return cls._async.setdefault(loop_id, asyncio.Semaphore(cls._size))


class JudgeConfigError(RuntimeError):
    """Raised for missing credentials or unknown providers."""


class BaseJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge. Subclasses only talk to their provider SDK."""

    provider: str = "base"
    # False for providers that authenticate through the environment instead of one API key
    # (e.g. Bedrock uses the AWS credential chain); such judges validate access in _build_client.
    requires_api_key: bool = True

    def __init__(
        self,
        cfg: JudgeProviderCfg,
        max_retries: int = 5,
        max_concurrent_calls: int = 2,
        json_retries: int = 2,
    ) -> None:
        api_key = os.environ.get(cfg.api_key_env or "") if cfg.api_key_env else None
        if self.requires_api_key and not api_key:
            raise JudgeConfigError(
                f"{self.provider} judge needs the {cfg.api_key_env or '<api_key_env>'} environment variable"
            )
        self.cfg = cfg
        self.max_retries = max_retries
        self.json_retries = json_retries
        _Limiter.configure(max_concurrent_calls)
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
        text = ""
        for _ in range(self.json_retries + 1):
            text = self._call(prompt)
            if schema is None or (clean := extract_json(text)) is not None:
                return text if schema is None else clean
        return text  # still invalid: let DeepEval raise its usual error

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        text = ""
        for _ in range(self.json_retries + 1):
            text = await self._acall(prompt)
            if schema is None or (clean := extract_json(text)) is not None:
                return text if schema is None else clean
        return text

    def _call(self, prompt: str) -> str:
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=2, max=60),
            reraise=True,
        ):
            with attempt, _Limiter.sync():
                return self._complete(prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _acall(self, prompt: str) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=2, max=60),
            reraise=True,
        ):
            with attempt:
                async with _Limiter.aio():
                    return await self._acomplete(prompt)
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.provider}:{self.cfg.model}"
