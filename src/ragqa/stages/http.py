"""HTTP helper shared by REST-backed stages.

Responsibilities
----------------
* build the request body from the YAML template (``{question}``, ``{chunks}``,
  ``{chunks_text}`` placeholders keep their native types when they are the whole value)
* apply auth (``none`` / ``bearer`` / ``header``) from environment variables
* time the call and retry transient failures (timeouts, connection errors, 5xx, 429)
* extract token / cost usage using the paths configured under ``usage``
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ragqa.config import EndpointCfg
from ragqa.jsonpath import extract


@dataclass
class HttpResult:
    body: Any
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None


def render(template: Any, values: dict[str, Any]) -> Any:
    """Substitute ``{placeholders}`` in a nested template."""
    if isinstance(template, str):
        stripped = template.strip()
        if stripped.startswith("{") and stripped.endswith("}") and stripped[1:-1] in values:
            return values[stripped[1:-1]]  # whole-value placeholder keeps its type
        out = template
        for k, v in values.items():
            if isinstance(v, str):
                out = out.replace("{" + k + "}", v)
        return out
    if isinstance(template, dict):
        return {k: render(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, values) for v in template]
    return template


def _headers(cfg: EndpointCfg) -> dict[str, str]:
    headers = dict(cfg.headers)
    auth = cfg.auth
    if auth.type == "none":
        return headers
    token = os.environ.get(auth.token_env or "")
    if not token:
        raise RuntimeError(f"auth type '{auth.type}' requires env var {auth.token_env}")
    if auth.type == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    elif auth.type == "header":
        headers[auth.header] = token
    else:
        raise RuntimeError(f"unsupported auth type '{auth.type}'")
    return headers


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


async def call(cfg: EndpointCfg, values: dict[str, Any], client: httpx.AsyncClient) -> HttpResult:
    """Perform the configured request. Raises on final failure."""
    if not cfg.url:
        raise RuntimeError("endpoint url is empty - set it in .env (see .env.example)")
    body = render(cfg.request, values)
    headers = _headers(cfg)
    last: Exception | None = None
    start = time.perf_counter()
    for attempt in range(cfg.retries + 1):
        try:
            resp = await client.request(
                cfg.method, cfg.url, json=body if cfg.method.upper() != "GET" else None,
                params=body if cfg.method.upper() == "GET" else None,
                headers=headers, timeout=cfg.timeout_s,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000
            data = resp.json()
            u = cfg.usage
            pt = _int(extract(data, u.prompt_tokens)) if u.prompt_tokens else None
            ct = _int(extract(data, u.completion_tokens)) if u.completion_tokens else None
            cost = extract(data, u.cost_usd) if u.cost_usd else None
            if cost is None and pt is not None and ct is not None and u.price_per_1k_prompt:
                cost = pt / 1000 * u.price_per_1k_prompt + ct / 1000 * (u.price_per_1k_completion or 0.0)
            return HttpResult(data, latency_ms, pt, ct, float(cost) if cost is not None else None)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            retryable = isinstance(exc, httpx.TransportError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if not retryable or attempt == cfg.retries:
                break
            await _sleep(0.5 * 2**attempt)
    raise RuntimeError(f"{cfg.method} {cfg.url} failed: {last}")


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
