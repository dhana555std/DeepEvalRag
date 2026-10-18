"""Configuration loading (``.env`` + YAML with ``${ENV}`` interpolation).

Sources, in order of precedence
-------------------------------
1. Real environment variables (CI secrets, shell exports)
2. ``.env`` file in the working directory (loaded with python-dotenv, never overrides 1)
3. YAML files in the config directory (``RAGQA_CONFIG_DIR``, default ``./config``)

YAML values may reference the environment with ``${VAR}`` or
``${VAR:-default}`` (default used when unset or empty) so that endpoints, models and secrets live in ``.env``
while the *shape* of the integration lives in version-controlled YAML.

Config files
------------
* ``rag_api.yaml``   - pipeline stages, retrieval/generation endpoints and payload mappings
* ``judges.yaml``    - default judge and per-provider model settings
* ``thresholds.yaml``- which metrics run and their pass/fail thresholds
* ``reporting.yaml`` - report title, output dir, branding (logo, CSS, template dir)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            # Shell ``:-`` semantics: the default also applies when the variable is set but EMPTY
            # (CI systems export unset variables/secrets as empty strings).
            return os.environ.get(name) or (default if default is not None else "")

        return _ENV_REF.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def config_dir() -> Path:
    return Path(os.environ.get("RAGQA_CONFIG_DIR", "config"))


def load_yaml(name: str, directory: Path | None = None) -> dict[str, Any]:
    """Load ``<config dir>/<name>`` with env interpolation. Loads ``.env`` once."""
    load_dotenv(override=False)
    path = (directory or config_dir()) / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        return _interpolate(yaml.safe_load(fh) or {})


# --------------------------------------------------------------------------- RAG API


class AuthCfg(BaseModel):
    type: str = "none"  # none | bearer | header
    token_env: str | None = None  # env var holding the token / header value
    header: str = "X-API-Key"  # header name when type == header


class UsageCfg(BaseModel):
    """Where to find token/cost info in the response, plus a price fallback."""

    prompt_tokens: str | None = None
    completion_tokens: str | None = None
    cost_usd: str | None = None
    price_per_1k_prompt: float | None = None
    price_per_1k_completion: float | None = None


class EndpointCfg(BaseModel):
    url: str
    method: str = "POST"
    timeout_s: float = 30.0
    retries: int = 2
    headers: dict[str, str] = Field(default_factory=dict)
    auth: AuthCfg = Field(default_factory=AuthCfg)
    # Request body template. A string that is exactly "{question}" etc. is replaced
    # by the value itself (so lists/dicts keep their type). Placeholders:
    # {question} {chunks} {chunks_text}
    request: dict[str, Any] = Field(default_factory=dict)
    # Response mapping (paths understood by ragqa.jsonpath)
    response: dict[str, Any] = Field(default_factory=dict)
    usage: UsageCfg = Field(default_factory=UsageCfg)


class PipelineCfg(BaseModel):
    stages: list[str] = Field(default_factory=lambda: ["retrieval", "generation"])
    retriever_stage: str = "retrieval"
    final_stage: str = "generation"
    # Optional stage whose output feeds Generator-layer metrics (generator tested in
    # isolation against golden context). Example: "generation_golden".
    generator_isolated_stage: str | None = None
    concurrency: int = 4


class RagApiCfg(BaseModel):
    pipeline: PipelineCfg = Field(default_factory=PipelineCfg)
    # One entry per stage name. Extra stages (reranker, ...) just add a key.
    endpoints: dict[str, EndpointCfg] = Field(default_factory=dict)


def load_rag_api(directory: Path | None = None) -> RagApiCfg:
    raw = load_yaml("rag_api.yaml", directory)
    endpoints = {k: v for k, v in raw.items() if k not in ("pipeline",)}
    return RagApiCfg(pipeline=PipelineCfg(**raw.get("pipeline", {})), endpoints=endpoints)


# --------------------------------------------------------------------------- Judges


class JudgeProviderCfg(BaseModel):
    model: str
    api_key_env: str | None = None  # None for providers that use ambient credentials (Bedrock/AWS)
    temperature: float = 0.0
    max_tokens: int = 2048
    base_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)  # provider-specific settings (e.g. region)


class JudgesCfg(BaseModel):
    default: str = "groq"
    max_retries: int = 5
    providers: dict[str, JudgeProviderCfg] = Field(default_factory=dict)


def load_judges(directory: Path | None = None) -> JudgesCfg:
    return JudgesCfg(**load_yaml("judges.yaml", directory))


# --------------------------------------------------------------------------- Thresholds


class MetricCfg(BaseModel):
    enabled: bool = True
    threshold: float | None = None  # None -> informational (never gates)
    aggregate: str = "mean"  # mean | p95


class ThresholdsCfg(BaseModel):
    metrics: dict[str, MetricCfg] = Field(default_factory=dict)
    fail_on_error: bool = True

    def for_key(self, key: str) -> MetricCfg:
        return self.metrics.get(key, MetricCfg(enabled=False))


def load_thresholds(directory: Path | None = None) -> ThresholdsCfg:
    return ThresholdsCfg(**load_yaml("thresholds.yaml", directory))


# --------------------------------------------------------------------------- Reporting


class ReportingCfg(BaseModel):
    title: str = "RAG Evaluation Report"
    output_dir: str = "reports"
    reporters: list[str] = Field(default_factory=lambda: ["html", "json"])
    logo: str | None = None  # path to a logo file copied next to the report
    favicon: str | None = None  # path to a favicon (svg/png/ico); default: bundled generic icon
    css: str | None = None  # extra/override stylesheet copied as brand.css
    template_dir: str | None = None  # directory with a custom report.html.j2
    page_size: int = 20  # cases shown per page in the HTML report (client-side); 0 = no pagination
    cases_order: str = "failed_first"  # failed_first | dataset
    theme: str = "dark"  # default colour theme of the HTML report: dark | light | system
    pager_pages: int = 20  # max page numbers shown at once ("sheet"); « » jump between sheets


def load_reporting(directory: Path | None = None) -> ReportingCfg:
    try:
        return ReportingCfg(**load_yaml("reporting.yaml", directory))
    except FileNotFoundError:
        return ReportingCfg()
