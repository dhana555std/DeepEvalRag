"""LLM-as-judge providers.

Importing this package auto-registers every judge module found in it, so a new
provider is added by dropping a file here (see ``ragqa.judges.base``). Use
``create_judge()`` to instantiate one from ``config/judges.yaml``; Groq is the
default and ``JUDGE_PROVIDER`` (env) or ``--judge`` (CLI) overrides it.
"""

from __future__ import annotations

import importlib
import os
import pkgutil

from ragqa.config import JudgesCfg, load_judges
from ragqa.judges.base import BaseJudge, JudgeConfigError, judges

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")


def create_judge(provider: str | None = None, cfg: JudgesCfg | None = None) -> BaseJudge:
    """Build the judge for ``provider`` (default: ``JUDGE_PROVIDER`` env, else config default)."""
    cfg = cfg or load_judges()
    name = provider or os.environ.get("JUDGE_PROVIDER") or cfg.default
    if name not in cfg.providers:
        raise JudgeConfigError(f"judge '{name}' has no entry in judges.yaml")
    return judges.get(name)(cfg.providers[name], max_retries=cfg.max_retries,
                                  max_concurrent_calls=cfg.max_concurrent_calls, json_retries=cfg.json_retries)


__all__ = ["BaseJudge", "JudgeConfigError", "create_judge", "judges"]
