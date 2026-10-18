"""Metric specs grouped by layer.

Importing this package auto-registers every module here, so a new metric (or a
whole new layer, e.g. ``reranker``) is added by dropping a file in this folder.
``enabled_specs()`` returns the instantiated specs switched on in
``config/thresholds.yaml`` (optionally filtered by ``fnmatch`` patterns).
"""

from __future__ import annotations

import fnmatch
import importlib
import pkgutil

from ragqa.config import ThresholdsCfg
from ragqa.metrics.base import MetricSpec, metric_specs

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")


def enabled_specs(cfg: ThresholdsCfg, only: list[str] | None = None) -> list[MetricSpec]:
    """Specs enabled in config, in registry order, filtered by ``only`` patterns (e.g. ``generator.*``)."""
    specs: list[MetricSpec] = []
    for key, cls in metric_specs.items():
        if not cfg.for_key(key).enabled:
            continue
        if only and not any(fnmatch.fnmatch(key, pat) for pat in only):
            continue
        specs.append(cls())
    return specs


__all__ = ["MetricSpec", "enabled_specs", "metric_specs"]
