"""Pluggable reporters (HTML, JSON; Allure can be added as another module).

``write_reports`` runs every reporter named in the config (or passed in) and
returns the created files. Unknown reporter names fail fast with the list of
registered ones.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from ragqa.config import ReportingCfg
from ragqa.models import RunReport
from ragqa.reporting.base import Reporter, reporters

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")


def write_reports(report: RunReport, cfg: ReportingCfg, out_dir: str | Path | None = None,
                  names: list[str] | None = None) -> list[Path]:
    out = Path(out_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names or cfg.reporters:
        written += reporters.get(name)().write(report, cfg, out)
    return written


__all__ = ["Reporter", "reporters", "write_reports"]
