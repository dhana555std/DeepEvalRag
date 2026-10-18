"""Reporter abstraction and registry.

A reporter turns a ``RunReport`` into files. Reporters only read the report
model, so adding one (for example Allure) never touches the evaluation code:

1. create ``src/ragqa/reporting/<name>.py`` with a ``@reporters.register("<name>")`` class;
2. list the name under ``reporters`` in ``config/reporting.yaml`` (or pass ``--reporters``).

``ragqa.reporting.__init__`` auto-imports every module in the package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ragqa.config import ReportingCfg
from ragqa.models import RunReport
from ragqa.registry import Registry

reporters: Registry[type["Reporter"]] = Registry("reporter")


class Reporter(ABC):
    name: str = ""

    @abstractmethod
    def write(self, report: RunReport, cfg: ReportingCfg, out_dir: Path) -> list[Path]:
        """Write the report into ``out_dir`` and return the created files."""
