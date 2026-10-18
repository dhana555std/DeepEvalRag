"""JSON reporter: the machine-readable source of truth (``results.json``).

CI can archive it, diff it between builds, or feed other tools. It is the exact
serialisation of ``RunReport`` (see ``ragqa.models``).
"""

from __future__ import annotations

from pathlib import Path

from ragqa.config import ReportingCfg
from ragqa.models import RunReport
from ragqa.reporting.base import Reporter, reporters


@reporters.register("json")
class JsonReporter(Reporter):
    name = "json"

    def write(self, report: RunReport, cfg: ReportingCfg, out_dir: Path) -> list[Path]:
        path = out_dir / "results.json"
        path.write_text(report.model_dump_json(indent=2))
        return [path]
