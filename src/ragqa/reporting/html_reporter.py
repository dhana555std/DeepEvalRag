"""HTML reporter: renders ``templates/report.html.j2`` with ``static/report.css``.

Branding (no Python changes needed)
-----------------------------------
Markup, styling and logic are separate:

* template  ``reporting/templates/report.html.j2``  (structure; digest data comes from ``reporting/summary.py``)
* stylesheet ``reporting/static/report.css``        (design tokens are CSS variables)

Customise through ``config/reporting.yaml``:

* ``css``          - extra stylesheet copied as ``brand.css`` and loaded *after* the default,
                     so overriding ``:root`` variables (colours, font) is enough
* ``logo``         - image copied next to the report and shown in the header
* ``favicon``      - browser-tab icon (svg/png/ico); a generic bundled icon is used when unset
* ``template_dir`` - directory containing your own ``report.html.j2`` (takes precedence)

Pagination (client-side)
------------------------
``report.html`` contains every case; the separate ``static/report.js`` (copied next to the report,
like the CSS) shows ``page_size`` cases at a time (default 20) and builds the pager. Clicking
``Next`` swaps the visible cases on the same page. Page numbers are shown in sheets of at most
``pager_pages`` (default 20); ``«`` / ``»`` jump between sheets. ``#page=3`` in the URL selects a
page. Without JavaScript (or when printing) all cases are visible. ``cases_order`` is
``failed_first`` (default; problems on page 1) or ``dataset``. ``page_size: 0`` disables paging.

``theme`` (``dark`` default, ``light`` or ``system``) is the theme the report opens in until the
viewer picks another with the switcher (remembered in the browser).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, select_autoescape

from ragqa.config import ReportingCfg
from ragqa.evaluation.aggregate import layer_sort_key
from ragqa.models import RunReport
from ragqa.reporting.base import Reporter, reporters
from ragqa.reporting.charts import GLOBALS as CHART_GLOBALS
from ragqa.reporting.summary import LAYER_TITLES, summarize
from ragqa.reporting.summary import fmt_value as fmt

_PKG = Path(__file__).parent


def short_error(error: str | None) -> str:
    """One readable line for a metric error (details live in the summary's root-cause list)."""
    if not error:
        return ""
    if error.startswith("StageFailure"):
        stage = error.split("'")[1] if "'" in error else "pipeline"
        return f"Not scored: the {stage} stage failed (see What needs attention)."
    line = error.strip().splitlines()[0]
    return line if len(line) <= 200 else line[:199] + "…"


def status_class(passed: bool | None) -> str:
    return {True: "pass", False: "fail", None: "info"}[passed]


def status_label(passed: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "INFO"}[passed]


def _group_by_layer(items: list) -> list[tuple[str, str, list]]:
    layers: dict[str, list] = {}
    for it in items:
        layers.setdefault(it.layer, []).append(it)
    return [(k, LAYER_TITLES.get(k, k.title()), v) for k, v in sorted(layers.items(), key=lambda kv: layer_sort_key(kv[0]))]


@reporters.register("html")
class HtmlReporter(Reporter):
    name = "html"

    def _env(self, cfg: ReportingCfg) -> Environment:
        loaders = []
        if cfg.template_dir:
            loaders.append(FileSystemLoader(cfg.template_dir))
        loaders.append(FileSystemLoader(_PKG / "templates"))
        env = Environment(loader=ChoiceLoader(loaders), autoescape=select_autoescape(["html", "j2"]))
        env.filters.update(fmt=fmt, status_class=status_class, status_label=status_label,
                           short_error=short_error)
        env.globals["group_by_layer"] = _group_by_layer
        env.globals.update(CHART_GLOBALS)
        return env

    def write(self, report: RunReport, cfg: ReportingCfg, out_dir: Path) -> list[Path]:
        written: list[Path] = []
        shutil.copy(_PKG / "static" / "report.css", out_dir / "report.css")
        written.append(out_dir / "report.css")

        brand_css = logo = None
        favicon_src = Path(cfg.favicon) if cfg.favicon and Path(cfg.favicon).exists() else _PKG / "static" / "favicon.svg"
        favicon = f"favicon{favicon_src.suffix}"
        shutil.copy(favicon_src, out_dir / favicon)
        written.append(out_dir / favicon)
        if cfg.css and Path(cfg.css).exists():
            shutil.copy(cfg.css, out_dir / "brand.css")
            brand_css = "brand.css"
            written.append(out_dir / "brand.css")
        if cfg.logo and Path(cfg.logo).exists():
            logo = f"logo{Path(cfg.logo).suffix}"
            shutil.copy(cfg.logo, out_dir / logo)
            written.append(out_dir / logo)

        shutil.copy(_PKG / "static" / "report.js", out_dir / "report.js")
        written.append(out_dir / "report.js")

        cases = list(report.runs[0].cases)  # multi mode: first run of the first judge
        if cfg.cases_order == "failed_first":
            cases.sort(key=lambda c: c.passed)  # stable: failing cases first, dataset order within groups

        for stale in out_dir.glob("cases-*.html"):  # leftovers of the earlier multi-file pagination
            stale.unlink()

        html = self._env(cfg).get_template("report.html.j2").render(
            report=report, summary=summarize(report), brand_css=brand_css, logo=logo, favicon=favicon,
            cases=cases, page_size=max(cfg.page_size, 0), pager_pages=max(cfg.pager_pages, 1),
            cases_order=cfg.cases_order, theme=cfg.theme if cfg.theme in ("dark", "light", "system") else "dark",
        )
        path = out_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        written.append(path)
        return written
