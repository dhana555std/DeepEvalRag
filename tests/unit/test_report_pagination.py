"""Client-side pagination of the HTML report (report.js): markup contract, pager logic, real browser behaviour.

* Python tests check what the reporter writes (all cases in one page, page-size attributes, report.js copied,
  ordering, stale files removed).
* ``node`` tests check the pure pager logic (``buildPager``) - skipped when node is not installed.
* Headless-Chrome tests load the generated ``report.html`` and inspect the resulting DOM (visible cases per
  page, hash navigation, a simulated click on Next) - skipped when Chrome/Chromium is not installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from ragqa import config as cfgmod
from ragqa.config import ReportingCfg
from ragqa.models import CaseResult, MetricResult, RunReport, RunResult
from ragqa.reporting import write_reports

JS = Path(__file__).resolve().parents[2] / "src" / "ragqa" / "reporting" / "static" / "report.js"


def _report(n: int, failing: set[int] = frozenset()) -> RunReport:
    cases = []
    for i in range(1, n + 1):
        bad = i in failing
        cases.append(CaseResult(
            case_id=f"C{i:04d}", question=f"question {i}", category="c", expected_answer="a", answer="x",
            metrics=[MetricResult(key="generator.citation", layer="generator", name="citation",
                                  score=0.0 if bad else 1.0, threshold=0.8, passed=not bad)]))
    run = RunResult(run_id="r1", judge="groq:x", cases=cases)
    return RunReport(title="T", mode="single", dataset_size=n, judges=["groq:x"], runs=[run])


def _case_ids(html: str) -> list[str]:
    return re.findall(r'id="case-(C\d+)"', html)


# ------------------------------------------------------------------------------ what the reporter writes
def test_all_cases_are_in_one_page_with_pagination_attributes(tmp_path):
    write_reports(_report(60), ReportingCfg(), tmp_path, ["html"])
    html = (tmp_path / "report.html").read_text()
    assert len(_case_ids(html)) == 60 and html.count("data-case>") == 60
    assert 'id="case-list" data-page-size="20" data-pager-pages="20"' in html
    assert '<script src="report.js" defer></script>' in html
    assert html.count("<nav class=\"pager\" data-pager") == 2  # top and bottom pager placeholders
    assert (tmp_path / "report.js").read_text() == JS.read_text()
    assert sorted(p.name for p in tmp_path.glob("*.html")) == ["report.html"]


def test_defaults_are_20_cases_per_page_and_20_page_numbers():
    cfg = cfgmod.load_reporting()
    assert cfg.page_size == 20 and cfg.pager_pages == 20 and cfg.cases_order == "failed_first"


def test_page_size_is_configurable_and_zero_disables(tmp_path):
    write_reports(_report(30), ReportingCfg(page_size=7, pager_pages=5), tmp_path / "a", ["html"])
    assert 'data-page-size="7" data-pager-pages="5"' in (tmp_path / "a" / "report.html").read_text()
    write_reports(_report(30), ReportingCfg(page_size=0), tmp_path / "b", ["html"])
    assert 'data-page-size="0"' in (tmp_path / "b" / "report.html").read_text()


def test_failed_cases_come_first_by_default_and_dataset_order_on_request(tmp_path):
    rep = _report(60, failing={55, 58})
    write_reports(rep, ReportingCfg(), tmp_path / "a", ["html"])
    html = (tmp_path / "a" / "report.html").read_text()
    assert _case_ids(html)[:3] == ["C0055", "C0058", "C0001"] and "problem cases first" in html
    write_reports(rep, ReportingCfg(cases_order="dataset"), tmp_path / "b", ["html"])
    html = (tmp_path / "b" / "report.html").read_text()
    assert _case_ids(html)[:3] == ["C0001", "C0002", "C0003"] and "problem cases first" not in html


def test_stale_page_files_from_the_previous_multi_file_layout_are_removed(tmp_path):
    (tmp_path / "cases-2.html").write_text("old")
    write_reports(_report(3), ReportingCfg(), tmp_path, ["html"])
    assert not (tmp_path / "cases-2.html").exists()


# ------------------------------------------------------------------------------ pager logic (node)
NODE = shutil.which("node")


def _build_pager(current: int, total: int, size: int | None = None) -> list[dict]:
    script = (
        "const m={exports:{}};new Function('module',require('fs').readFileSync(process.argv[1],'utf8'))(m);"
        "console.log(JSON.stringify(m.exports.buildPager(+process.argv[2],+process.argv[3],+process.argv[4]||undefined)))"
    )
    out = subprocess.check_output([NODE, "-e", script, str(JS), str(current), str(total), str(size or "")], text=True)
    return json.loads(out)


def _labels(items):
    return [i["label"] for i in items]


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestPagerLogic:
    def test_single_page_has_no_pager(self):
        assert _build_pager(1, 1) == []

    def test_few_pages_have_no_sheet_jumps(self):
        assert _labels(_build_pager(1, 3)) == ["‹ Prev", "1", "2", "3", "Next ›"]

    def test_at_most_20_page_numbers_are_shown(self):
        items = _build_pager(1, 100)
        assert [i["label"] for i in items if i["label"].isdigit()] == [str(n) for n in range(1, 21)]
        assert items[0]["target"] is None  # "Prev" is disabled on page 1
        assert items[-1]["label"] == "»" and items[-1]["target"] == 21 and items[-1]["title"] == "Pages 21-40"
        assert "«" not in _labels(items)  # no backward jump on the first sheet

    def test_next_and_sheet_jumps_cross_into_the_following_sheet(self):
        p20 = _build_pager(20, 45)
        assert next(i for i in p20 if i["label"] == "Next ›")["target"] == 21
        p21 = _build_pager(21, 45)
        assert [i["label"] for i in p21 if i["label"].isdigit()] == [str(n) for n in range(21, 41)]
        assert p21[0]["label"] == "«" and p21[0]["target"] == 1
        assert next(i for i in p21 if i["label"] == "‹ Prev")["target"] == 20
        assert next(i for i in p21 if i.get("current"))["label"] == "21"

    def test_last_sheet_is_partial_without_forward_jump(self):
        last = _build_pager(45, 45)
        assert [i["label"] for i in last if i["label"].isdigit()] == ["41", "42", "43", "44", "45"]
        assert "»" not in _labels(last) and next(i for i in last if i["label"] == "Next ›")["target"] is None

    def test_sheet_size_is_configurable(self):
        items = _build_pager(1, 10, 4)
        assert [i["label"] for i in items if i["label"].isdigit()] == ["1", "2", "3", "4"]
        assert items[-1]["label"] == "»" and items[-1]["target"] == 5


# ------------------------------------------------------------------------------ real browser (headless Chrome)
def _find_chrome() -> str | None:
    for candidate in (
        os.environ.get("CHROME"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


CHROME = _find_chrome()


class _Dom(HTMLParser):
    """Collects visible/hidden cases, open state and pager buttons from a dumped DOM."""

    def __init__(self):
        super().__init__()
        self.visible: list[str] = []
        self.opened: list[str] = []
        self.buttons: list[tuple[str, bool, bool]] = []  # (label, disabled, current)
        self.info = ""
        self._btn: dict | None = None
        self._in_info = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "details" and "data-case" in a:
            cid = a["id"].removeprefix("case-")
            if "hidden" not in a:
                self.visible.append(cid)
            if "open" in a:
                self.opened.append(cid)
        elif tag == "button" and "data-theme-set" not in a:  # theme switcher buttons are not pager buttons
            self._btn = {"disabled": "disabled" in a, "current": a.get("class") == "current", "label": ""}
        elif tag == "p" and "data-page-info" in a:
            self._in_info = True

    def handle_data(self, data):
        if self._btn is not None:
            self._btn["label"] += data
        elif self._in_info:
            self.info += data

    def handle_endtag(self, tag):
        if tag == "button" and self._btn is not None:
            self.buttons.append((self._btn["label"], self._btn["disabled"], self._btn["current"]))
            self._btn = None
        elif tag == "p":
            self._in_info = False


def _dump(report_dir: Path, suffix: str = "", inject: str = "") -> _Dom:
    page = report_dir / "report.html"
    if inject:
        copy = report_dir / "interaction.html"
        copy.write_text(page.read_text().replace("</body>", f"<script>window.addEventListener('load',function(){{{inject}}});</script></body>"))
        page = copy
    out = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--virtual-time-budget=4000", "--dump-dom",
         f"file://{page}{suffix}"], capture_output=True, text=True, timeout=60).stdout
    dom = _Dom()
    dom.feed(out)
    return dom


@pytest.fixture
def big_report(tmp_path):
    write_reports(_report(60, failing={3}), ReportingCfg(cases_order="dataset"), tmp_path, ["html"])
    return tmp_path


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
class TestInBrowser:
    def test_first_page_shows_20_cases_and_a_pager(self, big_report):
        dom = _dump(big_report)
        assert dom.visible == [f"C{i:04d}" for i in range(1, 21)]
        assert "Showing cases 1–20 of 60" in dom.info
        labels = [b[0] for b in dom.buttons[: len(dom.buttons) // 2]]  # top pager (bottom is identical)
        assert labels == ["‹ Prev", "1", "2", "3", "Next ›"]
        assert next(b for b in dom.buttons if b[0] == "‹ Prev")[1] is True  # disabled on page 1
        assert next(b for b in dom.buttons if b[2])[0] == "1"

    def test_clicking_next_shows_the_following_cases_on_the_same_page(self, big_report):
        dom = _dump(big_report, inject="Array.from(document.querySelectorAll('.pager button')).find(function(b){return b.textContent==='Next ›'}).click();")
        assert dom.visible == [f"C{i:04d}" for i in range(21, 41)]
        assert "Showing cases 21–40 of 60" in dom.info
        assert next(b for b in dom.buttons if b[2])[0] == "2"

    def test_clicking_a_page_number_and_last_page_is_partial(self, big_report):
        dom = _dump(big_report, inject="Array.from(document.querySelectorAll('.pager button')).find(function(b){return b.textContent==='3'}).click();")
        assert dom.visible == [f"C{i:04d}" for i in range(41, 61)]
        nxt = next(b for b in dom.buttons if b[0] == "Next ›")
        assert nxt[1] is True  # Next is disabled on the last page

    def test_url_hash_selects_a_page_and_a_case(self, big_report):
        assert _dump(big_report, "#page=2").visible[0] == "C0021"
        dom = _dump(big_report, "#case-C0055")
        assert dom.visible[0] == "C0041" and "C0055" in dom.opened

    def test_small_report_shows_everything_without_a_pager(self, tmp_path):
        write_reports(_report(12), ReportingCfg(), tmp_path, ["html"])
        dom = _dump(tmp_path)
        assert len(dom.visible) == 12 and dom.buttons == []
