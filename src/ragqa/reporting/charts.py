"""Chart data for the HTML report.

The report draws its charts with plain HTML + CSS (no JavaScript, no libraries, no network): this
module only turns results into the numbers the templates need (bar widths, histogram bins, donut
gradients) and ``templates/_charts.html.j2`` renders them. Charts therefore follow the active
theme (System / Light / Dark), scale with the page and print.

Registered as Jinja globals by ``html_reporter.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any

from ragqa.evaluation.aggregate import layer_sort_key
from ragqa.models import CaseResult, RunResult
from ragqa.reporting.summary import LAYER_TITLES


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _ok(value: float, threshold: float | None, direction: str) -> bool | None:
    if threshold is None:
        return None
    return value >= threshold if direction == "min" else value <= threshold


def _scale_top(unit: str, *values: float | None) -> float:
    """Upper end of a bar: ratio scores live on 0..1, other units scale to the data."""
    if unit == "score":
        return 1.0
    return (max((v for v in values if v is not None), default=0.0) * 1.2) or 1.0


def bullet(value: float | None, threshold: float | None = None, direction: str = "min", unit: str = "score") -> dict | None:
    """A bar for ``value`` with a tick at ``threshold`` (pass/fail tone via ``ok``)."""
    if value is None:
        return None
    top = _scale_top(unit, value, threshold)
    return {"fill": round(_clamp(value / top * 100), 1),
            "tick": None if threshold is None else round(_clamp(threshold / top * 100), 1),
            "ok": _ok(value, threshold, direction)}


def range_bar(lo: float | None, hi: float | None, avg: float | None, threshold: float | None = None,
              direction: str = "min", unit: str = "score") -> dict | None:
    """Min-max whisker with a dot at the mean (multi-run spread)."""
    if lo is None or hi is None or avg is None:
        return None
    top = _scale_top(unit, hi, threshold)
    left = _clamp(lo / top * 100)
    return {"left": round(left, 1), "width": round(max(_clamp(hi / top * 100) - left, 1.2), 1),
            "mean": round(_clamp(avg / top * 100), 1),
            "tick": None if threshold is None else round(_clamp(threshold / top * 100), 1),
            "ok": _ok(avg, threshold, direction)}


def histogram(values: list[float], threshold: float | None = None, direction: str = "min",
              unit: str = "score", bins: int = 10) -> dict | None:
    """Distribution of per-case scores; bars are toned by whether the bin meets the threshold."""
    if not values:
        return None
    lo, hi = (0.0, 1.0) if unit == "score" else (0.0, max(max(values), threshold or 0.0) or 1.0)
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, max(0, int((v - lo) / width)))] += 1
    peak = max(counts) or 1
    out = []
    for i, c in enumerate(counts):
        b_lo, b_hi = lo + i * width, lo + (i + 1) * width
        out.append({"count": c, "h": round(c / peak * 100), "lo": b_lo, "hi": b_hi,
                    "ok": _ok((b_lo + b_hi) / 2, threshold, direction)})
    return {"bins": out, "n": len(values), "mean": mean(values), "lo": lo, "hi": hi,
            "tick": None if threshold is None else round(_clamp((threshold - lo) / (hi - lo) * 100), 1)}


def donut(segments: Iterable[tuple[str, float, str]]) -> dict:
    """``segments``: (label, value, tone) with tone in pass/fail/warn/info -> CSS conic-gradient."""
    segs = [(label, v, tone) for label, v, tone in segments]
    total = sum(v for _, v, _ in segs)
    stops, at, items = [], 0.0, []
    for label, v, tone in segs:
        pct = v / total * 100 if total else 0.0
        if v > 0:
            stops.append(f"var(--{tone}) {at:.2f}% {at + pct:.2f}%")
        at += pct
        items.append({"label": label, "value": v, "tone": tone, "pct": round(pct)})
    return {"style": "conic-gradient(" + (", ".join(stops) if stops else "var(--info-bg) 0 100%") + ")",
            "total": total, "items": items}


def layer_health(items: list) -> list[dict]:
    """Per layer: how many metrics passed / failed / could not be scored / are informational.

    Works for run aggregates (``errors``, ``passed``) and multi-run aggregates (``mean is None`` = error).
    """
    layers: dict[str, dict[str, int]] = {}
    for it in items:
        row = layers.setdefault(it.layer, {"passed": 0, "failed": 0, "errored": 0, "info": 0})
        errored = bool(getattr(it, "errors", 0)) or (hasattr(it, "runs") and it.mean is None)
        if errored:
            row["errored"] += 1
        elif it.passed is True:
            row["passed"] += 1
        elif it.passed is False:
            row["failed"] += 1
        else:
            row["info"] += 1
    out = []
    for layer, row in sorted(layers.items(), key=lambda kv: layer_sort_key(kv[0])):
        total = sum(row.values())
        out.append({"layer": layer, "title": LAYER_TITLES.get(layer, layer.title()), "total": total, **row,
                    "pct": {k: round(v / total * 100, 1) for k, v in row.items()}})
    return out


def stage_latency(cases: list[CaseResult]) -> list[dict]:
    """Mean / p95 latency per pipeline stage over the cases where the stage succeeded."""
    by_stage: dict[str, list[float]] = {}
    failures: dict[str, int] = {}
    for c in cases:
        for s in c.stages:
            if s.ok:
                by_stage.setdefault(s.stage, []).append(s.latency_ms)
            else:
                failures[s.stage] = failures.get(s.stage, 0) + 1
                by_stage.setdefault(s.stage, [])
    rows = []
    for stage, vals in by_stage.items():
        vals.sort()
        p95 = vals[min(len(vals) - 1, round(0.95 * (len(vals) - 1)))] if vals else None
        rows.append({"stage": stage, "n": len(vals), "failed": failures.get(stage, 0),
                     "mean": mean(vals) if vals else None, "p95": p95})
    top = max((r["p95"] for r in rows if r["p95"] is not None), default=0.0) or 1.0
    for r in rows:
        r["mean_pct"] = None if r["mean"] is None else round(_clamp(r["mean"] / top * 100), 1)
        r["p95_pct"] = None if r["p95"] is None else round(_clamp(r["p95"] / top * 100), 1)
    return rows


def metric_scores(run: RunResult) -> dict[str, list[float]]:
    """metric key -> per-case scores that were actually computed."""
    out: dict[str, list[float]] = {}
    for c in run.cases:
        for m in c.metrics:
            if not m.skipped and m.error is None and m.score is not None:
                out.setdefault(m.key, []).append(m.score)
    return out


GLOBALS: dict[str, Any] = {
    "bullet": bullet, "range_bar": range_bar, "histogram": histogram, "donut": donut,
    "layer_health": layer_health, "stage_latency": stage_latency, "metric_scores": metric_scores,
}
