"""Chart data helpers behind the HTML report's graphs (src/ragqa/reporting/charts.py)."""

from ragqa.reporting.charts import bullet, donut, histogram, range_bar


def test_bullet_ratio_and_threshold_tone():
    b = bullet(0.9, 0.8, "min", "score")
    assert b == {"fill": 90.0, "tick": 80.0, "ok": True}
    assert bullet(0.5, 0.8, "min", "score")["ok"] is False
    assert bullet(None) is None and bullet(0.5)["tick"] is None and bullet(0.5)["ok"] is None


def test_bullet_latency_uses_max_direction_and_scales_to_data():
    b = bullet(6000, 5000, "max", "ms")
    assert b["ok"] is False and b["fill"] > b["tick"] and b["fill"] <= 100


def test_histogram_bins_and_threshold_tick():
    h = histogram([0.0, 0.05, 0.95, 1.0], 0.7, "min", "score")
    assert [b["count"] for b in h["bins"]] == [2, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert h["tick"] == 70.0 and h["bins"][0]["ok"] is False and h["bins"][-1]["ok"] is True
    assert histogram([], 0.5) is None


def test_donut_segments_and_empty():
    d = donut([("a", 3, "pass"), ("b", 1, "fail"), ("c", 0, "warn")])
    assert "var(--pass) 0.00% 75.00%" in d["style"] and "var(--fail) 75.00% 100.00%" in d["style"]
    assert "warn" not in d["style"] and d["total"] == 4
    assert "var(--info-bg)" in donut([("a", 0, "pass")])["style"]


def test_range_bar_has_minimum_width_and_mean_marker():
    r = range_bar(0.8, 0.8, 0.8, 0.7, "min", "score")
    assert r["width"] >= 1.2 and r["mean"] == 80.0 and r["ok"] is True
