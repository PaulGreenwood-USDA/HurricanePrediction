"""Tests for the long-run stage history card.

The parquet store is injected so these run without touching data/.
"""
from __future__ import annotations

import pytest

from hurricane_asheville import stage_history as sh

pd = pytest.importorskip("pandas")


def _frame(pairs):
    return pd.DataFrame({
        "ts": pd.to_datetime([t for t, _ in pairs], utc=True),
        "value": [v for _, v in pairs],
    })


def _synthetic():
    """Five years of hourly-ish readings with a Helene spike."""
    idx = pd.date_range("2021-06-01", "2026-08-10", freq="6h", tz="UTC")
    values = [1.5 + (i % 24) * 0.01 for i in range(len(idx))]
    frame = pd.DataFrame({"ts": idx, "value": values})
    spike = frame["ts"] == pd.Timestamp("2024-09-27 00:00", tz="UTC")
    frame.loc[spike, "value"] = 18.47
    return frame


# ---- empty / degraded input ----------------------------------------------

def test_empty_history_returns_empty_result():
    out = sh.build(site_id="X", current_ft=1.0, history_df=_frame([]))
    assert out.points == []
    assert out.percentile is None


def test_missing_columns_are_survived():
    bad = pd.DataFrame({"nope": [1, 2, 3]})
    assert sh.build(site_id="X", current_ft=1.0, history_df=bad).points == []


def test_chart_points_empty_for_empty_history():
    out = sh.build(site_id="X", current_ft=1.0, history_df=_frame([]))
    assert sh.chart_points(out) == {}


# ---- daily maxima ---------------------------------------------------------

def test_resamples_to_daily_maxima_not_means():
    """A daily mean would flatten a flood peak out of existence."""
    frame = _frame([("2024-09-27T00:00Z", 18.47),
                    ("2024-09-27T06:00Z", 2.0),
                    ("2024-09-27T12:00Z", 2.0),
                    ("2024-09-27T18:00Z", 2.0)])
    out = sh.build(site_id="X", current_ft=2.0, history_df=frame)
    assert len(out.points) == 1
    assert out.points[0]["ft"] == pytest.approx(18.47)


# ---- Helene + record ------------------------------------------------------

def test_finds_the_helene_peak():
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic(),
                   thresholds={"record": 24.82})
    assert out.helene_ft == pytest.approx(18.47)
    assert out.helene_date == "2024-09-27"


def test_flags_a_daily_mean_series():
    """The gauge did not fail during Helene -- the 15-minute record holds the
    full 24.82 ft crest. A daily *mean* series tops out ~6 ft lower, and
    presenting 18.47 as 'the peak' would be wrong."""
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic(),
                   thresholds={"record": 24.82})
    assert out.daily_mean_series is True


def test_not_flagged_when_the_series_reaches_the_record():
    """Once 15-minute data is stored the plotted peak matches the crest and
    the caveat must stop firing."""
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic(),
                   thresholds={"record": 18.5})
    assert out.daily_mean_series is False


def test_axis_holds_the_record_line():
    """If vmax ignored the record, its reference line would fall off-chart."""
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic(),
                   thresholds={"record": 24.82})
    assert out.vmax >= 24.82


# ---- percentile -----------------------------------------------------------

def test_percentile_is_month_of_year_specific():
    """1.7 ft means different things in August and March."""
    idx = pd.date_range("2021-01-01", "2026-01-01", freq="1D", tz="UTC")
    vals = [5.0 if ts.month == 3 else 1.0 for ts in idx]
    out = sh.build(site_id="X", current_ft=2.0,
                   history_df=pd.DataFrame({"ts": idx, "value": vals}))
    # Last timestamp is in January, where every reading is 1.0, so 2.0 is top.
    assert out.percentile == pytest.approx(100.0)
    assert out.percentile_month == "January"


def test_percentile_needs_enough_samples():
    frame = _frame([("2026-08-0{}T00:00Z".format(i), 1.0) for i in range(1, 6)])
    out = sh.build(site_id="X", current_ft=1.0, history_df=frame)
    assert out.percentile is None


def test_percentile_none_without_current_reading():
    out = sh.build(site_id="X", current_ft=None, history_df=_synthetic())
    assert out.percentile is None


# ---- chart geometry -------------------------------------------------------

def test_chart_points_scale_into_the_box():
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic(),
                   thresholds={"record": 24.82})
    c = sh.chart_points(out, width=1000, height=140)
    xs, ys = zip(*(map(float, p.split(",")) for p in c["points"].split()))
    assert min(xs) == 0.0 and max(xs) == pytest.approx(1000.0)
    assert all(-1 <= y <= 141 for y in ys)


def test_higher_stage_plots_higher_on_screen():
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic(),
                   thresholds={"record": 24.82})
    c = sh.chart_points(out)
    assert c["y_record"] < c["y_helene"] < c["y_current"]


def test_chart_marks_year_boundaries():
    out = sh.build(site_id="X", current_ft=1.7, history_df=_synthetic())
    labels = [t["label"] for t in sh.chart_points(out)["year_ticks"]]
    assert labels == sorted(labels)
    assert "2024" in labels
