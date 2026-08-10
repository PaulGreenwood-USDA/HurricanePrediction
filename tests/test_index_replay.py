"""Tests for the historical Flood Index replay.

The replay is the only evidence that the index's hand-tuned weights do
anything useful, so its reconstruction logic needs to be right -- particularly
the parts that are approximations, which must be approximations in the
conservative direction.
"""
from __future__ import annotations

import pytest

from hurricane_asheville import index_replay as ir

pd = pytest.importorskip("pandas")


def _hist(rows):
    """rows: [(ts, entity_id, metric, value, source)]"""
    return pd.DataFrame({
        "ts": pd.to_datetime([r[0] for r in rows], utc=True),
        "entity_id": [r[1] for r in rows],
        "metric": [r[2] for r in rows],
        "value": [r[3] for r in rows],
        "source": [r[4] if len(r) > 4 else "usgs_dv" for r in rows],
        "entity_type": ["gauge"] * len(rows),
    })


def _flood_history():
    """A calm baseline with one sharp rise, no storms involved."""
    days = pd.date_range("2024-09-20", "2024-09-30", freq="1D", tz="UTC")
    stages = [1.5, 1.5, 1.6, 1.6, 2.0, 3.0, 9.7, 18.5, 18.4, 13.1, 8.5]
    rows = [(d, "03451500", "stage_ft", s, "usgs_dv")
            for d, s in zip(days, stages)]
    rows += [(d, "asheville", "wx_precip_in_24h", p, "open_meteo_archive")
             for d, p in zip(days, [0, 0, 0.1, 0.4, 2.8, 4.0, 1.8, 0.2, 0, 0, 0])]
    rows += [(d, "asheville", "soil_era5_0_7cm", v, "open_meteo_archive")
             for d, v in zip(days, [0.30, 0.31, 0.35, 0.41, 0.44, 0.46,
                                     0.51, 0.51, 0.49, 0.48, 0.47])]
    return _hist(rows)


# ---- degraded input -------------------------------------------------------

def test_replay_without_history_is_empty():
    out = ir.replay(history_df=_hist([]))
    assert out.days == []
    assert out.notes


def test_replay_without_stage_for_site():
    hist = _hist([("2024-09-20", "other", "stage_ft", 1.0)])
    out = ir.replay(history_df=hist, site_id="03451500")
    assert out.days == []


# ---- scoring --------------------------------------------------------------

def test_replay_uses_the_live_scorer(monkeypatch):
    """A parallel reimplementation would drift from the dashboard."""
    calls = {"n": 0}
    real = ir.compute_index

    def spy(**kw):
        calls["n"] += 1
        return real(**kw)

    monkeypatch.setattr(ir, "compute_index", spy)
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    assert calls["n"] == len(out.days) > 0


def test_flood_day_scores_far_above_calm_days(monkeypatch):
    """With no storm and no alert credit, an 18.5 ft stage still has to
    separate decisively from a 1.5 ft baseline."""
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    by = {d.date: d for d in out.days}
    assert by["2024-09-21"].label == "CALM"
    assert by["2024-09-27"].score - by["2024-09-21"].score > 40
    assert by["2024-09-27"].label in ("ALERT", "WARNING", "EMERGENCY")
    # Stage should be pinned at its ceiling well before the crest.
    assert by["2024-09-27"].components["stage"] == 35.0


def test_storm_proximity_raises_the_score(monkeypatch):
    """Same river and rainfall, but a hurricane 90 mi away."""
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    without = {d.date: d.score for d in ir.replay(history_df=_flood_history()).days}

    monkeypatch.setattr(ir, "storm_distances",
                        lambda *a, **k: {"2024-09-27": (90.0, "HELENE")})
    with_storm = {d.date: d.score
                  for d in ir.replay(history_df=_flood_history()).days}
    assert with_storm["2024-09-27"] > without["2024-09-27"]


def test_storm_name_is_carried_through(monkeypatch):
    monkeypatch.setattr(ir, "storm_distances",
                        lambda *a, **k: {"2024-09-27": (90.0, "HELENE")})
    out = ir.replay(history_df=_flood_history())
    day = next(d for d in out.days if d.date == "2024-09-27")
    assert day.storm_name == "HELENE"
    assert day.nearest_storm_mi == 90.0


# ---- approximations must be conservative ----------------------------------

def test_alert_component_is_always_zero(monkeypatch):
    """NWS alert state is not archived, so the replay cannot credit it.
    That biases every replayed score down by up to 10 points."""
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    assert all(d.components["alert"] == 0.0 for d in out.days)


def test_rate_of_rise_is_a_daily_average(monkeypatch):
    """An 8.8 ft rise in one day reads as ~0.37 ft/hr, not the multiple
    ft/hr the river actually did -- the rise component is understated."""
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    day = next(d for d in out.days if d.date == "2024-09-27")
    assert day.components["rise"] < 5.0     # of a possible 10


def test_qpf_uses_forward_looking_observed_rain(monkeypatch):
    """The replay assumes a perfect forecast, so it is optimistic. The day
    before the peak must already see the rain that is coming."""
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    by = {d.date: d for d in out.days}
    # 24th onward: 2.8 + 4.0 + 1.8 falls over the following three days.
    assert by["2024-09-24"].precip_72h_in == pytest.approx(8.6, abs=0.01)


def test_rolling_forward_sum_looks_ahead_not_behind():
    s = pd.Series([1.0, 2.0, 3.0, 0.0, 0.0])
    fwd = ir._rolling_forward_sum(s, days=3)
    assert fwd.iloc[0] == pytest.approx(6.0)
    assert fwd.iloc[3] == pytest.approx(0.0)


def test_soil_saturation_threshold(monkeypatch):
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    by = {d.date: d for d in out.days}
    assert by["2024-09-20"].triggers["soil_saturated"] is False   # 0.30
    assert by["2024-09-27"].triggers["soil_saturated"] is True    # 0.51


# ---- validation summary ---------------------------------------------------

def test_validation_reports_event_score_and_lead_time(monkeypatch):
    monkeypatch.setattr(ir, "storm_distances",
                        lambda *a, **k: {"2024-09-26": (638.0, "HELENE"),
                                          "2024-09-27": (90.0, "HELENE")})
    out = ir.replay(history_df=_flood_history())
    v = ir.build_validation(out, event_date="2024-09-27", event_name="Helene")
    assert v["event"]["name"] == "Helene"
    assert v["event"]["score"] >= 60
    assert v["event"]["lead_days"] >= 1
    assert v["base_rate"]["total_days"] == len(out.days)


def test_validation_base_rate_counts_alert_days(monkeypatch):
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    v = ir.build_validation(out)
    alert_days = [d for d in out.days if d.score >= 40]
    assert v["base_rate"]["alert_or_above_days"] == len(alert_days)
    assert len(v["elevated_days"]) == len(alert_days)


def test_validation_empty_for_empty_replay():
    assert ir.build_validation(ir.ReplayResult()) == {}


def test_validation_carries_caveats(monkeypatch):
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    v = ir.build_validation(ir.replay(history_df=_flood_history()))
    joined = " ".join(v["caveats"]).lower()
    assert "alert" in joined and "perfect" in joined


def test_write_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    path = tmp_path / "validation.json"
    ir.write_validation(out, str(path))
    loaded = ir.load_validation(str(path))
    assert loaded["event"]["date"] == "2024-09-27"


def test_load_validation_missing_file_is_empty():
    assert ir.load_validation("/nonexistent/path.json") == {}


# ---- event window ---------------------------------------------------------

def test_event_summary_window(monkeypatch):
    monkeypatch.setattr(ir, "storm_distances", lambda *a, **k: {})
    out = ir.replay(history_df=_flood_history())
    window = ir.event_summary(out, "2024-09-27", window_days=2)
    assert [d.date for d in window] == ["2024-09-25", "2024-09-26",
                                         "2024-09-27", "2024-09-28",
                                         "2024-09-29"]
