"""Tests for the historical bootstrap loaders (USGS DV + ERA5 archive)."""
from __future__ import annotations

import pytest

from hurricane_asheville import bootstrap


def _usgs_dv_payload():
    return {
        "value": {
            "timeSeries": [
                {
                    "variable": {"variableCode": [{"value": "00065"}]},
                    "values": [{"value": [
                        {"value": "2.10", "dateTime": "2024-09-26"},
                        {"value": "8.50", "dateTime": "2024-09-27"},
                        {"value": "23.40", "dateTime": "2024-09-28"},
                        {"value": "-999999", "dateTime": "2024-09-29"},
                    ]}],
                },
                {
                    "variable": {"variableCode": [{"value": "00060"}]},
                    "values": [{"value": [
                        {"value": "750", "dateTime": "2024-09-26"},
                        {"value": "12000", "dateTime": "2024-09-27"},
                    ]}],
                },
            ],
        },
    }


def _archive_payload():
    return {
        "daily": {
            "time": ["2024-09-26", "2024-09-27", "2024-09-28"],
            "precipitation_sum":   [10.0,  254.0, 50.0],
            "temperature_2m_max":  [25.0,   18.0,  14.0],
            "temperature_2m_min":  [15.0,   12.0,   9.0],
            "wind_speed_10m_max":  [20.0,   60.0,  35.0],
        },
    }


def test_fetch_usgs_dv_parses_both_params(monkeypatch, fake_response):
    from hurricane_asheville import bootstrap as boot
    monkeypatch.setattr(boot.requests, "get",
                        lambda *a, **k: fake_response(json_data=_usgs_dv_payload()))
    rows = boot.fetch_usgs_dv("03451500", "2024-09-26", "2024-09-29")
    assert len(rows) == 5  # 3 stage (sentinel dropped) + 2 discharge
    assert all(r["source"] == "usgs_dv" for r in rows)
    assert all(r["entity_type"] == "gauge" for r in rows)
    metrics = {r["metric"] for r in rows}
    assert metrics == {"stage_ft", "discharge_cfs"}
    helene_peak = [r for r in rows
                   if r["metric"] == "stage_ft" and r["ts"] == "2024-09-28"]
    assert helene_peak[0]["value"] == 23.40


def test_bootstrap_gauges_iterates_all_sites(monkeypatch, fake_response):
    from hurricane_asheville import bootstrap as boot
    seen: list[str] = []

    def fake_get(*a, **k):
        seen.append(k.get("params", {}).get("sites"))
        return fake_response(json_data=_usgs_dv_payload())

    monkeypatch.setattr(boot.requests, "get", fake_get)
    rows = boot.bootstrap_gauges(years=2, site_ids=["03451500", "03446000"],
                                  pause_s=0)
    assert seen == ["03451500", "03446000"]
    # Both sites contribute rows, both with usgs_dv source
    eids = {r["entity_id"] for r in rows}
    assert eids == {"03451500", "03446000"}


def test_archive_to_rows_converts_units(monkeypatch, fake_response):
    from hurricane_asheville.bootstrap import _archive_to_rows
    rows = _archive_to_rows(_archive_payload(), "forest", "Pisgah")
    by = {(r["ts"], r["metric"]): r["value"] for r in rows}
    # 254 mm -> 10.0 inches
    assert abs(by[("2024-09-27", "wx_precip_in_24h")] - 10.0) < 0.01
    # 25 C max -> 77.0 F
    assert abs(by[("2024-09-26", "wx_temp_max_f")] - 77.0) < 0.1
    # 60 km/h -> ~37.3 mph
    assert abs(by[("2024-09-27", "wx_wind_max_mph")] - 37.28) < 0.1
    assert all(r["source"] == "open_meteo_archive" for r in rows)
    assert all(r["entity_type"] == "forest" for r in rows)


def test_bootstrap_weather_iterates_all_points(monkeypatch, fake_response):
    from hurricane_asheville import bootstrap as boot
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return fake_response(json_data=_archive_payload())

    monkeypatch.setattr(boot.requests, "get", fake_get)
    # 3 fake points -> 3 archive calls
    pts = [("point", "asheville", 35.6, -82.5),
           ("forest", "Pisgah", 35.78, -82.30),
           ("district", "Pisgah/Appalachian", 35.917, -82.30)]
    rows = boot.bootstrap_weather(years=1, points=pts, pause_s=0)
    assert calls["n"] == 3
    assert {r["entity_id"] for r in rows} == {"asheville", "Pisgah",
                                                "Pisgah/Appalachian"}


def test_bootstrap_all_writes_partitions(monkeypatch, fake_response, tmp_path):
    from hurricane_asheville import bootstrap as boot, history

    def fake_get(*a, **k):
        url = a[0] if a else k.get("url", "")
        if "archive-api" in url:
            return fake_response(json_data=_archive_payload())
        return fake_response(json_data=_usgs_dv_payload())

    monkeypatch.setattr(boot.requests, "get", fake_get)
    # Keep the dataset small: only one site, one weather point.
    monkeypatch.setattr(boot, "_all_site_ids", lambda: ["03451500"])
    monkeypatch.setattr(boot, "_all_weather_points",
                         lambda: [("point", "asheville", 35.6, -82.5)])

    summary = boot.bootstrap_all(years=1, base_dir=tmp_path)
    assert summary["gauge_rows"] > 0
    assert summary["weather_rows"] > 0
    assert summary["partitions_written"] >= 1

    df = history.load_history(base_dir=tmp_path)
    assert set(df["source"].unique()) == {"usgs_dv", "open_meteo_archive"}


# ---- reservoir pool elevation ---------------------------------------------

def test_usgs_dv_parses_pool_elevation(monkeypatch, fake_response):
    """Reservoirs report 00062, not 00065. Asking only for stage returned
    nothing at all for Falls Lake and almost nothing for Jordan Lake."""
    payload = {"value": {"timeSeries": [{
        "variable": {"variableCode": [{"value": "00062"}]},
        "values": [{"value": [
            {"dateTime": "2024-09-27T00:00:00.000", "value": "248.9"},
        ]}],
    }]}}
    monkeypatch.setattr(bootstrap.requests, "get",
                        lambda *a, **k: fake_response(json_data=payload))
    rows = bootstrap.fetch_usgs_dv("02087182", "2024-09-01", "2024-09-30")
    assert len(rows) == 1
    assert rows[0]["metric"] == "pool_elevation_ft"
    assert rows[0]["value"] == 248.9


def test_usgs_dv_requests_all_three_parameters(monkeypatch, fake_response):
    seen = {}

    def fake_get(url, params=None, **k):
        seen.update(params or {})
        return fake_response(json_data={"value": {"timeSeries": []}})

    monkeypatch.setattr(bootstrap.requests, "get", fake_get)
    bootstrap.fetch_usgs_dv("02087182", "2024-09-01", "2024-09-30")
    assert "00062" in seen["parameterCd"]


# ---- soil moisture backfill -----------------------------------------------

def test_soil_hourly_reduced_to_daily_mean():
    """Five years hourly at 13 points is ~1.1M values; consumers work daily."""
    data = {"hourly": {
        "time": ["2024-09-27T00:00", "2024-09-27T12:00", "2024-09-28T00:00"],
        "soil_moisture_0_to_7cm": [0.50, 0.54, 0.48],
        "soil_moisture_7_to_28cm": [0.40, 0.42, 0.38],
    }}
    rows = bootstrap._soil_to_daily_rows(data, "point", "asheville")
    top = {r["ts"]: r["value"] for r in rows if r["metric"] == "soil_era5_0_7cm"}
    assert top["2024-09-27"] == pytest.approx(0.52)
    assert top["2024-09-28"] == pytest.approx(0.48)
    assert {r["metric"] for r in rows} == {"soil_era5_0_7cm", "soil_era5_7_28cm"}


def test_soil_metric_names_do_not_collide_with_live_feed():
    """ERA5 offers 0-7 cm; the live feed reads 0-1 cm. Same quantity, different
    measurement -- they must not concatenate into one series that silently
    changes definition partway through."""
    assert "soil_era5" in "".join(bootstrap._SOIL_VAR_TO_METRIC.values())
    assert "soil_moisture_top" not in bootstrap._SOIL_VAR_TO_METRIC.values()


def test_soil_rows_skip_nulls():
    data = {"hourly": {"time": ["2024-09-27T00:00", "2024-09-27T01:00"],
                       "soil_moisture_0_to_7cm": [None, 0.5],
                       "soil_moisture_7_to_28cm": [None, None]}}
    rows = bootstrap._soil_to_daily_rows(data, "point", "asheville")
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(0.5)


def test_soil_empty_payload():
    assert bootstrap._soil_to_daily_rows({}, "point", "asheville") == []
