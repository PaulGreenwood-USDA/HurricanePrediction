"""Live-data tests with mocked HTTP. Patch the per-module `requests`."""
from __future__ import annotations

import pytest

from hurricane_asheville import gauge as gauge_mod


def test_fetch_gauge_parses_payload(monkeypatch, fake_response, usgs_iv_payload):
    monkeypatch.setattr(gauge_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=usgs_iv_payload))
    g = gauge_mod.fetch_gauge()
    assert g is not None
    assert g.site_id == "03451500"
    assert g.stage_ft == 1.6  # last point
    assert g.discharge_cfs == 750.0
    assert g.flood_category == "below action"


def test_fetch_gauge_handles_network_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(gauge_mod.requests, "get", boom)
    assert gauge_mod.fetch_gauge() is None


def test_fetch_gauge_handles_empty_series(monkeypatch, fake_response):
    monkeypatch.setattr(gauge_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data={"value": {"timeSeries": []}}))
    assert gauge_mod.fetch_gauge() is None


def test_fetch_gauge_history(monkeypatch, fake_response, usgs_history_payload):
    monkeypatch.setattr(gauge_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=usgs_history_payload))
    hist = gauge_mod.fetch_gauge_history("03451500")
    assert len(hist) == 96
    assert hist[0][1] == 1.00
    assert hist[-1][1] == 1.50


def test_rate_of_rise_positive(usgs_history_payload):
    pts = usgs_history_payload["value"]["timeSeries"][0]["values"][0]["value"]
    hist = [(p["dateTime"], float(p["value"])) for p in pts]
    rate = gauge_mod.rate_of_rise_ft_per_hr(hist)
    assert rate is not None
    assert rate > 0


def test_rate_of_rise_too_short_returns_none():
    assert gauge_mod.rate_of_rise_ft_per_hr([]) is None
    assert gauge_mod.rate_of_rise_ft_per_hr([("t", 1.0)]) is None


def test_eta_to_stage_basic():
    eta = gauge_mod.eta_to_stage_hours(current_ft=5.0, target_ft=9.5,
                                       rate_ft_per_hr=0.5)
    assert eta == pytest.approx(9.0, rel=1e-3)


def test_eta_when_not_rising():
    assert gauge_mod.eta_to_stage_hours(5.0, 9.5, 0.0) is None
    assert gauge_mod.eta_to_stage_hours(5.0, 9.5, None) is None


def test_eta_already_at_target():
    assert gauge_mod.eta_to_stage_hours(10.0, 9.5, 0.5) == 0.0


def test_fetch_nws_alerts_parses(monkeypatch, fake_response, nws_alerts_payload):
    monkeypatch.setattr(gauge_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=nws_alerts_payload))
    alerts = gauge_mod.fetch_nws_alerts(35.6, -82.5)
    assert len(alerts) == 1
    assert alerts[0]["event"] == "Flood Watch"
    assert alerts[0]["severity"] == "Moderate"


def test_fetch_nws_alerts_network_error_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("timeout")
    monkeypatch.setattr(gauge_mod.requests, "get", boom)
    assert gauge_mod.fetch_nws_alerts(35.6, -82.5) == []


# ---- NWPS forecast traces -------------------------------------------------

NWPS_PAYLOAD = {
    "issuedTime": "2026-08-10T09:00:00Z",
    "data": [
        {"validTime": "2026-08-10T12:00:00Z", "primary": 2.0},
        {"validTime": "2026-08-10T18:00:00Z", "primary": 4.5},
        {"validTime": "2026-08-11T00:00:00Z", "primary": 3.1},
    ],
}


def test_nwps_forecast_queries_by_usgs_site_id(monkeypatch, fake_response):
    """NWPS resolves USGS ids directly. The old NWSLI table held bad ids
    (ASHN7, CTON7 both 404), so the primary gauge never got a forecast."""
    seen = {}

    def fake_get(url, *a, **k):
        seen["url"] = url
        return fake_response(json_data=NWPS_PAYLOAD)

    monkeypatch.setattr(gauge_mod.requests, "get", fake_get)
    out = gauge_mod.fetch_nwps_forecast("03451500")
    assert "03451500" in seen["url"]
    assert out is not None
    assert out["peak_ft"] == 4.5
    assert out["peak_t"] == "2026-08-10T18:00:00Z"
    assert len(out["points"]) == 3


def test_nwps_forecast_caches_within_ttl(monkeypatch, fake_response):
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return fake_response(json_data=NWPS_PAYLOAD)

    monkeypatch.setattr(gauge_mod.requests, "get", fake_get)
    gauge_mod.fetch_nwps_forecast("03451500")
    gauge_mod.fetch_nwps_forecast("03451500")
    assert calls["n"] == 1


def test_nwps_forecast_404_is_cached_negative(monkeypatch, fake_response):
    """A gauge with no forecast point shouldn't burn a request every refresh."""
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return fake_response(json_data={}, status_code=404)

    monkeypatch.setattr(gauge_mod.requests, "get", fake_get)
    assert gauge_mod.fetch_nwps_forecast("02089000") is None
    assert gauge_mod.fetch_nwps_forecast("02089000") is None
    assert calls["n"] == 1


def test_nwps_forecast_429_triggers_backoff(monkeypatch, fake_response):
    """NWPS allows 10 requests / 5 min; a 429 must stop further calls."""
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return fake_response(json_data={}, status_code=429)

    monkeypatch.setattr(gauge_mod.requests, "get", fake_get)
    assert gauge_mod.fetch_nwps_forecast("03451500") is None
    assert gauge_mod._NWPS_BACKOFF_UNTIL > 0
    # A different site must not issue a request while backing off.
    assert gauge_mod.fetch_nwps_forecast("03443000") is None
    assert calls["n"] == 1


def test_nwps_forecast_site_list_stays_within_rate_budget():
    """Every site here costs one request per refresh against a 10-per-5-min
    budget shared with everything else NWPS-backed."""
    assert len(gauge_mod.NWPS_FORECAST_SITES) <= 5
    assert gauge_mod.SITE_FRENCH_BROAD_ASHEVILLE in gauge_mod.NWPS_FORECAST_SITES


def test_fetch_all_gauges_iterates(monkeypatch, fake_response,
                                    usgs_iv_payload, usgs_history_payload):
    """Each gauge call alternates iv, history. We just feed the same payload."""
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        # Alternate: even = iv, odd = history. Doesn't matter for this test.
        if (calls["n"] % 2) == 1:
            return fake_response(json_data=usgs_iv_payload)
        return fake_response(json_data=usgs_history_payload)

    monkeypatch.setattr(gauge_mod.requests, "get", fake_get)
    out = gauge_mod.fetch_all_gauges()
    assert len(out) == len(gauge_mod.UPSTREAM_GAUGES)
    for g in out:
        assert "site_id" in g
        assert "stage_ft" in g
        assert "history" in g
