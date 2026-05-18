"""Tests for the Flask dashboard. We patch _collect to avoid any real I/O."""
from __future__ import annotations

import pytest

from hurricane_asheville import dashboard


@pytest.fixture
def fake_state():
    return {
        "as_of": "2026-04-01 12:00:00 UTC",
        "as_of_epoch": 1_800_000_000,
        "index": {
            "score": 42,
            "label": "ALERT",
            "color": "#ef6c00",
            "components": {"stage": 10, "qpf": 12, "storm": 5,
                           "rise": 5, "alert": 10, "soil": 0},
            "triggers": {"stage_above_action": False,
                         "stage_above_minor": False,
                         "qpf_over_3in": True,
                         "qpf_over_1in": True,
                         "storm_within_500mi": False,
                         "storm_within_1000mi": True,
                         "river_rising_fast": False,
                         "nws_flood_or_tropical": True,
                         "soil_saturated": False,
                         "wet_week": False},
        },
        "gauges": [],
        "primary_site": "03451500",
        "flood_stages": {"action": 8.0, "minor": 9.5,
                         "moderate": 11.5, "major": 13.5},
        "storms": [],
        "alerts": [],
        "weather": {"next_72h_precip_in": 3.6},
        "soil": {"saturated": False, "soil_moisture_top": 0.20,
                 "past_7d_precip_in": 1.2, "condition": "moist"},
        "coastal": [],
        "forests": [],
        "season": {"named_storms": 13, "hurricanes": 6,
                   "major_hurricanes": 3, "ace": 90,
                   "p_us_major_landfall": 0.43,
                   "issued": "April 2026"},
        "asheville": {"lat": 35.5951, "lon": -82.5515},
    }


@pytest.fixture(autouse=True)
def _reset_cache():
    dashboard._CACHE["data"] = None
    dashboard._CACHE["ts"] = 0.0
    yield
    dashboard._CACHE["data"] = None
    dashboard._CACHE["ts"] = 0.0


@pytest.fixture
def client(monkeypatch, fake_state):
    monkeypatch.setattr(dashboard, "_collect", lambda: fake_state)
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


def test_index_route_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Sanity checks on the rendered HTML
    assert "Asheville" in body
    assert "Flood Index" in body or "flood index" in body.lower()


def test_api_state_returns_json(client, fake_state):
    resp = client.get("/api/state")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    assert data["index"]["score"] == 42
    assert data["index"]["label"] == "ALERT"
    for key in ("gauges", "storms", "alerts", "weather",
                "soil", "coastal", "forests", "season", "asheville"):
        assert key in data


def test_collect_uses_cache(monkeypatch, fake_state):
    """Second call within TTL should not re-invoke the upstream fetchers."""
    calls = {"n": 0}

    def fake_gauges():
        calls["n"] += 1
        return []

    monkeypatch.setattr(dashboard, "fetch_all_gauges", fake_gauges)
    monkeypatch.setattr(dashboard, "fetch_active_storms", lambda: [])
    monkeypatch.setattr(dashboard, "fetch_nws_alerts", lambda lat, lon: [])
    monkeypatch.setattr(dashboard, "fetch_current_weather",
                        lambda lat, lon: {"next_72h_precip_in": 0.0})
    monkeypatch.setattr(dashboard, "fetch_soil_state",
                        lambda lat, lon: {"saturated": False,
                                          "soil_moisture_top": 0.10,
                                          "past_7d_precip_in": 0.0})
    monkeypatch.setattr(dashboard, "fetch_all_coastal", lambda: [])
    monkeypatch.setattr(dashboard, "fetch_all_forests", lambda storms: [])

    a = dashboard._collect()
    b = dashboard._collect()
    assert calls["n"] == 1            # cache hit on second call
    assert a is b                     # same object reference


def test_collect_assembles_full_payload(monkeypatch):
    monkeypatch.setattr(dashboard, "fetch_all_gauges",
                        lambda: [{"site_id": "03451500", "stage_ft": 5.0,
                                  "rate_ft_per_hr": 0.05}])
    monkeypatch.setattr(dashboard, "fetch_active_storms", lambda: [])
    monkeypatch.setattr(dashboard, "fetch_nws_alerts", lambda lat, lon: [])
    monkeypatch.setattr(dashboard, "fetch_current_weather",
                        lambda lat, lon: {"next_72h_precip_in": 1.0})
    monkeypatch.setattr(dashboard, "fetch_soil_state",
                        lambda lat, lon: {"saturated": False,
                                          "soil_moisture_top": 0.20,
                                          "past_7d_precip_in": 0.5})
    monkeypatch.setattr(dashboard, "fetch_all_coastal", lambda: [])
    monkeypatch.setattr(dashboard, "fetch_all_forests", lambda storms: [])

    state = dashboard._collect()
    assert "index" in state
    assert state["index"]["score"] >= 0
    assert state["index"]["score"] <= 100
    assert state["primary_site"] == "03451500"
