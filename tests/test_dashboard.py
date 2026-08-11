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
        "weather": {
            "next_72h_precip_in": 3.6,
            "temp_f": 78.0,
            "humidity_pct": 65,
            "wind_mph": 8.0,
            "wind_dir_deg": 225,
            "pressure_mb": 1015.2,
            "precip_in": 0.0,
            "dew_point_f": 64.0,
            "apparent_temp_f": 80.0,
            "wet_bulb_f": None,
            "heat_index_f": None,
            "heat_category": "Normal",
            "heat_color": "#2e7d32",
            "hourly_temp_f": [],
            "hourly_apparent_f": [],
            "hourly_wet_bulb_f": [],
            "hourly_rh": [],
            "hourly_times": [],
            "as_of": "2026-04-01T12:00",
        },
        "soil": {"saturated": False, "soil_moisture_top": 0.20,
                 "past_7d_precip_in": 1.2, "condition": "moist"},
        "coastal": [],
        "buoys": [],
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


def test_page_declares_mobile_viewport(client):
    """Without this the page falls back to a 980px viewport and renders
    zoomed-out on phones -- the primary device during a flood event."""
    body = client.get("/").get_data(as_text=True)
    assert 'name="viewport"' in body
    assert "width=device-width" in body


def test_page_links_external_css_and_js(client):
    """Markup, styles and script are separate files now; a regression back to
    one inline blob would make the state payload ship twice again."""
    body = client.get("/").get_data(as_text=True)
    assert "dashboard.css" in body
    assert "dashboard.js" in body
    assert "const STATE = {" not in body


def test_map_payload_excludes_gauge_history(client, fake_state):
    """The map never reads per-gauge history; inlining it doubled page size."""
    fake_state["gauges"] = [
        {"site_id": "03451500", "label": "French Broad @ Asheville",
         "role": "primary", "lat": 35.6, "lon": -82.6, "stage_ft": 1.67,
         "display_ft": 1.67, "display_units": "ft", "pool_elevation_ft": None,
         "flood_category": "below action", "flood_class": "below-action",
         "thresholds": None, "thresholds_label": "", "rate_ft_per_hr": 0.0,
         "history": [{"t": f"2026-08-10T{h:02d}:00", "ft": 1.6} for h in range(24)],
         "eta_minor_hr": None, "eta_moderate_hr": None, "eta_major_hr": None,
         "nwps_forecast": None},
    ]
    payload = dashboard.map_state(fake_state)
    assert payload["gauges"][0]["site_id"] == "03451500"
    assert "history" not in payload["gauges"][0]
    assert "nwps_forecast" not in payload["gauges"][0]


def test_stage_pills_use_single_token_classes(client, fake_state):
    """Regression: the template derived pill classes by munging the label,
    so 'below action' produced class="stage-pill below action" and matched
    the '.stage-pill.action' rule -- every safe gauge looked like a warning."""
    fake_state["gauges"] = [
        {"site_id": "03451500", "label": "French Broad @ Asheville",
         "role": "primary", "lat": 35.6, "lon": -82.6, "stage_ft": 1.67,
         "pool_elevation_ft": None, "display_ft": 1.67, "display_units": "ft",
         "flood_category": "below action", "flood_class": "below-action",
         "thresholds": {"action": 6.5, "minor": 9.5,
                        "moderate": 13.0, "major": 18.0},
         "thresholds_label": "NWS flood stages here: action 6.5 ft",
         "rate_ft_per_hr": 0.0, "history": [],
         "eta_minor_hr": None, "eta_moderate_hr": None, "eta_major_hr": None,
         "nwps_forecast": None},
        {"site_id": "02087182", "label": "Falls Lake above dam",
         "role": "reservoir", "lat": 35.94, "lon": -78.58, "stage_ft": None,
         "pool_elevation_ft": 248.90, "display_ft": 248.90,
         "display_units": "ft pool elev",
         "flood_category": "below action", "flood_class": "below-action",
         "thresholds": {"action": 264.0, "minor": 265.0,
                        "moderate": 266.0, "major": 267.0},
         "thresholds_label": "NWS flood stages here: action 264 ft pool elev",
         "rate_ft_per_hr": 0.0, "history": [],
         "eta_minor_hr": None, "eta_moderate_hr": None, "eta_major_hr": None,
         "nwps_forecast": None},
    ]
    body = client.get("/").get_data(as_text=True)
    assert 'class="stage-pill below-action"' in body
    assert 'class="stage-pill below action"' not in body


def test_reservoir_row_shows_pool_elevation_with_units(client, fake_state):
    """A reservoir reports 00062 pool elevation, not river stage. Before the
    dashboard requested 00062 the row rendered a bare '?'."""
    fake_state["gauges"] = [
        {"site_id": "02087182", "label": "Falls Lake above dam",
         "role": "reservoir", "lat": 35.94, "lon": -78.58, "stage_ft": None,
         "pool_elevation_ft": 248.90, "display_ft": 248.90,
         "display_units": "ft pool elev",
         "flood_category": "below action", "flood_class": "below-action",
         "thresholds": None, "thresholds_label": "",
         "rate_ft_per_hr": 0.0, "history": [],
         "eta_minor_hr": None, "eta_moderate_hr": None, "eta_major_hr": None,
         "nwps_forecast": None},
    ]
    body = client.get("/").get_data(as_text=True)
    assert "248.90" in body
    assert "ft pool elev" in body


def test_api_state_returns_json(client, fake_state):
    resp = client.get("/api/state")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    assert data["index"]["score"] == 42
    assert data["index"]["label"] == "ALERT"
    for key in ("gauges", "storms", "alerts", "weather",
                "soil", "coastal", "buoys", "forests", "season", "asheville"):
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
    monkeypatch.setattr(dashboard, "fetch_all_buoys", lambda: [])
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
    monkeypatch.setattr(dashboard, "fetch_all_buoys", lambda: [])
    monkeypatch.setattr(dashboard, "fetch_all_forests", lambda storms: [])

    state = dashboard._collect()
    assert "index" in state
    assert state["index"]["score"] >= 0
    assert state["index"]["score"] <= 100
    assert state["primary_site"] == "03451500"
