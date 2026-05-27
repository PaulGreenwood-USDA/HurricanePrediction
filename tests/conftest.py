"""Pytest fixtures + offline mode for hurricane-asheville test suite.

We never want tests to hit live APIs. The `requests` module is patched at
session scope to fail loudly if any test forgets to mock its endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class FakeResponse:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, json_data=None, text="", status_code=200):
        self._json = json_data
        self.text = text
        self.status_code = status_code
        self.content = (text or "").encode()

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_response():
    """Factory: build a FakeResponse with arbitrary JSON / text / status."""
    def _make(json_data=None, text="", status_code=200):
        return FakeResponse(json_data=json_data, text=text, status_code=status_code)
    return _make


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Patch every per-module `requests.get` so any unmocked call fails loudly.

    Each network-touching module imports `requests` at module scope and calls
    `requests.get(...)`. We replace that bound symbol on every such module so
    a test that forgets to set up its own monkeypatch gets a clear failure
    instead of silently hitting the live API.

    Individual tests still override these patches via
    `monkeypatch.setattr(mod.requests, "get", ...)`.
    """
    import requests as _requests_pkg

    def _explode(*args, **kwargs):
        raise AssertionError(
            "Live network call attempted in test (unmocked requests.get): "
            f"args={args}, kwargs={kwargs}"
        )

    # Patch the canonical module first so any direct `requests.get` call fails.
    monkeypatch.setattr(_requests_pkg, "get", _explode)

    # Also patch each per-module rebound `requests` symbol so module-local
    # calls like `gauge_mod.requests.get(...)` are blocked until a test
    # overrides them.
    for modname in (
        "hurricane_asheville.active",
        "hurricane_asheville.buoys",
        "hurricane_asheville.dem",
        "hurricane_asheville.fire_weather",
        "hurricane_asheville.forests",
        "hurricane_asheville.gauge",
        "hurricane_asheville.hurdat",
        "hurricane_asheville.landslide",
        "hurricane_asheville.smoke_air",
        "hurricane_asheville.soil",
        "hurricane_asheville.tides",
        "hurricane_asheville.weather",
        "hurricane_asheville.wildfire",
    ):
        try:
            mod = __import__(modname, fromlist=["requests"])
        except ImportError:
            continue
        if hasattr(mod, "requests"):
            monkeypatch.setattr(mod.requests, "get", _explode, raising=False)
    yield


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Isolated cache dir per test."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


# ---------- Sample payloads ----------

@pytest.fixture
def usgs_iv_payload():
    """Realistic USGS NWIS instantaneous-values shape (single site, two params)."""
    return {
        "value": {
            "timeSeries": [
                {
                    "sourceInfo": {"siteName": "FRENCH BROAD RIVER AT ASHEVILLE, NC"},
                    "variable": {"variableCode": [{"value": "00065"}]},
                    "values": [{"value": [
                        {"value": "1.20", "dateTime": "2026-05-05T13:00:00.000-04:00"},
                        {"value": "1.40", "dateTime": "2026-05-05T13:15:00.000-04:00"},
                        {"value": "1.60", "dateTime": "2026-05-05T13:30:00.000-04:00"},
                    ]}],
                },
                {
                    "sourceInfo": {"siteName": "FRENCH BROAD RIVER AT ASHEVILLE, NC"},
                    "variable": {"variableCode": [{"value": "00060"}]},
                    "values": [{"value": [
                        {"value": "750.0", "dateTime": "2026-05-05T13:30:00.000-04:00"},
                    ]}],
                },
            ]
        }
    }


@pytest.fixture
def usgs_history_payload():
    """24h of stage values, gently rising 1.0 -> 1.5 ft."""
    pts = []
    for i in range(96):  # 15-min cadence
        ft = 1.00 + (0.5 * i / 95.0)
        # Synthesize ISO timestamps (4 per hour)
        hh = i // 4
        mm = (i % 4) * 15
        pts.append({"value": f"{ft:.2f}",
                    "dateTime": f"2026-05-05T{hh:02d}:{mm:02d}:00.000-04:00"})
    return {"value": {"timeSeries": [{
        "sourceInfo": {"siteName": "FRENCH BROAD RIVER AT ASHEVILLE, NC"},
        "variable": {"variableCode": [{"value": "00065"}]},
        "values": [{"value": pts}]}]}}


@pytest.fixture
def open_meteo_weather_payload():
    return {
        "current": {
            "time": "2026-05-05T14:00",
            "temperature_2m": 72.5,
            "relative_humidity_2m": 55,
            "precipitation": 0.0,
            "wind_speed_10m": 6.0,
            "wind_direction_10m": 180,
            "pressure_msl": 1015.0,
            "weather_code": 1,
        },
        "hourly": {"precipitation": [0.05] * 72 + [0.0] * 24},
    }


@pytest.fixture
def open_meteo_soil_payload():
    return {
        "current": {
            "time": "2026-05-05T14:00",
            "soil_moisture_0_to_1cm": 0.42,    # saturated
            "soil_moisture_1_to_3cm": 0.40,
            "soil_moisture_3_to_9cm": 0.35,
            "soil_moisture_9_to_27cm": 0.32,
        },
        "hourly": {"precipitation": [0.05] * (24 * 7) + [0.0] * 24},
    }


@pytest.fixture
def nhc_active_payload():
    return {
        "activeStorms": [
            {
                "id": "AL092024",
                "name": "TESTSTORM",
                "binNumber": "AT1",
                "classification": "HU",
                "intensity": "85",
                "latitudeNumeric": 30.0,
                "longitudeNumeric": -82.0,
                "movement": "N at 12 mph",
                "publicAdvisory": {"url": "https://example.com/adv"},
            },
            {
                "id": "EP012024",
                "name": "PACIFIC",
                "binNumber": "EP1",  # should be filtered out
                "classification": "TS",
                "intensity": "40",
                "latitudeNumeric": 15.0,
                "longitudeNumeric": -110.0,
                "movement": "W at 10 mph",
            },
        ]
    }


@pytest.fixture
def nws_alerts_payload():
    return {
        "features": [
            {"properties": {
                "event": "Flood Watch",
                "severity": "Moderate",
                "headline": "Flood Watch in effect through Friday",
                "onset": "2026-05-05T18:00:00-04:00",
                "ends": "2026-05-06T18:00:00-04:00",
            }},
        ]
    }


@pytest.fixture
def coops_water_level_payload():
    return {"data": [{"t": "2026-05-05 14:00", "v": "1.23"}]}


@pytest.fixture
def coops_wind_payload():
    return {"data": [{"t": "2026-05-05 14:00", "s": "10.5", "g": "15.0", "d": "180"}]}


@pytest.fixture
def coops_pressure_payload():
    return {"data": [{"t": "2026-05-05 14:00", "v": "1015.5"}]}
