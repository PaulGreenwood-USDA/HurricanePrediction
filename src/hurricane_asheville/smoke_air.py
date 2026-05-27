"""Air-quality and smoke proxy for the dashboard.

True NOAA HMS smoke polygons require either shapefile parsing or a private
KMZ download chain, neither of which we want to ship as a hard dependency.
And EPA AirNow's JSON API requires a free key.

What we DO use today (no auth required): **Open-Meteo Air Quality API**.
It exposes a free, hourly grid of PM2.5, PM10, ozone, NO2, SO2, AQI (US +
European), plus a `uv_index`. That is sufficient for "is the air safe in
Pisgah right now" context next to the wildfire pill.

If you later wire up an EPA AirNow key, you can replace ``fetch_air_quality``
with a real station-based lookup; the dashboard payload shape will stay the
same.
"""
from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


_AQI_BANDS = (
    (300, "HAZARDOUS",        "#7e0023"),
    (200, "VERY UNHEALTHY",   "#8f3f97"),
    (150, "UNHEALTHY",        "#c62828"),
    (100, "UNHEALTHY SENS.",  "#ef6c00"),
    ( 50, "MODERATE",         "#fdd835"),
    (  0, "GOOD",             "#4caf50"),
)


def _classify(aqi: float | None) -> tuple[str, str]:
    if aqi is None:
        return "n/a", "#9aa0aa"
    for cutoff, label, color in _AQI_BANDS:
        if aqi >= cutoff:
            return label, color
    return "GOOD", "#4caf50"


def fetch_air_quality(lat: float, lon: float, timeout: int = 15) -> dict:
    """Current PM2.5, PM10, ozone, and US-EPA AQI at (lat, lon)."""
    try:
        r = requests.get(
            URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": ("pm10,pm2_5,carbon_monoxide,ozone,"
                            "us_aqi,us_aqi_pm2_5,us_aqi_ozone"),
                "timezone": "America/New_York",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.info("Air quality fetch skipped (%s)", exc)
        return {"error": str(exc)}

    cur = data.get("current") or {}
    aqi = cur.get("us_aqi")
    label, color = _classify(aqi)
    return {
        "as_of": cur.get("time"),
        "pm2_5": cur.get("pm2_5"),
        "pm10": cur.get("pm10"),
        "ozone": cur.get("ozone"),
        "co": cur.get("carbon_monoxide"),
        "us_aqi": aqi,
        "us_aqi_pm2_5": cur.get("us_aqi_pm2_5"),
        "us_aqi_ozone": cur.get("us_aqi_ozone"),
        "label": label,
        "color": color,
    }
