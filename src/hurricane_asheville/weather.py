"""Open-Meteo current/forecast weather (no auth)."""
from __future__ import annotations

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_current_weather(lat: float, lon: float, timeout: int = 15) -> dict:
    try:
        r = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,"
                           "wind_speed_10m,wind_direction_10m,pressure_msl,"
                           "weather_code",
                "hourly": "precipitation",
                "forecast_days": 3,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "America/New_York",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

    cur = data.get("current", {})
    hourly = data.get("hourly", {})
    next_72h_precip = sum(hourly.get("precipitation", [])[:72])
    return {
        "temp_f": cur.get("temperature_2m"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "precip_in": cur.get("precipitation"),
        "wind_mph": cur.get("wind_speed_10m"),
        "wind_dir_deg": cur.get("wind_direction_10m"),
        "pressure_mb": cur.get("pressure_msl"),
        "weather_code": cur.get("weather_code"),
        "next_72h_precip_in": round(next_72h_precip, 2),
        "as_of": cur.get("time"),
    }
