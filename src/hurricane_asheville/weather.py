"""Open-Meteo current/forecast weather (no auth)."""
from __future__ import annotations

import math
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _wet_bulb_f(temp_f: float, rh: float) -> float:
    """Stull (2011) empirical wet-bulb approximation.
    Inputs: temp in °F, relative humidity in %. Returns °F.
    Valid roughly for 0–50 °C, 5–99 % RH.
    """
    t = (temp_f - 32.0) * 5.0 / 9.0
    tw = (
        t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw * 9.0 / 5.0 + 32.0, 1)


def _heat_index_f(temp_f: float, rh: float) -> float:
    """NWS Rothfusz heat index (°F).
    Uses the Steadman simple formula when T < 80 °F.
    """
    if temp_f < 80.0:
        hi = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
        return round(hi, 1)
    hi = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f ** 2
        - 0.05391553 * rh ** 2
        + 0.00122874 * temp_f ** 2 * rh
        + 0.00085282 * temp_f * rh ** 2
        - 0.00000199 * temp_f ** 2 * rh ** 2
    )
    if rh < 13 and 80 <= temp_f <= 112:
        hi -= (13 - rh) / 4 * math.sqrt((17 - abs(temp_f - 95)) / 17)
    elif rh > 85 and 80 <= temp_f <= 87:
        hi += (rh - 85) / 10 * (87 - temp_f) / 5
    return round(hi, 1)


def _heat_category(hi: float) -> tuple[str, str]:
    """Return (label, hex_color) for the NWS heat index category."""
    if hi >= 125:
        return "Extreme Danger", "#6a1b9a"
    if hi >= 103:
        return "Danger", "#c62828"
    if hi >= 90:
        return "Extreme Caution", "#ef6c00"
    if hi >= 80:
        return "Caution", "#f9a825"
    return "Normal", "#2e7d32"


def fetch_current_weather(lat: float, lon: float, timeout: int = 15) -> dict:
    try:
        r = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,"
                           "wind_speed_10m,wind_direction_10m,pressure_msl,"
                           "weather_code,dew_point_2m,apparent_temperature",
                "hourly": "precipitation,temperature_2m,apparent_temperature,"
                          "relative_humidity_2m",
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

    temp_f = cur.get("temperature_2m")
    rh = cur.get("relative_humidity_2m")
    wet_bulb = _wet_bulb_f(temp_f, rh) if (temp_f is not None and rh is not None) else None
    heat_index = _heat_index_f(temp_f, rh) if (temp_f is not None and rh is not None) else None
    hi_label, hi_color = _heat_category(heat_index) if heat_index is not None else ("Unknown", "#555")

    return {
        "temp_f": temp_f,
        "humidity_pct": rh,
        "precip_in": cur.get("precipitation"),
        "wind_mph": cur.get("wind_speed_10m"),
        "wind_dir_deg": cur.get("wind_direction_10m"),
        "pressure_mb": cur.get("pressure_msl"),
        "weather_code": cur.get("weather_code"),
        "next_72h_precip_in": round(next_72h_precip, 2),
        "as_of": cur.get("time"),
        "dew_point_f": cur.get("dew_point_2m"),
        "apparent_temp_f": cur.get("apparent_temperature"),
        "wet_bulb_f": wet_bulb,
        "heat_index_f": heat_index,
        "heat_category": hi_label,
        "heat_color": hi_color,
        "hourly_temp_f": [v for v in (hourly.get("temperature_2m") or [])[:24] if v is not None],
        "hourly_apparent_f": [v for v in (hourly.get("apparent_temperature") or [])[:24] if v is not None],
        "hourly_rh": [v for v in (hourly.get("relative_humidity_2m") or [])[:24] if v is not None],
        "hourly_times": list((hourly.get("time") or [])[:24]),
    }
