"""Soil moisture / antecedent precipitation - the missing flood pre-conditioner.

Open-Meteo's forecast API exposes hourly soil_moisture at multiple depths (m^3/m^3).
We hit it once per forest centroid + Asheville and report:
  - current 0-7 cm and 7-28 cm volumetric water content
  - 7-day antecedent precipitation total
  - saturation flag (>= 0.40 m^3/m^3 in topsoil = essentially saturated)
"""
from __future__ import annotations

import requests

URL = "https://api.open-meteo.com/v1/forecast"


def fetch_soil_state(lat: float, lon: float, timeout: int = 15) -> dict:
    """Return current soil moisture + 7-day antecedent precip at (lat,lon)."""
    try:
        r = requests.get(
            URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,"
                           "soil_moisture_3_to_9cm,soil_moisture_9_to_27cm",
                "hourly": "precipitation",
                "past_days": 7,
                "forecast_days": 1,
                "precipitation_unit": "inch",
                "timezone": "America/New_York",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

    cur = data.get("current", {}) or {}
    sm_top = cur.get("soil_moisture_0_to_1cm")
    sm_shallow = cur.get("soil_moisture_1_to_3cm")
    sm_mid = cur.get("soil_moisture_3_to_9cm")
    sm_root = cur.get("soil_moisture_9_to_27cm")

    # 7-day antecedent precipitation
    hourly = data.get("hourly", {}) or {}
    times = hourly.get("time", [])
    precip = hourly.get("precipitation", [])
    # take the first 7*24=168 hours (the past_days window)
    past_total = round(sum(precip[: 24 * 7]), 2) if precip else 0.0

    sm_topsoil = sm_top if sm_top is not None else sm_shallow
    saturated = sm_topsoil is not None and sm_topsoil >= 0.40
    very_dry = sm_topsoil is not None and sm_topsoil < 0.15

    if saturated:
        cond = "SATURATED"
    elif sm_topsoil is not None and sm_topsoil >= 0.30:
        cond = "wet"
    elif sm_topsoil is not None and sm_topsoil >= 0.20:
        cond = "moist"
    elif very_dry:
        cond = "very dry"
    else:
        cond = "normal"

    return {
        "as_of": cur.get("time"),
        "soil_moisture_top":     round(sm_top, 3)     if sm_top     is not None else None,
        "soil_moisture_shallow": round(sm_shallow, 3) if sm_shallow is not None else None,
        "soil_moisture_mid":     round(sm_mid, 3)     if sm_mid     is not None else None,
        "soil_moisture_root":    round(sm_root, 3)    if sm_root    is not None else None,
        "past_7d_precip_in": past_total,
        "saturated": saturated,
        "very_dry": very_dry,
        "condition": cond,
    }
