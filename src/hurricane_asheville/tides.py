"""NOAA Tides & Currents (CO-OPS) live water level + meteorology.

API: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
No auth. Stations along the NC coast give storm-surge early warning hours
before tropical cyclones impact Croatan NF and the eastern half of the state.
"""
from __future__ import annotations

import requests

COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# (station_id, label, lat, lon, role)
COASTAL_STATIONS = [
    ("8651370", "Duck, NC (Outer Banks N)",      36.1833, -75.7467, "outer-banks"),
    ("8654467", "Hatteras, NC",                  35.2086, -75.7036, "outer-banks"),
    ("8656483", "Beaufort, NC (Duke Marine Lab)", 34.7200, -76.6700, "central-coast"),
    ("8658120", "Wilmington, NC",                34.2275, -77.9536, "cape-fear"),
    ("8658163", "Wrightsville Beach, NC",        34.2133, -77.7867, "cape-fear"),
]


def _latest_value(station_id: str, product: str, units: str = "english",
                  timeout: int = 15) -> dict | None:
    """Pull the latest 6-min observation for a CO-OPS product."""
    try:
        r = requests.get(
            COOPS_URL,
            params={
                "station": station_id,
                "product": product,
                "date": "latest",
                "datum": "MLLW" if product == "water_level" else None,
                "units": units,
                "time_zone": "lst_ldt",
                "format": "json",
                "application": "hurricane-asheville",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return None
    if "data" not in data or not data["data"]:
        return None
    d = data["data"][-1]
    try:
        return {"t": d.get("t"), "v": float(d["v"])}
    except (TypeError, ValueError, KeyError):
        return None


def fetch_coastal_station(station_id: str, label: str,
                           lat: float, lon: float, role: str) -> dict:
    """Pull water level (ft MLLW), wind (kt + dir), and air pressure (mb)."""
    water = _latest_value(station_id, "water_level")
    wind = _latest_value(station_id, "wind")
    pres = _latest_value(station_id, "air_pressure")  # millibars

    # The wind product returns a different schema; refetch for direction.
    wind_dir = wind_speed = wind_gust = None
    wind_t = None
    try:
        r = requests.get(
            COOPS_URL,
            params={
                "station": station_id,
                "product": "wind",
                "date": "latest",
                "units": "english",
                "time_zone": "lst_ldt",
                "format": "json",
                "application": "hurricane-asheville",
            },
            timeout=15,
        )
        r.raise_for_status()
        d = r.json().get("data", [])
        if d:
            row = d[-1]
            wind_t = row.get("t")
            wind_speed = float(row["s"]) if row.get("s") not in (None, "") else None  # kt
            wind_gust  = float(row["g"]) if row.get("g") not in (None, "") else None
            wind_dir   = float(row["d"]) if row.get("d") not in (None, "") else None
    except Exception:  # noqa: BLE001
        pass

    return {
        "station_id": station_id,
        "label": label,
        "lat": lat,
        "lon": lon,
        "role": role,
        "water_level_ft": water["v"] if water else None,
        "water_level_t":  water["t"] if water else None,
        "air_pressure_mb": pres["v"] if pres else None,
        "wind_kt": wind_speed,
        "wind_gust_kt": wind_gust,
        "wind_dir_deg": wind_dir,
        "wind_t": wind_t,
    }


def fetch_all_coastal() -> list[dict]:
    out = []
    for sid, label, lat, lon, role in COASTAL_STATIONS:
        try:
            out.append(fetch_coastal_station(sid, label, lat, lon, role))
        except Exception:  # noqa: BLE001
            continue
    return out
