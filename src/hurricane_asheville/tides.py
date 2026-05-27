"""NOAA Tides & Currents (CO-OPS) live water level + meteorology.

API: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
No auth. Stations along the NC coast give storm-surge early warning hours
before tropical cyclones impact Croatan NF and the eastern half of the state.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# (station_id, label, lat, lon, role)
COASTAL_STATIONS = [
    ("8651370", "Duck, NC (Outer Banks N)",      36.1833, -75.7467, "outer-banks"),
    ("8654467", "Hatteras, NC",                  35.2086, -75.7036, "outer-banks"),
    ("8656483", "Beaufort, NC (Duke Marine Lab)", 34.7200, -76.6700, "central-coast"),
    ("8658120", "Wilmington, NC",                34.2275, -77.9536, "cape-fear"),
    ("8658163", "Wrightsville Beach, NC",        34.2133, -77.7867, "cape-fear"),
]


def _coops_request(station_id: str, product: str, *, datum: str | None = None,
                   timeout: int = 15) -> list[dict]:
    """Return the `data` array for the latest CO-OPS observation, or []."""
    params = {
        "station": station_id,
        "product": product,
        "date": "latest",
        "units": "english",
        "time_zone": "lst_ldt",
        "format": "json",
        "application": "hurricane-asheville",
    }
    if datum is not None:
        params["datum"] = datum
    try:
        r = requests.get(COOPS_URL, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json().get("data", []) or []
    except Exception as e:  # noqa: BLE001
        log.debug("CO-OPS %s/%s failed: %s", station_id, product, e)
        return []


def _scalar(rows: list[dict]) -> dict | None:
    """Pick the last row's `v` value as a float, with timestamp."""
    if not rows:
        return None
    last = rows[-1]
    try:
        return {"t": last.get("t"), "v": float(last["v"])}
    except (TypeError, ValueError, KeyError):
        return None


def fetch_coastal_station(station_id: str, label: str,
                           lat: float, lon: float, role: str) -> dict:
    """Pull water level (ft MLLW), wind (kt + dir + gust), and air pressure (mb)."""
    water = _scalar(_coops_request(station_id, "water_level", datum="MLLW"))
    pres  = _scalar(_coops_request(station_id, "air_pressure"))
    wind_rows = _coops_request(station_id, "wind")

    wind_t = wind_speed = wind_gust = wind_dir = None
    if wind_rows:
        row = wind_rows[-1]
        wind_t = row.get("t")

        def _f(key):
            v = row.get(key)
            if v in (None, ""):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        wind_speed = _f("s")
        wind_gust  = _f("g")
        wind_dir   = _f("d")

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
    """Pull every coastal station in parallel; skip any that error out."""
    def _one(entry):
        sid, label, lat, lon, role = entry
        try:
            return fetch_coastal_station(sid, label, lat, lon, role)
        except Exception as e:  # noqa: BLE001
            log.warning("coastal station %s failed: %s", sid, e)
            return None

    with ThreadPoolExecutor(max_workers=len(COASTAL_STATIONS)) as pool:
        results = list(pool.map(_one, COASTAL_STATIONS))
    return [r for r in results if r is not None]
