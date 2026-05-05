"""Real-time check: any active Atlantic storms threatening Asheville?"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import ASHEVILLE_LAT, ASHEVILLE_LON, NHC_ACTIVE_URL
from .geo import haversine_mi


@dataclass
class ActiveStorm:
    id: str
    name: str
    classification: str
    intensity_kt: float | None
    lat: float
    lon: float
    distance_mi: float
    movement: str
    public_advisory_url: str | None


def fetch_active_storms(timeout: int = 20) -> list[ActiveStorm]:
    try:
        r = requests.get(NHC_ACTIVE_URL, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not fetch NHC active storms: {e}")
        return []

    storms = data.get("activeStorms", []) if isinstance(data, dict) else []
    out: list[ActiveStorm] = []
    for s in storms:
        # NHC includes Atlantic + EastPac; restrict to Atlantic basin
        if (s.get("binNumber") or "").startswith("AT") or s.get("basin", "").lower().startswith("atl"):
            try:
                lat = float(s.get("latitudeNumeric", s.get("latitude", "nan")))
                lon = float(s.get("longitudeNumeric", s.get("longitude", "nan")))
            except (TypeError, ValueError):
                continue
            try:
                wind = float(s.get("intensity"))
            except (TypeError, ValueError):
                wind = None
            dist = float(haversine_mi(ASHEVILLE_LAT, ASHEVILLE_LON, lat, lon))
            out.append(
                ActiveStorm(
                    id=s.get("id", ""),
                    name=s.get("name", ""),
                    classification=s.get("classification", ""),
                    intensity_kt=wind,
                    lat=lat,
                    lon=lon,
                    distance_mi=dist,
                    movement=s.get("movement", ""),
                    public_advisory_url=(s.get("publicAdvisory") or {}).get("url"),
                )
            )
    out.sort(key=lambda x: x.distance_mi)
    return out
