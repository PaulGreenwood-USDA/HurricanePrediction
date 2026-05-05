"""Live river-gauge data for the French Broad River at Asheville.

USGS NWIS Instantaneous Values API (no auth required).
Site 03451500 = French Broad River at Asheville, NC.
  - 00060: discharge, cubic feet per second
  - 00065: gage height, feet

NWS AHPS flood thresholds for this site (current as of 2025):
  Action  :  7.0 ft
  Minor   :  9.5 ft   (formal flood stage)
  Moderate: 12.0 ft
  Major   : 16.0 ft
  Record  : 23.10 ft  (1916 flood; Helene 2024 reached ~24.7 ft preliminary)
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

USGS_NWIS_URL = "https://waterservices.usgs.gov/nwis/iv/"
SITE_FRENCH_BROAD_ASHEVILLE = "03451500"

FLOOD_STAGES_FT = {
    "action": 7.0,
    "minor": 9.5,
    "moderate": 12.0,
    "major": 16.0,
    "record": 23.10,
}


@dataclass
class GaugeReading:
    site_id: str
    site_name: str
    timestamp: str
    stage_ft: float | None
    discharge_cfs: float | None
    flood_category: str
    pct_to_minor: float | None
    pct_to_major: float | None


def _classify(stage_ft: float | None) -> str:
    if stage_ft is None:
        return "unknown"
    if stage_ft >= FLOOD_STAGES_FT["major"]:
        return "MAJOR FLOOD"
    if stage_ft >= FLOOD_STAGES_FT["moderate"]:
        return "MODERATE FLOOD"
    if stage_ft >= FLOOD_STAGES_FT["minor"]:
        return "MINOR FLOOD"
    if stage_ft >= FLOOD_STAGES_FT["action"]:
        return "action stage"
    return "below action"


def fetch_gauge(site_id: str = SITE_FRENCH_BROAD_ASHEVILLE,
                timeout: int = 20) -> GaugeReading | None:
    """Pull the latest stage + discharge from USGS NWIS."""
    try:
        r = requests.get(
            USGS_NWIS_URL,
            params={
                "sites": site_id,
                "parameterCd": "00060,00065",
                "format": "json",
                "siteStatus": "active",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] USGS NWIS fetch failed: {e}")
        return None

    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        return None

    site_name = series[0]["sourceInfo"]["siteName"]
    stage = discharge = None
    timestamp = ""
    for ts in series:
        code = ts["variable"]["variableCode"][0]["value"]
        values = ts["values"][0]["value"]
        if not values:
            continue
        latest = values[-1]
        try:
            v = float(latest["value"])
        except (TypeError, ValueError):
            continue
        if v == -999999:
            v = None
        timestamp = latest.get("dateTime", timestamp)
        if code == "00065":
            stage = v
        elif code == "00060":
            discharge = v

    return GaugeReading(
        site_id=site_id,
        site_name=site_name,
        timestamp=timestamp,
        stage_ft=stage,
        discharge_cfs=discharge,
        flood_category=_classify(stage),
        pct_to_minor=None if stage is None else 100.0 * stage / FLOOD_STAGES_FT["minor"],
        pct_to_major=None if stage is None else 100.0 * stage / FLOOD_STAGES_FT["major"],
    )


# ---- Bonus: NWS active alerts for the Asheville point ----------------------

NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"


def fetch_nws_alerts(lat: float, lon: float, timeout: int = 20) -> list[dict]:
    """Active NWS alerts (watches/warnings/advisories) at a point. No auth."""
    try:
        r = requests.get(
            NWS_ALERTS_URL,
            params={"point": f"{lat},{lon}"},
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1 (contact: local)"},
        )
        r.raise_for_status()
        feats = r.json().get("features", [])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] NWS alerts fetch failed: {e}")
        return []
    out = []
    for f in feats:
        p = f.get("properties", {})
        out.append({
            "event": p.get("event", ""),
            "severity": p.get("severity", ""),
            "headline": p.get("headline", ""),
            "onset": p.get("onset", ""),
            "ends": p.get("ends", ""),
        })
    return out
