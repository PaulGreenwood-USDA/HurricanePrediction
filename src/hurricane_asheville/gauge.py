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

import logging
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

USGS_NWIS_URL = "https://waterservices.usgs.gov/nwis/iv/"
SITE_FRENCH_BROAD_ASHEVILLE = "03451500"

FLOOD_STAGES_FT = {
    "action": 7.0,
    "minor": 9.5,
    "moderate": 12.0,
    "major": 16.0,
    "record": 23.10,
}

# Upstream / nearby gauges that give early warning (hours of lead-time on Asheville).
# (site_id, label, lat, lon, role)
UPSTREAM_GAUGES = [
    ("03439000", "French Broad @ Rosman (headwaters)", 35.1432, -82.8262, "headwaters"),
    ("03443000", "French Broad @ Blantyre",            35.3576, -82.6171, "upstream"),
    ("03446000", "Mills River nr Mills River",         35.3876, -82.5643, "tributary"),
    ("03451000", "Swannanoa River @ Biltmore",         35.5073, -82.5365, "tributary"),
    ("03451500", "French Broad @ Asheville",           35.6090, -82.5790, "primary"),
    # WNC mountain coverage (Pisgah/Nantahala forests, Helene hot zones)
    ("03456500", "Pigeon River @ Canton",              35.5326, -82.8376, "regional"),
    ("02151500", "Broad River nr Bat Cave (Lake Lure)", 35.4576, -82.2843, "regional"),
    ("03512000", "Oconaluftee @ Birdtown (Smokies)",   35.4623, -83.3457, "regional"),
    # ---- Statewide piedmont coverage (Catawba / Yadkin / Cape Fear basins) ----
    ("02146000", "Catawba River nr Charlotte (Mtn Is.)", 35.3409, -80.9598, "statewide"),
    ("02118500", "Yadkin River @ Yadkin College",       35.8454, -80.3853, "statewide"),
    ("02129000", "Pee Dee River nr Rockingham",         35.0079, -79.8703, "statewide"),
    ("02102000", "Cape Fear River @ Lillington",        35.3979, -78.8161, "statewide"),
    ("02105769", "Cape Fear River @ Lock 1 nr Kelly",   34.4040, -78.2980, "statewide"),
    # ---- Coastal-plain rivers (Tar, Neuse, Lumber) — tropical flood hot zones ----
    ("02083500", "Tar River @ Tarboro",                 35.8932, -77.5366, "statewide"),
    ("02089000", "Neuse River @ Kinston",               35.2596, -77.5811, "statewide"),
    ("02105500", "Lumber River @ Boardman",             34.4400, -79.0140, "statewide"),
    # ---- Reservoir / lake stages (USACE / Duke / TVA — tropical release decisions) ----
    ("0351706800", "Fontana Reservoir nr Fontana Dam",  35.4500, -83.8050, "reservoir"),
    ("0208732885", "Falls Lake @ Falls Dam",            35.9395, -78.5828, "reservoir"),
    ("02096960", "Jordan Lake @ Farrington",            35.7280, -79.0533, "reservoir"),
    ("02143040", "Lake Norman @ Marshall Steam Plt",    35.5933, -80.9614, "reservoir"),
]


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
        log.warning("USGS NWIS fetch failed: %s", e)
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


# ---- 24h history + rate-of-rise -------------------------------------------

def fetch_gauge_history(site_id: str, hours: int = 24,
                         timeout: int = 20) -> list[tuple[str, float]]:
    """Last N hours of stage readings as (iso_time, ft)."""
    try:
        r = requests.get(
            USGS_NWIS_URL,
            params={
                "sites": site_id,
                "parameterCd": "00065",
                "format": "json",
                "period": f"PT{hours}H",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("USGS history fetch failed for %s: %s", site_id, e)
        return []
    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        return []
    out = []
    for v in series[0]["values"][0]["value"]:
        try:
            ft = float(v["value"])
            if ft == -999999:
                continue
            out.append((v["dateTime"], ft))
        except (TypeError, ValueError):
            continue
    return out


def rate_of_rise_ft_per_hr(history: list[tuple[str, float]]) -> float | None:
    """Slope of last ~3 hours of stage readings, ft/hr. Positive = rising."""
    from datetime import datetime
    if len(history) < 2:
        return None
    # take last 3 hours of points
    cutoff_n = max(2, min(len(history), 12))  # ~12 readings = 3h at 15-min cadence
    recent = history[-cutoff_n:]
    try:
        t0 = datetime.fromisoformat(recent[0][0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(recent[-1][0].replace("Z", "+00:00"))
        hrs = (t1 - t0).total_seconds() / 3600.0
        if hrs <= 0:
            return None
        return (recent[-1][1] - recent[0][1]) / hrs
    except Exception:  # noqa: BLE001
        return None


def eta_to_stage_hours(current_ft: float, target_ft: float,
                       rate_ft_per_hr: float | None) -> float | None:
    """Linear extrapolation: hours until current_ft reaches target_ft.
    Returns None if not rising or already past target."""
    if rate_ft_per_hr is None or rate_ft_per_hr <= 0.05:
        return None
    if current_ft >= target_ft:
        return 0.0
    return (target_ft - current_ft) / rate_ft_per_hr


def fetch_all_gauges(hours_history: int = 24) -> list[dict]:
    """Pull current reading + 24h history for every gauge in UPSTREAM_GAUGES.

    Network calls run in parallel (one thread per gauge) to keep dashboard
    cold-load under a few seconds.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(entry):
        site_id, label, lat, lon, role = entry
        g = fetch_gauge(site_id)
        hist = fetch_gauge_history(site_id, hours=hours_history)
        rate = rate_of_rise_ft_per_hr(hist)
        eta_minor = eta_moderate = eta_major = None
        if g and g.stage_ft is not None and site_id == SITE_FRENCH_BROAD_ASHEVILLE:
            eta_minor    = eta_to_stage_hours(g.stage_ft, FLOOD_STAGES_FT["minor"], rate)
            eta_moderate = eta_to_stage_hours(g.stage_ft, FLOOD_STAGES_FT["moderate"], rate)
            eta_major    = eta_to_stage_hours(g.stage_ft, FLOOD_STAGES_FT["major"], rate)
        nwps = fetch_nwps_forecast(site_id) if site_id in NWS_LID_FOR_USGS else None
        return {
            "site_id": site_id,
            "label": label,
            "role": role,
            "lat": lat,
            "lon": lon,
            "stage_ft": g.stage_ft if g else None,
            "discharge_cfs": g.discharge_cfs if g else None,
            "flood_category": g.flood_category if g else "unknown",
            "timestamp": g.timestamp if g else "",
            "rate_ft_per_hr": rate,
            "history": [{"t": t, "ft": v} for t, v in hist[-96:]],
            "eta_minor_hr": eta_minor,
            "eta_moderate_hr": eta_moderate,
            "eta_major_hr": eta_major,
            "nwps_forecast": nwps,
        }

    with ThreadPoolExecutor(max_workers=min(8, len(UPSTREAM_GAUGES))) as pool:
        return list(pool.map(_one, UPSTREAM_GAUGES))


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
        log.warning("NWS alerts fetch failed: %s", e)
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


# ---- NWS National Water Prediction Service forecast traces -----------------

# Mapping from USGS site ID to NWS gauge ID (NWSLI / lid). NWPS uses NWS IDs,
# not USGS IDs, for forecast traces.  Lookups: https://water.noaa.gov/
NWS_LID_FOR_USGS = {
    "03451500": "ASHN7",   # French Broad @ Asheville
    "03451000": "BLTN7",   # Swannanoa @ Biltmore
    "03456500": "CTON7",   # Pigeon @ Canton
    "02151500": "BAVN7",   # Broad nr Bat Cave
    "03512000": "BDTN7",   # Oconaluftee @ Birdtown
    "03513000": "BRYN7",   # Tuckasegee @ Bryson City
}

NWPS_FORECAST_URL = (
    "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow/forecast"
)


def fetch_nwps_forecast(usgs_site_id: str, timeout: int = 15) -> dict | None:
    """Return the official NWS NWPS stage-flow forecast for a USGS site, or
    None if no NWS ID mapping is known or the call fails.

    Output: ``{"lid": str, "issued": iso, "points": [{"t": iso, "ft": float},
    ...], "peak_ft": float, "peak_t": iso}``.
    """
    lid = NWS_LID_FOR_USGS.get(usgs_site_id)
    if not lid:
        return None
    try:
        r = requests.get(
            NWPS_FORECAST_URL.format(lid=lid),
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.info("NWPS forecast fetch skipped (%s): %s", lid, exc)
        return None

    pts: list[dict] = []
    for p in (data.get("data") or data.get("forecast") or []):
        try:
            ft = float(p.get("primary") or p.get("stage") or p.get("value"))
            t = p.get("validTime") or p.get("time") or p.get("timestamp")
            if t:
                pts.append({"t": t, "ft": ft})
        except (TypeError, ValueError):
            continue

    peak = max(pts, key=lambda x: x["ft"]) if pts else None
    return {
        "lid": lid,
        "issued": data.get("issuedTime") or data.get("issued"),
        "points": pts[:120],
        "peak_ft": peak["ft"] if peak else None,
        "peak_t": peak["t"] if peak else None,
    }
