"""Live river-gauge data for the French Broad River at Asheville.

USGS NWIS Instantaneous Values API (no auth required).
Site 03451500 = French Broad River at Asheville, NC.
  - 00060: discharge, cubic feet per second
  - 00065: gage height, feet

Flood thresholds are **per site**. Every gauge sits on a different datum, so
Asheville's 9.5 ft "minor flood" means nothing on the Cape Fear at Lock 1,
whose ordinary fair-weather stage is higher than that. Thresholds come from
the NWS National Water Prediction Service and are baked into
``data/nws_flood_thresholds.json`` -- NWPS rate-limits to 10 requests per
5 minutes, which a 20-gauge refresh would blow instantly.

Regenerate that file with::

    uv run python scripts/refresh_flood_thresholds.py

Sites with no published NWS thresholds are classified ``"no thresholds"``
rather than being measured against some other river's numbers. Guessing is
worse than saying nothing on a flood dashboard.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

USGS_NWIS_URL = "https://waterservices.usgs.gov/nwis/iv/"
SITE_FRENCH_BROAD_ASHEVILLE = "03451500"

THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "data" / "nws_flood_thresholds.json"

# French Broad @ Asheville, kept as a module constant because the Flood Index,
# the CLI stage bar and the ML forecast card are all specifically about this
# one site. Overwritten from the JSON store below when it is present.
FLOOD_STAGES_FT = {
    "action": 7.0,
    "minor": 9.5,
    "moderate": 12.0,
    "major": 16.0,
    "record": 23.10,
}


def _load_thresholds() -> dict[str, dict]:
    """Per-USGS-site NWS flood thresholds, keyed by site id.

    Returns an empty mapping if the store is missing or unreadable; callers
    degrade to "no thresholds" rather than to another site's numbers.
    """
    try:
        raw = json.loads(THRESHOLDS_PATH.read_text())
    except FileNotFoundError:
        log.warning("flood threshold store not found at %s; gauges will be "
                    "reported without flood categories", THRESHOLDS_PATH)
        return {}
    except (OSError, ValueError) as exc:
        log.warning("flood threshold store unreadable (%s): %s",
                    THRESHOLDS_PATH, exc)
        return {}
    sites: dict[str, dict] = {}
    for site_id, entry in (raw.get("sites") or {}).items():
        if not isinstance(entry, dict):
            continue
        # Defence in depth: NWPS writes -9999 for "level not defined". If one
        # ever survives into the store, every reading would exceed it and the
        # gauge would report MAJOR FLOOD forever.
        clean = dict(entry)
        for level in ("action", "minor", "moderate", "major", "record"):
            v = clean.get(level)
            if isinstance(v, (int, float)) and v <= -999:
                log.warning("dropping sentinel %s=%s for site %s", level, v, site_id)
                clean[level] = None
        sites[site_id] = clean
    return sites


FLOOD_STAGES_BY_SITE: dict[str, dict] = _load_thresholds()

if SITE_FRENCH_BROAD_ASHEVILLE in FLOOD_STAGES_BY_SITE:
    # Merge over the defaults rather than replacing them: index_score, the CLI
    # stage bar and the ML card all index this dict directly, so every key has
    # to stay present even if NWPS omits a level.
    _ash = FLOOD_STAGES_BY_SITE[SITE_FRENCH_BROAD_ASHEVILLE]
    FLOOD_STAGES_FT = {
        **FLOOD_STAGES_FT,
        **{k: float(_ash[k])
           for k in ("action", "minor", "moderate", "major", "record")
           if _ash.get(k) is not None},
    }

# Gauges measuring reservoir pool elevation rather than river stage. A lake
# has no NWS flood stage in the river sense, so classifying it against one
# would be meaningless.
RESERVOIR_ROLES = {"reservoir"}

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
    flood_class: str = "unknown"
    thresholds: dict | None = field(default=None)


# Display label -> CSS class. The template used to derive this by string
# munging the label, which turned "below action" into the two classes
# "below" and "action" and painted every safe gauge in action-stage yellow.
_FLOOD_CLASS = {
    "MAJOR FLOOD":    "major",
    "MODERATE FLOOD": "moderate",
    "MINOR FLOOD":    "minor",
    "action stage":   "action",
    "below action":   "below-action",
    "pool stage":     "pool",
    "no thresholds":  "no-thresholds",
    "unknown":        "unknown",
}


def flood_class(category: str) -> str:
    """CSS-safe single-token slug for a flood category label."""
    return _FLOOD_CLASS.get(category, "unknown")


def thresholds_for(site_id: str) -> dict | None:
    """Published NWS flood thresholds for a USGS site, or None if we have none."""
    return FLOOD_STAGES_BY_SITE.get(site_id)


def format_thresholds(t: dict | None) -> str:
    """Human-readable threshold summary for a tooltip.

    Skips levels NWS leaves undefined at a gauge, so the UI never shows
    "moderate None ft".
    """
    if not t:
        return ""
    parts = [f"{level} {t[level]:g} ft"
             for level in ("action", "minor", "moderate", "major")
             if t.get(level) is not None]
    return "NWS flood stages here: " + ", ".join(parts) if parts else ""


def _classify(stage_ft: float | None,
              site_id: str = SITE_FRENCH_BROAD_ASHEVILLE,
              role: str | None = None) -> str:
    """Classify a stage reading against *that site's* NWS thresholds.

    Never falls back to another site's thresholds: a gauge we have no
    published numbers for is reported as "no thresholds", not as safe.
    """
    if stage_ft is None:
        return "unknown"
    t = FLOOD_STAGES_BY_SITE.get(site_id)
    if not t:
        # Only a fallback: if NWS publishes flood stages for a site we use
        # them, whatever role the site is tagged with.
        return "pool stage" if role in RESERVOIR_ROLES else "no thresholds"
    for level, label in (("major", "MAJOR FLOOD"),
                         ("moderate", "MODERATE FLOOD"),
                         ("minor", "MINOR FLOOD"),
                         ("action", "action stage")):
        v = t.get(level)
        if v is not None and stage_ft >= v:
            return label
    return "below action"


def fetch_gauge(site_id: str = SITE_FRENCH_BROAD_ASHEVILLE,
                timeout: int = 20,
                role: str | None = None) -> GaugeReading | None:
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

    t = thresholds_for(site_id)
    category = _classify(stage, site_id=site_id, role=role)

    def _pct(level: str) -> float | None:
        if stage is None or not t or not t.get(level):
            return None
        return 100.0 * stage / t[level]

    return GaugeReading(
        site_id=site_id,
        site_name=site_name,
        timestamp=timestamp,
        stage_ft=stage,
        discharge_cfs=discharge,
        flood_category=category,
        pct_to_minor=_pct("minor"),
        pct_to_major=_pct("major"),
        flood_class=flood_class(category),
        thresholds=t,
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
        g = fetch_gauge(site_id, role=role)
        hist = fetch_gauge_history(site_id, hours=hours_history)
        rate = rate_of_rise_ft_per_hr(hist)
        t = thresholds_for(site_id)
        eta_minor = eta_moderate = eta_major = None
        # ETAs are only meaningful against this site's own published stages.
        if g and g.stage_ft is not None and t:
            def _eta(level):
                target = t.get(level)
                return (None if target is None
                        else eta_to_stage_hours(g.stage_ft, target, rate))
            eta_minor, eta_moderate, eta_major = (
                _eta("minor"), _eta("moderate"), _eta("major"))
        nwps = (fetch_nwps_forecast(site_id)
                if site_id in NWPS_FORECAST_SITES else None)
        category = g.flood_category if g else "unknown"
        return {
            "site_id": site_id,
            "label": label,
            "role": role,
            "lat": lat,
            "lon": lon,
            "stage_ft": g.stage_ft if g else None,
            "discharge_cfs": g.discharge_cfs if g else None,
            "flood_category": category,
            "flood_class": flood_class(category),
            "thresholds": t,
            "thresholds_label": format_thresholds(t),
            "timestamp": g.timestamp if g else "",
            "rate_ft_per_hr": rate,
            "history": [{"t": t_, "ft": v} for t_, v in hist[-96:]],
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

# NWPS resolves USGS site ids directly on /v1/gauges/{id}, so no NWSLI lookup
# table is needed. The one this replaced was also wrong -- ASHN7 and CTON7 are
# not valid NWPS ids, so the primary Asheville gauge never rendered a forecast.
NWPS_FORECAST_URL = (
    "https://api.water.noaa.gov/nwps/v1/gauges/{ident}/stageflow/forecast"
)

# Sites we pull forecast traces for. Deliberately minimal: NWPS allows only
# 10 requests per 5 minutes *per client*, and the dashboard renders the NWPS
# trace for the primary gauge alone. Add upstream sites here only alongside
# UI that actually shows them -- otherwise it is a wasted request per refresh.
NWPS_FORECAST_SITES = (
    SITE_FRENCH_BROAD_ASHEVILLE,  # French Broad @ Asheville (primary)
)

# The dashboard's own cache is 60 s. Without a longer-lived cache here, a
# locally-run Flask instance would burn the whole NWPS budget in a minute.
# Forecast traces are reissued a few times a day, so 30 minutes costs nothing.
_NWPS_TTL_SECONDS = 1800.0
_NWPS_CACHE: dict[str, tuple[float, dict | None]] = {}
# Set when NWPS returns 429; suppresses all calls until it passes.
_NWPS_BACKOFF_UNTIL = 0.0
_NWPS_BACKOFF_SECONDS = 300.0


def fetch_nwps_forecast(usgs_site_id: str, timeout: int = 15) -> dict | None:
    """Return the official NWS NWPS stage-flow forecast for a USGS site.

    Returns None if the site has no NWPS forecast point, the call fails, or we
    are backing off from a rate-limit response.

    Output: ``{"lid": str, "issued": iso, "points": [{"t": iso, "ft": float},
    ...], "peak_ft": float, "peak_t": iso}``.
    """
    global _NWPS_BACKOFF_UNTIL

    now = time.time()
    cached = _NWPS_CACHE.get(usgs_site_id)
    if cached is not None and now - cached[0] < _NWPS_TTL_SECONDS:
        return cached[1]

    if now < _NWPS_BACKOFF_UNTIL:
        log.info("NWPS forecast skipped (%s): backing off from rate limit",
                 usgs_site_id)
        return cached[1] if cached else None

    try:
        r = requests.get(
            NWPS_FORECAST_URL.format(ident=usgs_site_id),
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        if r.status_code == 429:
            _NWPS_BACKOFF_UNTIL = now + _NWPS_BACKOFF_SECONDS
            log.warning("NWPS rate limit hit; pausing forecast fetches for %.0f s",
                        _NWPS_BACKOFF_SECONDS)
            return cached[1] if cached else None
        if r.status_code == 404:
            # No forecast point at this gauge -- cache the negative so we do
            # not spend a request on it again next refresh.
            _NWPS_CACHE[usgs_site_id] = (now, None)
            return None
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.info("NWPS forecast fetch skipped (%s): %s", usgs_site_id, exc)
        return cached[1] if cached else None

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
    out = {
        "lid": (thresholds_for(usgs_site_id) or {}).get("lid"),
        "issued": data.get("issuedTime") or data.get("issued"),
        "points": pts[:120],
        "peak_ft": peak["ft"] if peak else None,
        "peak_t": peak["t"] if peak else None,
    }
    _NWPS_CACHE[usgs_site_id] = (now, out)
    return out
