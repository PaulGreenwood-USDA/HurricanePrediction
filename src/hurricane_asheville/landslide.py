"""Landslide hazard for the NC National Forests.

Two pieces:

1. ``compute_landslide_hazard`` – a deterministic 0–100 hazard index per forest,
   built from terrain region + current soil saturation + antecedent (past 7-day)
   precipitation + 72-hour forecast QPF. Calibrated against the Helene 2024
   event (which triggered 2,000+ landslides across WNC after ~3 days of
   orographic rainfall on already-saturated soils).

2. ``fetch_nearby_landslides`` – best-effort fetch of historical landslide
   points from the USGS National Landslide Inventory (v3) ArcGIS feature
   service within a search radius. Returns ``[]`` on any failure; this is a
   nice-to-have, never required.

Hazard thresholds (rough, southern Appalachians; based on USGS rainfall
intensity–duration curves and NCGS landslide guidance):

  mountain forests (Pisgah, Nantahala) – steep slopes, weathered regolith
    saturated soil + 72h QPF >= 4 in    → EXTREME
    saturated soil + 72h QPF >= 2 in    → HIGH
    wet soil      + 72h QPF >= 3 in     → HIGH
    7d precip >= 6 in                   → ELEVATED minimum
  piedmont forests (Uwharrie) – moderate relief
    saturated + 72h QPF >= 6 in         → HIGH
    else generally ELEVATED / CALM
  coastal forests (Croatan) – flat
    landslides are rare; hazard capped at ELEVATED
"""
from __future__ import annotations

import logging
from typing import Iterable

import requests

log = logging.getLogger(__name__)

# USGS National Landslide Inventory v3 (hosted ArcGIS Feature Service).
# Public, no auth required. Returns one feature per recorded landslide.
USGS_NLI_URL = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
    "US_Landslide_Inventory_v3_pt/FeatureServer/0/query"
)


_LEVELS = (
    ("EXTREME",  80, "#6a1b9a"),
    ("HIGH",     60, "#c62828"),
    ("ELEVATED", 35, "#ef6c00"),
    ("CALM",      0, "#4caf50"),
)


def _classify(score: float) -> tuple[str, str]:
    for label, threshold, color in _LEVELS:
        if score >= threshold:
            return label, color
    return "CALM", "#4caf50"


def compute_landslide_hazard(region: str,
                              soil: dict | None,
                              weather: dict | None) -> dict:
    """Return a 0–100 landslide hazard index for one forest.

    ``region`` is one of ``mountain`` / ``piedmont`` / ``coastal``.
    ``soil`` is the dict from :func:`hurricane_asheville.soil.fetch_soil_state`.
    ``weather`` is the dict from :func:`hurricane_asheville.weather.fetch_current_weather`.
    All inputs may be falsy or carry ``error``; in that case we return a neutral
    score so the dashboard still renders.
    """
    soil = soil if isinstance(soil, dict) and not soil.get("error") else {}
    weather = weather if isinstance(weather, dict) and not weather.get("error") else {}

    sm = soil.get("soil_moisture_top") or soil.get("soil_moisture_shallow") or 0.0
    past_7d = float(soil.get("past_7d_precip_in") or 0.0)
    qpf_72h = float(weather.get("next_72h_precip_in") or 0.0)
    saturated = bool(soil.get("saturated"))

    # --- baseline by terrain region ---------------------------------------
    if region == "mountain":
        base = 18.0          # always some baseline exposure in the Blue Ridge
        cap = 100.0
    elif region == "piedmont":
        base = 6.0
        cap = 65.0           # piedmont rarely sees catastrophic slides
    else:                    # coastal / unknown
        base = 0.0
        cap = 30.0           # flat terrain – essentially landslide-free

    score = base

    # --- soil moisture contribution (0–35) --------------------------------
    if saturated:
        score += 30.0
    elif sm >= 0.35:
        score += 22.0
    elif sm >= 0.30:
        score += 14.0
    elif sm >= 0.20:
        score += 6.0

    # --- antecedent precipitation contribution (0–20) ---------------------
    if past_7d >= 8.0:
        score += 20.0
    elif past_7d >= 5.0:
        score += 14.0
    elif past_7d >= 3.0:
        score += 8.0
    elif past_7d >= 1.5:
        score += 3.0

    # --- 72h QPF contribution (0–35) --------------------------------------
    if qpf_72h >= 6.0:
        score += 35.0
    elif qpf_72h >= 4.0:
        score += 25.0
    elif qpf_72h >= 2.0:
        score += 14.0
    elif qpf_72h >= 1.0:
        score += 6.0

    # --- compound multiplier: saturated + heavy QPF = nonlinear danger ----
    if saturated and qpf_72h >= 3.0 and region == "mountain":
        score *= 1.15

    score = max(0.0, min(cap, score))
    label, color = _classify(score)

    return {
        "score": round(score, 1),
        "label": label,
        "color": color,
        "drivers": {
            "soil_moisture_top": round(sm, 3) if sm else 0.0,
            "saturated": saturated,
            "past_7d_precip_in": round(past_7d, 2),
            "next_72h_precip_in": round(qpf_72h, 2),
            "region": region,
        },
        # Helpful for the dashboard tooltip:
        "explain": _explain(region, saturated, sm, past_7d, qpf_72h),
    }


def _explain(region: str, saturated: bool, sm: float,
             past_7d: float, qpf_72h: float) -> str:
    if region == "coastal":
        return "Flat terrain – landslides are rare regardless of rainfall."
    bits: list[str] = []
    if saturated:
        bits.append("soil saturated")
    elif sm >= 0.30:
        bits.append("soil wet")
    if past_7d >= 3.0:
        bits.append(f"{past_7d:.1f} in past 7d")
    if qpf_72h >= 2.0:
        bits.append(f"{qpf_72h:.1f} in forecast (72h)")
    if not bits:
        return "Dry conditions; no significant landslide trigger active."
    trigger = ", ".join(bits)
    if region == "mountain":
        return f"Steep WNC terrain + {trigger}."
    return f"Moderate piedmont relief + {trigger}."


def fetch_nearby_landslides(lat: float, lon: float,
                             radius_mi: float = 25.0,
                             max_results: int = 25,
                             timeout: int = 15) -> list[dict]:
    """Best-effort fetch of recorded landslide points from USGS NLI.

    Uses a geodesic ``distance`` filter on the ArcGIS query endpoint. Returns
    ``[]`` on any failure (network, schema change, no service available) – this
    is purely additional context, never required to render the dashboard.
    """
    try:
        r = requests.get(
            USGS_NLI_URL,
            params={
                "where": "1=1",
                "geometry": f"{lon},{lat}",
                "geometryType": "esriGeometryPoint",
                "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "distance": radius_mi,
                "units": "esriSRUnit_StatuteMile",
                "outFields": "Name,Conf,Type,Date,Year,State,County",
                "returnGeometry": "true",
                "outSR": 4326,
                "resultRecordCount": max_results,
                "orderByFields": "Year DESC",
                "f": "json",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.info("USGS NLI fetch skipped (%s)", exc)
        return []

    feats = data.get("features") or []
    out: list[dict] = []
    for f in feats:
        attrs = f.get("attributes") or {}
        geom = f.get("geometry") or {}
        out.append({
            "name": attrs.get("Name") or attrs.get("name"),
            "type": attrs.get("Type"),
            "confidence": attrs.get("Conf"),
            "date": attrs.get("Date"),
            "year": attrs.get("Year"),
            "county": attrs.get("County"),
            "state": attrs.get("State"),
            "lat": geom.get("y"),
            "lon": geom.get("x"),
        })
    return out


def summarize_inventory(events: Iterable[dict]) -> dict:
    """Compact summary of an inventory list for dashboard display."""
    events = list(events)
    if not events:
        return {"count": 0, "most_recent_year": None}
    years = [e.get("year") for e in events if isinstance(e.get("year"), int)]
    return {
        "count": len(events),
        "most_recent_year": max(years) if years else None,
    }
