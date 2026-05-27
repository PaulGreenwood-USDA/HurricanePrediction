"""Active wildfire data from the interagency NIFC / WFIGS feed.

NIFC publishes a public ArcGIS feature service of current wildland fire
incidents (the canonical interagency dataset, populated from IRWIN). No
authentication required.

  - Locations: WFIGS_Incident_Locations_Current (points)
  - Perimeters: WFIGS_Interagency_Perimeters_Current (polygons)

For the dashboard we only need point locations + basic attributes:
  IncidentName, DailyAcres, PercentContained, FireCause, FireDiscoveryDateTime,
  POOState (Point Of Origin state), IncidentTypeCategory.
"""
from __future__ import annotations

import logging

import requests

from .geo import haversine_mi

log = logging.getLogger(__name__)

WFIGS_LOCATIONS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations_Current/FeatureServer/0/query"
)


def fetch_active_wildfires(state: str = "NC",
                            timeout: int = 20) -> list[dict]:
    """Return active wildland-fire incidents whose Point of Origin is in
    *state* (default North Carolina). Empty list on any failure."""
    try:
        r = requests.get(
            WFIGS_LOCATIONS_URL,
            params={
                "where": f"POOState='US-{state}' AND IncidentTypeCategory='WF'",
                "outFields": ("IncidentName,DailyAcres,PercentContained,"
                              "FireCause,FireDiscoveryDateTime,POOState,"
                              "IncidentTypeCategory,IrwinID"),
                "returnGeometry": "true",
                "outSR": 4326,
                "f": "json",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.info("WFIGS fetch skipped (%s)", exc)
        return []

    out: list[dict] = []
    for f in data.get("features") or []:
        a = f.get("attributes") or {}
        g = f.get("geometry") or {}
        if g.get("x") is None or g.get("y") is None:
            continue
        out.append({
            "name": a.get("IncidentName") or "Unnamed",
            "acres": a.get("DailyAcres"),
            "contained_pct": a.get("PercentContained"),
            "cause": a.get("FireCause"),
            "discovered": a.get("FireDiscoveryDateTime"),
            "irwin_id": a.get("IrwinID"),
            "lat": g.get("y"),
            "lon": g.get("x"),
        })
    return out


def fires_near(forest_lat: float, forest_lon: float,
               fires: list[dict],
               radius_mi: float = 50.0) -> list[dict]:
    """Subset of *fires* within *radius_mi* of a forest centroid, sorted by
    distance, each annotated with `distance_mi`."""
    out = []
    for f in fires:
        if f.get("lat") is None or f.get("lon") is None:
            continue
        d = haversine_mi(forest_lat, forest_lon, f["lat"], f["lon"])
        if d <= radius_mi:
            out.append({**f, "distance_mi": round(d, 1)})
    out.sort(key=lambda x: x["distance_mi"])
    return out


def summarize_fires(fires: list[dict]) -> dict:
    """Compact totals for a list of fire dicts (e.g. fires near one forest)."""
    if not fires:
        return {"count": 0, "total_acres": 0.0,
                "max_acres": 0.0, "min_contained_pct": None}
    acres = [float(f["acres"]) for f in fires
             if isinstance(f.get("acres"), (int, float))]
    contained = [float(f["contained_pct"]) for f in fires
                 if isinstance(f.get("contained_pct"), (int, float))]
    return {
        "count": len(fires),
        "total_acres": round(sum(acres), 1),
        "max_acres": round(max(acres), 1) if acres else 0.0,
        "min_contained_pct": min(contained) if contained else None,
    }
