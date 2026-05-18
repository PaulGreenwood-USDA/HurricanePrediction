"""National Forests in North Carolina.

Four units administered by the USDA Forest Service:
  - Pisgah NF      (~512,000 ac) - WNC mountains, surrounds Asheville
  - Nantahala NF   (~531,000 ac) - SW NC mountains
  - Uwharrie NF    (~ 50,000 ac) - central NC piedmont
  - Croatan NF     (~160,000 ac) - coastal, between New Bern & Cape Lookout

Each has a representative center, headquarters location, and ranger districts.
We pull live weather + NWS alerts at each centroid for situational awareness,
and (when storms are active) the distance from each forest to the nearest TC.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geo import haversine_mi
from .gauge import fetch_nws_alerts
from .weather import fetch_current_weather


@dataclass(frozen=True)
class NationalForest:
    name: str
    short: str
    acres: int
    established: int
    hq_city: str
    center_lat: float        # rough geographic centroid for queries
    center_lon: float
    districts: tuple[str, ...]
    region: str              # mountain / piedmont / coastal
    notes: str


NC_NATIONAL_FORESTS: tuple[NationalForest, ...] = (
    NationalForest(
        name="Pisgah National Forest",
        short="Pisgah",
        acres=512_758,
        established=1916,
        hq_city="Asheville, NC",
        center_lat=35.78,
        center_lon=-82.30,
        districts=("Appalachian", "Grandfather", "Pisgah"),
        region="mountain",
        notes=("Surrounds Asheville. Highest orographic rainfall in the eastern US; "
               "Helene 2024 caused catastrophic landslides and infrastructure damage."),
    ),
    NationalForest(
        name="Nantahala National Forest",
        short="Nantahala",
        acres=531_286,
        established=1920,
        hq_city="Asheville, NC",
        center_lat=35.20,
        center_lon=-83.60,
        districts=("Cheoah", "Nantahala", "Tusquitee"),
        region="mountain",
        notes=("Largest NF in NC. Steep gorges (Whitewater, Cullasaja) - "
               "high flash-flood and landslide exposure during tropical events."),
    ),
    NationalForest(
        name="Uwharrie National Forest",
        short="Uwharrie",
        acres=50_645,
        established=1961,
        hq_city="Troy, NC",
        center_lat=35.40,
        center_lon=-79.95,
        districts=("Uwharrie",),
        region="piedmont",
        notes=("Smallest NC NF. Ancient eroded mountains; "
               "moderate exposure to tropical rainfall as storms move inland."),
    ),
    NationalForest(
        name="Croatan National Forest",
        short="Croatan",
        acres=160_481,
        established=1936,
        hq_city="New Bern, NC",
        center_lat=34.85,
        center_lon=-77.00,
        districts=("Croatan",),
        region="coastal",
        notes=("Coastal - between Neuse and White Oak rivers. "
               "Highest tropical-cyclone wind/storm-surge exposure of any NC NF. "
               "Pocosin wetlands and longleaf pine."),
    ),
)


def fetch_forest_state(forest: NationalForest,
                        active_storms: list | None = None) -> dict:
    """Pull live weather + alerts at forest centroid, plus storm distances."""
    weather = fetch_current_weather(forest.center_lat, forest.center_lon)
    alerts = fetch_nws_alerts(forest.center_lat, forest.center_lon)

    nearest_storm = None
    nearest_mi = None
    if active_storms:
        for s in active_storms:
            if s.lat is None or s.lon is None:
                continue
            d = haversine_mi(forest.center_lat, forest.center_lon, s.lat, s.lon)
            if nearest_mi is None or d < nearest_mi:
                nearest_mi = d
                nearest_storm = s.name

    return {
        "name": forest.name,
        "short": forest.short,
        "acres": forest.acres,
        "established": forest.established,
        "hq_city": forest.hq_city,
        "center_lat": forest.center_lat,
        "center_lon": forest.center_lon,
        "districts": list(forest.districts),
        "region": forest.region,
        "notes": forest.notes,
        "weather": weather,
        "alerts": alerts,
        "nearest_storm": nearest_storm,
        "nearest_storm_mi": nearest_mi,
    }


def fetch_all_forests(active_storms: list | None = None) -> list[dict]:
    return [fetch_forest_state(f, active_storms) for f in NC_NATIONAL_FORESTS]
