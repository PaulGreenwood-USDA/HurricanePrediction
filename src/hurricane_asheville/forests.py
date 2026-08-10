"""National Forests in North Carolina.

Four units administered by the USDA Forest Service:
  - Pisgah NF      (~512,000 ac) - WNC mountains, surrounds Asheville
  - Nantahala NF   (~531,000 ac) - SW NC mountains
  - Uwharrie NF    (~ 50,000 ac) - central NC piedmont
  - Croatan NF     (~160,000 ac) - coastal, between New Bern & Cape Lookout

For each forest we pull:
  * live weather + NWS alerts at the centroid
  * the closest USGS streamgauges (river stage / discharge)
  * a computed landslide hazard index (terrain + soil + 72h QPF)
  * a best-effort fetch of historical landslide events from the USGS NLI
  * distance to the nearest active Atlantic tropical cyclone
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .gauge import (fetch_gauge, fetch_gauge_history, fetch_nws_alerts,
                    flood_class, rate_of_rise_ft_per_hr)
from .geo import haversine_mi
from .landslide import (compute_landslide_hazard, fetch_nearby_landslides,
                        summarize_inventory)
from .fire_weather import compute_fire_weather
from .smoke_air import fetch_air_quality
from .wildfire import fires_near, summarize_fires
from .soil import fetch_soil_state
from .weather import fetch_current_weather

# (site_id, label, lat, lon, role) — per-forest USGS streamgauges.
# Same shape as gauge.UPSTREAM_GAUGES so we can reuse `fetch_gauge`.
FOREST_GAUGES: dict[str, tuple[tuple[str, str, float, float, str], ...]] = {
    "Pisgah": (
        ("03451500", "French Broad @ Asheville",        35.6090, -82.5790, "primary"),
        ("03451000", "Swannanoa River @ Biltmore",      35.5073, -82.5365, "tributary"),
        ("03456500", "Pigeon River @ Canton",           35.5326, -82.8376, "regional"),
        ("02151500", "Broad River nr Bat Cave",         35.4576, -82.2843, "regional"),
    ),
    "Nantahala": (
        ("03512000", "Oconaluftee @ Birdtown",          35.4623, -83.3457, "primary"),
        ("03513000", "Tuckasegee @ Bryson City",        35.4334, -83.4471, "tributary"),
        ("03500000", "Little Tennessee nr Prentiss",    35.0871, -83.3879, "headwaters"),
        ("03550000", "Cheoah River nr Robbinsville",    35.3409, -83.8338, "regional"),
    ),
    "Uwharrie": (
        ("02126000", "Pee Dee River nr Wadesboro",      34.9893, -80.0648, "primary"),
        ("02125000", "Rocky River nr Norwood",          35.2540, -80.1284, "tributary"),
        ("02118500", "Yadkin River at Yadkin College",  35.8454, -80.3853, "regional"),
    ),
    "Croatan": (
        ("02092500", "Trent River nr Trenton",          35.0651, -77.4047, "tributary"),
        ("02091814", "Neuse River at Kinston",          35.2596, -77.5811, "primary"),
        ("02093229", "White Oak River at Maysville",    34.9020, -77.2369, "regional"),
    ),
}


# USDA Forest Service Ranger District offices. Coordinates are the public
# district-office addresses (close enough to the district's geographic
# population centroid for weather/alerts/AQI/fire-weather purposes).
# Each tuple: (district_name, office_city, lat, lon, notes)
DISTRICT_OFFICES: dict[str, tuple[tuple[str, str, float, float, str], ...]] = {
    "Pisgah": (
        ("Appalachian", "Burnsville",
         35.9170, -82.3000,
         "Black Mtns, Mt. Mitchell, Roan Mtn; steep slopes + frequent rime."),
        ("Grandfather", "Nebo",
         35.7380, -81.9430,
         "Linville Gorge, Wilson Creek wilderness; landslide-prone slopes."),
        ("Pisgah", "Pisgah Forest",
         35.2900, -82.7120,
         "Brevard / DuPont edge; Davidson + French Broad headwaters."),
    ),
    "Nantahala": (
        ("Cheoah", "Robbinsville",
         35.3230, -83.8060,
         "Joyce Kilmer Memorial Forest, Cheoah & Santeetlah lakes."),
        ("Nantahala", "Franklin",
         35.1820, -83.3820,
         "Nantahala Gorge, Standing Indian; high orographic precip."),
        ("Tusquitee", "Murphy",
         35.0900, -84.0300,
         "Hiwassee, Shooting Creek; SW corner of NC."),
    ),
    "Uwharrie": (
        ("Uwharrie", "Troy",
         35.3580, -79.8910,
         "Single district; OHV trails, ancient Uwharrie Mtns."),
    ),
    "Croatan": (
        ("Croatan", "New Bern",
         34.9400, -77.0400,
         "Single district; pocosin wetlands between Neuse and White Oak."),
    ),
}


@dataclass(frozen=True)
class NationalForest:
    name: str
    short: str
    acres: int
    established: int
    hq_city: str
    center_lat: float
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


def _gauges_for(forest_short: str) -> list[dict]:
    """Fetch every USGS gauge associated with one forest, in parallel.

    Each entry returned mirrors the shape used by the dashboard's main gauge
    network (so the UI can reuse `flood_category` styling, etc.).
    """
    entries = FOREST_GAUGES.get(forest_short, ())
    if not entries:
        return []

    def _one(entry):
        site_id, label, lat, lon, role = entry
        g = fetch_gauge(site_id, role=role)
        hist = fetch_gauge_history(site_id, hours=12)
        rate = rate_of_rise_ft_per_hr(hist)
        category = g.flood_category if g else "unknown"
        return {
            "site_id": site_id,
            "label": label,
            "role": role,
            "lat": lat,
            "lon": lon,
            "stage_ft": g.stage_ft if g else None,
            "flood_category": category,
            "flood_class": flood_class(category),
            "rate_ft_per_hr": rate,
        }

    with ThreadPoolExecutor(max_workers=min(4, len(entries))) as pool:
        return list(pool.map(_one, entries))


def _districts_for(forest: NationalForest,
                    active_fires: list | None) -> list[dict]:
    """Pull live weather / alerts / AQI / fire-weather / landslide for every
    ranger district office in this forest, in parallel.

    Per-district payload is intentionally lighter than the forest-level call:
    no USGS gauges (those are forest-scope), no landslide inventory lookup
    (one ArcGIS hit per forest is enough — we only want hazard *score* per
    district), so we can fan out across 1–3 districts cheaply.
    """
    entries = DISTRICT_OFFICES.get(forest.short, ())
    if not entries:
        return []

    def _one(entry):
        name, office, lat, lon, notes = entry
        weather = fetch_current_weather(lat, lon)
        alerts = fetch_nws_alerts(lat, lon)
        soil = fetch_soil_state(lat, lon)
        air = fetch_air_quality(lat, lon)
        landslide = compute_landslide_hazard(forest.region, soil, weather)
        # Skip per-district inventory fetch — heavy and changes slowly.
        landslide["inventory"] = {"count": 0}
        fire_wx = compute_fire_weather(weather, alerts, forest.region)
        nearby_fires = fires_near(lat, lon, active_fires or [], radius_mi=25.0)
        return {
            "name": name,
            "office": office,
            "lat": lat,
            "lon": lon,
            "notes": notes,
            "weather": weather,
            "alerts": alerts,
            "soil": soil,
            "air_quality": air,
            "landslide": landslide,
            "fire_weather": fire_wx,
            "fires_summary": summarize_fires(nearby_fires),
        }

    with ThreadPoolExecutor(max_workers=min(4, len(entries))) as pool:
        return list(pool.map(_one, entries))


def fetch_forest_state(forest: NationalForest,
                        active_storms: list | None = None,
                        active_fires: list | None = None,
                        include_landslide_inventory: bool = True) -> dict:
    """Pull live weather, alerts, gauges, landslide + fire-weather hazards,
    air quality, nearby active wildfires, and (optionally) the USGS National
    Landslide Inventory points for one forest."""
    weather = fetch_current_weather(forest.center_lat, forest.center_lon)
    alerts = fetch_nws_alerts(forest.center_lat, forest.center_lon)
    soil = fetch_soil_state(forest.center_lat, forest.center_lon)
    gauges = _gauges_for(forest.short)
    air = fetch_air_quality(forest.center_lat, forest.center_lon)

    landslide = compute_landslide_hazard(forest.region, soil, weather)
    fire_wx = compute_fire_weather(weather, alerts, forest.region)

    inventory: list[dict] = []
    if include_landslide_inventory:
        inventory = fetch_nearby_landslides(
            forest.center_lat, forest.center_lon, radius_mi=25.0)
    landslide["inventory"] = summarize_inventory(inventory)

    nearby_fires = fires_near(forest.center_lat, forest.center_lon,
                              active_fires or [], radius_mi=50.0)
    fires_summary = summarize_fires(nearby_fires)

    districts_data = _districts_for(forest, active_fires)

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
        "districts_data": districts_data,
        "center_lon": forest.center_lon,
        "districts": list(forest.districts),
        "region": forest.region,
        "notes": forest.notes,
        "weather": weather,
        "alerts": alerts,
        "soil": soil,
        "gauges": gauges,
        "landslide": landslide,
        "fire_weather": fire_wx,
        "air_quality": air,
        "fires_nearby": nearby_fires[:6],   # cap for payload size
        "fires_summary": fires_summary,
        "nearest_storm": nearest_storm,
        "nearest_storm_mi": nearest_mi,
    }


def fetch_all_forests(active_storms: list | None = None) -> list[dict]:
    """Pull live state for every NC National Forest in parallel.

    Fetches the statewide active-wildfire list once up front and passes it
    into each per-forest worker, so we hit the NIFC service exactly once per
    dashboard refresh instead of four times.
    """
    from .wildfire import fetch_active_wildfires
    active_fires = fetch_active_wildfires(state="NC")

    with ThreadPoolExecutor(max_workers=len(NC_NATIONAL_FORESTS)) as pool:
        return list(pool.map(
            lambda f: fetch_forest_state(f, active_storms, active_fires),
            NC_NATIONAL_FORESTS,
        ))
