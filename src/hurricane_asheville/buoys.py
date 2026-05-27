"""NOAA NDBC offshore buoy + C-MAN station realtime feed.

The National Data Buoy Center publishes a fixed-format text endpoint per
station at:
    https://www.ndbc.noaa.gov/data/realtime2/{station}.txt

These offshore platforms give 12-24h advance warning of hurricane approach
before storms reach the NC coast — wave height + period + wind +
pressure are the canonical leading indicators of an approaching tropical
cyclone.

We pull a short list of stations along/off the NC coast plus the South
Atlantic Bight to surface that lead-time signal on the dashboard.

Each station file's first data row is the most recent observation, columns:
    YY  MM DD hh mm  WDIR WSPD GST  WVHT  DPD  APD MWD  PRES  ATMP  WTMP ...
Missing values are encoded as 'MM'.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

NDBC_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"

# (station_id, label, lat, lon, role)
#   offshore = open-ocean buoys (deep water, full wave field)
#   nearshore = coastal/sound buoys
#   cman = Coastal-Marine Automated Network (lighthouses, towers)
NC_BUOYS: tuple[tuple[str, str, float, float, str], ...] = (
    ("41025", "Diamond Shoals (off Hatteras)",  35.006, -75.402, "offshore"),
    ("41013", "Frying Pan Shoals (off Cape Fear)", 33.441, -77.764, "offshore"),
    ("41036", "Onslow Bay Outer",               34.207, -76.949, "offshore"),
    ("41037", "Wrightsville Beach Nearshore",   34.144, -77.715, "nearshore"),
    ("CLKN7", "Cape Lookout C-MAN",             34.622, -76.525, "cman"),
    ("DUKN7", "Duck Pier C-MAN (FRF)",          36.183, -75.747, "cman"),
    ("HCGN7", "Hatteras C-MAN",                 35.209, -75.703, "cman"),
    # Adjacent state buoys give extra spatial coverage of approaching systems:
    ("41002", "South Hatteras (S Atlantic Bight)", 31.760, -74.840, "offshore"),
    ("41004", "Edisto SC (S of NC)",            32.501, -79.099, "offshore"),
)


def _parse_realtime2(text: str) -> dict | None:
    """Return the most recent obs row as a dict, or None if unparseable."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    # Two header rows (units + variable names); then most recent obs first.
    header = lines[0].lstrip("#").split()
    for raw in lines[2:]:
        parts = raw.split()
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))

        def _f(key: str) -> float | None:
            v = row.get(key)
            if v in (None, "MM", ""):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        # Skip rows where everything important is missing.
        if all(_f(k) is None for k in ("WVHT", "WSPD", "PRES")):
            continue

        return {
            "obs_time": " ".join(parts[:5]),
            "wind_dir_deg": _f("WDIR"),
            "wind_kt": _ms_to_kt(_f("WSPD")),
            "wind_gust_kt": _ms_to_kt(_f("GST")),
            "wave_ht_ft": _m_to_ft(_f("WVHT")),
            "dominant_period_s": _f("DPD"),
            "avg_period_s": _f("APD"),
            "wave_dir_deg": _f("MWD"),
            "pressure_mb": _f("PRES"),
            "air_temp_f": _c_to_f(_f("ATMP")),
            "water_temp_f": _c_to_f(_f("WTMP")),
        }
    return None


def _ms_to_kt(v):  # m/s -> kt
    return None if v is None else round(v * 1.94384, 1)


def _m_to_ft(v):
    return None if v is None else round(v * 3.28084, 1)


def _c_to_f(v):
    return None if v is None else round(v * 9 / 5 + 32, 1)


def _classify_seas(wave_ft: float | None, wind_kt: float | None) -> str:
    """Coarse sea-state pill for the UI."""
    if wave_ft is None and wind_kt is None:
        return "unknown"
    wh = wave_ft or 0
    wk = wind_kt or 0
    if wh >= 25 or wk >= 64:
        return "PHENOMENAL"
    if wh >= 13 or wk >= 48:
        return "VERY ROUGH"
    if wh >= 8 or wk >= 34:
        return "ROUGH"
    if wh >= 4 or wk >= 22:
        return "MODERATE"
    return "CALM"


_SEA_COLOR = {
    "CALM": "#4caf50",
    "MODERATE": "#9e9d24",
    "ROUGH": "#ef6c00",
    "VERY ROUGH": "#c62828",
    "PHENOMENAL": "#6a1b9a",
    "unknown": "#555",
}


def fetch_buoy(station: str, timeout: int = 15) -> dict | None:
    try:
        r = requests.get(
            NDBC_URL.format(station=station),
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        return _parse_realtime2(r.text)
    except Exception as exc:  # noqa: BLE001
        log.info("NDBC fetch skipped (%s): %s", station, exc)
        return None


def fetch_all_buoys() -> list[dict]:
    """Pull every NC buoy/C-MAN in parallel; one dict per station."""

    def _one(entry):
        station, label, lat, lon, role = entry
        obs = fetch_buoy(station) or {}
        # Guarantee every key the dashboard template expects, even on miss.
        defaults = {
            "obs_time": None, "wind_dir_deg": None, "wind_kt": None,
            "wind_gust_kt": None, "wave_ht_ft": None, "dominant_period_s": None,
            "avg_period_s": None, "wave_dir_deg": None, "pressure_mb": None,
            "air_temp_f": None, "water_temp_f": None,
        }
        merged = {**defaults, **obs}
        seas = _classify_seas(merged.get("wave_ht_ft"), merged.get("wind_kt"))
        return {
            "station_id": station,
            "label": label,
            "role": role,
            "lat": lat,
            "lon": lon,
            "seas": seas,
            "color": _SEA_COLOR.get(seas, "#555"),
            **merged,
        }

    with ThreadPoolExecutor(max_workers=min(8, len(NC_BUOYS))) as pool:
        return list(pool.map(_one, NC_BUOYS))
