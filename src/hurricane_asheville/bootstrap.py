"""One-time historical bootstrap of the parquet history store.

Hourly snapshots take time to accumulate. To skip that wait we pull
multi-year archives from two free, public, no-auth APIs:

* **USGS NWIS Daily Values** — daily mean stage_ft + discharge_cfs for every
  gauge we already track (UPSTREAM_GAUGES + FOREST_GAUGES).
* **Open-Meteo ERA5 archive** — daily precip + temp + wind for every weather
  point on the dashboard (Asheville + 4 forest centroids + 8 ranger-district
  offices).

Output rows match the schema documented in :mod:`hurricane_asheville.history`.
The ``source`` column ("usgs_dv" / "open_meteo_archive") makes the bootstrap
rows trivially separable from the live "snapshot" rows at training time.
"""
from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Iterable, Sequence

import requests

from .config import ASHEVILLE_LAT, ASHEVILLE_LON
from .forests import (DISTRICT_OFFICES, FOREST_GAUGES, NC_NATIONAL_FORESTS)
from .gauge import UPSTREAM_GAUGES

log = logging.getLogger(__name__)

USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# All distinct USGS sites we care about (statewide gauges + per-forest).
def _all_site_ids() -> list[str]:
    ids = {entry[0] for entry in UPSTREAM_GAUGES}
    for entries in FOREST_GAUGES.values():
        ids.update(entry[0] for entry in entries)
    return sorted(ids)


# All distinct weather points (Asheville + forest centroids + district offices).
def _all_weather_points() -> list[tuple[str, str, float, float]]:
    """Return ``(entity_type, entity_id, lat, lon)`` tuples."""
    pts: list[tuple[str, str, float, float]] = [
        ("point", "asheville", ASHEVILLE_LAT, ASHEVILLE_LON),
    ]
    for f in NC_NATIONAL_FORESTS:
        pts.append(("forest", f.short, f.center_lat, f.center_lon))
    for short, entries in DISTRICT_OFFICES.items():
        for name, _office, lat, lon, _notes in entries:
            pts.append(("district", f"{short}/{name}", lat, lon))
    return pts


# ---- USGS daily values ----------------------------------------------------

def _parse_usgs_dv_payload(data: dict, site_id: str
                            ) -> list[tuple[str, str, float]]:
    """Returns [(iso_date, metric, value)] from a NWIS DV JSON response."""
    series = (data.get("value") or {}).get("timeSeries") or []
    out: list[tuple[str, str, float]] = []
    for ts in series:
        code = ts["variable"]["variableCode"][0]["value"]
        metric = {"00065": "stage_ft",
                   "00060": "discharge_cfs",
                   # Reservoirs report pool elevation, not river stage. Asking
                   # only for 00060/00065 returned nothing at all for Falls
                   # Lake and almost nothing for Jordan Lake.
                   "00062": "pool_elevation_ft"}.get(code)
        if not metric:
            continue
        values = (ts.get("values") or [{}])[0].get("value") or []
        for v in values:
            try:
                f = float(v["value"])
            except (TypeError, ValueError):
                continue
            if f == -999999:
                continue
            t = v.get("dateTime")
            if not t:
                continue
            out.append((t, metric, f))
    return out


def fetch_usgs_dv(site_id: str, start: str, end: str,
                   timeout: int = 30) -> list[dict]:
    """Pull daily-mean stage + discharge for one USGS site over [start, end]."""
    try:
        r = requests.get(
            USGS_DV_URL,
            params={
                "sites": site_id,
                "parameterCd": "00060,00062,00065",
                "startDT": start,
                "endDT": end,
                "format": "json",
                "statCd": "00003",  # daily mean
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("USGS DV fetch failed for %s: %s", site_id, exc)
        return []

    rows = []
    for t, metric, value in _parse_usgs_dv_payload(data, site_id):
        rows.append({
            "ts": t,
            "source": "usgs_dv",
            "entity_type": "gauge",
            "entity_id": site_id,
            "metric": metric,
            "value": value,
        })
    return rows


def bootstrap_gauges(years: int = 5,
                      site_ids: Sequence[str] | None = None,
                      end: str | None = None,
                      pause_s: float = 0.5) -> list[dict]:
    """Daily mean stage_ft / discharge_cfs for every NC gauge, last N years."""
    end_d = _dt.date.today() if end is None else _dt.date.fromisoformat(end)
    start_d = end_d.replace(year=end_d.year - years)
    sites = list(site_ids) if site_ids else _all_site_ids()
    log.info("bootstrap_gauges: %d sites, %s -> %s", len(sites),
              start_d, end_d)

    rows: list[dict] = []
    for sid in sites:
        chunk = fetch_usgs_dv(sid, start_d.isoformat(), end_d.isoformat())
        log.info("  %s: %d rows", sid, len(chunk))
        rows.extend(chunk)
        if pause_s:
            time.sleep(pause_s)
    return rows


# ---- USGS instantaneous values (15-minute) --------------------------------

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"

#: Gauges worth pulling at full resolution: the ML target plus its upstream
#: predictors. Every other gauge is display-only, and 15-minute data for all
#: 28 would be ~5M rows in a git-committed store for no modelling gain.
IV_GAUGE_IDS = ("03451500", "03439000", "03443000", "03446000", "03451000")


def fetch_usgs_iv(site_id: str, start: str, end: str,
                   timeout: int = 180) -> list[tuple[str, float]]:
    """15-minute gage height for one site over [start, end].

    Returns ``[(iso_time, feet)]``. USGS 301-redirects this endpoint to
    nwis.waterservices.usgs.gov, which requests follows automatically.
    """
    try:
        r = requests.get(
            USGS_IV_URL,
            params={
                "sites": site_id,
                "parameterCd": "00065",
                "startDT": start,
                "endDT": end,
                "format": "json",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("USGS IV fetch failed for %s %s..%s: %s",
                    site_id, start, end, exc)
        return []

    series = (data.get("value") or {}).get("timeSeries") or []
    out: list[tuple[str, float]] = []
    for ts in series:
        for v in (ts.get("values") or [{}])[0].get("value") or []:
            try:
                f = float(v["value"])
            except (TypeError, ValueError):
                continue
            if f == -999999:
                continue
            t = v.get("dateTime")
            if t:
                out.append((t, f))
    return out


def _iv_to_hourly_max_rows(points: list[tuple[str, float]],
                            site_id: str) -> list[dict]:
    """Reduce 15-minute readings to hourly **maxima**.

    Maxima, not means: a mean is what made the stored Helene peak read 18.47 ft
    when the river actually crested at 24.82: averaging smooths a flash crest
    away. The feature pipeline consumes hourly data, so storing all four
    samples per hour would quadruple the store for no modelling gain -- but
    taking the max keeps the peak that matters intact.
    """
    import pandas as pd

    if not points:
        return []
    s = pd.Series({pd.to_datetime(t, utc=True): v for t, v in points})
    s = s.sort_index().resample("1h").max().dropna()
    return [{"ts": ts.isoformat(), "source": "usgs_iv",
             "entity_type": "gauge", "entity_id": site_id,
             "metric": "stage_ft", "value": float(v)}
            for ts, v in s.items()]


def bootstrap_gauges_iv(years: int = 5,
                         site_ids: Sequence[str] | None = None,
                         end: str | None = None,
                         chunk_days: int = 365,
                         pause_s: float = 1.0) -> list[dict]:
    """High-resolution stage history for the modelling gauges.

    The daily-mean backfill leaves the training frame ~93% forward-filled:
    every lag feature reads the same repeated value and within-day dynamics
    simply are not present, so no model can learn a rising limb from it.
    Requests are chunked by year because a decade in one call is a large
    response and a single timeout would lose the lot.
    """
    end_d = _dt.date.today() if end is None else _dt.date.fromisoformat(end)
    start_d = end_d.replace(year=end_d.year - years)
    sites = list(site_ids) if site_ids else list(IV_GAUGE_IDS)
    log.info("bootstrap_gauges_iv: %d sites, %s -> %s", len(sites),
              start_d, end_d)

    rows: list[dict] = []
    for sid in sites:
        site_rows: list[dict] = []
        cursor = start_d
        while cursor < end_d:
            chunk_end = min(cursor + _dt.timedelta(days=chunk_days), end_d)
            pts = fetch_usgs_iv(sid, cursor.isoformat(), chunk_end.isoformat())
            site_rows.extend(_iv_to_hourly_max_rows(pts, sid))
            log.info("  %s %s..%s: %d raw -> %d hourly",
                      sid, cursor, chunk_end, len(pts), len(site_rows))
            cursor = chunk_end + _dt.timedelta(days=1)
            if pause_s:
                time.sleep(pause_s)
        rows.extend(site_rows)
    return rows


# ---- Open-Meteo ERA5 archive ---------------------------------------------

_DAILY_VARS = ("precipitation_sum", "temperature_2m_max",
                "temperature_2m_min", "wind_speed_10m_max")

_VAR_TO_METRIC = {
    "precipitation_sum":  ("wx_precip_in_24h",      lambda mm: mm / 25.4),
    "temperature_2m_max": ("wx_temp_max_f",         lambda c: c * 9 / 5 + 32),
    "temperature_2m_min": ("wx_temp_min_f",         lambda c: c * 9 / 5 + 32),
    "wind_speed_10m_max": ("wx_wind_max_mph",       lambda kmh: kmh * 0.621371),
}


def fetch_open_meteo_archive(lat: float, lon: float,
                              start: str, end: str,
                              timeout: int = 60) -> dict:
    """Raw daily archive payload for one point. Empty dict on failure."""
    try:
        r = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "daily": ",".join(_DAILY_VARS),
                "timezone": "UTC",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Open-Meteo archive fetch failed (%.3f, %.3f): %s",
                    lat, lon, exc)
        return {}


def _archive_to_rows(data: dict, entity_type: str, entity_id: str) -> list[dict]:
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return []
    out: list[dict] = []
    for api_var, (metric, convert) in _VAR_TO_METRIC.items():
        values = daily.get(api_var) or []
        for t, v in zip(times, values):
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            out.append({
                "ts": t,
                "source": "open_meteo_archive",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": metric,
                "value": convert(f),
            })
    return out


# ---- Open-Meteo ERA5 soil moisture ---------------------------------------

# Soil moisture is the pre-conditioner that turns a wet tropical system into a
# catastrophic one, and it was the single input with no history at all -- the
# daily archive above carries only precip, temp and wind. ERA5 publishes it
# hourly, so we pull hourly and reduce to a daily mean.
#
# Depths differ from the live feed. soil.py reads the forecast model's
# 0-1/1-3/3-9/9-27 cm layers; ERA5 offers 0-7/7-28 cm. They are the same
# quantity in the same units but not the same measurement, so they get
# distinct metric names rather than being silently concatenated into one
# series that changes definition partway through.
_SOIL_HOURLY_VARS = ("soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm")

_SOIL_VAR_TO_METRIC = {
    "soil_moisture_0_to_7cm":  "soil_era5_0_7cm",
    "soil_moisture_7_to_28cm": "soil_era5_7_28cm",
}


def fetch_open_meteo_soil(lat: float, lon: float, start: str, end: str,
                           timeout: int = 120) -> dict:
    """Raw hourly ERA5 soil-moisture payload for one point."""
    try:
        r = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "hourly": ",".join(_SOIL_HOURLY_VARS),
                "timezone": "UTC",
            },
            timeout=timeout,
            headers={"User-Agent": "hurricane-asheville/0.1"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Open-Meteo soil fetch failed (%.3f, %.3f): %s",
                    lat, lon, exc)
        return {}


def _soil_to_daily_rows(data: dict, entity_type: str,
                         entity_id: str) -> list[dict]:
    """Reduce hourly soil moisture to one daily mean per metric.

    Five years hourly at 13 points is ~1.1M values; the index replay and the
    ML feature builder both work daily, so storing hourly would be dead weight.
    """
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return []

    out: list[dict] = []
    for api_var, metric in _SOIL_VAR_TO_METRIC.items():
        values = hourly.get(api_var) or []
        buckets: dict[str, list[float]] = {}
        for t, v in zip(times, values):
            if v is None:
                continue
            try:
                buckets.setdefault(t[:10], []).append(float(v))
            except (TypeError, ValueError):
                continue
        for day, vals in buckets.items():
            out.append({
                "ts": day,
                "source": "open_meteo_archive",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": metric,
                "value": sum(vals) / len(vals),
            })
    return out


def bootstrap_soil(years: int = 5,
                    points: Iterable[tuple[str, str, float, float]] | None = None,
                    end: str | None = None,
                    pause_s: float = 0.6) -> list[dict]:
    """Daily-mean ERA5 soil moisture for every weather point, last N years."""
    end_d = _dt.date.today() if end is None else _dt.date.fromisoformat(end)
    start_d = end_d.replace(year=end_d.year - years)
    pts = list(points) if points else _all_weather_points()
    log.info("bootstrap_soil: %d points, %s -> %s", len(pts), start_d, end_d)

    rows: list[dict] = []
    for etype, eid, lat, lon in pts:
        data = fetch_open_meteo_soil(lat, lon, start_d.isoformat(),
                                      end_d.isoformat())
        chunk = _soil_to_daily_rows(data, etype, eid)
        log.info("  %s/%s: %d rows", etype, eid, len(chunk))
        rows.extend(chunk)
        if pause_s:
            time.sleep(pause_s)
    return rows


def bootstrap_weather(years: int = 5,
                       points: Iterable[tuple[str, str, float, float]]
                          | None = None,
                       end: str | None = None,
                       pause_s: float = 0.4) -> list[dict]:
    """Daily ERA5 precip / temp / wind for every weather point on the
    dashboard, last N years."""
    end_d = _dt.date.today() if end is None else _dt.date.fromisoformat(end)
    start_d = end_d.replace(year=end_d.year - years)
    pts = list(points) if points else _all_weather_points()
    log.info("bootstrap_weather: %d points, %s -> %s", len(pts),
              start_d, end_d)

    rows: list[dict] = []
    for etype, eid, lat, lon in pts:
        data = fetch_open_meteo_archive(lat, lon,
                                         start_d.isoformat(), end_d.isoformat())
        chunk = _archive_to_rows(data, etype, eid)
        log.info("  %s/%s: %d rows", etype, eid, len(chunk))
        rows.extend(chunk)
        if pause_s:
            time.sleep(pause_s)
    return rows


# ---- orchestrator ---------------------------------------------------------

def bootstrap_all(years: int = 5, base_dir=None) -> dict:
    """Run gauge + weather bootstrap, append to the history parquet store.

    Returns a small summary dict suitable for printing in the CLI.
    """
    from . import history

    gauge_rows = bootstrap_gauges(years=years)
    weather_rows = bootstrap_weather(years=years)
    soil_rows = bootstrap_soil(years=years)

    kw = {} if base_dir is None else {"base_dir": base_dir}
    g_files = history.append_long_rows(gauge_rows, **kw)
    w_files = history.append_long_rows(weather_rows, **kw)
    s_files = history.append_long_rows(soil_rows, **kw)

    return {
        "years": years,
        "gauge_rows": len(gauge_rows),
        "weather_rows": len(weather_rows),
        "soil_rows": len(soil_rows),
        "partitions_written": len({*g_files, *w_files, *s_files}),
    }
