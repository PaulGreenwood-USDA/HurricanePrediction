"""Digital elevation model (DEM) for the Asheville region.

We use the free, no-auth Open-Meteo Elevation API to pull a coarse elevation
grid (~0.1 degrees, ~7 km) covering the southern Appalachians. The grid is
cached on disk after the first download.

Real elevation lets us compute the terrain gradient grad(h) and the actual
upslope wind component V . grad(h) along each storm track, instead of the
"longitude of Asheville" heuristic in terrain.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests

ELEV_API = "https://api.open-meteo.com/v1/elevation"

# Region: enough to cover the SE Blue Ridge escarpment and storms approaching
# Asheville from the SE.
LAT_MIN, LAT_MAX = 33.5, 37.5
LON_MIN, LON_MAX = -84.5, -80.0
# Coarser 0.20 deg (~14 km) keeps total points under Open-Meteo's free-tier
# rate window. The Blue Ridge escarpment is ~50 km wide so this still resolves
# the gradient adequately for an upslope-flow calculation.
GRID_DEG = 0.20


def _grid_axes() -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_DEG / 2, GRID_DEG)
    lons = np.arange(LON_MIN, LON_MAX + GRID_DEG / 2, GRID_DEG)
    return lats, lons


def _fetch_batch(lats: list[float], lons: list[float], retries: int = 5) -> list[float]:
    delay = 1.5
    for attempt in range(retries):
        r = requests.get(
            ELEV_API,
            params={
                "latitude": ",".join(f"{x:.4f}" for x in lats),
                "longitude": ",".join(f"{x:.4f}" for x in lons),
            },
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["elevation"]
        if r.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
    raise RuntimeError("elevation API: still 429 after retries")


def download_dem(cache_path: str | Path = "data/dem.npz") -> Path:
    """Download elevation grid (one-shot) and cache as .npz."""
    cache = Path(cache_path)
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    lats, lons = _grid_axes()
    LL, NN = np.meshgrid(lats, lons, indexing="ij")
    flat_lat = LL.ravel().tolist()
    flat_lon = NN.ravel().tolist()
    elev = np.full(len(flat_lat), np.nan)
    n = len(flat_lat)
    print(f"Downloading {n} elevation points from Open-Meteo ...")
    BATCH = 100
    for i in range(0, n, BATCH):
        chunk_lat = flat_lat[i:i + BATCH]
        chunk_lon = flat_lon[i:i + BATCH]
        try:
            elev[i:i + BATCH] = _fetch_batch(chunk_lat, chunk_lon)
        except Exception as e:  # noqa: BLE001
            print(f"  batch {i//BATCH} failed: {e}")
        time.sleep(0.6)  # gentle throttle
    missing = int(np.isnan(elev).sum())
    if missing > n // 4:
        raise RuntimeError(f"Too many DEM cells missing ({missing}/{n}); aborting cache.")
    if missing:
        # Fill any small holes with nearest-neighbour mean of valid values
        elev[np.isnan(elev)] = float(np.nanmean(elev))
    elev_grid = elev.reshape(LL.shape)
    np.savez_compressed(cache, lats=lats, lons=lons, elev=elev_grid)
    print(f"  cached -> {cache} ({n} cells, {missing} filled)")
    return cache


def load_dem(cache_path: str | Path = "data/dem.npz") -> dict:
    p = download_dem(cache_path)
    z = np.load(p)
    lats = z["lats"]
    lons = z["lons"]
    elev = z["elev"]
    # Gradient in m per degree. Convert lon-direction to per-meter via cos(lat).
    # We'll keep degree gradients and convert when used.
    dh_dlat, dh_dlon = np.gradient(elev, lats, lons)  # m/deg
    return {
        "lats": lats,
        "lons": lons,
        "elev": elev,
        "dh_dlat": dh_dlat,
        "dh_dlon": dh_dlon,
    }


def _bilinear(grid: np.ndarray, lats: np.ndarray, lons: np.ndarray,
              lat: float, lon: float) -> float:
    """Bilinear interpolation. Returns NaN outside the grid."""
    if lat < lats[0] or lat > lats[-1] or lon < lons[0] or lon > lons[-1]:
        return float("nan")
    i = np.searchsorted(lats, lat) - 1
    j = np.searchsorted(lons, lon) - 1
    i = max(0, min(i, len(lats) - 2))
    j = max(0, min(j, len(lons) - 2))
    fy = (lat - lats[i]) / (lats[i + 1] - lats[i])
    fx = (lon - lons[j]) / (lons[j + 1] - lons[j])
    v00 = grid[i, j]
    v01 = grid[i, j + 1]
    v10 = grid[i + 1, j]
    v11 = grid[i + 1, j + 1]
    return float((1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11))


def upslope_component(
    dem: dict,
    lat: float,
    lon: float,
    wind_from_deg: float,
    wind_speed_kt: float,
) -> float:
    """V . grad(h)  in (m/s)*(m/m), i.e. instantaneous vertical velocity at
    the surface forced by terrain. Positive = upslope (rainfall enhancement).

    `wind_from_deg` is meteorological convention (wind FROM 90 deg = east wind,
    i.e. blowing toward 270 deg).
    """
    dh_dlat = _bilinear(dem["dh_dlat"], dem["lats"], dem["lons"], lat, lon)  # m/deg
    dh_dlon = _bilinear(dem["dh_dlon"], dem["lats"], dem["lons"], lat, lon)  # m/deg
    if np.isnan(dh_dlat) or np.isnan(dh_dlon):
        return 0.0
    # Convert to m/m
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat))
    gh_lat = dh_dlat / m_per_deg_lat  # rise per meter north
    gh_lon = dh_dlon / m_per_deg_lon  # rise per meter east

    # Wind vector (toward) in m/s
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    speed_ms = float(wind_speed_kt) * 0.5144
    u = speed_ms * np.sin(np.radians(wind_to_deg))   # east component
    v = speed_ms * np.cos(np.radians(wind_to_deg))   # north component
    return float(u * gh_lon + v * gh_lat)
