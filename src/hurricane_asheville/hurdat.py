"""HURDAT2 best-track loader.

HURDAT2 format reference: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format.pdf

Each storm starts with a header line:
    AL092023, IDALIA, 38,
followed by N data lines:
    20230826, 1200,  , TD, 18.0N,  85.5W,  30, 1006, ... (16 fields)
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from .config import HURDAT2_URL

log = logging.getLogger(__name__)


def _parse_lat(s: str) -> float:
    s = s.strip()
    v = float(s[:-1])
    return v if s[-1] == "N" else -v


def _parse_lon(s: str) -> float:
    s = s.strip()
    v = float(s[:-1])
    return v if s[-1] == "E" else -v


def download_hurdat2(cache_path: str | os.PathLike, url: str = HURDAT2_URL) -> Path:
    """Download HURDAT2 to cache_path if missing. Returns the path."""
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 100_000:
        return cache
    log.info("Downloading HURDAT2 from %s ...", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return cache


def parse_hurdat2(path: str | os.PathLike) -> pd.DataFrame:
    """Parse HURDAT2 file to a long-form DataFrame of track points."""
    rows = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    buf = io.StringIO(text)
    storm_id = name = None
    for raw in buf:
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # Header lines are 4 fields; data lines have 21 fields (20 + trailing comma).
        if len(parts) == 4 and parts[0].startswith(("AL", "EP", "CP")):
            storm_id = parts[0]
            name = parts[1]
            continue
        if len(parts) < 8 or storm_id is None:
            continue
        try:
            date = parts[0]
            time = parts[1]
            record_id = parts[2]      # "L"=landfall, etc.
            status = parts[3]         # TD, TS, HU, EX, SS, ...
            lat = _parse_lat(parts[4])
            lon = _parse_lon(parts[5])
            wind = int(parts[6])      # knots, -99 = missing
            pres = int(parts[7])      # mb, -999 = missing
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "storm_id": storm_id,
                "name": name,
                "datetime": pd.to_datetime(f"{date}{time}", format="%Y%m%d%H%M", errors="coerce"),
                "year": int(date[:4]),
                "record_id": record_id,
                "status": status,
                "lat": lat,
                "lon": lon,
                "wind_kt": None if wind < 0 else wind,
                "pres_mb": None if pres <= 0 else pres,
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["datetime"])
    return df


def load_hurdat2(cache_dir: str | os.PathLike = "data") -> pd.DataFrame:
    path = download_hurdat2(Path(cache_dir) / "hurdat2.txt")
    return parse_hurdat2(path)
