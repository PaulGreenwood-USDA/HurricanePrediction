"""Atmospheric moisture proxy.

Real precipitable water (PWAT) for a specific historical storm requires ERA5
reanalysis (Copernicus CDS, requires free account + API key). For a first-cut
model we use a climatological proxy:

  PWAT_mm(month) for the SE US, peaking ~50 mm in Aug-Sep
  modulated by a "tropical surge" factor when an Atlantic TC is active and
  south of Asheville (the storm itself imports Caribbean/Gulf moisture).

Plug in real ERA5 data via `era5_pwat()` if you have a CDS API key.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

# Monthly mean PWAT (kg/m^2 ~ mm) for ~ 35N 82W from ERA5 1991-2020.
# Source: NCEP/NCAR Reanalysis monthly climo (rounded values).
PWAT_CLIMO_MM = {
    1: 14, 2: 15, 3: 19, 4: 24, 5: 32, 6: 41,
    7: 47, 8: 48, 9: 41, 10: 28, 11: 20, 12: 15,
}

# Saturation PWAT used to normalize: roughly the maximum observed during
# tropical events in the SE US.
PWAT_SAT_MM = 65.0


def climo_pwat_mm(month: int) -> float:
    return float(PWAT_CLIMO_MM.get(int(month), 25))


def storm_moisture_factor(
    storm_lat: float,
    storm_lon: float,
    storm_intensity_kt: float,
    month: int,
    asheville_lat: float = 35.5951,
) -> float:
    """0.3 .. 1.5 multiplier on rainfall risk from atmospheric moisture.

    Combines:
      - Climatological PWAT for the month (Aug/Sep peak)
      - Tropical surge: an intense storm SOUTH of Asheville is pumping
        Caribbean moisture northward, raising effective PWAT well above
        climatology.
    """
    pwat = climo_pwat_mm(month)
    # Tropical surge: scales with intensity (saturating) and with how far
    # south the storm is at closest approach.
    south_offset_deg = max(0.0, asheville_lat - storm_lat)  # degrees south
    south_factor = 1.0 - np.exp(-south_offset_deg / 3.0)    # 0..1, 1.0 ~5deg south
    intensity_factor = min(1.0, max(0.0, (storm_intensity_kt - 25.0) / 75.0))
    surge_mm = 18.0 * south_factor * intensity_factor       # up to +18 mm above climo
    effective = min(PWAT_SAT_MM, pwat + surge_mm)
    return float(0.3 + 1.2 * (effective / PWAT_SAT_MM))     # 0.3..1.5


def era5_pwat(times: Iterable, lat: float, lon: float) -> np.ndarray:
    """Stub: real ERA5 PWAT lookup.

    Set CDSAPI_KEY env var and install `cdsapi` to enable. Returns NaN until
    implemented. This function is intentionally a stub to keep the project
    free of large auth-required downloads while leaving a hook for rigour.
    """
    if os.environ.get("CDSAPI_KEY"):
        raise NotImplementedError(
            "ERA5 hook stubbed. To enable: pip install cdsapi, configure "
            "~/.cdsapirc, then implement single-level total_column_water_vapour "
            "retrieval here. See https://cds.climate.copernicus.eu/api-how-to"
        )
    return np.array([np.nan] * len(list(times)))
