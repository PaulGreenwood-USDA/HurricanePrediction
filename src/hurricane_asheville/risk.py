"""Climatology + season-scaled risk model for Asheville."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ASHEVILLE_LAT,
    ASHEVILLE_LON,
    CSU_2026_FORECAST,
    CSU_CLIMO_1991_2020,
    DEFAULT_RADIUS_MI,
)
from .geo import haversine_mi


@dataclass
class AshevilleRisk:
    radius_mi: float
    years_analyzed: tuple[int, int]
    n_years: int
    n_storms_in_radius: int
    storms_per_year: float
    p_at_least_one_storm: float
    p_hurricane_strength_in_radius: float
    seasonal_scale: float
    seasonal_p_at_least_one_storm: float
    season_label: str


def storms_near_asheville(
    tracks: pd.DataFrame,
    radius_mi: float = DEFAULT_RADIUS_MI,
) -> pd.DataFrame:
    """Return one row per (storm_id, year) whose track entered the radius.

    Includes the closest-approach distance, peak status while inside the
    radius, and peak wind while inside the radius.
    """
    d = haversine_mi(ASHEVILLE_LAT, ASHEVILLE_LON, tracks["lat"].to_numpy(), tracks["lon"].to_numpy())
    inside = tracks.assign(dist_mi=d).query("dist_mi <= @radius_mi")
    if inside.empty:
        return inside
    agg = (
        inside.sort_values("datetime")
        .groupby(["storm_id", "name", "year"], as_index=False)
        .agg(
            min_dist_mi=("dist_mi", "min"),
            peak_wind_kt=("wind_kt", "max"),
            min_pres_mb=("pres_mb", "min"),
            statuses=("status", lambda s: ",".join(sorted(set(s)))),
            first_time=("datetime", "min"),
            last_time=("datetime", "max"),
        )
        .sort_values("first_time")
    )
    return agg


def compute_climatology(
    tracks: pd.DataFrame,
    radius_mi: float = DEFAULT_RADIUS_MI,
    start_year: int = 1950,
) -> dict:
    """Climatological per-year rate of TCs affecting Asheville."""
    sub = tracks.query("year >= @start_year")
    yr_min, yr_max = int(sub["year"].min()), int(sub["year"].max())
    n_years = yr_max - yr_min + 1
    near = storms_near_asheville(sub, radius_mi=radius_mi)
    n_storms = len(near)
    rate = n_storms / n_years
    # Probability >= 1 storm in a year, assuming Poisson with rate `rate`
    p_any = 1.0 - np.exp(-rate)
    # Fraction of those that were >= TS (>=34kt) AND >= hurricane (>=64kt) within radius
    hur = near.query("peak_wind_kt >= 64")
    p_hur = (1.0 - np.exp(-len(hur) / n_years))
    return {
        "year_range": (yr_min, yr_max),
        "n_years": n_years,
        "n_storms": n_storms,
        "rate_per_year": rate,
        "p_at_least_one": p_any,
        "p_at_least_one_hurricane": p_hur,
        "near_table": near,
    }


def seasonal_scale_factor(
    season_ace: float | None = None,
    climo_ace: float = CSU_CLIMO_1991_2020["ace"],
) -> float:
    """Scale climatological rate by ratio of forecast ACE to climo ACE.

    A below-average season (ACE 90 vs climo 123) gives ~0.73x risk.
    """
    if season_ace is None:
        return 1.0
    return float(season_ace) / float(climo_ace)


def build_risk(
    tracks: pd.DataFrame,
    radius_mi: float = DEFAULT_RADIUS_MI,
    start_year: int = 1950,
    season_forecast: dict | None = None,
    season_label: str = "2026 (CSU April outlook)",
) -> AshevilleRisk:
    if season_forecast is None:
        season_forecast = CSU_2026_FORECAST
    climo = compute_climatology(tracks, radius_mi=radius_mi, start_year=start_year)
    scale = seasonal_scale_factor(season_forecast.get("ace"))
    scaled_rate = climo["rate_per_year"] * scale
    p_any_scaled = 1.0 - np.exp(-scaled_rate)
    return AshevilleRisk(
        radius_mi=radius_mi,
        years_analyzed=climo["year_range"],
        n_years=climo["n_years"],
        n_storms_in_radius=climo["n_storms"],
        storms_per_year=climo["rate_per_year"],
        p_at_least_one_storm=climo["p_at_least_one"],
        p_hurricane_strength_in_radius=climo["p_at_least_one_hurricane"],
        seasonal_scale=scale,
        seasonal_p_at_least_one_storm=p_any_scaled,
        season_label=season_label,
    )
