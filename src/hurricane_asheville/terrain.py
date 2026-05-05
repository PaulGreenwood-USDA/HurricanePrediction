"""Terrain-aware adjustments for Asheville hurricane risk.

Asheville sits at ~2,134 ft elevation in the French Broad River valley, on the
windward (SE) side of the Blue Ridge escarpment which rises to 5,000-6,000 ft
in a SW-NE arc immediately south and east of the city.

The dominant tropical-system hazard for Asheville is *orographic rainfall*:
when a tropical cyclone passes WEST of the Blue Ridge with NORTHWARD motion,
its eastern semicircle pumps warm, moist Gulf/Atlantic air against the SE-facing
escarpment, producing 2-4x the rainfall the same storm would drop in flat
terrain. This is the Helene 2024, Frances 2004, Ivan 2004, and Frances/Floyd
1999 pattern.

We model two terrain effects:

  1. orographic_rainfall_factor(storm_lon, storm_motion_deg): 0..3
     Peaks when the storm is west of Asheville with northward motion (forcing
     SE -> NW flow into the escarpment) and is ~zero when east of the mountains.

  2. inland_wind_decay(distance_inland_mi, peak_wind_kt): scaled wind
     Tropical cyclones lose ~50% of surface wind in the first 12 hours after
     landfall (Kaplan & DeMaria 1995). Mountainous terrain accelerates this.

Together with the climatological frequency, these give a more realistic
*flood* risk for Asheville than wind-based metrics alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ASHEVILLE_LAT, ASHEVILLE_LON
from .geo import haversine_mi

# Blue Ridge escarpment crest, simplified: a line from
#   (35.0N, 83.0W)  near Highlands, NC
# to
#   (36.2N, 81.5W)  near Boone, NC
# Bearing of the crest line ~ 040 deg (NE).
BLUE_RIDGE_CREST_BEARING_DEG = 40.0

# Asheville sits on the windward side of this crest for SE flow.
# The "moisture-impinging" wind direction is perpendicular to the crest,
# from the SE -- i.e. ~130 deg (wind FROM 130 deg = blowing toward 310 deg).
UPSLOPE_FLOW_FROM_DEG = 130.0

# Asheville is ~250 mi from the nearest open ocean (SC/GA coast).
ASHEVILLE_DIST_INLAND_MI = 250.0


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing FROM point 1 TO point 2, in degrees from north (0..360)."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    brng = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return brng


def angular_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def orographic_rainfall_factor(
    storm_lon: float,
    storm_lat: float,
    storm_motion_bearing_deg: float | None,
) -> float:
    """0..3 multiplier for rainfall risk based on storm geometry.

    Two ingredients:
      A) Side of mountains. Storms WEST of the Blue Ridge crest (~ -82.5W at
         Asheville's latitude) get the full upslope effect on Asheville.
         Storms EAST of the crest are shielded.
      B) Storm motion. A northward-moving storm to the west of Asheville
         drives its eastern semicircle's southerly flow up the escarpment.
         A storm moving south or stationary does not.
    """
    # A) Side of mountains, smooth transition over ~50 mi
    # West of Asheville (more negative lon) -> factor approaches 1.0
    # East of Asheville -> factor falls toward ~0.2 (shielded)
    delta_lon_mi = (ASHEVILLE_LON - storm_lon) * 69.0 * np.cos(np.radians(storm_lat))
    side = 0.2 + 0.8 / (1.0 + np.exp(-delta_lon_mi / 25.0))  # logistic, 0.2..1.0

    # B) Motion alignment. Want storm moving roughly north (bearing 0/360)
    # to maximize southerly flow on its east side. Score = cos(motion - 0)
    # clamped to [0, 1]. If motion unknown, assume modest factor 0.5.
    if storm_motion_bearing_deg is None or np.isnan(storm_motion_bearing_deg):
        motion_score = 0.5
    else:
        # Best alignment is northward motion (bearing 0). Worst is southward (180).
        motion_score = max(0.0, np.cos(np.radians(storm_motion_bearing_deg)))

    # Base factor (no terrain) = 1.0; max plausible amplification ~ 3x
    return float(1.0 + 2.0 * side * motion_score)


def inland_wind_decay(distance_inland_mi: float, peak_wind_kt: float) -> float:
    """Kaplan-DeMaria style inland wind decay, with mountain enhancement.

    Empirical: V(t) = Vb + (V0 - Vb) * exp(-alpha * t),  Vb = 26 kt
    We translate hours-since-landfall to inland distance assuming 12 mph
    forward motion. Mountains roughly double the decay constant.
    """
    Vb = 26.0
    if peak_wind_kt is None or peak_wind_kt <= Vb:
        return float(peak_wind_kt or 0.0)
    forward_mph = 12.0
    hours_inland = max(0.0, distance_inland_mi) / forward_mph
    alpha_flat = 0.095   # per hour, Kaplan-DeMaria mid-Atlantic
    alpha_mtn = 0.18     # roughly 2x in the southern Appalachians
    return float(Vb + (peak_wind_kt - Vb) * np.exp(-alpha_mtn * hours_inland))


@dataclass
class StormTerrainScore:
    storm_id: str
    name: str
    year: int
    min_dist_mi: float
    closest_lat: float
    closest_lon: float
    motion_bearing_at_closest_deg: float | None
    peak_wind_kt: float | None
    decayed_wind_kt: float
    orographic_factor: float
    rainfall_risk_score: float


def score_storm_terrain(track: pd.DataFrame) -> StormTerrainScore | None:
    """Compute a terrain-aware rainfall risk score for one storm's full track.

    `track` is the full HURDAT2 track (not just points within radius) so we
    can compute motion at closest approach.
    """
    track = track.sort_values("datetime").reset_index(drop=True)
    if track.empty:
        return None

    d = haversine_mi(ASHEVILLE_LAT, ASHEVILLE_LON, track["lat"].to_numpy(), track["lon"].to_numpy())
    i = int(np.argmin(d))
    closest_lat = float(track.loc[i, "lat"])
    closest_lon = float(track.loc[i, "lon"])
    min_dist = float(d[i])

    # Motion at closest approach: bearing from previous point to next point
    j_prev = max(0, i - 1)
    j_next = min(len(track) - 1, i + 1)
    if j_next > j_prev:
        motion = float(
            bearing_deg(
                track.loc[j_prev, "lat"], track.loc[j_prev, "lon"],
                track.loc[j_next, "lat"], track.loc[j_next, "lon"],
            )
        )
    else:
        motion = float("nan")

    peak = track["wind_kt"].max(skipna=True)
    peak = None if pd.isna(peak) else float(peak)

    decayed = inland_wind_decay(min_dist, peak or 0.0)
    oro = orographic_rainfall_factor(closest_lon, closest_lat, motion if not np.isnan(motion) else None)

    # Rainfall risk score: scales with proximity, intensity, and orographic factor.
    # Distance kernel: exp(-d / 75 mi); a TS-strength remnant at the closest
    # point of the Blue Ridge gets a benchmark score ~ peak_wind / 2.
    dist_kernel = float(np.exp(-min_dist / 75.0))
    intensity = peak or 25.0
    rainfall_score = float(intensity * dist_kernel * oro / 2.0)

    return StormTerrainScore(
        storm_id=str(track["storm_id"].iloc[0]),
        name=str(track["name"].iloc[0]),
        year=int(track["year"].iloc[0]),
        min_dist_mi=min_dist,
        closest_lat=closest_lat,
        closest_lon=closest_lon,
        motion_bearing_at_closest_deg=None if np.isnan(motion) else motion,
        peak_wind_kt=peak,
        decayed_wind_kt=decayed,
        orographic_factor=oro,
        rainfall_risk_score=rainfall_score,
    )


def score_all_near_storms(
    tracks: pd.DataFrame,
    radius_mi: float,
    start_year: int = 1950,
) -> pd.DataFrame:
    """Score every storm whose track came within `radius_mi` of Asheville."""
    d = haversine_mi(ASHEVILLE_LAT, ASHEVILLE_LON, tracks["lat"].to_numpy(), tracks["lon"].to_numpy())
    near_ids = set(tracks.assign(_d=d).query("_d <= @radius_mi and year >= @start_year")["storm_id"])
    rows = []
    for sid, g in tracks[tracks["storm_id"].isin(near_ids)].groupby("storm_id"):
        s = score_storm_terrain(g)
        if s is not None:
            rows.append(s.__dict__)
    df = pd.DataFrame(rows).sort_values("rainfall_risk_score", ascending=False)
    return df
