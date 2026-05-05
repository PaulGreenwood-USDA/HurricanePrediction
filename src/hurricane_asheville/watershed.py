"""French Broad watershed (upstream of Asheville).

The French Broad River drains roughly 945 sq mi above the USGS gauge at
Asheville (03451500). Unusually for the SE US, this watershed drains
NORTHWARD through Asheville, so any rainfall over the southern Blue Ridge
ends up rolling through downtown.

We approximate the upstream-of-Asheville watershed (HUC-8 06010105 plus
upper portions of 06010106) with a hand-drawn polygon. Point-in-polygon
test is the standard ray-casting algorithm so we don't need shapely.
"""
from __future__ import annotations

import numpy as np

# (lat, lon) vertices, ordered. Approximate but follows the actual ridgeline.
# - SW corner: near Rosman/Brevard headwaters of French Broad
# - South:    along NC/SC and NC/GA state line ridge
# - East:     Pisgah Ridge / Blue Ridge Parkway
# - North:    just south of Weaverville (downstream of Asheville cut off)
# - West:     Newfound + Cold Mountain ridges
FRENCH_BROAD_POLY = np.array([
    (35.65, -82.55),  # north (just downstream of Asheville)
    (35.55, -82.20),  # NE -- Black Mountains
    (35.40, -82.10),  # E -- top of Pisgah Ridge near Mt Mitchell area
    (35.20, -82.20),  # SE -- Pisgah / Blue Ridge crest
    (35.05, -82.55),  # S  -- Brevard / Rosman headwaters
    (35.10, -82.95),  # SW -- Newfound Mountains
    (35.40, -83.10),  # W  -- Cold Mountain area
    (35.60, -82.85),  # NW -- Hominy Creek divide
    (35.65, -82.55),  # close
])


def in_french_broad(lat: float, lon: float,
                    poly: np.ndarray = FRENCH_BROAD_POLY) -> bool:
    """Ray-casting point-in-polygon. poly is (N,2) of (lat, lon)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def fraction_track_in_watershed(track_lats, track_lons) -> float:
    """Fraction of consecutive track segments whose midpoint falls inside the
    French Broad polygon. A storm whose center never touches the watershed
    can still bring rain (eastern semicircle), but a center that DOES pass
    over it is unambiguously a flood threat.
    """
    track_lats = np.asarray(track_lats)
    track_lons = np.asarray(track_lons)
    if len(track_lats) < 2:
        return 0.0
    mid_lat = 0.5 * (track_lats[:-1] + track_lats[1:])
    mid_lon = 0.5 * (track_lons[:-1] + track_lons[1:])
    hits = sum(in_french_broad(la, lo) for la, lo in zip(mid_lat, mid_lon))
    return hits / len(mid_lat)


def watershed_proximity_factor(min_dist_to_watershed_mi: float) -> float:
    """1.0 right at/over the watershed, decaying with distance.

    The eastern-semicircle moisture plume of a tropical system is ~150-200
    mi wide, so a storm 100 mi west can still rain heavily on the basin.
    """
    return float(np.exp(-max(0.0, min_dist_to_watershed_mi) / 100.0))


def watershed_distance_mi(lat: float, lon: float,
                          poly: np.ndarray = FRENCH_BROAD_POLY) -> float:
    """Approximate distance (mi) from a point to the polygon.
    0 if inside; otherwise distance to nearest vertex.
    """
    if in_french_broad(lat, lon, poly):
        return 0.0
    from .geo import haversine_mi
    d = haversine_mi(lat, lon, poly[:, 0], poly[:, 1])
    return float(np.min(d))
