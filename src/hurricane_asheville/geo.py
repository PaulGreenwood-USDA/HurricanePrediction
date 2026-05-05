"""Geographic helpers."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_MI = 3958.7613


def haversine_mi(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Inputs may be scalars or numpy arrays."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))
