"""Constants for Asheville, NC analysis."""
from __future__ import annotations

# Asheville, NC city center (lat, lon)
ASHEVILLE_LAT = 35.5951
ASHEVILLE_LON = -82.5515

# Default radius (miles) around Asheville to count a storm as "affecting" the area.
# 150 mi captures direct hits + significant rainfall events (Helene 2024 passed
# ~50 mi west at peak rainfall; Frances/Ivan 2004 tracked within ~100 mi).
DEFAULT_RADIUS_MI = 150.0

# CSU April 2026 Atlantic Hurricane Season forecast (from "Helene PMP HUB.pdf")
# Source: Klotzbach et al., CSU Tropical Meteorology Project, 9 April 2026.
CSU_2026_FORECAST = {
    "named_storms": 13,
    "named_storm_days": 55,
    "hurricanes": 6,
    "hurricane_days": 20,
    "major_hurricanes": 2,
    "major_hurricane_days": 5,
    "ace": 90,
    "ntc_pct": 100,
    # P(>=1 major-hurricane landfall) on entire continental US coast
    "p_us_major_landfall": 0.32,
    # P(>=1 major-hurricane landfall) on US East Coast incl. peninsula FL
    "p_east_coast_major_landfall": 0.15,
    # P(>=1 major-hurricane landfall) on Gulf Coast
    "p_gulf_major_landfall": 0.20,
    "issued": "2026-04-09",
}

# 1991-2020 climatological averages quoted by CSU
CSU_CLIMO_1991_2020 = {
    "named_storms": 14.4,
    "named_storm_days": 69.4,
    "hurricanes": 7.2,
    "hurricane_days": 27.0,
    "major_hurricanes": 3.2,
    "major_hurricane_days": 7.4,
    "ace": 123,
    "ntc_pct": 135,
}

# HURDAT2 Atlantic best-track URL (updated annually by NHC).
# This file (released 2026-02-27) covers 1851-2025 and INCLUDES Helene (2024).
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"

# NHC active storms (Atlantic) JSON-ish feed
NHC_ACTIVE_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
