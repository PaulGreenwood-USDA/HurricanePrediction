"""Hurricane risk analysis for the Asheville, NC region.

Combines:
  - NOAA HURDAT2 historical Atlantic best-track data (climatology)
  - CSU Tropical Meteorology Project seasonal forecast (activity scaling)
  - NHC active-storms feed (real-time risk)

Asheville is inland (~250 mi from coast) so the dominant hazard is
tropical-system rainfall (e.g. Helene 2024, Frances/Ivan 2004).
"""

__version__ = "0.1.0"
