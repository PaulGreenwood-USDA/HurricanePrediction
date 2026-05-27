# Hurricane Risk: Asheville, NC

> 🌪 **Live dashboard:** <https://paulgreenwood-usda.github.io/HurricanePrediction/>
> — rebuilt hourly by GitHub Actions from USGS, NWS, NHC, Open-Meteo, and NOAA CO-OPS feeds.

A Python toolkit (managed with [uv](https://docs.astral.sh/uv/)) that estimates
the risk of an Atlantic tropical cyclone — and the resulting **flood hazard** —
for **Asheville, NC**, by fusing climatology, seasonal outlooks, real-time
storm tracks, live river-gauge data, terrain-aware orographic rainfall
modeling, soil-moisture pre-conditioning, and a live web dashboard.

It is built around the lesson of Helene (Sep 2024): for an inland city in the
Blue Ridge, the danger is not landfalling wind but **decaying tropical systems
dumping orographic rainfall into a north-flowing mountain watershed on already
saturated soil**.

## What it does

1. **Climatology** — NOAA HURDAT2 best-track data (1851–2025, includes Helene)
   filtered to storms that came within a configurable radius of Asheville
   (default 150 mi).
2. **Seasonal context** — Colorado State University (CSU) Tropical
   Meteorology Project April 2026 outlook (extracted from *Helene PMP HUB.pdf*).
   Climatological frequency is scaled by `ACE_forecast / ACE_climo`.
3. **Real-time storm tracking** — National Hurricane Center *Active Storms*
   feed, with distance from each current Atlantic storm to Asheville.
4. **Terrain-aware orographic rainfall** — real DEM (Open-Meteo Elevation),
   the upslope wind component `V · ∇h` at the SE Blue Ridge escarpment,
   Kaplan–DeMaria inland wind decay with mountain enhancement, monthly PWAT
   climatology + tropical-moisture surge, and the fraction of each track that
   sits inside the French Broad watershed upstream of Asheville.
5. **Live river gauge + flood thresholds** — USGS NWIS for French Broad at
   Asheville (site 03451500), with NWS AHPS action / minor / moderate / major
   stage thresholds and Helene's preliminary record stage for reference.
6. **Active NWS alerts** — api.weather.gov alerts for the Asheville point and
   national-forest centroids.
7. **Soil moisture / antecedent precipitation** — Open-Meteo volumetric water
   content at multiple depths plus 7-day rainfall totals (the missing
   pre-conditioner that turned Helene from "wet" into "catastrophic").
8. **Surrounding context** — current weather + alerts for the four NC
   National Forests (Pisgah, Nantahala, Uwharrie, Croatan) and live tide /
   storm-surge observations from NOAA CO-OPS stations along the NC coast.
9. **Asheville Flood Index (AFI)** — a single 0–100 composite
   (CALM → ELEVATED → ALERT → WARNING → EMERGENCY) computed from the live
   gauge, soil saturation, antecedent rainfall, active alerts, and proximity
   of any active tropical system.
10. **Live web dashboard** — Flask app that renders all of the above with
    60-second caching; also publishable as a static snapshot to GitHub Pages
    and deployable to Azure App Service.

> Asheville is ~250 mi inland in the Blue Ridge mountains at ~2,134 ft. The
> hazard is almost always **rainfall and flooding from decaying tropical
> systems** (Helene 2024, Frances + Ivan 2004), not wind from a landfalling
> hurricane. The `P(hurricane-strength inside radius)` metric is therefore
> expected to be very low — that is physically correct.

## Setup

```powershell
uv sync
```

Python 3.12+ is required.

## CLI usage

The package installs a single console script, `hurricane-asheville`.

```powershell
# Climatology + 2026 CSU-adjusted seasonal probability
uv run hurricane-asheville risk

# List the historical storms that came within 150 mi (sorted by date)
uv run hurricane-asheville history --start-year 1990

# 10 closest passes ever
uv run hurricane-asheville history --top 10

# Different radius (e.g. 100 mi for direct rainfall hits)
uv run hurricane-asheville --radius 100 risk

# Real-time check of any active Atlantic storms
uv run hurricane-asheville active

# Plot all historical tracks that came within radius
uv run hurricane-asheville plot --output output/tracks.png

# Rank historical storms by terrain-aware orographic rainfall risk
uv run hurricane-asheville terrain --top 15
# (add --no-dem to skip the real DEM upslope calc and use the longitude proxy)

# Pre-download the regional elevation grid into data/dem.npz
uv run hurricane-asheville dem

# Live USGS gauge + NWS alerts for Asheville
uv run hurricane-asheville gauge

# Run the live web dashboard (Flask) at http://127.0.0.1:5000
uv run hurricane-asheville dashboard
uv run hurricane-asheville dashboard --host 0.0.0.0 --port 8000 --debug
```

Global options (placed before the subcommand):

| Flag           | Default | Description                                          |
| -------------- | ------- | ---------------------------------------------------- |
| `--cache-dir`  | `data`  | Where HURDAT2 and the DEM grid are cached on disk.   |
| `--radius`     | `150`   | Search radius (mi) around Asheville for climatology. |
| `--start-year` | `1950`  | First year included in climatology / plots.          |

## Web dashboard

```powershell
uv run hurricane-asheville dashboard
```

Then open <http://127.0.0.1:5000>. The dashboard refreshes its underlying data
every 60 seconds (in-process cache) and shows:

- Asheville Flood Index (0–100) with color-coded label and component breakdown
- Current French Broad stage vs action / minor / moderate / major / record
- Active NHC Atlantic storms and distance to Asheville
- Active NWS alerts for the Asheville point
- Current weather (Open-Meteo)
- Soil moisture and 7-day antecedent precipitation
- Live NOAA CO-OPS tide / surge observations for NC coastal stations
- Per-forest weather + alerts for Pisgah / Nantahala / Uwharrie / Croatan

### Static snapshot (GitHub Pages)

```powershell
uv run python build_static.py
```

Renders the dashboard once and writes `site/index.html` + `site/state.json`.

### Azure App Service

The repo ships an [azure.yaml](azure.yaml) and Bicep under [infra/](infra/) for
[`azd`](https://learn.microsoft.com/azure/developer/azure-developer-cli/):

```powershell
azd up
```

Oryx runs gunicorn against [wsgi.py](wsgi.py), which adds `src/` to
`sys.path` and exposes `app` from `hurricane_asheville.dashboard`.

## Method notes

- **Climatology probability** uses a Poisson model:
  $P(\ge 1 \text{ storm in a year}) = 1 - e^{-\lambda}$ where
  $\lambda$ = mean storms per year within the radius.
- **Seasonal scaling** multiplies $\lambda$ by `ACE_forecast / ACE_climo`
  (90 / 123 ≈ 0.73 for 2026). This is a deliberately simple adjustment — CSU
  itself notes seasonal forecasts have no skill at predicting *where* storms
  go, only basin-wide activity.
- **Inland wind decay** follows Kaplan & DeMaria (1995), with an additional
  mountain-enhancement term in [src/hurricane_asheville/terrain.py](src/hurricane_asheville/terrain.py).
- **Orographic factor** combines storm position / motion with the real terrain
  gradient sampled from a cached Open-Meteo elevation grid
  ([src/hurricane_asheville/dem.py](src/hurricane_asheville/dem.py),
  [src/hurricane_asheville/terrain.py](src/hurricane_asheville/terrain.py)).
  When no DEM is available it falls back to a longitude / motion heuristic.
- **Moisture factor** uses monthly PWAT climatology for ~35 N 82 W with a
  tropical-surge bonus when an Atlantic TC sits south of Asheville
  ([src/hurricane_asheville/moisture.py](src/hurricane_asheville/moisture.py)).
  Plug in real ERA5 via the `era5_pwat()` hook if you have a CDS API key.
- **Watershed weighting** intersects each track with a hand-drawn polygon
  approximating the French Broad watershed upstream of Asheville
  ([src/hurricane_asheville/watershed.py](src/hurricane_asheville/watershed.py)).
- **Flood Index** combines the live gauge percentile, topsoil saturation
  (≥ 0.40 m³/m³), 7-day antecedent precipitation, active flood / tropical
  alerts, and proximity of any active TC
  ([src/hurricane_asheville/index_score.py](src/hurricane_asheville/index_score.py)).
- HURDAT2 is cached in [data/hurdat2.txt](data/hurdat2.txt) and the elevation
  grid in [data/dem.npz](data/dem.npz) after first download.

## Data sources

| Source                                  | Auth      | Used for                                  |
| --------------------------------------- | --------- | ----------------------------------------- |
| NOAA NHC HURDAT2                        | none      | Historical Atlantic best-tracks 1851–2025 |
| NOAA NHC CurrentStorms.json             | none      | Active Atlantic storms                    |
| USGS NWIS Instantaneous Values          | none      | French Broad gauge (site 03451500)        |
| api.weather.gov                         | none      | NWS active alerts                         |
| Open-Meteo Forecast API                 | none      | Current weather + soil moisture           |
| Open-Meteo Elevation API                | none      | DEM grid for orographic calc              |
| NOAA CO-OPS datagetter                  | none      | Live NC tide / surge observations         |
| CSU Tropical Meteorology Project (PDF)  | bundled   | 2026 seasonal forecast constants          |

All HTTP calls have timeouts and degrade gracefully when a service is offline;
the CLI and dashboard remain usable on cached climatology alone.

## Repository layout

```
azure.yaml                    azd manifest (App Service, Python)
build_static.py               render dashboard once for GitHub Pages
extract_pdf.py                pull text out of the bundled CSU PDF
wsgi.py                       gunicorn entry point for Azure App Service
infra/                        Bicep for App Service + supporting resources
site/                         static snapshot output (index.html, state.json)
data/                         hurdat2.txt + dem.npz cache
output/                       generated plots (e.g. tracks.png)
src/hurricane_asheville/
  config.py        constants, Asheville lat/lon, CSU 2026 numbers, URLs
  geo.py           haversine distance
  hurdat.py        HURDAT2 download + parser
  risk.py          climatology + seasonal scaling
  active.py        NHC current-storms client
  terrain.py       orographic factor, inland wind decay, storm scoring
  dem.py           Open-Meteo elevation grid + grad(h) / upslope component
  moisture.py      PWAT climatology + tropical-surge factor
  watershed.py     French Broad upstream-of-Asheville polygon test
  gauge.py         USGS NWIS gauge + NWS alerts + flood thresholds
  weather.py       Open-Meteo current/forecast weather
  soil.py          soil moisture + 7-day antecedent precip
  forests.py       NC National Forests (Pisgah/Nantahala/Uwharrie/Croatan)
  tides.py         NOAA CO-OPS tide + meteorology stations
  index_score.py   Asheville Flood Index (AFI) composite 0-100
  dashboard.py     Flask web dashboard
  cli.py           argparse entry point
tests/                        pytest suite (conftest stubs out network)
```

## Tests

```powershell
uv run pytest
```

Network-touching modules are stubbed out via [tests/conftest.py](tests/conftest.py)
so the suite runs offline.
