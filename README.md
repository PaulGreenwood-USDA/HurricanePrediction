# Hurricane Risk: Asheville, NC

A small Python tool (managed with [uv](https://docs.astral.sh/uv/)) that estimates the
risk of an Atlantic tropical cyclone affecting **Asheville, NC** by combining:

1. **Climatology** — NOAA HURDAT2 best-track data (1851–present) filtered to
   storms that came within a configurable radius of Asheville (default 150 mi).
2. **Seasonal context** — the Colorado State University (CSU) Tropical
   Meteorology Project April 2026 outlook (the source PDF in this repo,
   *Helene PMP HUB.pdf*). Climatological frequency is scaled by the ratio of
   forecast ACE to the 1991–2020 climatological ACE.
3. **Real-time** — National Hurricane Center *Active Storms* feed, showing
   distance from any current Atlantic storm to Asheville.

> Asheville is ~250 mi inland in the Blue Ridge mountains. The hazard is almost
> always **rainfall and flooding from decaying tropical systems** (Helene
> 2024, Frances + Ivan 2004), not wind from a landfalling hurricane. The
> "P(hurricane-strength inside radius)" metric is therefore expected to be
> very low — that is physically correct.

## Setup

```powershell
uv sync
```

## Use

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
```

## Method notes

- **Climatology probability** uses a Poisson model:
  $P(\ge 1 \text{ storm in a year}) = 1 - e^{-\lambda}$ where
  $\lambda$ = mean storms per year within the radius.
- **Seasonal scaling** multiplies $\lambda$ by `ACE_forecast / ACE_climo`
  (90/123 ≈ 0.73 for 2026). This is a deliberately simple adjustment — CSU
  itself notes seasonal forecasts have no skill at predicting *where* storms
  go, only basin-wide activity.
- HURDAT2 is cached in `data/hurdat2.txt` after first download.

## Layout

```
src/hurricane_asheville/
  config.py    constants, Asheville lat/lon, CSU 2026 numbers
  geo.py       haversine distance
  hurdat.py    HURDAT2 download + parser
  risk.py      climatology + seasonal scaling
  active.py    NHC current-storms client
  cli.py       argparse entry point
```
