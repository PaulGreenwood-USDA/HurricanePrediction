"""CLI entry point: `uv run hurricane-asheville`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .active import fetch_active_storms
from .config import ASHEVILLE_LAT, ASHEVILLE_LON, CSU_2026_FORECAST, DEFAULT_RADIUS_MI
from .gauge import FLOOD_STAGES_FT, fetch_gauge, fetch_nws_alerts
from .hurdat import load_hurdat2
from .risk import build_risk, storms_near_asheville
from .terrain import score_all_near_storms


def cmd_history(args):
    tracks = load_hurdat2(args.cache_dir)
    near = storms_near_asheville(tracks, radius_mi=args.radius)
    near = near.query("year >= @args.start_year")
    if args.top:
        near = near.sort_values("min_dist_mi").head(args.top)
    cols = ["year", "name", "storm_id", "min_dist_mi", "peak_wind_kt", "min_pres_mb", "statuses"]
    print(near[cols].to_string(index=False))
    print(f"\n{len(near)} storms passed within {args.radius:.0f} mi of Asheville since {args.start_year}.")


def cmd_risk(args):
    tracks = load_hurdat2(args.cache_dir)
    risk = build_risk(
        tracks,
        radius_mi=args.radius,
        start_year=args.start_year,
        season_forecast=CSU_2026_FORECAST,
    )
    print(f"--- Asheville, NC tropical-cyclone risk ---")
    print(f"Center                : {ASHEVILLE_LAT:.4f}N, {ASHEVILLE_LON:.4f}W")
    print(f"Search radius         : {risk.radius_mi:.0f} mi")
    print(f"Climatology window    : {risk.years_analyzed[0]}-{risk.years_analyzed[1]} ({risk.n_years} yrs)")
    print(f"Storms within radius  : {risk.n_storms_in_radius}")
    print(f"Mean storms / year    : {risk.storms_per_year:.3f}")
    print(f"P(>=1 TC in a year)   : {risk.p_at_least_one_storm*100:.1f}%   [climatology]")
    print(f"P(>=1 hurricane-strength inside radius / yr): "
          f"{risk.p_hurricane_strength_in_radius*100:.1f}%")
    print()
    print(f"Season label          : {risk.season_label}")
    print(f"  ACE forecast        : {CSU_2026_FORECAST['ace']} (climo 123)")
    print(f"  Named storms        : {CSU_2026_FORECAST['named_storms']} (climo 14.4)")
    print(f"  Major hurricanes    : {CSU_2026_FORECAST['major_hurricanes']} (climo 3.2)")
    print(f"  Seasonal scale      : x{risk.seasonal_scale:.2f}")
    print(f"  Season-adjusted P(>=1 TC affects Asheville): "
          f"{risk.seasonal_p_at_least_one_storm*100:.1f}%")


def cmd_active(args):
    storms = fetch_active_storms()
    if not storms:
        print("No active Atlantic storms (or feed unavailable).")
        return
    print(f"{'Name':<14}{'Class':<8}{'Wind kt':>8}{'  Pos':<18}{'Dist (mi)':>11}  Movement")
    print("-" * 78)
    for s in storms:
        pos = f"{s.lat:.1f},{s.lon:.1f}"
        wind = f"{s.intensity_kt:.0f}" if s.intensity_kt is not None else "?"
        print(f"{s.name:<14}{s.classification:<8}{wind:>8}  {pos:<16}{s.distance_mi:>11.0f}  {s.movement}")
        if s.public_advisory_url:
            print(f"             advisory: {s.public_advisory_url}")


def cmd_plot(args):
    tracks = load_hurdat2(args.cache_dir)
    near = storms_near_asheville(tracks, radius_mi=args.radius)
    near_ids = set(near["storm_id"])
    sub = tracks[tracks["storm_id"].isin(near_ids) & (tracks["year"] >= args.start_year)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for sid, g in sub.groupby("storm_id"):
        g = g.sort_values("datetime")
        ax.plot(g["lon"], g["lat"], color="steelblue", alpha=0.35, linewidth=1)
    ax.plot(ASHEVILLE_LON, ASHEVILLE_LAT, "r*", markersize=14, label="Asheville")
    # radius circle (rough; degrees ~ miles/69 at this latitude)
    import numpy as np
    th = np.linspace(0, 2 * np.pi, 200)
    deg = args.radius / 69.0
    ax.plot(ASHEVILLE_LON + deg * np.cos(th) / np.cos(np.radians(ASHEVILLE_LAT)),
            ASHEVILLE_LAT + deg * np.sin(th), "r--", linewidth=1, label=f"{args.radius:.0f} mi")
    ax.set_xlim(-100, -60)
    ax.set_ylim(20, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Atlantic TCs passing within {args.radius:.0f} mi of Asheville ({args.start_year}+)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def cmd_terrain(args):
    tracks = load_hurdat2(args.cache_dir)
    dem_path = str(Path(args.cache_dir) / "dem.npz")
    df = score_all_near_storms(tracks, radius_mi=args.radius, start_year=args.start_year,
                               use_dem=not args.no_dem, dem_path=dem_path)
    if df.empty:
        print("No storms found.")
        return
    df = df.head(args.top)
    cols = ["year", "name", "month", "min_dist_mi", "peak_wind_kt", "decayed_wind_kt",
            "orographic_factor", "upslope_w_ms", "moisture_factor",
            "watershed_proximity", "watershed_track_frac", "rainfall_risk_score"]
    fmt = df[cols].copy()
    fmt["min_dist_mi"] = fmt["min_dist_mi"].round(0)
    fmt["decayed_wind_kt"] = fmt["decayed_wind_kt"].round(0)
    fmt["orographic_factor"] = fmt["orographic_factor"].round(2)
    fmt["upslope_w_ms"] = fmt["upslope_w_ms"].round(2)
    fmt["moisture_factor"] = fmt["moisture_factor"].round(2)
    fmt["watershed_proximity"] = fmt["watershed_proximity"].round(2)
    fmt["watershed_track_frac"] = fmt["watershed_track_frac"].round(2)
    fmt["rainfall_risk_score"] = fmt["rainfall_risk_score"].round(1)
    print(fmt.to_string(index=False))
    print()
    print("Cols: month=peak month, decayed_wind_kt=Kaplan-DeMaria w/ mountain enhancement,")
    print("upslope_w_ms = real V . grad(h) at SE escarpment (m/s, +=upslope rainfall),")
    print("moisture_factor = monthly PWAT climo + tropical surge,")
    print("watershed_track_frac = fraction of track in French Broad upstream of Asheville.")


def cmd_dem(args):
    from .dem import download_dem
    download_dem(Path(args.cache_dir) / "dem.npz")


def cmd_gauge(args):
    g = fetch_gauge()
    if g is None:
        print("USGS gauge data unavailable.")
        return
    print(f"--- USGS {g.site_id}  {g.site_name} ---")
    print(f"As of           : {g.timestamp}")
    if g.stage_ft is not None:
        bar_len = 40
        frac = min(1.0, g.stage_ft / FLOOD_STAGES_FT['major'])
        bar = "#" * int(bar_len * frac) + "-" * (bar_len - int(bar_len * frac))
        print(f"Stage           : {g.stage_ft:6.2f} ft  [{bar}] {g.flood_category}")
    else:
        print("Stage           : (no data)")
    if g.discharge_cfs is not None:
        print(f"Discharge       : {g.discharge_cfs:8.0f} cfs")
    print("Flood thresholds: " + ", ".join(
        f"{k}={v}ft" for k, v in FLOOD_STAGES_FT.items()))
    if g.stage_ft is not None:
        print(f"  -> {g.pct_to_minor:5.1f}% of minor-flood stage")
        print(f"  -> {g.pct_to_major:5.1f}% of major-flood stage")
    print()
    alerts = fetch_nws_alerts(ASHEVILLE_LAT, ASHEVILLE_LON)
    if not alerts:
        print("No active NWS alerts for Asheville point.")
        return
    if alerts:
        print(f"Active NWS alerts ({len(alerts)}):")
        for a in alerts:
            print(f"  [{a['severity']}] {a['event']}")
            if a['headline']:
                print(f"     {a['headline']}")
    else:
        print("No active NWS alerts for Asheville point.")


def cmd_dashboard(args):
    from .dashboard import run
    run(host=args.host, port=args.port, debug=args.debug)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hurricane-asheville",
                                description="Hurricane risk analysis for Asheville, NC.")
    p.add_argument("--cache-dir", default="data", help="Where to cache HURDAT2 data.")
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS_MI,
                   help=f"Radius (mi) around Asheville. Default {DEFAULT_RADIUS_MI:.0f}.")
    p.add_argument("--start-year", type=int, default=1950, help="Climatology start year.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("risk", help="Climatology + seasonal-adjusted risk for Asheville.")\
       .set_defaults(func=cmd_risk)

    h = sub.add_parser("history", help="List historical storms that affected Asheville.")
    h.add_argument("--top", type=int, default=0, help="Show only N closest passes.")
    h.set_defaults(func=cmd_history)

    sub.add_parser("active", help="Check NHC active Atlantic storms vs Asheville.")\
       .set_defaults(func=cmd_active)

    pl = sub.add_parser("plot", help="Plot historical tracks within radius.")
    pl.add_argument("--output", default="output/asheville_tracks.png")
    pl.set_defaults(func=cmd_plot)

    tr = sub.add_parser("terrain",
                        help="Rank storms by terrain-aware orographic rainfall risk.")
    tr.add_argument("--top", type=int, default=15)
    tr.add_argument("--no-dem", action="store_true",
                    help="Skip real-DEM upslope calculation (faster, no download).")
    tr.set_defaults(func=cmd_terrain)

    de = sub.add_parser("dem", help="Pre-download the regional elevation grid.")
    de.set_defaults(func=cmd_dem)

    ga = sub.add_parser("gauge",
                        help="Live USGS French Broad gauge + NWS alerts.")
    ga.set_defaults(func=cmd_gauge)

    db = sub.add_parser("dashboard",
                        help="Run the live web dashboard (Flask).")
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=5000)
    db.add_argument("--debug", action="store_true")
    db.set_defaults(func=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
