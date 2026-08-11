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


def cmd_ml_bootstrap(args):
    """One-shot historical backfill of the parquet history store."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .bootstrap import bootstrap_all
    summary = bootstrap_all(years=args.years)
    print("Bootstrap complete:")
    for k, v in summary.items():
        print(f"  {k:>22}: {v}")


def cmd_ml_history_info(args):
    """Print an audit summary of the parquet history store."""
    from .history import history_stats
    stats = history_stats()
    if stats["partitions"] == 0:
        print("No history yet. Run `ml-bootstrap` or wait for hourly Pages refreshes.")
        return
    print(f"Partitions   : {stats['partitions']}")
    print(f"Total rows   : {stats['rows']:,}")
    print(f"First ts     : {stats['first_ts']}")
    print(f"Last ts      : {stats['last_ts']}")
    print(f"Sources      : {', '.join(stats['sources'])}")
    print(f"Entity counts: {stats['entity_count']}")
    print(f"Metrics ({len(stats['metrics'])}):")
    for m in stats["metrics"]:
        print(f"  - {m}")


def cmd_ml_features(args):
    """Build a feature+target frame for a gauge and (optionally) save it."""
    from .features import build_training_frame
    from .history import load_history

    df = load_history()
    if df.empty:
        print("No history yet. Run `ml-bootstrap` first.")
        return
    horizons = tuple(int(h) for h in args.horizons.split(","))
    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                  if args.thresholds else None)
    frame = build_training_frame(
        df, args.target,
        horizons=horizons,
        precip_entity_id=args.precip_entity_id,
        thresholds=thresholds,
        dropna_features=args.dropna_features,
    )
    if frame.empty:
        print(f"No features built for {args.target} -- is there history for that site?")
        return
    feat_cols = [c for c in frame.columns if not c.startswith("y_")]
    y_cols = [c for c in frame.columns if c.startswith("y_")]
    print(f"Built features for {args.target}: rows={len(frame):,}")
    print(f"  feature cols: {len(feat_cols)}")
    print(f"  target cols : {y_cols}")
    print(f"  ts range    : {frame.index.min()}  ->  {frame.index.max()}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out)
        print(f"Wrote {out}  ({out.stat().st_size/1024:.1f} KB)")


def cmd_ml_train(args):
    """Train LightGBM model(s) for a gauge with walk-forward backtest."""
    from .features import build_training_frame
    from .history import load_history
    from .models import (default_model_path, train_with_backtest)

    df = load_history()
    if df.empty:
        print("No history yet. Run `ml-bootstrap` first.")
        return
    horizons = tuple(int(h) for h in args.horizons.split(","))
    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                  if args.thresholds else ())
    frame = build_training_frame(
        df, args.target, horizons=horizons,
        precip_entity_id=args.precip_entity_id,
        thresholds=thresholds,
    )
    if frame.empty:
        print(f"No training frame for {args.target}.")
        return
    print(f"Training frame: {len(frame):,} rows x {len(frame.columns)} cols")

    kinds: list[tuple[str, float | None]] = []
    if not args.classification_only:
        kinds.append(("regression", None))
    if thresholds:
        for thr in thresholds:
            kinds.append(("classification", thr))

    for h in horizons:
        for kind, thr in kinds:
            try:
                bundle = train_with_backtest(
                    frame, args.target, h,
                    kind=kind, threshold=thr, n_folds=args.folds,
                )
            except (ValueError, KeyError) as exc:
                print(f"  [{kind} h{h} thr={thr}] skipped: {exc}")
                continue
            key = ("overall_mae" if kind == "regression"
                   else "overall_auc")
            primary = bundle.metrics.get(key)
            label = (f"{kind}" if thr is None
                     else f"{kind} thr={thr}")
            baseline = bundle.metrics.get("baseline") or {}
            beats = bundle.metrics.get("beats_baseline")
            ref = baseline.get("mae" if kind == "regression" else "auc")

            verdict = ""
            if ref is not None and primary is not None:
                if beats:
                    verdict = f"  BEATS persistence ({ref:.3f})"
                else:
                    ratio = (primary / ref if kind == "regression"
                             else ref / primary)
                    verdict = (f"  LOSES to persistence ({ref:.3f}, "
                               f"{ratio:.1f}x worse)")

            out = default_model_path(
                args.target,
                f"{kind}" + (f"_thr{thr}" if thr is not None else ""),
                h)
            bundle.save(out)
            print(f"  [h={h}h {label}] {key}="
                  f"{primary:.4f}" if primary is not None else
                  f"  [h={h}h {label}] {key}=None")
            print(f"      {verdict.strip() or 'no baseline available'}  -> {out}")

    print("\nA model that loses to persistence should not be served. "
          "serving.py reads metrics.beats_baseline.")


def cmd_ml_predict(args):
    """Predict for the latest timestamp using a saved bundle."""
    import json as _json

    from .history import load_history
    from .models import ModelBundle

    bundle = ModelBundle.load(args.model)
    df = load_history()
    if df.empty:
        print("No history yet.")
        return
    upstream = (args.upstream.split(",") if args.upstream else None)
    out = bundle.__class__  # for static checkers
    from .models import predict_latest

    result = predict_latest(bundle, df,
                              precip_entity_id=args.precip_entity_id,
                              upstream_ids=upstream)
    print(_json.dumps(result, indent=2, default=str))


def cmd_ml_backtest(args):
    """Walk-forward backtest + write plots to site/ml/<target>/."""
    from .backtest import backtest_and_plot
    from .features import build_training_frame
    from .history import load_history

    df = load_history()
    if df.empty:
        print("No history yet. Run `ml-bootstrap` first.")
        return
    horizons = tuple(int(h) for h in args.horizons.split(","))
    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                   if args.thresholds else ())
    frame = build_training_frame(
        df, args.target, horizons=horizons,
        precip_entity_id=args.precip_entity_id,
        thresholds=thresholds,
    )
    if frame.empty:
        print(f"No training frame for {args.target}.")
        return
    results = backtest_and_plot(
        frame, args.target,
        horizons=horizons, thresholds=thresholds,
        out_dir=args.out, n_folds=args.folds,
    )
    for r in results:
        label = (f"{r.kind} thr={r.threshold}" if r.threshold is not None
                  else r.kind)
        primary = (r.metrics.get("mae") if r.kind == "regression"
                    else r.metrics.get("auc"))
        print(f"  [h={r.horizon_h}h {label}] n={r.n_rows}  primary={primary}")
    out = Path(args.out) / args.target
    print(f"Wrote plots and summary -> {out}")


def cmd_index_replay(args):
    """Replay the Flood Index over the historical record and validate it."""
    from .index_replay import (VALIDATION_PATH, build_validation,
                                event_summary, replay, write_validation)

    result = replay(start=args.start, end=args.end)
    if not result.days:
        print("No replay produced. Notes: " + "; ".join(result.notes))
        return

    print(f"Replayed {len(result.days):,} days "
          f"({result.first_date} -> {result.last_date})")
    print("Label distribution: " + ", ".join(
        f"{k}={v}" for k, v in sorted(result.label_counts.items(),
                                       key=lambda kv: -kv[1])))

    summary = build_validation(result, event_date=args.event_date,
                                event_name=args.event_name)
    ev = summary.get("event", {})
    base = summary.get("base_rate", {})
    print()
    print(f"{args.event_name} ({args.event_date}): "
          f"score {ev.get('score')} / 100 = {ev.get('label')}, "
          f"{ev.get('lead_days')} days of ALERT+ warning beforehand")
    print(f"Base rate: {base.get('alert_or_above_days')} of "
          f"{base.get('total_days'):,} days at ALERT or above "
          f"({base.get('pct')}%), {base.get('with_named_storm')} of those "
          f"with a named storm in range")

    print()
    print(f"{'date':12} {'score':>5} {'label':11} {'stage':>7} {'72h in':>7}  storm")
    top = sorted(result.days, key=lambda d: -d.score)[:args.top]
    for d in sorted(top, key=lambda d: d.date):
        stage = f"{d.stage_ft:.2f}" if d.stage_ft is not None else "-"
        precip = f"{d.precip_72h_in:.2f}" if d.precip_72h_in is not None else "-"
        storm = (f"{d.storm_name} {d.nearest_storm_mi:.0f}mi"
                 if d.storm_name and d.nearest_storm_mi is not None else "")
        print(f"{d.date:12} {d.score:>5} {d.label:11} {stage:>7} {precip:>7}  {storm}")

    print()
    print("Caveats:")
    for n in result.notes:
        print(f"  - {n}")

    if not args.no_write:
        path = write_validation(result, args.write or VALIDATION_PATH,
                                 event_date=args.event_date,
                                 event_name=args.event_name)
        print(f"\nwrote {path}")


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

    mb = sub.add_parser("ml-bootstrap",
                        help="Backfill the parquet history store from USGS + ERA5.")
    mb.add_argument("--years", type=int, default=5)
    mb.set_defaults(func=cmd_ml_bootstrap)

    mi = sub.add_parser("ml-history-info",
                        help="Print an audit summary of the parquet history store.")
    mi.set_defaults(func=cmd_ml_history_info)

    mf = sub.add_parser("ml-features",
                        help="Build a feature+target frame for a gauge.")
    mf.add_argument("--target", default="03451500",
                     help="USGS site id to predict (default: French Broad @ Asheville).")
    mf.add_argument("--horizons", default="6,24,72",
                     help="Comma-separated forecast horizons in hours.")
    mf.add_argument("--thresholds", default="",
                     help="Comma-separated stage thresholds (ft) for peak-above targets.")
    mf.add_argument("--precip-entity-id", default="asheville",
                     dest="precip_entity_id",
                     help="Entity id whose precip series to use (default: 'asheville').")
    mf.add_argument("--out", default="",
                     help="Optional parquet output path.")
    mf.add_argument("--dropna-features", action="store_true",
                     dest="dropna_features")
    mf.set_defaults(func=cmd_ml_features)

    mt = sub.add_parser("ml-train",
                        help="Train LightGBM models with walk-forward backtest.")
    mt.add_argument("--target", default="03451500")
    mt.add_argument("--horizons", default="6,24,72")
    mt.add_argument("--thresholds", default="",
                     help="Comma-separated stage thresholds for classification heads.")
    mt.add_argument("--precip-entity-id", default="asheville",
                     dest="precip_entity_id")
    mt.add_argument("--folds", type=int, default=5)
    mt.add_argument("--classification-only", action="store_true",
                     dest="classification_only")
    mt.set_defaults(func=cmd_ml_train)

    mp = sub.add_parser("ml-predict",
                        help="Predict for the latest snapshot using a saved bundle.")
    mp.add_argument("--model", required=True,
                     help="Path to a .joblib bundle (sidecar .json must exist).")
    mp.add_argument("--precip-entity-id", default="asheville",
                     dest="precip_entity_id")
    mp.add_argument("--upstream", default="",
                     help="Comma-separated upstream gauge ids (default: auto for primary).")
    mp.set_defaults(func=cmd_ml_predict)

    mbt = sub.add_parser("ml-backtest",
                         help="Replay walk-forward folds and write plots.")
    mbt.add_argument("--target", default="03451500")
    mbt.add_argument("--horizons", default="6,24,72")
    mbt.add_argument("--thresholds", default="")
    mbt.add_argument("--precip-entity-id", default="asheville",
                      dest="precip_entity_id")
    mbt.add_argument("--folds", type=int, default=5)
    mbt.add_argument("--out", default="site/ml",
                      help="Output directory (default: site/ml).")
    mbt.set_defaults(func=cmd_ml_backtest)

    ir = sub.add_parser(
        "index-replay",
        help="Replay the Flood Index across history and validate it on Helene.")
    ir.add_argument("--start", default=None)
    ir.add_argument("--end", default=None)
    ir.add_argument("--event-date", default="2024-09-27", dest="event_date")
    ir.add_argument("--event-name", default="Helene", dest="event_name")
    ir.add_argument("--top", type=int, default=15,
                     help="How many highest-scoring days to print.")
    ir.add_argument("--write", default=None,
                     help="Write the validation summary to this path "
                          "(default: data/index_validation.json).")
    ir.add_argument("--no-write", action="store_true", dest="no_write")
    ir.set_defaults(func=cmd_index_replay)
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
