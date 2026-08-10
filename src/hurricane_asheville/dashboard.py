"""Real-time dashboard for Asheville hurricane risk.

Run with:
    uv run hurricane-asheville dashboard
Then open http://127.0.0.1:5000

Markup lives in ``templates/``, styles and script in ``static/``, and the
arithmetic that used to run inside Jinja lives in :mod:`viewmodel` where it can
be tested. This module is only responsible for collecting upstream data,
assembling the view model, and serving it.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import asdict

from flask import Flask, jsonify, render_template
from markupsafe import Markup, escape

from . import stage_history, viewmodel
from .active import fetch_active_storms
from .buoys import fetch_all_buoys
from .config import ASHEVILLE_LAT, ASHEVILLE_LON, CSU_2026_FORECAST
from .forests import fetch_all_forests
from .gauge import (FLOOD_STAGES_FT, SITE_FRENCH_BROAD_ASHEVILLE,
                    fetch_all_gauges, fetch_nws_alerts)
from .index_score import compute_index
from .soil import fetch_soil_state
from .storm_track import attach_forecasts
from .tides import fetch_all_coastal
from .weather import fetch_current_weather

app = Flask(__name__)
_log = logging.getLogger(__name__)

_CACHE: dict = {"data": None, "ts": 0.0}
_TTL_SECONDS = 60.0


def _safe(label, fn, default, *args, **kwargs):
    """Run an upstream fetcher; swallow exceptions so one bad feed cannot
    break the whole dashboard render (important for the static GitHub Pages
    build, where any unhandled exception fails the deploy)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("dashboard fetch failed (%s): %s", label, exc)
        traceback.print_exc()
        return default


def _collect():
    now = time.time()
    if _CACHE["data"] is not None and now - _CACHE["ts"] < _TTL_SECONDS:
        return _CACHE["data"]

    gauges = _safe("gauges", fetch_all_gauges, [])
    primary = next(
        (g for g in gauges if g["site_id"] == SITE_FRENCH_BROAD_ASHEVILLE), None)
    storms = _safe("storms", fetch_active_storms, [])
    alerts = _safe("alerts", fetch_nws_alerts, [], ASHEVILLE_LAT, ASHEVILLE_LON)
    weather = _safe("weather", fetch_current_weather, {"error": "unavailable"},
                    ASHEVILLE_LAT, ASHEVILLE_LON)
    soil = _safe("soil", fetch_soil_state, {"error": "unavailable"},
                 ASHEVILLE_LAT, ASHEVILLE_LON)
    coastal = _safe("coastal", fetch_all_coastal, [])
    buoys = _safe("buoys", fetch_all_buoys, [])
    forests = _safe("forests", fetch_all_forests, [], storms)

    idx = compute_index(
        primary_gauge=primary,
        rate_ft_per_hr=(primary or {}).get("rate_ft_per_hr"),
        storms=[asdict(s) for s in storms],
        alerts=alerts,
        weather=weather,
        soil=soil,
    )

    storm_dicts = [asdict(s) for s in storms]
    # The current position alone is not what people look at during a storm.
    storm_dicts = _safe("storm_forecasts", attach_forecasts, storm_dicts,
                        storm_dicts)

    data = {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "as_of_epoch": int(now),
        "index": {
            "score": idx.score,
            "label": idx.label,
            "color": idx.color,
            "components": idx.components,
            "triggers": idx.triggers,
        },
        "gauges": gauges,
        "primary_site": SITE_FRENCH_BROAD_ASHEVILLE,
        "flood_stages": FLOOD_STAGES_FT,
        "storms": storm_dicts,
        "alerts": alerts,
        "weather": weather,
        "soil": soil,
        "buoys": buoys,
        "coastal": coastal,
        "forests": forests,
        "season": CSU_2026_FORECAST,
        "asheville": {"lat": ASHEVILLE_LAT, "lon": ASHEVILLE_LON},
    }

    # Optional ML forecasts -- silently no-op if no trained models are present.
    try:
        from .history import load_history
        from .serving import forecast_all, load_model_metrics
        hist = load_history()
        if not hist.empty:
            ml = forecast_all(hist)
            if ml:
                data["ml_forecasts"] = ml
                # Accuracy metadata comes from the sidecars written at training
                # time, which ship with the repo. The old code read a
                # `site/ml/summary.json` that only an explicit ml-backtest run
                # produces, so the deployed card never had any.
                data["ml_metrics"] = load_model_metrics(SITE_FRENCH_BROAD_ASHEVILLE)
    except Exception as exc:  # noqa: BLE001
        _log.warning("ml forecasts skipped: %s", exc)

    # Narrative TL;DR + glossary (always cheap, always populated).
    try:
        from .narrative import GLOSSARY, summarize
        data["narrative"] = summarize(data)
        data["glossary"] = GLOSSARY
    except Exception as exc:  # noqa: BLE001
        _log.warning("narrative skipped: %s", exc)

    _CACHE["data"] = data
    _CACHE["ts"] = now
    return data


# ---- view model -----------------------------------------------------------

# Only the fields the map script actually reads. The page used to inline the
# entire state via `tojson` -- ~260 KB, most of it gauge history the map never
# touches -- and then ship it again as state.json.
_MAP_GAUGE_FIELDS = ("site_id", "label", "lat", "lon", "stage_ft",
                     "pool_elevation_ft", "display_ft", "display_units",
                     "flood_category", "flood_class", "rate_ft_per_hr")
_MAP_STORM_FIELDS = ("id", "name", "classification", "intensity_kt", "lat",
                     "lon", "distance_mi", "movement", "forecast_track", "cone")
_MAP_BUOY_FIELDS = ("station_id", "label", "lat", "lon", "seas", "color",
                    "wave_ht_ft", "dominant_period_s", "wind_kt",
                    "wind_gust_kt", "pressure_mb", "water_temp_f")
_MAP_COASTAL_FIELDS = ("station_id", "label", "lat", "lon", "water_level_ft",
                       "wind_kt", "wind_gust_kt", "wind_dir_deg",
                       "air_pressure_mb")
_MAP_FOREST_FIELDS = ("short", "name", "region", "acres", "center_lat",
                      "center_lon", "notes", "fires_nearby")


def _pick(rows, fields):
    return [{k: r.get(k) for k in fields if k in r} for r in (rows or [])]


def map_state(data: dict, *, state_url: str = "state.json") -> dict:
    """The slim payload the client script needs, and nothing else."""
    return {
        "asheville": data.get("asheville"),
        "primary_site": data.get("primary_site"),
        "state_url": state_url,
        "gauges": _pick(data.get("gauges"), _MAP_GAUGE_FIELDS),
        "storms": _pick(data.get("storms"), _MAP_STORM_FIELDS),
        "buoys": _pick(data.get("buoys"), _MAP_BUOY_FIELDS),
        "coastal": _pick(data.get("coastal"), _MAP_COASTAL_FIELDS),
        "forests": _pick(data.get("forests"), _MAP_FOREST_FIELDS),
    }


def build_view_model(data: dict, *, state_url: str = "state.json") -> dict:
    """Everything the template needs, pre-computed."""
    gauges = data.get("gauges") or []
    primary = next((g for g in gauges
                    if g.get("site_id") == data.get("primary_site")), None)
    weather = data.get("weather") or {}

    history = chart = None
    try:
        history = stage_history.build(
            site_id=data.get("primary_site"),
            current_ft=(primary or {}).get("stage_ft"),
            thresholds=data.get("flood_stages"))
        chart = stage_history.chart_points(history) if history.points else None
        if not history.points:
            history = None
    except Exception as exc:  # noqa: BLE001
        _log.warning("stage history skipped: %s", exc)

    wet_bulb = weather.get("wet_bulb_f")
    return {
        "freshness": viewmodel.freshness(data),
        "dial": viewmodel.index_dial(data.get("index") or {}),
        "triggers": viewmodel.triggers(data.get("index") or {}),
        "networks": viewmodel.gauge_networks(gauges),
        "sparklines": {g["site_id"]: viewmodel.gauge_sparkline(g)
                       for g in gauges if g.get("site_id")},
        "primary_gauge": primary,
        "ml": viewmodel.ml_card(data),
        "qpf": viewmodel.qpf_chart(weather),
        "soil": viewmodel.soil_card(data.get("soil") or {}),
        "heat_chart": viewmodel.heat_chart(weather),
        "heat_scale": viewmodel.category_scale(
            weather.get("heat_index_f"), viewmodel.HEAT_INDEX_BANDS,
            lo=60, hi=130),
        "wet_bulb_scale": viewmodel.category_scale(
            wet_bulb, viewmodel.WET_BULB_BANDS, lo=55, hi=90),
        "wet_bulb_color": viewmodel.wet_bulb_color(wet_bulb),
        "history": history,
        "history_chart": chart,
        "map_state_json": Markup(json.dumps(map_state(data, state_url=state_url),
                                            default=str)),
    }


# ---- template helpers -----------------------------------------------------

_SEVERITY_MARKS = {
    "below-action": "○",   # hollow circle
    "action": "◔",         # quarter-filled
    "minor": "◑",          # half-filled
    "moderate": "◕",       # three-quarters
    "major": "●",          # filled
    "pool": "□",           # square: a different quantity entirely
}


@app.template_global()
def severity_mark(flood_class: str) -> str:
    """A shape for each severity so colour is never the only cue."""
    return _SEVERITY_MARKS.get(flood_class, "—")


@app.template_global()
def jargon(term: str, label: str | None = None) -> Markup:
    """A glossary term that works on touch and keyboard, not just hover.

    ``title=`` tooltips are invisible on a phone, which is exactly where the
    "how to read this" affordance is needed.
    """
    from .narrative import GLOSSARY
    definition = GLOSSARY.get(term)
    text = escape(label or term)
    if not definition:
        return Markup(str(text))
    slug = escape(term.replace(" ", "-").lower())
    return Markup(
        f'<button type="button" class="jargon" data-term="{slug}" '
        f'aria-expanded="false" aria-controls="note-{slug}" '
        f'title="{escape(definition)}">{text}</button>'
        f'<span class="jargon-note" id="note-{slug}">{escape(definition)}</span>'
    )


# ---- routes ---------------------------------------------------------------

@app.route("/")
def index():
    from .narrative import GLOSSARY, summarize
    data = _collect()
    data.setdefault("glossary", GLOSSARY)
    data.setdefault("narrative", summarize(data))
    # The Pages snapshot sits next to state.json; the live app serves /api/state.
    state_url = "state.json" if app.config.get("STATIC_BUILD") else "/api/state"
    vm = build_view_model(data, state_url=state_url)
    return render_template("dashboard.html", data=data, vm=vm)


@app.route("/api/state")
def api_state():
    return jsonify(_collect())


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    print(f"Asheville hurricane dashboard -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
