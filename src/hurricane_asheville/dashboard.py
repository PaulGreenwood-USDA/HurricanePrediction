"""Real-time dashboard for Asheville hurricane risk.

Run with:
    uv run hurricane-asheville dashboard
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import logging
import time
import traceback
import json
from dataclasses import asdict

from flask import Flask, jsonify, render_template_string

from .active import fetch_active_storms
from .buoys import fetch_all_buoys
from .config import ASHEVILLE_LAT, ASHEVILLE_LON, CSU_2026_FORECAST
from .forests import fetch_all_forests
from .gauge import (FLOOD_STAGES_FT, SITE_FRENCH_BROAD_ASHEVILLE,
                    fetch_all_gauges, fetch_nws_alerts)
from .index_score import compute_index
from .soil import fetch_soil_state
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
        "storms": [asdict(s) for s in storms],
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
        from .serving import forecast_all
        from .history import load_history
        hist = load_history()
        if not hist.empty:
            ml = forecast_all(hist)
            if ml:
                data["ml_forecasts"] = ml
    except Exception as exc:  # noqa: BLE001
        _log.warning("ml forecasts skipped: %s", exc)
    # Optional ML backtest summaries (one per target) for the UI links.
    try:
        from pathlib import Path as _Path
        bt_root = _Path("site/ml")
        if bt_root.exists():
            bt: dict = {}
            for summary in bt_root.glob("*/summary.json"):
                try:
                    bt[summary.parent.name] = json.loads(summary.read_text())
                except Exception:
                    continue
            if bt:
                data["ml_backtest"] = bt
    except Exception as exc:  # noqa: BLE001
        _log.warning("ml backtest summaries skipped: %s", exc)

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


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Asheville Hurricane Risk - Live</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
<style>
  :root {
    --bg: #0f1115; --panel: #1a1d24; --panel2: #232732;
    --text: #e6e6e6; --dim: #9aa0aa; --accent: #4fc3f7;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background: var(--bg); color: var(--text); }
  header { padding: 1rem 1.5rem; background: var(--panel);
           display: flex; align-items: center; justify-content: space-between;
           border-bottom: 1px solid #000; gap: 1rem; flex-wrap: wrap; }
  header h1 { margin:0; font-size: 1.25rem; font-weight: 600; }
  .updated { font-size:.75rem; color: var(--dim); }
  .pulse { display:inline-block; width:8px; height:8px; border-radius:50%;
           background:#4caf50; margin-right:.4rem; vertical-align:middle;
           animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }

  .grid { display: grid; gap: 1rem; padding: 1rem; grid-template-columns: repeat(12, 1fr); }
  .card { background: var(--panel); border-radius: 10px; padding: 1rem 1.2rem;
          box-shadow: 0 2px 6px rgba(0,0,0,.4); grid-column: span 12; }
  @media (min-width: 900px) {
    .span6 { grid-column: span 6; }
    .span4 { grid-column: span 4; }
    .span8 { grid-column: span 8; }
  }
  .card h2 { margin: 0 0 .6rem 0; font-size: .9rem; color: var(--accent);
             text-transform: uppercase; letter-spacing: .08em; }
  .big { font-size: 2.2rem; font-weight: 700; line-height:1; }
  .dim { color: var(--dim); font-size: .85rem; }
  .row { display:flex; justify-content: space-between; padding: .25rem 0;
         border-bottom: 1px solid #2a2e36; }
  .row:last-child { border-bottom: 0; }
  .alert { background: #b71c1c; padding:.6rem; border-radius: 6px; margin:.4rem 0; }
  table { width: 100%; border-collapse: collapse; font-size:.88rem; }
  th, td { padding: .35rem .4rem; text-align:left; border-bottom: 1px solid #2a2e36; }
  th { color: var(--dim); font-weight: 500; }

  .dial-wrap { display:flex; align-items:center; gap: 1.2rem; flex-wrap: wrap; }
  .dial { width: 130px; height: 130px; flex: 0 0 130px; }
  .dial-label { letter-spacing:.15em; font-weight:700; padding:.2rem .7rem;
                border-radius:999px; color:#fff; display:inline-block; }
  .breakdown { flex: 1; min-width: 200px; }
  .seg { display:flex; height:14px; border-radius: 4px; overflow:hidden;
         background: var(--panel2); margin-top:.4rem; }
  .seg span { height:100%; }

  .lights { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: .5rem; }
  .light { display:flex; align-items:center; padding:.5rem .7rem; border-radius:6px;
           background: var(--panel2); font-size:.85rem; }
  .light .led { width:12px; height:12px; border-radius:50%; margin-right:.6rem;
                background:#444; flex:0 0 12px; }
  .light.on .led { background: #c62828; box-shadow: 0 0 8px #c62828; }

  .spark { height: 28px; width: 100%; }
  .gauge-row { display:grid; grid-template-columns: 1.6fr auto 1.4fr auto;
               gap:.6rem; align-items:center; padding:.5rem 0;
               border-bottom: 1px solid #2a2e36; }
  .gauge-row:last-child { border-bottom: 0; }
  .gauge-name b { display:block; font-size:.95rem; }
  .stage-pill { padding:.15rem .5rem; border-radius:4px; font-size:.78rem;
                background: var(--panel2); white-space: nowrap; }
  .stage-pill.action { background:#9e9d24; color:#000; }
  .stage-pill.minor  { background:#ef6c00; }
  .stage-pill.moderate { background:#c62828; }
  .stage-pill.major  { background:#6a1b9a; }

  /* gauge network filter tabs */
  .net-tabs { display:flex; flex-wrap:wrap; gap:.35rem;
              margin:.3rem 0 .7rem 0; }
  .net-tab { padding:.3rem .7rem; border-radius:999px; font-size:.78rem;
             background:#11141a; color: var(--dim); cursor: pointer;
             border:1px solid #2a2e36; user-select:none;
             transition: all .15s ease; }
  .net-tab:hover { color: var(--fg); border-color: var(--accent); }
  .net-tab.active { background: var(--accent); color:#000;
                    border-color: var(--accent); font-weight:600; }
  .net-tab .count { opacity:.7; margin-left:.3rem; font-size:.7rem; }
  .gauge-row.hidden { display:none; }
  .rate-up { color: #ff8a65; }
  .rate-down { color: #81c784; }

  #map { height: 380px; border-radius: 8px; }
  footer { text-align:center; padding: 1rem; color: var(--dim); font-size:.8rem; }

  .forest-grid { display:grid; gap:.7rem;
                 grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
  .forest { background: var(--panel2); border-radius:8px; padding:.7rem .9rem;
            border-left: 4px solid var(--accent); }
  .forest.region-mountain { border-left-color:#4caf50; }
  .forest.region-piedmont { border-left-color:#fdd835; }
  .forest.region-coastal  { border-left-color:#ef5350; }
  .forest h3 { margin:0 0 .2rem 0; font-size:.95rem; }
  .forest .meta { font-size:.75rem; color: var(--dim); margin-bottom:.4rem; }
  .forest .stats { display:grid; grid-template-columns: 1fr 1fr; gap:.2rem .6rem;
                   font-size:.82rem; }
  .forest .stats span:nth-child(odd) { color: var(--dim); }
  .forest .alert-mini { background:#b71c1c; padding:.2rem .4rem; border-radius:4px;
                        font-size:.72rem; margin-top:.3rem; display:inline-block; }
  .forest .storm-near { background:#c62828; padding:.2rem .4rem; border-radius:4px;
                        font-size:.72rem; margin-top:.3rem; display:inline-block; }
  .forest-landslide { margin-top:.5rem; display:flex; flex-wrap:wrap;
                      align-items:center; gap:.4rem; }
  .ll-pill { padding:.2rem .5rem; border-radius:999px; font-size:.72rem;
             font-weight:700; color:#fff; letter-spacing:.05em;
             text-transform:uppercase; }
  .ll-inv  { font-size:.72rem; color: var(--dim); }
  .forest-gauges { margin-top:.45rem; padding-top:.4rem;
                   border-top: 1px dashed #2a2e36; }
  .ll-section-title { font-size:.7rem; color: var(--dim); letter-spacing:.08em;
                      text-transform:uppercase; margin-bottom:.2rem; }
  .forest-gauge { display:grid;
                  grid-template-columns: 1fr auto auto;
                  gap:.4rem; align-items:center;
                  font-size:.76rem; padding:.15rem 0;
                  border-bottom: 1px solid #2a2e36; }
  .forest-gauge:last-child { border-bottom: 0; }
  .fg-name { color: var(--dim); }
  .fg-stage { color: var(--text); font-variant-numeric: tabular-nums; }

  .districts { margin-top:.6rem; padding-top:.5rem;
               border-top: 1px dashed #2a2e36; display:grid; gap:.4rem;
               grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
  .district { background:#11141a; border-radius:6px; padding:.45rem .6rem;
              border-left: 3px solid #4fc3f7; }
  .district h4 { margin:0 0 .15rem 0; font-size:.82rem; }
  .district .d-meta { font-size:.68rem; color: var(--dim);
                      margin-bottom:.35rem; }
  .district .d-stats { display:grid; grid-template-columns: auto 1fr;
                       gap:.1rem .5rem; font-size:.74rem; }
  .district .d-stats span:nth-child(odd) { color: var(--dim); }
  .district .d-pills { display:flex; flex-wrap:wrap; gap:.25rem;
                       margin-top:.35rem; }
  .d-pill { padding:.1rem .4rem; border-radius:999px; font-size:.66rem;
            font-weight:700; color:#fff; letter-spacing:.04em;
            text-transform:uppercase; }
  .district .d-notes { font-size:.66rem; color: var(--dim); margin-top:.3rem;
                       font-style: italic; }

  /* --- Phase 6: TL;DR hero + jargon tooltips + compact triggers --- */
  .hero { grid-column: span 12; padding: 1.4rem 1.6rem;
          border-left: 8px solid var(--hero-color, #2e7d32); }
  .hero .level { display:inline-block; padding:.25rem .8rem;
                  border-radius:999px; font-size:.7rem; font-weight:800;
                  letter-spacing:.18em; text-transform:uppercase;
                  color:#fff; }
  .hero h2.headline { margin:.6rem 0 .2rem 0; font-size:1.45rem;
                       font-weight:700; color:#fff; text-transform:none;
                       letter-spacing:0; }
  .hero .sub { color: var(--dim); font-size:.92rem; margin-bottom:.6rem; }
  .hero .rec { background:#11141a; border-radius:6px; padding:.5rem .8rem;
                font-size:.9rem; line-height:1.4; }
  .hero .facts { margin-top:.7rem; display:grid; gap:.3rem;
                  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
  .hero .fact { background:#11141a; border-radius:6px;
                 padding:.4rem .6rem; font-size:.82rem; }
  .hero .fact b { color: var(--accent); }

  /* tooltip on jargon: dotted underline + native title= */
  .jargon { border-bottom: 1px dotted var(--dim); cursor: help; }

  .ml-subtitle {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    margin-left: .45rem;
    font-size: .72rem;
    font-weight: 500;
    color: var(--dim);
    letter-spacing: 0;
    text-transform: none;
    vertical-align: middle;
  }
  .ml-subtitle .dot { opacity: .65; }
  .ml-subtitle .site-chip {
    display: inline-block;
    padding: .08rem .38rem;
    border-radius: 999px;
    border: 1px solid #2a2e36;
    background: #11141a;
    color: var(--text);
    font-weight: 600;
  }

  /* ── ML Forecast card ──────────────────────────────────────────── */
  .ml-grid {
    display: grid;
    gap: .75rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin-top: .75rem;
  }

  .ml-horizon {
    background: var(--panel2);
    border-radius: 8px;
    padding: .75rem .9rem;
    border-top: 3px solid var(--accent);
  }

  .h-label {
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: .3rem;
  }

  .h-stage {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
  }

  .h-stage small {
    font-size: .85rem;
    font-weight: 400;
    color: var(--dim);
    margin-left: .15rem;
  }

  .h-delta {
    font-size: .78rem;
    margin-top: .3rem;
    font-variant-numeric: tabular-nums;
  }

  .h-delta.up   { color: #ff8a65; }
  .h-delta.down { color: #81c784; }
  .h-delta.flat { color: var(--dim); }

  /* flood-stage progress bar */
  .ml-bar {
    position: relative;
    height: 10px;
    border-radius: 4px;
    overflow: hidden;
    background: #11141a;
    margin-top: .6rem;
  }

  .ml-bar .band {
    position: absolute;
    top: 0; bottom: 0;
    opacity: .55;
  }

  .ml-bar .band.action   { background: #9e9d24; }
  .ml-bar .band.minor    { background: #ef6c00; }
  .ml-bar .band.moderate { background: #c62828; }
  .ml-bar .band.major    { background: #6a1b9a; }

  .ml-bar .now,
  .ml-bar .pred {
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    transform: translateX(-1px);
  }

  .ml-bar .now  { background: rgba(255,255,255,.45); }
  .ml-bar .pred { background: #fff; box-shadow: 0 0 4px #fff; }
  .ml-bar.high .pred { background: #ef5350; box-shadow: 0 0 4px #ef5350; }

  .ml-bar-legend {
    display: flex;
    justify-content: space-between;
    font-size: .6rem;
    color: var(--dim);
    margin-top: .25rem;
    font-variant-numeric: tabular-nums;
  }

  /* probability section */
  .ml-probs {
    margin-top: 1rem;
    padding-top: .75rem;
    border-top: 1px solid #2a2e36;
    display: grid;
    gap: .5rem;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  }

  .ml-prob {
    background: var(--panel2);
    border-radius: 6px;
    padding: .55rem .7rem;
    font-size: .8rem;
  }

  .pbar {
    height: 8px;
    background: #11141a;
    border-radius: 4px;
    overflow: hidden;
    margin: .35rem 0 .25rem;
  }

  .pbar .fill {
    height: 100%;
    border-radius: 4px;
    min-width: 3px;
  }

  .pct {
    font-size: 1.1rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1;
  }

  .pct.high { color: #ef5350; }
  .pct.med  { color: #fb8c00; }
  .pct.low  { color: #7cb342; }

  /* backtest plot links */
  .ml-meta {
    margin-top: .75rem;
    padding-top: .6rem;
    border-top: 1px solid #2a2e36;
    font-size: .78rem;
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    align-items: center;
  }

  .ml-plot-link {
    display: inline-block;
    padding: .2rem .55rem;
    border-radius: 4px;
    background: #11141a;
    border: 1px solid #2a2e36;
    color: var(--accent);
    font-size: .73rem;
    text-decoration: none;
  }

  .ml-plot-link:hover {
    border-color: var(--accent);
    background: #1a1d24;
  }

  /* ── Heat stress & wet-bulb card ─────────────────────────────── */
  .heat-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1rem;
    margin-top: .6rem;
    align-items: start;
  }
  .heat-index-block {
    text-align: center;
    background: var(--panel2);
    border-radius: 8px;
    padding: .9rem 1.1rem;
    min-width: 120px;
  }
  .heat-index-num {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
  }
  .heat-cat-badge {
    display: inline-block;
    margin-top: .4rem;
    padding: .2rem .65rem;
    border-radius: 999px;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #fff;
  }
  .heat-metrics { display: grid; gap: .25rem; }
  .heat-row {
    display: flex;
    justify-content: space-between;
    padding: .3rem .55rem;
    background: var(--panel2);
    border-radius: 6px;
    font-size: .88rem;
  }
  .heat-row b { font-variant-numeric: tabular-nums; }
  .heat-sparkline {
    margin-top: .8rem;
    padding-top: .65rem;
    border-top: 1px solid #2a2e36;
  }
  .heat-spark-title {
    font-size: .65rem;
    color: var(--dim);
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .3rem;
  }
  .heat-spark { height: 52px; width: 100%; display: block; }
  .heat-spark-legend {
    display: flex;
    gap: 1rem;
    font-size: .68rem;
    color: var(--dim);
    margin-top: .25rem;
  }
  .heat-spark-legend .leg-actual::before { content: '\2014\00a0'; color: #4fc3f7; }
  .heat-spark-legend .leg-app::before    { content: '\2013\00a0'; color: #ff8a65; }
  .heat-nws-cats {
    margin-top: .8rem;
    padding-top: .65rem;
    border-top: 1px solid #2a2e36;
    display: flex;
    flex-wrap: wrap;
    gap: .45rem .9rem;
    font-size: .72rem;
  }
  .heat-nws-cat { display: flex; align-items: center; gap: .35rem; }
  .heat-nws-swatch { width: 10px; height: 10px; border-radius: 2px; flex: 0 0 10px; }

  /* compact triggers row: fired ones bright, off ones muted */
  .trig-strip { display:flex; flex-wrap:wrap; gap:.35rem; }
  .trig { padding:.25rem .55rem; border-radius:999px; font-size:.74rem;
          background:#11141a; color: var(--dim);
          border:1px solid #2a2e36; }
  .trig.on { background:#7f1d1d; color:#fff; border-color:#7f1d1d;
             font-weight:600; }

  /* legend / 'how to read' card */
  .legend dt { font-weight:600; color: var(--accent);
                margin-top:.5rem; font-size:.85rem; }
  .legend dd { margin:.1rem 0 .4rem 0; color: var(--dim);
                font-size:.82rem; line-height:1.4; }
</style>
</head>
<body>
<header>
  <h1>&#127786; Asheville Hurricane Risk - Live</h1>
  <span class="updated"><span class="pulse"></span>
    Updated <span id="age">just now</span> &middot;
    <span id="ts">{{ data.as_of }}</span></span>
</header>

<div class="grid">

  {% set n = data.get('narrative') %}
  {% if n %}
  <div class="card hero" style="--hero-color: {{ n.color }};">
    <span class="level" style="background: {{ n.color }};">{{ n.level }}</span>
    <h2 class="headline">{{ n.headline }}</h2>
    <div class="sub">{{ n.subheadline }}</div>
    <div class="rec">{{ n.recommendation }}</div>
    {% if n.key_facts %}
    <div class="facts">
      {% for f in n.key_facts %}<div class="fact">{{ f }}</div>{% endfor %}
    </div>
    {% endif %}
  </div>
  {% endif %}

  <div class="card span6">
    <h2><span class="jargon" title="{{ data.glossary['Flood Index'] }}">Asheville Flood Index</span></h2>
    <div class="dial-wrap">
      <svg class="dial" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="52" fill="none" stroke="#2a2e36" stroke-width="12"/>
        <circle cx="60" cy="60" r="52" fill="none"
                stroke="{{ data.index.color }}" stroke-width="12"
                stroke-linecap="round"
                stroke-dasharray="{{ (data.index.score / 100 * 326)|round(1) }} 326"
                transform="rotate(-90 60 60)"/>
        <text x="60" y="62" text-anchor="middle"
              fill="#e6e6e6" font-size="28" font-weight="800">{{ data.index.score }}</text>
        <text x="60" y="80" text-anchor="middle" fill="#9aa0aa" font-size="9"
              letter-spacing="2">/ 100</text>
      </svg>
      <div class="breakdown">
        <span class="dial-label" style="background: {{ data.index.color }}">{{ data.index.label }}</span>
        <div class="seg">
          {% set cmap = {'stage':'#2196f3','qpf':'#4fc3f7','storm':'#ef5350','rise':'#ff8a65','alert':'#ba68c8','soil':'#8d6e63'} %}
          {% for k,v in data.index.components.items() %}
            <span style="width: {{ v }}%; background: {{ cmap[k] }}" title="{{ k }} {{ v }}"></span>
          {% endfor %}
        </div>
        <div class="dim" style="margin-top:.5rem; font-size:.78rem;">
          stage {{ data.index.components.stage }} &middot;
          qpf {{ data.index.components.qpf }} &middot;
          storm {{ data.index.components.storm }} &middot;
          rise {{ data.index.components.rise }} &middot;
          alert {{ data.index.components.alert }} &middot;
          soil {{ data.index.components.soil }}
        </div>
      </div>
    </div>
  </div>

  <div class="card span6">
    <h2>Triggers</h2>
    <div class="dim" style="margin-bottom:.4rem; font-size:.78rem;">
      Discrete risk conditions. Bright pills are currently firing.
    </div>
    {% set names = [
        ('stage_above_action','River \u2265 action stage'),
        ('stage_above_minor','River \u2265 minor flood'),
        ('qpf_over_1in','72h QPF \u2265 1 in'),
        ('qpf_over_3in','72h QPF \u2265 3 in'),
        ('storm_within_1000mi','TC within 1000 mi'),
        ('storm_within_500mi','TC within 500 mi'),
        ('river_rising_fast','River rising > 0.3 ft/hr'),
        ('nws_flood_or_tropical','NWS flood/tropical alert'),
        ('soil_saturated','Soil saturated'),
        ('wet_week','Past 7d precip \u2265 3 in'),
      ] %}
    {# sort: fired first, then dormant #}
    {% set fired = [] %}
    {% set dormant = [] %}
    {% for k,label in names %}
      {% if data.index.triggers[k] %}{% set _ = fired.append((k,label)) %}
      {% else %}{% set _ = dormant.append((k,label)) %}{% endif %}
    {% endfor %}
    <div class="trig-strip">
      {% for k,label in fired %}<span class="trig on">{{ label }}</span>{% endfor %}
      {% for k,label in dormant %}<span class="trig">{{ label }}</span>{% endfor %}
    </div>
  </div>

  <div class="card span8">
    <h2>Gauge networks</h2>
    {% set NETWORKS = [
        ('french_broad', 'French Broad', ['primary','upstream','headwaters','tributary']),
        ('regional',     'WNC regional', ['regional']),
        ('statewide',    'Statewide',    ['statewide']),
        ('reservoirs',   'Reservoirs',   ['reservoir']),
      ] %}
    {% set role_to_net = {} %}
    {% for key, label, roles in NETWORKS %}
      {% for r in roles %}{% set _ = role_to_net.update({r: key}) %}{% endfor %}
    {% endfor %}
    {% set counts = {'all': data.gauges|length} %}
    {% for key, label, roles in NETWORKS %}
      {% set _ = counts.update({key: data.gauges | selectattr('role','in',roles) | list | length}) %}
    {% endfor %}
    <div class="net-tabs" id="net-tabs">
      <span class="net-tab active" data-net="all">All<span class="count">{{ counts['all'] }}</span></span>
      {% for key, label, roles in NETWORKS %}
        {% if counts[key] > 0 %}
        <span class="net-tab" data-net="{{ key }}">{{ label }}<span class="count">{{ counts[key] }}</span></span>
        {% endif %}
      {% endfor %}
    </div>
    {% for g in data.gauges %}
      <div class="gauge-row" data-net="{{ role_to_net.get(g.role, 'other') }}">
        <div class="gauge-name">
          <b>{{ g.label }}</b>
          <span class="dim">{{ g.role }} &middot; USGS {{ g.site_id }}</span>
        </div>
        <div style="text-align:right;">
          <div><b>{{ "%.2f"|format(g.stage_ft) if g.stage_ft is not none else "?" }}</b> ft</div>
          {% if g.rate_ft_per_hr is not none %}
            {% if g.rate_ft_per_hr > 0.05 %}
              <span class="dim rate-up">&uarr; {{ "%.2f"|format(g.rate_ft_per_hr) }} ft/hr</span>
            {% elif g.rate_ft_per_hr < -0.05 %}
              <span class="dim rate-down">&darr; {{ "%.2f"|format(-g.rate_ft_per_hr) }} ft/hr</span>
            {% else %}
              <span class="dim">flat</span>
            {% endif %}
          {% endif %}
        </div>
        <svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none">
          {% set hist = g.history %}
          {% if hist and hist|length > 1 %}
            {% set vals = hist | map(attribute='ft') | list %}
            {% set vmin = vals | min %}
            {% set vmax = vals | max %}
            {% set vrng = (vmax - vmin) if (vmax - vmin) > 0.01 else 0.01 %}
            {% set n = vals | length %}
            {% set pts = [] %}
            {% for v in vals %}
              {% set _ = pts.append("%.1f,%.1f"|format((loop.index0 / (n - 1)) * 100, 28 - ((v - vmin) / vrng) * 26)) %}
            {% endfor %}
            <polyline points="{{ pts|join(' ') }}" fill="none"
                      stroke="{{ '#ef5350' if (g.rate_ft_per_hr or 0) > 0.05 else '#4fc3f7' }}"
                      stroke-width="1.5"/>
          {% endif %}
        </svg>
        <span class="stage-pill {{ g.flood_category|lower|replace(' flood','')|replace(' stage','') }}">
          {{ g.flood_category }}
        </span>
      </div>
    {% endfor %}

    {% set p = data.gauges | selectattr('site_id','equalto',data.primary_site) | first %}
    {% if p and (p.eta_minor_hr is not none or p.eta_moderate_hr is not none or p.eta_major_hr is not none) %}
      <div class="dim" style="margin-top:.6rem;">
        <b>Linear ETA at current rate</b>:
        {% if p.eta_minor_hr is not none %} minor {{ "%.1f"|format(p.eta_minor_hr) }}h{% endif %}
        {% if p.eta_moderate_hr is not none %} &middot; moderate {{ "%.1f"|format(p.eta_moderate_hr) }}h{% endif %}
        {% if p.eta_major_hr is not none %} &middot; major {{ "%.1f"|format(p.eta_major_hr) }}h{% endif %}
      </div>
    {% endif %}
    {% if p and p.nwps_forecast and p.nwps_forecast.peak_ft is not none %}
      <div class="dim" style="margin-top:.3rem;">
        <b>NWS NWPS forecast</b>:
        peak {{ "%.2f"|format(p.nwps_forecast.peak_ft) }} ft
        {% if p.nwps_forecast.peak_t %} at {{ p.nwps_forecast.peak_t }}{% endif %}
        ({{ p.nwps_forecast.points|length }} pts, issued {{ p.nwps_forecast.issued or 'n/a' }})
      </div>
    {% endif %}
  </div>

  {% set ml = data.get('ml_forecasts', {}).get(data.primary_site) %}
  {% set bt = data.get('ml_backtest', {}).get(data.primary_site) %}
  {% if ml and (ml.regression or ml.classification) %}
  {% set p = data.gauges | selectattr('site_id','equalto',data.primary_site) | first %}
  {% set fs = data.flood_stages %}
  {# stage axis goes 0 -> 1.25 * major so the major band is visible #}
  {% set axis_max = (fs.major * 1.25) if fs and fs.major else 20 %}
  {# Reduce regression list to a dict by horizon for easy join with classes. #}
  {% set reg_by_h = {} %}
  {% for r in ml.regression %}{% set _ = reg_by_h.update({r.horizon_h: r.predicted_stage_ft}) %}{% endfor %}
  {% set horizons = ml.regression | map(attribute='horizon_h') | list %}
  {# backtest metrics keyed by (kind, horizon, threshold) #}
  {% set mae_by_h = {} %}
  {% set auc_by_ht = {} %}
  {% if bt and bt.results %}
    {% for r in bt.results %}
      {% if r.kind == 'regression' and r.metrics.mae %}
        {% set _ = mae_by_h.update({r.horizon_h: r.metrics.mae}) %}
      {% elif r.kind == 'classification' and r.metrics.auc %}
        {% set _ = auc_by_ht.update({(r.horizon_h, r.threshold): r.metrics.auc}) %}
      {% endif %}
    {% endfor %}
  {% endif %}
  <div class="card span12">
    <h2>
      <span class="jargon" title="{{ data.glossary['ML forecast'] }}">ML stage forecast</span>
      <span class="ml-subtitle">
        <span>LightGBM model</span>
        <span class="dot" aria-hidden="true">&middot;</span>
        <span>trained on USGS gauge history</span>
        <span class="dot" aria-hidden="true">&middot;</span>
        <span class="site-chip">site {{ data.primary_site }}</span>
      </span>
    </h2>

    <div class="ml-grid">
      {% for h in horizons %}
        {% set pred = reg_by_h[h] %}
        {% set cur = (p.stage_ft if p else None) %}
        {% set delta = (pred - cur) if (cur is not none) else None %}
        {% set cls_h = ml.classification | selectattr('horizon_h','equalto',h) | list %}
        {% set bar_class = 'high' if (pred and fs and pred >= fs.minor) else '' %}
        <div class="ml-horizon">
          <div class="h-label">+{{ h }} hours</div>
          <div class="h-stage">{{ "%.2f"|format(pred) }}<small> ft</small></div>
          {% if delta is not none %}
            {% set cls = 'up' if delta > 0.05 else ('down' if delta < -0.05 else 'flat') %}
            {% set arrow = '↑' if delta > 0.05 else ('↓' if delta < -0.05 else '→') %}
            <div class="h-delta {{ cls }}">
              {{ arrow }} {{ "%+.2f"|format(delta) }} ft vs now ({{ "%.2f"|format(cur) }} ft)
            </div>
          {% endif %}
          {% if fs %}
          {% set act_pct  = (fs.action   / axis_max) * 100 %}
          {% set min_pct  = (fs.minor    / axis_max) * 100 %}
          {% set mod_pct  = (fs.moderate / axis_max) * 100 %}
          {% set maj_pct  = (fs.major    / axis_max) * 100 %}
          {% set now_pct  = ((cur or 0) / axis_max) * 100 %}
          {% set pred_pct = (pred / axis_max) * 100 %}
          <div class="ml-bar {{ bar_class }}">
            <div class="band action"
                 style="left:{{ act_pct }}%; width:{{ min_pct - act_pct }}%;"></div>
            <div class="band minor"
                 style="left:{{ min_pct }}%; width:{{ mod_pct - min_pct }}%;"></div>
            <div class="band moderate"
                 style="left:{{ mod_pct }}%; width:{{ maj_pct - mod_pct }}%;"></div>
            <div class="band major"
                 style="left:{{ maj_pct }}%; right:0;"></div>
            <div class="now"  style="left:{{ now_pct }}%;"
                 title="Current {{ '%.2f'|format(cur or 0) }} ft"></div>
            <div class="pred" style="left:{{ pred_pct }}%;"
                 title="Predicted {{ '%.2f'|format(pred) }} ft"></div>
          </div>
          <div class="ml-bar-legend">
            <span>0</span>
            <span title="action stage">A {{ fs.action }}</span>
            <span title="minor flood">Mi {{ fs.minor }}</span>
            <span title="moderate flood">Mo {{ fs.moderate }}</span>
            <span title="major flood">Mj {{ fs.major }}</span>
          </div>
          {% endif %}
          {% if mae_by_h.get(h) %}
          <div class="h-delta flat" style="margin-top:.4rem; font-size:.68rem;">
            backtest MAE ±{{ "%.2f"|format(mae_by_h[h]) }} ft
          </div>
          {% endif %}
        </div>
      {% endfor %}
    </div>

    {% if ml.classification %}
    <div class="ml-probs">
      <div class="dim" style="font-size:.72rem; margin-top:.5rem;">
        Probability of exceeding flood thresholds
      </div>
      {% for c in ml.classification | sort(attribute='horizon_h') %}
        {% set pct = (c.probability * 100) | round(0, 'floor') | int %}
        {% set sev = 'high' if pct >= 50 else ('med' if pct >= 25 else 'low') %}
        {% set color = '#ef5350' if pct >= 50 else ('#fb8c00' if pct >= 25 else '#7cb342') %}
        {% set auc = auc_by_ht.get((c.horizon_h, c.threshold)) %}
        <div class="ml-prob">
          <div>&gt; {{ c.threshold }} ft within {{ c.horizon_h }}h</div>
          <div class="pbar">
            <div class="fill" style="width:{{ pct }}%; background:{{ color }};"></div>
          </div>
          <div class="pct {{ sev }}">
            {{ pct }}%
            {% if auc %}<span class="dim" style="font-weight:400; font-size:.65rem;"><br>AUC {{ "%.2f"|format(auc) }}</span>{% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
    {% endif %}

    {% if bt and bt.results %}
    <div class="ml-meta">
      <span><b>Backtest plots:</b></span>
      {% for r in bt.results %}
        {% for label, path in r.plots.items() %}
          <a class="ml-plot-link" href="ml/{{ path }}" target="_blank">
            {{ r.kind[:3] }} h{{ r.horizon_h }}{% if r.threshold %} &gt;{{ r.threshold }}ft{% endif %} · {{ label }}
          </a>
        {% endfor %}
      {% endfor %}
    </div>
    {% endif %}
  </div>
  {% endif %}

  <div class="card span4">
    <h2>Asheville weather</h2>
    {% if data.weather and not data.weather.error %}
      <div class="big">{{ data.weather.temp_f }}&deg;F</div>
      <div class="row"><span>Humidity</span><span>{{ data.weather.humidity_pct }}%</span></div>
      <div class="row"><span>Wind</span><span>{{ data.weather.wind_mph }} mph @ {{ data.weather.wind_dir_deg }}&deg;</span></div>
      <div class="row"><span>Pressure</span><span>{{ data.weather.pressure_mb }} mb</span></div>
      <div class="row"><span>Current precip</span><span>{{ data.weather.precip_in }} in</span></div>
      <div class="row"><span>Next 72h precip</span><span><b>{{ data.weather.next_72h_precip_in }} in</b></span></div>
      <div class="dim">Open-Meteo &middot; {{ data.weather.as_of }}</div>
    {% else %}
      <div class="dim">Weather feed unavailable</div>
    {% endif %}
  </div>

  <div class="card span8">
    <h2>Heat stress &amp; wet-bulb
      <span class="ml-subtitle">
        <span>NWS heat index</span>
        <span class="dot" aria-hidden="true">&middot;</span>
        <span>Stull wet-bulb</span>
        <span class="dot" aria-hidden="true">&middot;</span>
        <span class="site-chip">Asheville</span>
      </span>
    </h2>
    {% set w = data.weather %}
    {% if w and not w.error and w.heat_index_f is not none %}
    <div class="heat-grid">
      <div class="heat-index-block">
        <div class="dim" style="font-size:.63rem; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.2rem;">Heat Index</div>
        <div class="heat-index-num" style="color: {{ w.heat_color }};">{{ w.heat_index_f }}&deg;</div>
        <div class="heat-cat-badge" style="background: {{ w.heat_color }};">{{ w.heat_category }}</div>
      </div>
      <div class="heat-metrics">
        <div class="heat-row"><span>Actual temp</span><b>{{ w.temp_f }}&deg;F</b></div>
        {% if w.apparent_temp_f is not none %}
        <div class="heat-row"><span>Feels like</span><b>{{ w.apparent_temp_f }}&deg;F</b></div>
        {% endif %}
        {% if w.wet_bulb_f is not none %}
        <div class="heat-row"><span><span class="jargon" title="Lowest temperature achievable by evaporative cooling; key heat-stress indicator.">Wet-bulb temp</span></span><b>{{ w.wet_bulb_f }}&deg;F</b></div>
        {% endif %}
        {% if w.dew_point_f is not none %}
        <div class="heat-row"><span>Dew point</span><b>{{ w.dew_point_f }}&deg;F</b></div>
        {% endif %}
        <div class="heat-row"><span>Relative humidity</span><b>{{ w.humidity_pct }}%</b></div>
      </div>
    </div>
    {% set ht = w.hourly_temp_f %}
    {% set ha = w.hourly_apparent_f %}
    {% if ht and ha and ht | length > 1 %}
    <div class="heat-sparkline">
      <div class="heat-spark-title">24-hour temperature forecast</div>
      {% set n = ht | length %}
      {% set _all = ht + ha %}
      {% set vmin = _all | min - 2 %}
      {% set vmax = _all | max + 2 %}
      {% set vrng = (vmax - vmin) if (vmax - vmin) > 0.1 else 0.1 %}
      <svg class="heat-spark" viewBox="0 0 200 52" preserveAspectRatio="none">
        {% set pts = [] %}
        {% for v in ht %}
          {% set _ = pts.append("%.1f,%.1f" | format((loop.index0 / (n - 1)) * 200, 50 - ((v - vmin) / vrng) * 48)) %}
        {% endfor %}
        <polyline points="{{ pts | join(' ') }}" fill="none" stroke="#4fc3f7" stroke-width="1.5"/>
        {% set pts2 = [] %}
        {% set n2 = ha | length %}
        {% for v in ha %}
          {% set _ = pts2.append("%.1f,%.1f" | format((loop.index0 / (n2 - 1)) * 200, 50 - ((v - vmin) / vrng) * 48)) %}
        {% endfor %}
        <polyline points="{{ pts2 | join(' ') }}" fill="none" stroke="#ff8a65" stroke-width="1.5" stroke-dasharray="4,2"/>
      </svg>
      <div class="heat-spark-legend">
        <span class="leg-actual">Actual temp</span>
        <span class="leg-app">Feels-like</span>
      </div>
    </div>
    {% endif %}
    <div class="heat-nws-cats">
      <span class="dim" style="width:100%; font-size:.63rem; letter-spacing:.08em; text-transform:uppercase;">NWS heat index scale</span>
      {% for label, color, rng in [
          ('Normal',          '#2e7d32', '< 80°F'),
          ('Caution',         '#f9a825', '80–90°F'),
          ('Extreme Caution', '#ef6c00', '90–103°F'),
          ('Danger',          '#c62828', '103–124°F'),
          ('Extreme Danger',  '#6a1b9a', '≥ 125°F'),
        ] %}
      <div class="heat-nws-cat">
        <div class="heat-nws-swatch" style="background: {{ color }};"></div>
        <span>{{ label }} <span class="dim">({{ rng }})</span></span>
      </div>
      {% endfor %}
    </div>
    {% else %}
    <div class="dim">Heat data unavailable</div>
    {% endif %}
  </div>

  <div class="card span8">
    <h2>Situational map</h2>
    <div id="map"></div>
  </div>

  <div class="card span4">
    <h2>Active Atlantic storms</h2>
    {% if data.storms %}
      <table>
        <tr><th>Name</th><th>Class</th><th>kt</th><th>mi</th></tr>
        {% for s in data.storms %}
        <tr>
          <td><b>{{ s.name }}</b></td>
          <td>{{ s.classification }}</td>
          <td>{{ s.intensity_kt|round(0)|int if s.intensity_kt else "?" }}</td>
          <td>{{ s.distance_mi|round(0)|int }}</td>
        </tr>
        {% endfor %}
      </table>
    {% else %}
      <div class="dim">No active Atlantic storms.</div>
    {% endif %}
  </div>

  <div class="card span6">
    <h2>NWS alerts (Asheville)</h2>
    {% if data.alerts %}
      {% for a in data.alerts %}
        <div class="alert">
          <b>{{ a.event }}</b> [{{ a.severity }}]<br>
          <small>{{ a.headline }}</small>
        </div>
      {% endfor %}
    {% else %}
      <div class="dim">No active watches/warnings/advisories.</div>
    {% endif %}
  </div>

  <div class="card span6">
    <h2>2026 CSU Atlantic outlook</h2>
    <div class="row"><span>Named storms</span><span>{{ data.season.named_storms }} (climo 14.4)</span></div>
    <div class="row"><span>Hurricanes</span><span>{{ data.season.hurricanes }} (climo 7.2)</span></div>
    <div class="row"><span>Major hurricanes</span><span>{{ data.season.major_hurricanes }} (climo 3.2)</span></div>
    <div class="row"><span>ACE</span><span>{{ data.season.ace }} (climo 123)</span></div>
    <div class="row"><span>P(major US landfall)</span><span>{{ (data.season.p_us_major_landfall * 100)|round(0)|int }}%</span></div>
    <div class="dim">Klotzbach et al., CSU, {{ data.season.issued }}</div>
  </div>

  <div class="card span6">
    <h2>Soil + antecedent precip (Asheville)</h2>
    {% if data.soil and not data.soil.error %}
      {% set sm = data.soil.soil_moisture_top or data.soil.soil_moisture_shallow or 0 %}
      {% set sat_pct = (sm / 0.45 * 100)|round(0)|int %}
      <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
        <div>
          <div class="big">{{ "%.2f"|format(sm) }}</div>
          <div class="dim">m&sup3;/m&sup3; topsoil ({{ data.soil.condition }})</div>
        </div>
        <div class="breakdown">
          <div class="seg" title="0-0.45 m3/m3">
            <span style="width: {{ sat_pct if sat_pct < 100 else 100 }}%;
                         background: {% if data.soil.saturated %}#c62828
                                     {% elif sm >= 0.30 %}#ff8a65
                                     {% elif sm >= 0.20 %}#9e9d24
                                     {% else %}#4caf50{% endif %};"></span>
          </div>
          <div class="row"><span>Past 7d precip</span><span><b>{{ data.soil.past_7d_precip_in }} in</b></span></div>
          <div class="row"><span>Topsoil 0-1 cm</span><span>{{ data.soil.soil_moisture_top }}</span></div>
          <div class="row"><span>Shallow 1-3 cm</span><span>{{ data.soil.soil_moisture_shallow }}</span></div>
          <div class="row"><span>Mid 3-9 cm</span><span>{{ data.soil.soil_moisture_mid }}</span></div>
          <div class="row"><span>Root 9-27 cm</span><span>{{ data.soil.soil_moisture_root }}</span></div>
        </div>
      </div>
      <div class="dim" style="margin-top:.5rem;">
        Open-Meteo &middot; saturated &gt;= 0.40 m&sup3;/m&sup3; &middot;
        amplifier x{% if data.soil.saturated %}1.25{% elif sm >= 0.30 %}1.10{% else %}1.00{% endif %}
      </div>
    {% else %}
      <div class="dim">Soil data unavailable</div>
    {% endif %}
  </div>

  <div class="card span6">
    <h2>NC Coast - water level + wind (NOAA CO-OPS)</h2>
    {% if data.coastal %}
      <table>
        <tr><th>Station</th><th>Water (ft MLLW)</th><th>Wind (kt)</th><th>Press (mb)</th></tr>
        {% for c in data.coastal %}
        <tr>
          <td><b>{{ c.label }}</b><br><span class="dim" style="font-size:.7rem;">{{ c.role }} &middot; {{ c.station_id }}</span></td>
          <td>{{ "%.2f"|format(c.water_level_ft) if c.water_level_ft is not none else "?" }}</td>
          <td>
            {% if c.wind_kt is not none %}
              {{ "%.0f"|format(c.wind_kt) }}
              {% if c.wind_gust_kt %} (g {{ "%.0f"|format(c.wind_gust_kt) }}){% endif %}
              {% if c.wind_dir_deg is not none %} @ {{ c.wind_dir_deg|round(0)|int }}&deg;{% endif %}
            {% else %}?{% endif %}
          </td>
          <td>{{ "%.1f"|format(c.air_pressure_mb) if c.air_pressure_mb is not none else "?" }}</td>
        </tr>
        {% endfor %}
      </table>
      <div class="dim" style="margin-top:.4rem;">
        Storm-surge early-warning chain: pressure drops + wind backing + water rising = TC approach.
      </div>
    {% else %}
      <div class="dim">CO-OPS feed unavailable</div>
    {% endif %}
  </div>

  <div class="card" style="grid-column: 1 / -1;">
    <h2>NOAA NDBC offshore buoys + C-MAN (hurricane lead-time)</h2>
    {% if data.buoys %}
      <table>
        <tr>
          <th>Station</th><th>Seas</th><th>Wave (ft / s)</th>
          <th>Wind (kt)</th><th>Pressure (mb)</th><th>Water (&deg;F)</th>
        </tr>
        {% for b in data.buoys %}
        <tr>
          <td>
            <b>{{ b.label }}</b>
            <br><span class="dim" style="font-size:.7rem;">{{ b.role }} &middot; NDBC {{ b.station_id }}</span>
          </td>
          <td>
            <span class="ll-pill" style="background: {{ b.color }}">{{ b.seas }}</span>
          </td>
          <td>
            {% if b.wave_ht_ft is not none %}
              {{ "%.1f"|format(b.wave_ht_ft) }} ft
              {% if b.dominant_period_s %} / {{ "%.0f"|format(b.dominant_period_s) }} s{% endif %}
            {% else %}?{% endif %}
          </td>
          <td>
            {% if b.wind_kt is not none %}
              {{ "%.0f"|format(b.wind_kt) }}
              {% if b.wind_gust_kt %} (g {{ "%.0f"|format(b.wind_gust_kt) }}){% endif %}
              {% if b.wind_dir_deg is not none %} @ {{ b.wind_dir_deg|round(0)|int }}&deg;{% endif %}
            {% else %}?{% endif %}
          </td>
          <td>{{ "%.1f"|format(b.pressure_mb) if b.pressure_mb is not none else "?" }}</td>
          <td>{{ "%.1f"|format(b.water_temp_f) if b.water_temp_f is not none else "?" }}</td>
        </tr>
        {% endfor %}
      </table>
      <div class="dim" style="margin-top:.4rem;">
        Wave height &amp; long-period swell at Diamond Shoals / Frying Pan Shoals are
        the canonical 12-24h leading indicators of a TC approaching the NC coast.
      </div>
    {% else %}
      <div class="dim">NDBC feed unavailable</div>
    {% endif %}
  </div>

  <div class="card" style="grid-column: 1 / -1;">
    <h2>NC National Forests</h2>
    <div class="forest-grid">
      {% for f in data.forests %}
        <div class="forest region-{{ f.region }}">
          <h3>{{ f.name }}</h3>
          <div class="meta">{{ "{:,}".format(f.acres) }} ac &middot; est. {{ f.established }} &middot;
                            HQ {{ f.hq_city }} &middot; {{ f.region }}</div>
          <div class="stats">
            {% if f.weather and not f.weather.error %}
              <span>Temp</span><span>{{ f.weather.temp_f }}&deg;F</span>
              <span>72h precip</span><span><b>{{ f.weather.next_72h_precip_in }} in</b></span>
              <span>Wind</span><span>{{ f.weather.wind_mph }} mph</span>
              <span>RH</span><span>{{ f.weather.humidity_pct }}%</span>
            {% else %}
              <span>Weather</span><span class="dim">unavailable</span>
            {% endif %}
            {% if f.nearest_storm %}
              <span>Nearest TC</span>
              <span><b>{{ f.nearest_storm }}</b> {{ f.nearest_storm_mi|round(0)|int }} mi</span>
            {% endif %}
            <span>Districts</span><span>{{ f.districts|join(', ') }}</span>
          </div>

          {% if f.landslide %}
            <div class="forest-landslide" title="{{ f.landslide.explain }}">
              <span class="ll-pill" style="background: {{ f.landslide.color }}">
                Landslide: {{ f.landslide.label }} ({{ f.landslide.score }})
              </span>
              {% if f.landslide.inventory and f.landslide.inventory.count %}
                <span class="ll-inv">{{ f.landslide.inventory.count }} historical event{{ '' if f.landslide.inventory.count == 1 else 's' }} within 25 mi{% if f.landslide.inventory.most_recent_year %} (latest {{ f.landslide.inventory.most_recent_year }}){% endif %}</span>
              {% endif %}
            </div>
          {% endif %}

          {% if f.fire_weather %}
            <div class="forest-landslide" title="{{ f.fire_weather.explain }}">
              <span class="ll-pill" style="background: {{ f.fire_weather.color }}">
                Fire wx: {{ f.fire_weather.label }} ({{ f.fire_weather.score }})
              </span>
              {% if f.fires_summary and f.fires_summary.count %}
                <span class="ll-inv">
                  {{ f.fires_summary.count }} active fire{{ '' if f.fires_summary.count == 1 else 's' }} within 50 mi
                  ({{ "{:,.0f}".format(f.fires_summary.total_acres) }} ac{% if f.fires_summary.min_contained_pct is not none %}, min {{ f.fires_summary.min_contained_pct|int }}% contained{% endif %})
                </span>
              {% endif %}
            </div>
          {% endif %}

          {% if f.air_quality and not f.air_quality.error and f.air_quality.us_aqi is not none %}
            <div class="forest-landslide">
              <span class="ll-pill" style="background: {{ f.air_quality.color }}">
                AQI: {{ f.air_quality.label }} ({{ f.air_quality.us_aqi|int }})
              </span>
              {% if f.air_quality.pm2_5 is not none %}
                <span class="ll-inv">PM2.5 {{ f.air_quality.pm2_5 }} &micro;g/m&sup3;</span>
              {% endif %}
            </div>
          {% endif %}

          {% if f.gauges %}
            <div class="forest-gauges">
              <div class="ll-section-title">Nearby USGS gauges</div>
              {% for g in f.gauges %}
                <div class="forest-gauge">
                  <span class="fg-name">{{ g.label }}</span>
                  <span class="fg-stage">
                    {% if g.stage_ft is not none %}{{ "%.2f"|format(g.stage_ft) }} ft{% else %}?{% endif %}
                    {% if g.rate_ft_per_hr is not none and g.rate_ft_per_hr > 0.05 %}
                      <span class="rate-up">&uarr;</span>
                    {% elif g.rate_ft_per_hr is not none and g.rate_ft_per_hr < -0.05 %}
                      <span class="rate-down">&darr;</span>
                    {% endif %}
                  </span>
                  <span class="stage-pill {{ g.flood_category|lower|replace(' flood','')|replace(' stage','') }}">
                    {{ g.flood_category }}
                  </span>
                </div>
              {% endfor %}
            </div>
          {% endif %}

          {% if f.alerts %}
            {% for a in f.alerts[:2] %}
              <span class="alert-mini">{{ a.event }}</span>
            {% endfor %}
          {% endif %}
          {% if f.nearest_storm_mi is not none and f.nearest_storm_mi < 300 %}
            <span class="storm-near">!! TC within 300 mi !!</span>
          {% endif %}
          <div class="dim" style="margin-top:.4rem; font-size:.74rem;">{{ f.notes }}</div>

          {% if f.districts_data %}
            <div class="ll-section-title" style="margin-top:.6rem;">Ranger districts</div>
            <div class="districts">
              {% for d in f.districts_data %}
                <div class="district">
                  <h4>{{ d.name }} RD</h4>
                  <div class="d-meta">Office: {{ d.office }}</div>
                  <div class="d-stats">
                    {% if d.weather and not d.weather.error %}
                      <span>Temp</span><span>{{ d.weather.temp_f }}&deg;F</span>
                      <span>72h precip</span><span><b>{{ d.weather.next_72h_precip_in }} in</b></span>
                      <span>Wind</span><span>{{ d.weather.wind_mph }} mph</span>
                      <span>RH</span><span>{{ d.weather.humidity_pct }}%</span>
                    {% else %}
                      <span>Weather</span><span class="dim">unavailable</span>
                    {% endif %}
                  </div>
                  <div class="d-pills">
                    {% if d.landslide %}
                      <span class="d-pill" style="background: {{ d.landslide.color }}"
                            title="{{ d.landslide.explain }}">
                        Slide {{ d.landslide.label }} ({{ d.landslide.score }})
                      </span>
                    {% endif %}
                    {% if d.fire_weather %}
                      <span class="d-pill" style="background: {{ d.fire_weather.color }}"
                            title="{{ d.fire_weather.explain }}">
                        Fire {{ d.fire_weather.label }} ({{ d.fire_weather.score }})
                      </span>
                    {% endif %}
                    {% if d.air_quality and not d.air_quality.error and d.air_quality.us_aqi is not none %}
                      <span class="d-pill" style="background: {{ d.air_quality.color }}">
                        AQI {{ d.air_quality.label }} ({{ d.air_quality.us_aqi|int }})
                      </span>
                    {% endif %}
                    {% if d.fires_summary and d.fires_summary.count %}
                      <span class="d-pill" style="background:#ff3d00">
                        {{ d.fires_summary.count }} fire{{ '' if d.fires_summary.count == 1 else 's' }}
                      </span>
                    {% endif %}
                    {% if d.alerts %}
                      <span class="d-pill" style="background:#b71c1c">
                        {{ d.alerts|length }} alert{{ '' if d.alerts|length == 1 else 's' }}
                      </span>
                    {% endif %}
                  </div>
                  <div class="d-notes">{{ d.notes }}</div>
                </div>
              {% endfor %}
            </div>
          {% endif %}
        </div>
      {% endfor %}
    </div>
  </div>

  {% if data.glossary %}
  <div class="card span12 legend">
    <h2>How to read this dashboard</h2>
    <div class="dim" style="margin-bottom:.5rem; font-size:.78rem;">
      Plain-English definitions for the jargon on this page. Hover any
      <span class="jargon" title="Like this. Hover to read.">dotted term</span>
      for a quick tip.
    </div>
    <dl style="column-count: 2; column-gap: 1.5rem;">
      {% for term in [
          'Flood Index','QPF','stage','action stage','minor flood',
          'moderate flood','major flood','discharge','rate',
          'NWPS forecast','ML forecast','backtest','MAE','AUC',
          'soil saturated','TC','ETA',
        ] %}
        {% if data.glossary.get(term) %}
        <dt>{{ term }}</dt>
        <dd>{{ data.glossary[term] }}</dd>
        {% endif %}
      {% endfor %}
    </dl>
  </div>
  {% endif %}

</div>

<footer>
  Data: USGS NWIS, NWS api.weather.gov, NHC CurrentStorms, Open-Meteo, CSU Tropical Met Project.
</footer>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script>
  const STATE = {{ data | tojson }};
  const ASH = [STATE.asheville.lat, STATE.asheville.lon];

  const map = L.map('map', { zoomControl: true }).setView([35.4, -78.5], 7);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 18,
  }).addTo(map);

  L.circleMarker(ASH, {radius:8, color:'#4fc3f7', weight:2, fillColor:'#4fc3f7',
                       fillOpacity:.6}).addTo(map).bindPopup('<b>Asheville</b>');

  // Coastal NOAA stations (storm-surge early warning)
  (STATE.coastal || []).forEach(c => {
    const wind = c.wind_kt || 0;
    let color = '#1e88e5';
    if (wind >= 34) color = '#fb8c00';
    if (wind >= 64) color = '#c62828';
    L.marker([c.lat, c.lon], {
      icon: L.divIcon({
        className: 'coast-icon',
        html: '<div style=\"background:'+color+';color:#fff;font-weight:600;'+
              'padding:1px 5px;border-radius:3px;font-size:10px;border:1px solid #000;\">'+
              '&#127754; '+(c.water_level_ft != null ? c.water_level_ft.toFixed(1)+"'" : '?')+'</div>',
        iconSize: null,
      })
    }).addTo(map).bindPopup(
      '<b>' + c.label + '</b><br>' +
      'Water: ' + (c.water_level_ft != null ? c.water_level_ft.toFixed(2) + ' ft MLLW' : 'n/a') + '<br>' +
      'Wind: ' + (c.wind_kt != null ? c.wind_kt.toFixed(0) + ' kt' : 'n/a') +
        (c.wind_gust_kt ? ' (g ' + c.wind_gust_kt.toFixed(0) + ')' : '') +
        (c.wind_dir_deg != null ? ' @ ' + Math.round(c.wind_dir_deg) + '&deg;' : '') + '<br>' +
      'Pressure: ' + (c.air_pressure_mb != null ? c.air_pressure_mb.toFixed(1) + ' mb' : 'n/a') + '<br>' +
      '<small>NOAA CO-OPS ' + c.station_id + '</small>');
  });

  // NDBC offshore buoys + C-MAN stations — hurricane lead-time markers
  (STATE.buoys || []).forEach(b => {
    const wave = b.wave_ht_ft || 0;
    const radius = Math.max(5, Math.min(16, 5 + wave * 0.8));
    L.circleMarker([b.lat, b.lon], {
      radius, color: b.color || '#1976d2', fillColor: b.color || '#1976d2',
      fillOpacity: .55, weight: 2,
    }).addTo(map).bindPopup(
      '<b>&#9883; ' + b.label + '</b><br>' +
      'Seas: <b>' + b.seas + '</b><br>' +
      'Wave: ' + (b.wave_ht_ft != null ? b.wave_ht_ft.toFixed(1) + ' ft' : 'n/a') +
        (b.dominant_period_s ? ' / ' + b.dominant_period_s.toFixed(0) + ' s' : '') + '<br>' +
      'Wind: ' + (b.wind_kt != null ? b.wind_kt.toFixed(0) + ' kt' : 'n/a') +
        (b.wind_gust_kt ? ' (g ' + b.wind_gust_kt.toFixed(0) + ')' : '') + '<br>' +
      'Pressure: ' + (b.pressure_mb != null ? b.pressure_mb.toFixed(1) + ' mb' : 'n/a') + '<br>' +
      'Water: ' + (b.water_temp_f != null ? b.water_temp_f.toFixed(1) + '&deg;F' : 'n/a') + '<br>' +
      '<small>NDBC ' + b.station_id + ' &middot; ' + b.role + '</small>');
  });

  // National Forests
  const FOREST_COLOR = {mountain:'#4caf50', piedmont:'#fdd835', coastal:'#ef5350'};
  (STATE.forests || []).forEach(f => {
    const c = FOREST_COLOR[f.region] || '#4caf50';
    const radius = Math.max(10, Math.sqrt(f.acres / 1000) * 0.6);
    L.circle([f.center_lat, f.center_lon], {
      radius: radius * 1000,           // visual footprint, meters
      color: c, fillColor: c, fillOpacity: .12, weight: 1.5,
    }).addTo(map);
    L.marker([f.center_lat, f.center_lon], {
      icon: L.divIcon({
        className: 'nf-icon',
        html: '<div style=\"background:'+c+';color:#000;font-weight:700;'+
              'padding:2px 6px;border-radius:4px;font-size:11px;border:1px solid #000;\">'+
              '&#127794; '+f.short+'</div>',
        iconSize: null,
      })
    }).addTo(map).bindPopup(
      '<b>' + f.name + '</b><br>' +
      f.acres.toLocaleString() + ' ac &middot; est. ' + f.established + '<br>' +
      'HQ: ' + f.hq_city + '<br>' +
      (f.weather && !f.weather.error
        ? 'Temp: ' + f.weather.temp_f + '&deg;F &middot; 72h precip: ' +
          f.weather.next_72h_precip_in + ' in<br>'
        : '') +
      (f.nearest_storm
        ? 'Nearest TC: <b>' + f.nearest_storm + '</b> ' +
          Math.round(f.nearest_storm_mi) + ' mi<br>'
        : '') +
      '<small>' + f.notes + '</small>');
    (f.districts_data || []).forEach(d => {
      L.circleMarker([d.lat, d.lon], {
        radius: 4, color: c, fillColor: '#fff', fillOpacity: .9, weight: 2,
      }).addTo(map).bindPopup(
        '<b>' + d.name + ' RD</b> &middot; ' + f.short + ' NF<br>' +
        'Office: ' + d.office + '<br>' +
        (d.weather && !d.weather.error
          ? 'Temp: ' + d.weather.temp_f + '&deg;F &middot; 72h precip: ' +
            d.weather.next_72h_precip_in + ' in<br>'
          : '') +
        (d.fire_weather ? 'Fire wx: <b>' + d.fire_weather.label + '</b> (' +
          d.fire_weather.score + ')<br>' : '') +
        (d.landslide ? 'Landslide: <b>' + d.landslide.label + '</b> (' +
          d.landslide.score + ')<br>' : '') +
        '<small>' + d.notes + '</small>');
    });
  });

  STATE.gauges.forEach(g => {
    const cat = (g.flood_category || '').toLowerCase();
    let color = '#4caf50';
    if (cat.includes('action')) color = '#fdd835';
    if (cat.includes('minor')) color = '#ef6c00';
    if (cat.includes('moderate')) color = '#c62828';
    if (cat.includes('major')) color = '#6a1b9a';
    const m = L.circleMarker([g.lat, g.lon], {
      radius: g.site_id === STATE.primary_site ? 9 : 6,
      color, fillColor: color, fillOpacity: .7, weight: 2,
    }).addTo(map);
    const r = g.rate_ft_per_hr;
    const rateStr = r == null ? 'n/a' :
      (r > 0.05 ? '&uarr; ' : (r < -0.05 ? '&darr; ' : '')) +
      Math.abs(r).toFixed(2) + ' ft/hr';
    m.bindPopup(
      '<b>' + g.label + '</b><br>' +
      'Stage: ' + (g.stage_ft != null ? g.stage_ft.toFixed(2) + ' ft' : 'n/a') + '<br>' +
      'Rate: ' + rateStr + '<br>' +
      'Cat: ' + g.flood_category + '<br>' +
      '<small>USGS ' + g.site_id + '</small>');
  });

  (STATE.storms || []).forEach(s => {
    if (s.lat == null || s.lon == null) return;
    const wind = s.intensity_kt || 0;
    let color = '#90caf9';
    if (wind >= 64) color = '#ef5350';
    if (wind >= 96) color = '#c62828';
    if (wind >= 113) color = '#6a1b9a';
    L.circleMarker([s.lat, s.lon], {
      radius: 10, color, fillColor: color, fillOpacity: .5, weight: 2,
    }).addTo(map).bindPopup(
      '<b>' + s.name + '</b><br>' + s.classification + ' ' + wind + ' kt<br>' +
      (s.distance_mi || 0).toFixed(0) + ' mi from Asheville<br>' + (s.movement || ''));
    L.polyline([[s.lat, s.lon], ASH], {color: '#888', weight: 1, dashArray: '4,4'}).addTo(map);
  });

  // Active wildfires (deduplicate across forests; key by IRWIN id or name+coord)
  const seenFires = new Set();
  (STATE.forests || []).forEach(f => {
    (f.fires_nearby || []).forEach(fire => {
      const key = fire.irwin_id || (fire.name + ':' + fire.lat + ',' + fire.lon);
      if (seenFires.has(key)) return;
      seenFires.add(key);
      const acres = fire.acres || 0;
      const radius = Math.max(6, Math.min(18, Math.sqrt(acres) * 0.6));
      L.circleMarker([fire.lat, fire.lon], {
        radius, color: '#ff6f00', fillColor: '#ff3d00', fillOpacity: .7,
        weight: 2,
      }).addTo(map).bindPopup(
        '<b>&#128293; ' + fire.name + '</b><br>' +
        (acres ? acres.toLocaleString() + ' ac' : 'size unknown') +
        (fire.contained_pct != null ? ' &middot; ' + fire.contained_pct + '% contained' : '') + '<br>' +
        (fire.cause ? 'Cause: ' + fire.cause + '<br>' : '') +
        '<small>NIFC WFIGS' + (fire.irwin_id ? ' &middot; ' + fire.irwin_id : '') + '</small>');
    });
  });

  function ageTick() {
    const secs = Math.floor((Date.now()/1000) - STATE.as_of_epoch);
    const el = document.getElementById('age');
    if (secs < 60) el.textContent = secs + 's ago';
    else el.textContent = Math.floor(secs/60) + 'm ago';
  }
  setInterval(ageTick, 1000);
  setTimeout(() => location.reload(), 15 * 60_000);

  // Gauge network filter.
  (function() {
    const tabs = document.querySelectorAll('#net-tabs .net-tab');
    const rows = document.querySelectorAll('.gauge-row[data-net]');
    function apply(net) {
      rows.forEach(r => {
        const show = (net === 'all') || (r.dataset.net === net);
        r.classList.toggle('hidden', !show);
      });
      tabs.forEach(t => t.classList.toggle('active', t.dataset.net === net));
      try { localStorage.setItem('gaugeNet', net); } catch (e) {}
    }
    tabs.forEach(t => t.addEventListener('click', () => apply(t.dataset.net)));
    let saved = 'all';
    try { saved = localStorage.getItem('gaugeNet') || 'all'; } catch (e) {}
    if ([...tabs].some(t => t.dataset.net === saved)) apply(saved);
  })();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    from .narrative import GLOSSARY, summarize
    data = _collect()
    data.setdefault("glossary", GLOSSARY)
    data.setdefault("narrative", summarize(data))
    return render_template_string(PAGE, data=data)


@app.route("/api/state")
def api_state():
    return jsonify(_collect())


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    print(f"Asheville hurricane dashboard -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
