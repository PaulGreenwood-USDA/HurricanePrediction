"""Human-readable summary of the dashboard state.

Turns the kitchen-sink ``/api/state`` payload into a one-paragraph TL;DR
that answers three questions a non-expert actually asks:

1. **What's happening?** -- one headline sentence.
2. **Why?** -- 2-4 key facts that drove the verdict.
3. **What should I do?** -- a recommended action keyed to the severity.

This module is *pure*: it takes the already-collected state dict and
returns a plain dict. No I/O, no fetching. Easy to unit-test, easy to
inline into the Jinja template.
"""
from __future__ import annotations

from typing import Iterable

# Severity ladder. Matches the index_score labels but uses friendlier
# wording so the headline reads like English, not a SCADA panel.
_LEVEL_BY_LABEL = {
    "CALM":      ("calm",      "#2e7d32"),
    "ELEVATED":  ("watch",     "#9e9d24"),
    "ALERT":     ("alert",     "#ef6c00"),
    "WARNING":   ("warning",   "#c62828"),
    "EMERGENCY": ("emergency", "#6a1b9a"),
}

_RECOMMENDATIONS = {
    "calm":      "Normal conditions. No flood action needed.",
    "watch":     "Stay aware. Some risk factors are elevated — check back in a few hours.",
    "alert":     "Heightened flood potential. If you live near the French Broad, monitor closely and avoid low-lying areas.",
    "warning":   "Flooding likely. Move valuables to high ground, avoid flooded roads, follow local guidance.",
    "emergency": "Active flood emergency. Follow evacuation orders from local authorities.",
}


def _fmt_ft(v) -> str:
    try:
        return f"{float(v):.1f} ft"
    except (TypeError, ValueError):
        return "?"


def _primary_gauge(state: dict) -> dict | None:
    site = state.get("primary_site")
    for g in state.get("gauges", []) or []:
        if g.get("site_id") == site:
            return g
    return None


def _ml_for_primary(state: dict) -> dict | None:
    site = state.get("primary_site")
    block = (state.get("ml_forecasts") or {}).get(site)
    return block


def _key_facts(state: dict) -> list[str]:
    """Pick out the 2-4 most informative numbers to surface."""
    facts: list[str] = []
    p = _primary_gauge(state)
    if p:
        stage = p.get("stage_ft")
        cat = p.get("flood_category") or "unknown"
        if stage is not None:
            facts.append(f"French Broad at Asheville: {_fmt_ft(stage)} ({cat})")
        rate = p.get("rate_ft_per_hr")
        if rate is not None and abs(rate) >= 0.05:
            arrow = "rising" if rate > 0 else "falling"
            facts.append(f"Stage {arrow} at {abs(rate):.2f} ft/hr")

    w = state.get("weather") or {}
    if not w.get("error"):
        next72 = w.get("next_72h_precip_in")
        if next72 is not None and next72 >= 0.5:
            facts.append(f"Next 72h rainfall forecast: {next72:.1f} in")

    storms = state.get("storms") or []
    near = [s for s in storms
            if (s.get("distance_to_asheville_mi") or 9999) < 1000]
    if near:
        s = min(near, key=lambda x: x.get("distance_to_asheville_mi", 9999))
        d = s.get("distance_to_asheville_mi")
        name = s.get("name") or s.get("id") or "Tropical system"
        facts.append(
            f"{name} {d:.0f} mi from Asheville" if d is not None
            else f"{name} active in the Atlantic")

    soil = state.get("soil") or {}
    if not soil.get("error"):
        sat = soil.get("saturation_pct") or soil.get("top_sat_pct")
        if sat is not None and sat >= 80:
            facts.append(f"Soils saturated ({sat:.0f}%) — runoff will be fast")

    ml = _ml_for_primary(state)
    if ml:
        # Pick the 24h regression if present, else any
        reg = ml.get("regression") or []
        pick = next((r for r in reg if r.get("horizon_h") == 24), None) \
                or (reg[0] if reg else None)
        if pick:
            facts.append(
                f"ML forecast +{pick['horizon_h']}h: "
                f"{_fmt_ft(pick['predicted_stage_ft'])}")
        # Highest-probability flood classification
        cls = ml.get("classification") or []
        hot = max((c for c in cls if c.get("probability") is not None),
                   key=lambda c: c["probability"], default=None)
        if hot and hot["probability"] >= 0.25:
            pct = int(hot["probability"] * 100)
            facts.append(
                f"Model: {pct}% chance >"
                f"{hot['threshold']}ft within {hot['horizon_h']}h")

    alerts = state.get("alerts") or []
    flood_alerts = [a for a in alerts
                     if "flood" in (a.get("event") or "").lower()
                     or "tropical" in (a.get("event") or "").lower()]
    if flood_alerts:
        a = flood_alerts[0]
        facts.append(f"NWS: {a.get('event')} active")

    return facts[:4]


def _headline(state: dict, level_key: str) -> str:
    p = _primary_gauge(state) or {}
    stage = p.get("stage_ft")
    cat = (p.get("flood_category") or "").lower()
    storms_near_500 = any((s.get("distance_to_asheville_mi") or 9999) < 500
                           for s in (state.get("storms") or []))
    qpf = (state.get("weather") or {}).get("next_72h_precip_in") or 0

    if level_key == "emergency":
        return "Active flood emergency in the French Broad watershed."
    if level_key == "warning":
        if "flood" in cat:
            return f"French Broad is in {p.get('flood_category', 'flood').lower()} — flooding underway."
        return "Multiple severe risk factors firing — flooding likely."
    if level_key == "alert":
        if storms_near_500:
            return "Tropical system within 500 mi and rainfall building — flood risk rising."
        if qpf >= 3:
            return f"Heavy rain expected ({qpf:.1f}\" in next 72h) — flood risk rising."
        return "Several risk factors elevated — monitor closely."
    if level_key == "watch":
        if qpf >= 1:
            return f"Wet pattern incoming ({qpf:.1f}\" in next 72h). Stay aware."
        if storms_near_500:
            return "Tropical activity in the Atlantic warrants attention."
        return "Conditions are slightly elevated but no immediate threat."
    # calm
    if stage is not None:
        return (f"All clear. French Broad at Asheville is "
                f"{_fmt_ft(stage)} and steady.")
    return "All clear — no active flood risk factors."


def _subheadline(state: dict, level_key: str) -> str:
    """One-line context behind the headline."""
    idx = state.get("index") or {}
    score = idx.get("score")
    triggers_on = sum(1 for v in (idx.get("triggers") or {}).values() if v)
    total = len(idx.get("triggers") or {}) or 10
    base = f"Flood Index {score}/100"
    if triggers_on:
        base += f" — {triggers_on} of {total} risk triggers firing"
    return base


def summarize(state: dict) -> dict:
    """Return the narrative payload to embed under ``state['narrative']``."""
    idx = state.get("index") or {}
    label = (idx.get("label") or "CALM").upper()
    level_key, color = _LEVEL_BY_LABEL.get(label, ("calm", "#2e7d32"))
    return {
        "level": level_key,
        "color": color,
        "headline": _headline(state, level_key),
        "subheadline": _subheadline(state, level_key),
        "recommendation": _RECOMMENDATIONS[level_key],
        "key_facts": _key_facts(state),
    }


# ---- glossary -------------------------------------------------------------

# Plain-English explanations for jargon that appears in the dashboard.
# Used as hover tooltips on the rendered page.
GLOSSARY: dict[str, str] = {
    "Flood Index": "0-100 score blending river stage, rainfall forecast, "
                    "active storms, rise rate, NWS alerts, and soil saturation.",
    "QPF": "Quantitative Precipitation Forecast — predicted rainfall total.",
    "stage": "Height of the water surface above a gauge datum, in feet.",
    "action stage": "Level at which protective action by emergency managers begins.",
    "minor flood": "Stage at which minor flooding of low-lying areas starts.",
    "moderate flood": "Stage at which some structures and roads flood.",
    "major flood": "Stage at which extensive inundation occurs.",
    "discharge": "Volume of water flowing past the gauge, in cubic feet per second.",
    "rate": "How fast the river is rising or falling, in feet per hour.",
    "NWPS forecast": "National Water Prediction Service — official NOAA river forecast.",
    "ML forecast": "Machine-learning prediction from a LightGBM model trained "
                    "on 5 years of hourly history.",
    "backtest": "Walk-forward out-of-sample evaluation of the ML models.",
    "MAE": "Mean Absolute Error — average miss in feet between forecast and actual.",
    "AUC": "Area Under the ROC Curve — 0.5 is random, 1.0 is perfect classification.",
    "soil saturated": "Top-layer soil moisture is at or near field capacity; "
                       "any further rain will run off rather than soak in.",
    "TC": "Tropical Cyclone — any tropical depression, storm, or hurricane.",
    "ETA": "Linearly-extrapolated time to reach the next flood threshold "
            "if the current rise rate holds.",
}
