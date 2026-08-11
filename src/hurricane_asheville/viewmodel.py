"""Render-ready view model for the dashboard.

Everything here is a pure function from the ``_collect()`` state dict to
primitives the template can print without arithmetic. The template used to do
this work inline -- sparkline coordinates, flood-band percentages, backtest
metric joins -- which meant none of it could be tested and a divide-by-zero
showed up as a blank card rather than a failing test.

Rules for this module:

* no I/O, no fetching -- it takes the already-collected dict;
* never raise on missing or malformed input, return ``None`` and let the
  template omit the block;
* return plain floats/strings/lists so the template stays declarative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- sparklines


@dataclass
class Sparkline:
    """A polyline in a 0..width by 0..height viewBox."""
    points: str
    width: float
    height: float
    vmin: float
    vmax: float
    rising: bool = False


def sparkline(values, *, width: float = 100.0, height: float = 30.0,
              pad: float = 2.0, rising: bool = False) -> Sparkline | None:
    """Scale ``values`` into an SVG polyline string.

    Returns None for fewer than two points -- a one-point line is a dot that
    reads as data when it is not.
    """
    vals = [v for v in (values or []) if isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    vmin, vmax = min(vals), max(vals)
    # A flat series would divide by zero; give it a hairline range so it
    # renders as a straight line through the middle.
    vrng = (vmax - vmin) or 0.01
    n = len(vals)
    span = height - pad * 2
    pts = []
    for i, v in enumerate(vals):
        x = (i / (n - 1)) * width
        y = (height - pad) - ((v - vmin) / vrng) * span
        pts.append(f"{x:.1f},{y:.1f}")
    return Sparkline(points=" ".join(pts), width=width, height=height,
                     vmin=vmin, vmax=vmax, rising=rising)


def gauge_sparkline(gauge: dict) -> Sparkline | None:
    hist = gauge.get("history") or []
    rate = gauge.get("rate_ft_per_hr") or 0.0
    return sparkline([h.get("ft") for h in hist], rising=rate > 0.05)


# ------------------------------------------------------------ gauge networks

NETWORKS = (
    ("french_broad", "French Broad",
     ("primary", "upstream", "headwaters", "tributary")),
    ("regional",     "WNC regional", ("regional",)),
    ("statewide",    "Statewide",    ("statewide",)),
    ("reservoirs",   "Reservoirs",   ("reservoir",)),
)


def gauge_networks(gauges: list[dict]) -> dict:
    """Tab keys, labels and counts for the gauge-network filter."""
    role_to_net = {role: key for key, _label, roles in NETWORKS for role in roles}
    tabs = [{"key": "all", "label": "All", "count": len(gauges or [])}]
    for key, label, roles in NETWORKS:
        count = sum(1 for g in (gauges or []) if g.get("role") in roles)
        if count:
            tabs.append({"key": key, "label": label, "count": count})
    return {"tabs": tabs, "role_to_net": role_to_net}


def gauge_net_key(gauge: dict, role_to_net: dict) -> str:
    return role_to_net.get(gauge.get("role"), "other")


# ------------------------------------------------------------------ triggers

TRIGGER_LABELS = (
    ("stage_above_action",   "River ≥ action stage"),
    ("stage_above_minor",    "River ≥ minor flood"),
    ("qpf_over_1in",         "72h QPF ≥ 1 in"),
    ("qpf_over_3in",         "72h QPF ≥ 3 in"),
    ("storm_within_1000mi",  "TC within 1000 mi"),
    ("storm_within_500mi",   "TC within 500 mi"),
    ("river_rising_fast",    "River rising > 0.3 ft/hr"),
    ("nws_flood_or_tropical", "NWS flood/tropical alert"),
    ("soil_saturated",       "Soil saturated"),
    ("wet_week",             "Past 7d precip ≥ 3 in"),
)


def triggers(index: dict) -> list[dict]:
    """Trigger pills, fired ones first, each carrying its own on/off state."""
    fired_map = (index or {}).get("triggers") or {}
    items = [{"key": k, "label": label, "on": bool(fired_map.get(k))}
             for k, label in TRIGGER_LABELS]
    items.sort(key=lambda t: not t["on"])
    return items


# --------------------------------------------------------------- flood index

INDEX_COMPONENT_COLORS = {
    "stage": "#2196f3", "qpf": "#4fc3f7", "storm": "#ef5350",
    "rise": "#ff8a65", "alert": "#ba68c8", "soil": "#8d6e63",
}

_DIAL_CIRCUMFERENCE = 326.0


def index_dial(index: dict) -> dict:
    """Arc length for the flood-index donut plus its component segments."""
    score = (index or {}).get("score") or 0
    try:
        score = max(0.0, min(100.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    segments = []
    for key, value in ((index or {}).get("components") or {}).items():
        try:
            width = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
        segments.append({"key": key, "width": width,
                         "color": INDEX_COMPONENT_COLORS.get(key, "#607d8b")})
    return {
        "dash": f"{score / 100.0 * _DIAL_CIRCUMFERENCE:.1f} {_DIAL_CIRCUMFERENCE:.0f}",
        "segments": segments,
    }


# ------------------------------------------------------------------- ML card

def _band_pct(value, axis_max: float) -> float | None:
    if value is None or not axis_max:
        return None
    return max(0.0, min(100.0, (float(value) / axis_max) * 100.0))


def ml_card(state: dict) -> dict | None:
    """Join ML forecasts to the primary gauge, flood stages and model metrics.

    Returns None when there is nothing trustworthy to show.
    """
    site = state.get("primary_site")
    ml = (state.get("ml_forecasts") or {}).get(site)
    if not ml or not (ml.get("regression") or ml.get("classification")):
        return None

    primary = next((g for g in state.get("gauges") or []
                    if g.get("site_id") == site), None)
    current = (primary or {}).get("stage_ft")
    stages = state.get("flood_stages") or {}
    metrics = state.get("ml_metrics") or {}

    axis_max = (stages.get("major") or 20) * 1.25
    bands = []
    for level, color in (("action", "#9e9d24"), ("minor", "#ef6c00"),
                         ("moderate", "#c62828"), ("major", "#6a1b9a")):
        left = _band_pct(stages.get(level), axis_max)
        if left is None:
            continue
        bands.append({"level": level, "color": color, "left": left,
                      "stage": stages.get(level)})
    for i, band in enumerate(bands):
        nxt = bands[i + 1]["left"] if i + 1 < len(bands) else 100.0
        band["width"] = max(0.0, nxt - band["left"])

    horizons = []
    for r in ml.get("regression") or []:
        h = r.get("horizon_h")
        pred = r.get("predicted_stage_ft")
        if h is None or pred is None:
            continue
        delta = None if current is None else pred - current
        reg_metrics = metrics.get(f"regression_h{h}") or {}
        horizons.append({
            "horizon_h": h,
            "predicted_ft": pred,
            "current_ft": current,
            "delta": delta,
            "direction": ("up" if (delta or 0) > 0.05
                          else "down" if (delta or 0) < -0.05 else "flat"),
            "arrow": ("↑" if (delta or 0) > 0.05
                      else "↓" if (delta or 0) < -0.05 else "→"),
            "now_pct": _band_pct(current, axis_max),
            "pred_pct": _band_pct(pred, axis_max),
            "above_minor": bool(stages.get("minor") and pred >= stages["minor"]),
            "mae": reg_metrics.get("mae"),
            "event_mae": reg_metrics.get("event_mae"),
            "event_baseline_mae": reg_metrics.get("event_baseline_mae"),
            "event_regime": reg_metrics.get("event_regime"),
            "beats_overall": reg_metrics.get("beats_baseline_overall"),
        })
    horizons.sort(key=lambda h: h["horizon_h"])

    probabilities = []
    for c in ml.get("classification") or []:
        h, thr = c.get("horizon_h"), c.get("threshold")
        prob = c.get("probability")
        if h is None or prob is None:
            continue
        pct = int(max(0.0, min(1.0, prob)) * 100)
        key = f"classification_thr{thr}_h{h}"
        m = metrics.get(key) or {}
        probabilities.append({
            "horizon_h": h,
            "threshold": thr,
            "threshold_label": _threshold_label(thr, stages),
            "pct": pct,
            "severity": "high" if pct >= 50 else "med" if pct >= 25 else "low",
            "color": ("#ef5350" if pct >= 50
                      else "#fb8c00" if pct >= 25 else "#7cb342"),
            "auc": m.get("auc"),
            "positive_events": m.get("positive_events"),
            "folds_with_events": m.get("folds_with_events"),
            "n_folds": m.get("n_folds"),
            "trustworthy": bool(m.get("trustworthy")),
        })
    probabilities.sort(key=lambda p: (p["horizon_h"], p["threshold"] or 0))

    # The stage regression is withheld when it loses to persistence, which it
    # currently does at every horizon. The card then carries exceedance
    # probabilities only, and says why the forecast is absent rather than
    # silently dropping a section the reader saw yesterday.
    withheld = [k.replace("regression_h", "") for k, m in metrics.items()
                if k.startswith("regression_h") and m.get("beats_baseline") is False]

    return {
        "site": site,
        "bands": bands,
        "axis_max": axis_max,
        "horizons": horizons,
        "probabilities": probabilities,
        "any_untrustworthy": any(not p["trustworthy"] for p in probabilities),
        "trained_ts": ml.get("trained_ts") or _first_trained(ml),
        "withheld_regression_horizons": sorted(withheld, key=lambda h: int(h)),
        "baseline_mae": {
            k.replace("regression_h", ""): m.get("baseline_mae")
            for k, m in metrics.items() if k.startswith("regression_h")
        },
        "current_ft": current,
    }


def _threshold_label(threshold, stages: dict) -> str:
    """Name the flood category a threshold corresponds to, if it is one."""
    if threshold is None:
        return ""
    for level in ("action", "minor", "moderate", "major"):
        v = stages.get(level)
        if v is not None and abs(float(v) - float(threshold)) < 0.01:
            return level
    return ""


def _first_trained(ml: dict):
    for group in ("regression", "classification"):
        for item in ml.get(group) or []:
            if item.get("trained_ts"):
                return item["trained_ts"]
    return None


# ------------------------------------------------------------- heat / charts

@dataclass
class MultiLine:
    """Several series scaled into one shared viewBox."""
    series: list = field(default_factory=list)
    vmin: float = 0.0
    vmax: float = 1.0
    width: float = 200.0
    height: float = 52.0
    ticks: list = field(default_factory=list)
    reference_lines: list = field(default_factory=list)


def heat_chart(weather: dict, *, width: float = 200.0, height: float = 52.0,
               references=((90, "#ef6c00"), (80, "#f9a825"))) -> MultiLine | None:
    """Temperature / feels-like / wet-bulb over the next 24 h."""
    if not weather or weather.get("error"):
        return None
    specs = [
        ("actual", weather.get("hourly_temp_f"), "#4fc3f7", None),
        ("apparent", weather.get("hourly_apparent_f"), "#ff8a65", "4,2"),
        ("wet_bulb", weather.get("hourly_wet_bulb_f"), "#69f0ae", "2,3"),
    ]
    present = [(name, vals, color, dash) for name, vals, color, dash in specs
               if vals and len(vals) > 1]
    if not present:
        return None

    everything = [v for _n, vals, _c, _d in present for v in vals
                  if isinstance(v, (int, float))]
    if not everything:
        return None
    vmin, vmax = min(everything) - 1, max(everything) + 1
    vrng = (vmax - vmin) or 0.1
    plot_h = height - 2

    def scale(vals):
        n = len(vals)
        return " ".join(
            f"{(i / (n - 1)) * width:.1f},"
            f"{plot_h - ((v - vmin) / vrng) * (plot_h - 2):.1f}"
            for i, v in enumerate(vals))

    series = [{"name": name, "points": scale(vals), "color": color, "dash": dash}
              for name, vals, color, dash in present]

    n = len(present[0][1])
    times = weather.get("hourly_times") or []
    idxs = [i for i in (0, 6, 12, 18) if i < n]
    if (n - 1) not in idxs and n > 1:
        idxs.append(n - 1)
    ticks = []
    for i in idxs:
        label = times[i][-5:-3] if i < len(times) and times[i] else str(i)
        ticks.append({"x_pct": (i / (n - 1)) * 100 if n > 1 else 0,
                      "label": f"{label}h",
                      "anchor": "start" if i == 0
                                else "end" if i >= n - 2 else "middle"})

    ref = [{"value": v, "color": c,
            "y": plot_h - ((v - vmin) / vrng) * (plot_h - 2)}
           for v, c in references if vmin < v < vmax]

    return MultiLine(series=series, vmin=vmin, vmax=vmax, width=width,
                     height=height, ticks=ticks, reference_lines=ref)


def scale_marker(value, lo: float, span: float, width: float = 200.0) -> float | None:
    """X position of a marker on a coloured category scale, clamped inside."""
    if value is None or not span:
        return None
    x = (float(value) - lo) / span * width
    return max(1.0, min(width - 1.0, x))


HEAT_INDEX_BANDS = (
    ("#2e7d32", 60, 80, "Normal"),
    ("#f9a825", 80, 90, "Caution"),
    ("#ef6c00", 90, 103, "Ext. Caution"),
    ("#c62828", 103, 124, "Danger"),
    ("#6a1b9a", 124, 130, "Extreme"),
)

WET_BULB_BANDS = (
    ("#2e7d32", 55, 65, "Low Risk"),
    ("#f9a825", 65, 72, "Caution"),
    ("#ef6c00", 72, 78, "High Risk"),
    ("#c62828", 78, 83, "Danger"),
    ("#6a1b9a", 83, 90, "Extreme"),
)


def category_scale(value, bands, *, lo: float, hi: float,
                   width: float = 200.0) -> dict:
    """Coloured category strip with a marker for the current value."""
    span = hi - lo
    out = []
    for color, band_lo, band_hi, label in bands:
        x1 = (band_lo - lo) / span * width
        x2 = (min(band_hi, hi) - lo) / span * width
        out.append({"color": color, "label": label, "x1": x1,
                    "width": x2 - x1, "mid": (x1 + x2) / 2})
    marker = scale_marker(value, lo, span, width)
    anchor = "middle"
    if marker is not None:
        anchor = ("start" if marker < width * 0.125
                  else "end" if marker > width * 0.875 else "middle")
    return {"bands": out, "marker": marker, "anchor": anchor}


def wet_bulb_color(value) -> str:
    """Danger colour for a wet-bulb reading; ≥78 °F is the illness zone."""
    if value is None:
        return "#2e7d32"
    for color, _lo, hi, _label in WET_BULB_BANDS:
        if value < hi:
            return color
    return "#6a1b9a"


# ---------------------------------------------------------------- freshness

# The GitHub Pages workflow rebuilds on a "17 * * * *" cron, so a healthy page
# is at most ~1 hour old. Anything beyond that means a build was missed, and
# the indicator should stop claiming the data is live.
REBUILD_INTERVAL_MINUTES = 60
_FRESHNESS_BANDS = (
    (75, "fresh", "Live"),
    (150, "aging", "One rebuild missed"),
    (360, "stale", "Several rebuilds missed"),
)


def freshness(state: dict, *, now_epoch: float | None = None,
              interval_minutes: int = REBUILD_INTERVAL_MINUTES) -> dict:
    """How old the data is, and whether the page may still call itself live.

    The old header pulsed a green "live" dot on a page GitHub Actions rebuilds
    hourly, so a build that silently stopped looked identical to a healthy one.
    """
    import time as _time

    as_of = state.get("as_of_epoch")
    now = now_epoch if now_epoch is not None else _time.time()
    if not isinstance(as_of, (int, float)) or as_of <= 0:
        return {"level": "frozen", "label": "Unknown age", "age_minutes": None,
                "as_of_epoch": None, "interval_minutes": interval_minutes,
                "cadence": f"rebuilt every {interval_minutes} min"}

    age_minutes = max(0.0, (now - as_of) / 60.0)
    level, label = "frozen", "Stale — build may have failed"
    for limit, lvl, lbl in _FRESHNESS_BANDS:
        if age_minutes < limit:
            level, label = lvl, lbl
            break
    return {
        "level": level,
        "label": label,
        "age_minutes": round(age_minutes, 1),
        "as_of_epoch": int(as_of),
        "interval_minutes": interval_minutes,
        "cadence": f"rebuilt every {interval_minutes} min",
    }


# ------------------------------------------------------------ rainfall (QPF)

def qpf_chart(weather: dict, *, width: float = 300.0,
              height: float = 90.0) -> dict | None:
    """Hourly rainfall bars plus a cumulative curve.

    A single "next 72 h" total cannot distinguish a wet week from a flash
    flood. The bars show when it lands; the curve shows the running total.
    """
    if not weather or weather.get("error"):
        return None
    series = weather.get("hourly_precip_in") or []
    series = [v for v in series if isinstance(v, (int, float))]
    if not series or sum(series) <= 0:
        return None

    n = len(series)
    peak = max(series) or 0.01
    bar_w = width / n
    bars = []
    for i, v in enumerate(series):
        h = (v / peak) * (height - 2)
        # Thresholds are per-hour intensities that matter in the Blue Ridge.
        cls = "extreme" if v >= 0.5 else "heavy" if v >= 0.25 else ""
        bars.append({"x": i * bar_w, "y": height - h, "w": max(0.6, bar_w * 0.8),
                     "h": h, "cls": cls, "hour": i, "inches": round(v, 2)})

    total = sum(series)
    running = 0.0
    cum_pts = []
    for i, v in enumerate(series):
        running += v
        cum_pts.append(f"{i * bar_w + bar_w / 2:.1f},"
                       f"{height - (running / total) * (height - 2):.1f}")

    times = weather.get("hourly_precip_times") or []
    ticks = []
    for i in range(0, n, 12):
        label = times[i][-5:] if i < len(times) and times[i] else f"+{i}h"
        ticks.append({"x_pct": (i / max(1, n - 1)) * 100, "label": label})

    return {
        "bars": bars,
        "cumulative": " ".join(cum_pts),
        "width": width,
        "height": height,
        "total_in": round(total, 2),
        "peak_hour_in": round(peak, 2),
        "max_6h_in": weather.get("max_6h_precip_in"),
        "max_24h_in": weather.get("max_24h_precip_in"),
        "ticks": ticks,
        "hours": n,
    }


# ----------------------------------------------------------------- soil card

def soil_card(soil: dict) -> dict | None:
    if not soil or soil.get("error"):
        return None
    moisture = (soil.get("soil_moisture_top")
                or soil.get("soil_moisture_shallow") or 0)
    saturated = bool(soil.get("saturated"))
    if saturated:
        color, amplifier = "#c62828", 1.25
    elif moisture >= 0.30:
        color, amplifier = "#ff8a65", 1.10
    elif moisture >= 0.20:
        color, amplifier = "#9e9d24", 1.00
    else:
        color, amplifier = "#4caf50", 1.00
    return {
        "moisture": moisture,
        "saturation_pct": min(100.0, moisture / 0.45 * 100.0),
        "color": color,
        "amplifier": amplifier,
        "saturated": saturated,
    }
