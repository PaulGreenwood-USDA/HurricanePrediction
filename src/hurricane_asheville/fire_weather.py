"""Fire-weather hazard for the NC National Forests.

Combines two free, no-auth sources:

1. **NWS active alerts** (`api.weather.gov/alerts/active?point=lat,lon`) —
   filter for Red Flag Warning / Fire Weather Watch events. This is the
   authoritative federal "today's ignition risk" signal.

2. **Open-Meteo current** — relative humidity, 10-m wind, temperature. Used
   to compute a simple Fosberg-like Fire Weather Index (FWI) for situations
   where there is no active alert but conditions are still dangerous.

We return a `compute_fire_weather` summary with a CALM / ELEVATED / HIGH /
EXTREME label and color that mirrors the landslide hazard pill so the UI
can render them side-by-side.

Note: the simplified index used here is a transparent approximation of the
Fosberg Fire Weather Index and is *not* a substitute for NFDRS / NWS Spot
Forecasts. It is intended as situational context only.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


_FW_LEVELS = (
    ("EXTREME",  75, "#6a1b9a"),
    ("HIGH",     55, "#c62828"),
    ("ELEVATED", 30, "#ef6c00"),
    ("CALM",      0, "#4caf50"),
)


def _classify(score: float) -> tuple[str, str]:
    for label, threshold, color in _FW_LEVELS:
        if score >= threshold:
            return label, color
    return "CALM", "#4caf50"


def _fuel_moisture_pct(temp_f: float, rh_pct: float) -> float:
    """Equilibrium-moisture-content approximation used in the Fosberg index
    (Simard 1968 piecewise form). Returns moisture % of 1-h fuels."""
    rh = max(0.0, min(100.0, rh_pct))
    if rh < 10.0:
        m = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp_f
    elif rh < 50.0:
        m = 2.22749 + 0.160107 * rh - 0.014784 * temp_f
    else:
        m = (21.0606 + 0.005565 * rh * rh
             - 0.00035 * rh * temp_f - 0.483199 * rh)
    return max(1.0, min(40.0, m))


def fosberg_fwi(temp_f: float | None, rh_pct: float | None,
                wind_mph: float | None) -> float | None:
    """Compute the Fosberg Fire Weather Index (0–100). Returns None if any
    input is missing."""
    if temp_f is None or rh_pct is None or wind_mph is None:
        return None
    try:
        m = _fuel_moisture_pct(float(temp_f), float(rh_pct))
        # Normalize fuel moisture to a damping coefficient (0–1)
        n = 1.0 - 2.0 * (m / 30.0) + 1.5 * (m / 30.0) ** 2 - 0.5 * (m / 30.0) ** 3
        n = max(0.0, min(1.0, n))
        u = float(wind_mph)
        fwi = (n * math.sqrt(1.0 + u * u)) / 0.3002
        return round(max(0.0, min(100.0, fwi)), 1)
    except (TypeError, ValueError):
        return None


def is_fire_weather_alert(alert: dict) -> bool:
    event = (alert.get("event") or "").lower()
    return ("red flag" in event) or ("fire weather" in event)


def compute_fire_weather(weather: dict | None,
                          alerts: list[dict] | None,
                          region: str = "mountain") -> dict:
    """Combine forecast conditions + active fire-weather alerts into a single
    hazard summary for one forest centroid.

    ``region`` lets us very slightly bias the score: WNC mountain fuels are
    typically slower to dry than coastal fuels, so we cap coastal slightly
    lower in the absence of an alert (alerts always override)."""
    weather = weather if isinstance(weather, dict) and not weather.get("error") else {}
    alerts = alerts or []

    fw_alerts = [a for a in alerts if is_fire_weather_alert(a)]
    fwi = fosberg_fwi(weather.get("temp_f"),
                      weather.get("humidity_pct"),
                      weather.get("wind_mph"))

    score = fwi or 0.0
    if region == "coastal" and not fw_alerts:
        score = min(score, 65.0)

    # An active Red Flag dominates the visual: pin minimum HIGH.
    has_red_flag = any("red flag" in (a.get("event") or "").lower()
                       for a in fw_alerts)
    if has_red_flag:
        score = max(score, 70.0)
    elif fw_alerts:  # Fire Weather Watch only
        score = max(score, 45.0)

    label, color = _classify(score)

    return {
        "score": round(score, 1),
        "label": label,
        "color": color,
        "fwi": fwi,
        "drivers": {
            "temp_f": weather.get("temp_f"),
            "humidity_pct": weather.get("humidity_pct"),
            "wind_mph": weather.get("wind_mph"),
        },
        "active_alerts": [
            {"event": a.get("event"), "severity": a.get("severity"),
             "headline": a.get("headline")}
            for a in fw_alerts
        ],
        "explain": _explain(fwi, fw_alerts, weather),
    }


def _explain(fwi, fw_alerts, weather):
    if fw_alerts:
        events = ", ".join(sorted({a.get("event") or "alert" for a in fw_alerts}))
        return f"Active {events}."
    if fwi is None:
        return "Weather feed unavailable; no fire-weather index computed."
    rh = weather.get("humidity_pct")
    wind = weather.get("wind_mph")
    if rh is not None and wind is not None:
        return (f"FWI {fwi} (RH {rh}%, wind {wind} mph). "
                f"No active NWS fire-weather alert.")
    return f"FWI {fwi}. No active NWS fire-weather alert."
