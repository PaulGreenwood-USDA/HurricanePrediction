"""Asheville Flood Index (AFI): single 0-100 composite from live signals.

Designed so each driver is independently visible in the breakdown so the
dashboard can show *why* the score moved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .gauge import FLOOD_STAGES_FT


@dataclass
class IndexResult:
    score: int            # 0-100
    label: str            # CALM/ELEVATED/ALERT/WARNING/EMERGENCY
    color: str
    components: dict = field(default_factory=dict)
    triggers: dict = field(default_factory=dict)  # boolean threshold lights


def _label_color(score: int) -> tuple[str, str]:
    if score >= 80:
        return "EMERGENCY", "#6a1b9a"
    if score >= 60:
        return "WARNING", "#c62828"
    if score >= 40:
        return "ALERT", "#ef6c00"
    if score >= 20:
        return "ELEVATED", "#9e9d24"
    return "CALM", "#2e7d32"


def compute_index(*, primary_gauge: dict | None,
                  rate_ft_per_hr: float | None,
                  storms: list[dict],
                  alerts: list[dict],
                  weather: dict | None,
                  soil: dict | None = None) -> IndexResult:
    """Aggregate live signals into a 0-100 score.

    Weights (max contribution, sums to 100):
      stage     35   - current river stage vs. major flood
      qpf       20   - 72h forecast precipitation at Asheville
      storm     20   - nearest active Atlantic TC distance
      rise      10   - rate-of-rise on French Broad
      alert     10   - active NWS tropical/flood alerts
      soil       5   - soil saturation (antecedent moisture)
    Score is also multiplied 1.0..1.25 by a soil-saturation amplifier so the
    same rain on soaked ground reads higher.
    """
    comps = {}
    trig = {}

    # 1. Stage: 0..35
    stage_ft = (primary_gauge or {}).get("stage_ft")
    if stage_ft is not None:
        denom = FLOOD_STAGES_FT["major"]
        comps["stage"] = round(min(35.0, max(0.0, stage_ft / denom * 35.0)), 1)
        trig["stage_above_action"] = stage_ft >= FLOOD_STAGES_FT["action"]
        trig["stage_above_minor"]  = stage_ft >= FLOOD_STAGES_FT["minor"]
    else:
        comps["stage"] = 0.0
        trig["stage_above_action"] = False
        trig["stage_above_minor"]  = False

    # 2. QPF: 0..20 over 0..5"
    qpf = (weather or {}).get("next_72h_precip_in") or 0.0
    comps["qpf"] = round(min(20.0, qpf / 5.0 * 20.0), 1)
    trig["qpf_over_3in"] = qpf >= 3.0
    trig["qpf_over_1in"] = qpf >= 1.0

    # 3. Storm distance: 0..20
    storm_score = 0.0
    nearest_mi = None
    if storms:
        nearest_mi = min(s.get("distance_mi", 9e9) for s in storms)
        if nearest_mi < 1500:
            storm_score = max(0.0, 20.0 * (1500.0 - nearest_mi) / 1300.0)
            storm_score = min(20.0, storm_score)
    comps["storm"] = round(storm_score, 1)
    trig["storm_within_500mi"]  = nearest_mi is not None and nearest_mi < 500
    trig["storm_within_1000mi"] = nearest_mi is not None and nearest_mi < 1000

    # 4. Rate of rise: 0..10
    rate = rate_ft_per_hr or 0.0
    comps["rise"] = round(min(10.0, max(0.0, rate * 10.0)), 1)
    trig["river_rising_fast"] = rate >= 0.3

    # 5. Alerts: 0..10
    alert_hit = any(
        any(k in (a.get("event") or "") for k in ("Tropical", "Hurricane", "Flood"))
        for a in alerts
    )
    comps["alert"] = 10.0 if alert_hit else 0.0
    trig["nws_flood_or_tropical"] = alert_hit

    # 6. Soil: 0..5 + amplifier
    soil_amp = 1.0
    if soil and not soil.get("error"):
        sm = soil.get("soil_moisture_top") or soil.get("soil_moisture_shallow")
        past = soil.get("past_7d_precip_in") or 0.0
        soil_score = 0.0
        if sm is not None:
            # 0 at sm=0.15 (dry), 5 at sm=0.45 (saturated)
            soil_score = max(0.0, min(5.0, (sm - 0.15) / 0.30 * 5.0))
        # Bonus if a lot of rain has already fallen this week
        soil_score = max(soil_score, min(5.0, past / 4.0 * 5.0))
        comps["soil"] = round(soil_score, 1)
        if soil.get("saturated"):
            soil_amp = 1.25
        elif sm is not None and sm >= 0.30:
            soil_amp = 1.10
        trig["soil_saturated"] = bool(soil.get("saturated"))
        trig["wet_week"] = past >= 3.0
    else:
        comps["soil"] = 0.0
        trig["soil_saturated"] = False
        trig["wet_week"] = False

    raw = sum(comps.values()) * soil_amp
    score = int(round(min(100.0, raw)))
    label, color = _label_color(score)
    return IndexResult(score=score, label=label, color=color,
                       components=comps, triggers=trig)
