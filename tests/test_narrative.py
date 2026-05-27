"""Tests for the narrative TL;DR module."""
from __future__ import annotations

from hurricane_asheville.narrative import GLOSSARY, summarize


def _state(label="CALM", score=10, triggers=None, **extra):
    base = {
        "index": {
            "label": label,
            "score": score,
            "triggers": triggers or {f"t{i}": False for i in range(10)},
        },
        "primary_site": "03451500",
        "gauges": [],
        "weather": {},
        "storms": [],
        "soil": {},
        "alerts": [],
        "ml_forecasts": {},
    }
    base.update(extra)
    return base


def test_calm_headline_and_recommendation():
    s = _state(
        label="CALM", score=8,
        gauges=[{"site_id": "03451500", "stage_ft": 4.4,
                 "flood_category": "Normal", "rate_ft_per_hr": 0.0}],
    )
    out = summarize(s)
    assert out["level"] == "calm"
    assert out["color"] == "#2e7d32"
    assert "All clear" in out["headline"]
    assert "4.4 ft" in out["headline"]
    assert out["recommendation"].lower().startswith("normal")
    assert "Flood Index 8/100" in out["subheadline"]


def test_emergency_headline():
    s = _state(label="EMERGENCY", score=92,
               triggers={"a": True, "b": True, "c": True})
    out = summarize(s)
    assert out["level"] == "emergency"
    assert out["color"] == "#6a1b9a"
    assert "emergency" in out["headline"].lower()
    assert "Active flood emergency" in out["recommendation"]
    assert "3 of" in out["subheadline"]


def test_warning_with_flood_category():
    s = _state(
        label="WARNING", score=65,
        gauges=[{"site_id": "03451500", "stage_ft": 14.2,
                 "flood_category": "Minor", "rate_ft_per_hr": 0.4}],
    )
    out = summarize(s)
    assert out["level"] == "warning"
    assert "flood" in out["headline"].lower()


def test_alert_with_heavy_rain():
    s = _state(
        label="ALERT", score=45,
        weather={"next_72h_precip_in": 4.2},
    )
    out = summarize(s)
    assert out["level"] == "alert"
    assert "4.2" in out["headline"]


def test_key_facts_include_stage_and_ml():
    s = _state(
        label="ELEVATED", score=25,
        gauges=[{"site_id": "03451500", "stage_ft": 6.1,
                 "flood_category": "Normal", "rate_ft_per_hr": 0.15}],
        ml_forecasts={"03451500": {
            "regression": [{"horizon_h": 24, "predicted_stage_ft": 7.8}],
            "classification": [
                {"horizon_h": 24, "threshold": 8.0, "probability": 0.42},
            ],
        }},
    )
    out = summarize(s)
    facts = " | ".join(out["key_facts"])
    assert "6.1 ft" in facts
    assert "rising" in facts
    assert "7.8 ft" in facts
    assert "42%" in facts
    assert len(out["key_facts"]) <= 4


def test_low_probability_classification_excluded():
    s = _state(
        label="CALM", score=5,
        ml_forecasts={"03451500": {
            "regression": [],
            "classification": [
                {"horizon_h": 24, "threshold": 8.0, "probability": 0.10},
            ],
        }},
    )
    out = summarize(s)
    assert all("%" not in f for f in out["key_facts"])


def test_nws_flood_alert_surfaces():
    s = _state(
        label="ALERT", score=50,
        alerts=[{"event": "Flood Watch"}],
    )
    out = summarize(s)
    assert any("Flood Watch" in f for f in out["key_facts"])


def test_storm_distance_fact():
    s = _state(
        label="ALERT", score=45,
        storms=[
            {"name": "Helene", "distance_to_asheville_mi": 420},
            {"name": "Farsy", "distance_to_asheville_mi": 3000},
        ],
    )
    out = summarize(s)
    facts = " | ".join(out["key_facts"])
    assert "Helene" in facts
    assert "420" in facts
    assert "Farsy" not in facts  # too far away


def test_glossary_has_expected_keys():
    expected = {"Flood Index", "QPF", "stage", "action stage",
                "minor flood", "moderate flood", "major flood",
                "discharge", "rate", "NWPS forecast", "ML forecast",
                "backtest", "MAE", "AUC", "soil saturated", "TC", "ETA"}
    missing = expected - set(GLOSSARY)
    assert not missing, f"glossary missing: {missing}"
    # Every entry has a non-trivial definition.
    for k, v in GLOSSARY.items():
        assert isinstance(v, str) and len(v) > 10, k


def test_unknown_label_falls_back_to_calm():
    s = _state(label="BOGUS", score=0)
    out = summarize(s)
    assert out["level"] == "calm"
