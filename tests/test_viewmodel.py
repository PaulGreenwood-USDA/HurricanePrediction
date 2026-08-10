"""Tests for the render-ready view model.

This maths used to live inside the Jinja template, where a divide-by-zero
surfaced as a silently blank card rather than a failing test.
"""
from __future__ import annotations

import pytest

from hurricane_asheville import viewmodel as vm


# ---- sparklines -----------------------------------------------------------

def test_sparkline_needs_two_points():
    assert vm.sparkline([]) is None
    assert vm.sparkline([1.0]) is None


def test_sparkline_scales_into_viewbox():
    s = vm.sparkline([0.0, 5.0, 10.0], width=100, height=30)
    xs, ys = zip(*(map(float, p.split(",")) for p in s.points.split()))
    assert min(xs) == 0.0 and max(xs) == 100.0
    assert all(0 <= y <= 30 for y in ys)
    # Higher stage must sit higher on screen, i.e. smaller y.
    assert ys[2] < ys[0]


def test_sparkline_flat_series_does_not_divide_by_zero():
    s = vm.sparkline([3.0, 3.0, 3.0])
    assert s is not None
    assert all(p.count(",") == 1 for p in s.points.split())


def test_sparkline_ignores_non_numeric():
    s = vm.sparkline([1.0, None, 2.0, "x", 3.0])
    assert len(s.points.split()) == 3


# ---- triggers -------------------------------------------------------------

def test_triggers_put_fired_first():
    idx = {"triggers": {"soil_saturated": True, "qpf_over_1in": False,
                        "stage_above_action": True}}
    out = vm.triggers(idx)
    fired = [t["label"] for t in out if t["on"]]
    assert out[0]["on"] and out[1]["on"]
    assert len(fired) == 2


def test_triggers_handle_missing_index():
    out = vm.triggers({})
    assert len(out) == len(vm.TRIGGER_LABELS)
    assert not any(t["on"] for t in out)


# ---- flood index dial -----------------------------------------------------

def test_dial_dash_scales_with_score():
    assert vm.index_dial({"score": 0})["dash"].startswith("0.0 ")
    full = vm.index_dial({"score": 100})["dash"]
    assert full.startswith("326.0 ")


def test_dial_clamps_out_of_range_score():
    assert vm.index_dial({"score": 250})["dash"].startswith("326.0 ")
    assert vm.index_dial({"score": -5})["dash"].startswith("0.0 ")


def test_dial_survives_garbage_score():
    assert vm.index_dial({"score": "n/a"})["dash"].startswith("0.0 ")


# ---- gauge networks -------------------------------------------------------

def test_gauge_networks_counts_by_role():
    gauges = [{"role": "primary"}, {"role": "regional"},
              {"role": "regional"}, {"role": "reservoir"}]
    nets = vm.gauge_networks(gauges)
    counts = {t["key"]: t["count"] for t in nets["tabs"]}
    assert counts["all"] == 4
    assert counts["regional"] == 2
    assert counts["reservoirs"] == 1


def test_gauge_networks_omits_empty_tabs():
    nets = vm.gauge_networks([{"role": "primary"}])
    assert "reservoirs" not in {t["key"] for t in nets["tabs"]}


# ---- freshness ------------------------------------------------------------

def test_freshness_fresh_within_one_rebuild():
    f = vm.freshness({"as_of_epoch": 1000}, now_epoch=1000 + 30 * 60)
    assert f["level"] == "fresh"
    assert f["age_minutes"] == 30.0


def test_freshness_degrades_as_builds_are_missed():
    base = 1000
    assert vm.freshness({"as_of_epoch": base},
                        now_epoch=base + 90 * 60)["level"] == "aging"
    assert vm.freshness({"as_of_epoch": base},
                        now_epoch=base + 200 * 60)["level"] == "stale"
    assert vm.freshness({"as_of_epoch": base},
                        now_epoch=base + 600 * 60)["level"] == "frozen"


def test_freshness_never_claims_live_without_a_timestamp():
    f = vm.freshness({})
    assert f["level"] == "frozen"
    assert f["age_minutes"] is None


def test_freshness_states_the_cadence():
    assert "60" in vm.freshness({"as_of_epoch": 1}, now_epoch=2)["cadence"]


# ---- ML card --------------------------------------------------------------

def _ml_state(**over):
    state = {
        "primary_site": "X",
        "gauges": [{"site_id": "X", "stage_ft": 2.0}],
        "flood_stages": {"action": 6.5, "minor": 9.5,
                         "moderate": 13.0, "major": 18.0},
        "ml_forecasts": {"X": {
            "regression": [{"horizon_h": 24, "predicted_stage_ft": 2.5}],
            "classification": [
                {"horizon_h": 24, "threshold": 6.5, "probability": 0.1},
                {"horizon_h": 24, "threshold": 9.5, "probability": 0.0},
            ],
        }},
        "ml_metrics": {
            "regression_h24": {"mae": 0.21},
            "classification_thr6.5_h24": {"auc": 0.99, "trustworthy": True,
                                          "positive_events": 214,
                                          "folds_with_events": 2, "n_folds": 5},
            "classification_thr9.5_h24": {"auc": None, "trustworthy": False,
                                          "positive_events": 119,
                                          "folds_with_events": 1, "n_folds": 5},
        },
    }
    state.update(over)
    return state


def test_ml_card_none_without_forecasts():
    assert vm.ml_card({"primary_site": "X"}) is None
    assert vm.ml_card(_ml_state(ml_forecasts={"X": {"regression": [],
                                                    "classification": []}})) is None


def test_ml_card_joins_backtest_error_to_horizon():
    card = vm.ml_card(_ml_state())
    assert card["horizons"][0]["mae"] == 0.21
    assert card["horizons"][0]["delta"] == pytest.approx(0.5)
    assert card["horizons"][0]["direction"] == "up"


def test_ml_card_labels_thresholds_with_flood_category():
    card = vm.ml_card(_ml_state())
    labels = {p["threshold"]: p["threshold_label"] for p in card["probabilities"]}
    assert labels[6.5] == "action"
    assert labels[9.5] == "minor"


def test_ml_card_flags_uncalibrated_probabilities():
    """A classifier whose positive class appears in one fold is untested,
    and a confident-looking 0% must not imply flooding is unlikely."""
    card = vm.ml_card(_ml_state())
    by_thr = {p["threshold"]: p for p in card["probabilities"]}
    assert by_thr[6.5]["trustworthy"] is True
    assert by_thr[9.5]["trustworthy"] is False
    assert card["any_untrustworthy"] is True


def test_ml_card_bands_are_ordered_and_bounded():
    card = vm.ml_card(_ml_state())
    lefts = [b["left"] for b in card["bands"]]
    assert lefts == sorted(lefts)
    assert all(0 <= b["left"] <= 100 for b in card["bands"])
    assert all(b["width"] >= 0 for b in card["bands"])


def test_ml_card_handles_missing_current_stage():
    card = vm.ml_card(_ml_state(gauges=[{"site_id": "X", "stage_ft": None}]))
    assert card["horizons"][0]["delta"] is None


# ---- QPF ------------------------------------------------------------------

def test_qpf_none_when_dry():
    assert vm.qpf_chart({"hourly_precip_in": [0.0] * 24}) is None
    assert vm.qpf_chart({}) is None
    assert vm.qpf_chart({"error": "down"}) is None


def test_qpf_bars_and_cumulative():
    series = [0.0] * 10 + [0.6] + [0.1] * 5
    q = vm.qpf_chart({"hourly_precip_in": series, "max_6h_precip_in": 1.0,
                      "max_24h_precip_in": 1.1})
    assert len(q["bars"]) == len(series)
    assert q["total_in"] == pytest.approx(1.1, abs=0.01)
    # The heaviest hour is flagged so a burst is visible, not averaged away.
    assert q["bars"][10]["cls"] == "extreme"
    assert q["bars"][11]["cls"] == ""
    assert len(q["cumulative"].split()) == len(series)


# ---- scales ---------------------------------------------------------------

def test_category_scale_marker_clamped_inside():
    low = vm.category_scale(-100, vm.HEAT_INDEX_BANDS, lo=60, hi=130)
    high = vm.category_scale(999, vm.HEAT_INDEX_BANDS, lo=60, hi=130)
    assert low["marker"] == 1.0
    assert high["marker"] == 199.0


def test_category_scale_bands_tile_the_axis():
    scale = vm.category_scale(80, vm.HEAT_INDEX_BANDS, lo=60, hi=130)
    assert scale["bands"][0]["x1"] == 0
    assert sum(b["width"] for b in scale["bands"]) == pytest.approx(200, abs=1)


def test_wet_bulb_color_crosses_danger_at_78():
    assert vm.wet_bulb_color(70) != vm.wet_bulb_color(80)
    assert vm.wet_bulb_color(None) == "#2e7d32"


# ---- soil -----------------------------------------------------------------

def test_soil_card_amplifier_matches_saturation():
    assert vm.soil_card({"soil_moisture_top": 0.45,
                         "saturated": True})["amplifier"] == 1.25
    assert vm.soil_card({"soil_moisture_top": 0.32})["amplifier"] == 1.10
    assert vm.soil_card({"soil_moisture_top": 0.10})["amplifier"] == 1.00


def test_soil_card_saturation_never_exceeds_full_bar():
    assert vm.soil_card({"soil_moisture_top": 0.9})["saturation_pct"] == 100.0


def test_soil_card_none_on_error():
    assert vm.soil_card({"error": "unavailable"}) is None
