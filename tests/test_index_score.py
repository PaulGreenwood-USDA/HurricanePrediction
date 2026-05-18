from hurricane_asheville.index_score import compute_index


def _idx(**overrides):
    """Helper: build an index call with sensible defaults."""
    kwargs = dict(
        primary_gauge=None,
        rate_ft_per_hr=0.0,
        storms=[],
        alerts=[],
        weather={"next_72h_precip_in": 0.0},
        soil={"soil_moisture_top": 0.10, "past_7d_precip_in": 0.0,
              "saturated": False},
    )
    kwargs.update(overrides)
    return compute_index(**kwargs)


def test_baseline_calm_score():
    r = _idx()
    assert r.score < 10
    assert r.label == "CALM"
    assert r.color.startswith("#")


def test_high_stage_drives_score_up():
    r = _idx(primary_gauge={"stage_ft": 16.0})  # major
    assert r.components["stage"] >= 30
    assert r.score >= 30


def test_qpf_component():
    r = _idx(weather={"next_72h_precip_in": 5.0})
    assert r.components["qpf"] >= 18
    assert r.triggers["qpf_over_3in"] is True


def test_storm_component_decays_with_distance():
    near = _idx(storms=[{"distance_mi": 100}])
    far = _idx(storms=[{"distance_mi": 1400}])
    nope = _idx(storms=[{"distance_mi": 5000}])
    assert near.components["storm"] > far.components["storm"] > nope.components["storm"]
    assert near.triggers["storm_within_500mi"] is True
    assert far.triggers["storm_within_500mi"] is False


def test_rate_component_caps_at_10():
    r = _idx(rate_ft_per_hr=2.0)
    assert r.components["rise"] == 10.0
    assert r.triggers["river_rising_fast"] is True


def test_flood_alert_triggers_component():
    r = _idx(alerts=[{"event": "Flash Flood Warning"}])
    assert r.components["alert"] == 10.0
    assert r.triggers["nws_flood_or_tropical"] is True


def test_non_relevant_alert_is_ignored():
    r = _idx(alerts=[{"event": "Air Quality Advisory"}])
    assert r.components["alert"] == 0.0


def test_saturated_soil_amplifies_score():
    base = _idx(weather={"next_72h_precip_in": 3.0})
    sat = _idx(weather={"next_72h_precip_in": 3.0},
               soil={"soil_moisture_top": 0.45, "past_7d_precip_in": 5.0,
                     "saturated": True})
    assert sat.score > base.score
    assert sat.triggers["soil_saturated"] is True
    assert sat.triggers["wet_week"] is True


def test_score_caps_at_100():
    r = _idx(primary_gauge={"stage_ft": 30.0},
             rate_ft_per_hr=5.0,
             storms=[{"distance_mi": 50}],
             alerts=[{"event": "Hurricane Warning"}],
             weather={"next_72h_precip_in": 15.0},
             soil={"soil_moisture_top": 0.50, "past_7d_precip_in": 10.0,
                   "saturated": True})
    assert r.score == 100
    assert r.label == "EMERGENCY"


def test_label_thresholds():
    # The label is monotone in score
    labels_seen = []
    for stage in [0, 4, 8, 12, 16]:
        r = _idx(primary_gauge={"stage_ft": stage},
                 weather={"next_72h_precip_in": 4.0},
                 storms=[{"distance_mi": 200}])
        labels_seen.append((r.score, r.label))
    scores = [s for s, _ in labels_seen]
    assert scores == sorted(scores)
