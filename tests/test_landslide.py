from hurricane_asheville.landslide import (compute_landslide_hazard,
                                            summarize_inventory)


def test_dry_mountain_is_calm_or_elevated():
    h = compute_landslide_hazard(
        "mountain",
        soil={"soil_moisture_top": 0.15, "past_7d_precip_in": 0.2, "saturated": False},
        weather={"next_72h_precip_in": 0.0},
    )
    assert h["label"] in {"CALM", "ELEVATED"}
    assert h["score"] < 35


def test_helene_like_mountain_is_extreme():
    h = compute_landslide_hazard(
        "mountain",
        soil={"soil_moisture_top": 0.45, "past_7d_precip_in": 9.0, "saturated": True},
        weather={"next_72h_precip_in": 8.0},
    )
    assert h["label"] == "EXTREME"
    assert h["score"] >= 80


def test_coastal_is_capped():
    """Croatan is flat – even Helene-magnitude inputs should not exceed ELEVATED."""
    h = compute_landslide_hazard(
        "coastal",
        soil={"soil_moisture_top": 0.45, "past_7d_precip_in": 9.0, "saturated": True},
        weather={"next_72h_precip_in": 8.0},
    )
    assert h["label"] in {"CALM", "ELEVATED"}
    assert h["score"] <= 30


def test_piedmont_capped_below_extreme():
    h = compute_landslide_hazard(
        "piedmont",
        soil={"soil_moisture_top": 0.45, "past_7d_precip_in": 9.0, "saturated": True},
        weather={"next_72h_precip_in": 8.0},
    )
    assert h["score"] <= 65
    assert h["label"] != "EXTREME"


def test_missing_inputs_do_not_crash():
    h = compute_landslide_hazard("mountain", soil=None, weather=None)
    assert "score" in h and "label" in h and "color" in h
    h2 = compute_landslide_hazard("mountain",
                                   soil={"error": "boom"},
                                   weather={"error": "boom"})
    assert h2["score"] >= 0


def test_summarize_inventory_handles_empty_and_populated():
    assert summarize_inventory([]) == {"count": 0, "most_recent_year": None}
    events = [{"year": 2018}, {"year": 2024}, {"year": None}]
    s = summarize_inventory(events)
    assert s["count"] == 3
    assert s["most_recent_year"] == 2024
