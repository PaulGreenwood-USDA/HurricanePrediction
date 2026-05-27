from hurricane_asheville.fire_weather import (compute_fire_weather, fosberg_fwi,
                                              is_fire_weather_alert)


def test_fosberg_fwi_dry_windy_high():
    fwi = fosberg_fwi(temp_f=90.0, rh_pct=15.0, wind_mph=25.0)
    assert fwi is not None and fwi > 50


def test_fosberg_fwi_wet_calm_low():
    fwi = fosberg_fwi(temp_f=60.0, rh_pct=90.0, wind_mph=2.0)
    assert fwi is not None and fwi < 20


def test_fosberg_fwi_missing_input_returns_none():
    assert fosberg_fwi(None, 50, 10) is None
    assert fosberg_fwi(70, None, 10) is None
    assert fosberg_fwi(70, 50, None) is None


def test_red_flag_warning_forces_high():
    summary = compute_fire_weather(
        weather={"temp_f": 60, "humidity_pct": 80, "wind_mph": 2},
        alerts=[{"event": "Red Flag Warning", "severity": "Severe",
                  "headline": "Critical fire weather"}],
        region="mountain",
    )
    assert summary["label"] in {"HIGH", "EXTREME"}
    assert summary["score"] >= 70


def test_fire_weather_watch_forces_elevated():
    summary = compute_fire_weather(
        weather={"temp_f": 60, "humidity_pct": 80, "wind_mph": 2},
        alerts=[{"event": "Fire Weather Watch"}],
        region="mountain",
    )
    assert summary["label"] in {"ELEVATED", "HIGH"}
    assert summary["score"] >= 30


def test_calm_conditions_calm_label():
    summary = compute_fire_weather(
        weather={"temp_f": 60, "humidity_pct": 90, "wind_mph": 1},
        alerts=[],
        region="mountain",
    )
    assert summary["label"] == "CALM"


def test_is_fire_weather_alert_matches_both_phrasings():
    assert is_fire_weather_alert({"event": "Red Flag Warning"})
    assert is_fire_weather_alert({"event": "Fire Weather Watch"})
    assert not is_fire_weather_alert({"event": "Flood Warning"})
