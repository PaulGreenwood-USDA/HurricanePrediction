from hurricane_asheville import weather as weather_mod


def test_fetch_current_weather_parses(monkeypatch, fake_response,
                                       open_meteo_weather_payload):
    monkeypatch.setattr(weather_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=open_meteo_weather_payload))
    w = weather_mod.fetch_current_weather(35.6, -82.5)
    assert w["temp_f"] == 72.5
    assert w["humidity_pct"] == 55
    assert w["wind_mph"] == 6.0
    assert w["pressure_mb"] == 1015.0
    # 72 hourly points * 0.05 in = 3.6 in
    assert w["next_72h_precip_in"] == 3.6
    assert w.get("error") is None


def test_fetch_current_weather_handles_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(weather_mod.requests, "get", boom)
    out = weather_mod.fetch_current_weather(35.6, -82.5)
    assert "error" in out


def test_fetch_current_weather_empty_response(monkeypatch, fake_response):
    monkeypatch.setattr(weather_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data={}))
    out = weather_mod.fetch_current_weather(35.6, -82.5)
    # No exception, just None values everywhere
    assert out["temp_f"] is None
    assert out["next_72h_precip_in"] == 0.0
