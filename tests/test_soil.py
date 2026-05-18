from hurricane_asheville import soil as soil_mod


def test_soil_saturated_payload(monkeypatch, fake_response, open_meteo_soil_payload):
    monkeypatch.setattr(soil_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=open_meteo_soil_payload))
    s = soil_mod.fetch_soil_state(35.6, -82.5)
    assert s["soil_moisture_top"] == 0.42
    assert s["saturated"] is True
    assert s["very_dry"] is False
    assert s["condition"] == "SATURATED"
    # 168 hours * 0.05 in = 8.4 in
    assert s["past_7d_precip_in"] == 8.4


def test_soil_dry(monkeypatch, fake_response):
    payload = {
        "current": {"time": "t",
                    "soil_moisture_0_to_1cm": 0.10,
                    "soil_moisture_1_to_3cm": 0.10,
                    "soil_moisture_3_to_9cm": 0.10,
                    "soil_moisture_9_to_27cm": 0.12},
        "hourly": {"precipitation": [0.0] * 168},
    }
    monkeypatch.setattr(soil_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=payload))
    s = soil_mod.fetch_soil_state(35.6, -82.5)
    assert s["very_dry"] is True
    assert s["saturated"] is False
    assert s["condition"] == "very dry"


def test_soil_handles_network_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("dns fail")
    monkeypatch.setattr(soil_mod.requests, "get", boom)
    s = soil_mod.fetch_soil_state(35.6, -82.5)
    assert "error" in s
