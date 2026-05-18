from hurricane_asheville import tides as tides_mod


def test_fetch_coastal_station(monkeypatch, fake_response,
                                coops_water_level_payload,
                                coops_wind_payload,
                                coops_pressure_payload):
    """Module makes 4 GETs per station: water, wind (via _latest_value),
    pressure, then a separate wind detail. Feed them in order."""
    payloads = [
        coops_water_level_payload,
        coops_wind_payload,
        coops_pressure_payload,
        coops_wind_payload,  # second wind call inside fetch_coastal_station
    ]
    idx = {"n": 0}

    def fake_get(*a, **k):
        i = idx["n"]
        idx["n"] += 1
        # Inspect the product param to be robust to call order
        product = (k.get("params") or {}).get("product", "")
        if product == "water_level":
            return fake_response(json_data=coops_water_level_payload)
        if product == "air_pressure":
            return fake_response(json_data=coops_pressure_payload)
        if product == "wind":
            return fake_response(json_data=coops_wind_payload)
        return fake_response(json_data=payloads[min(i, len(payloads) - 1)])

    monkeypatch.setattr(tides_mod.requests, "get", fake_get)
    s = tides_mod.fetch_coastal_station("8658120", "Wilmington, NC",
                                        34.2275, -77.9536, "cape-fear")
    assert s["water_level_ft"] == 1.23
    assert s["air_pressure_mb"] == 1015.5
    assert s["wind_kt"] == 10.5
    assert s["wind_gust_kt"] == 15.0
    assert s["wind_dir_deg"] == 180.0


def test_fetch_all_coastal_returns_list(monkeypatch, fake_response,
                                         coops_water_level_payload,
                                         coops_wind_payload,
                                         coops_pressure_payload):
    def fake_get(*a, **k):
        product = (k.get("params") or {}).get("product", "")
        if product == "water_level":
            return fake_response(json_data=coops_water_level_payload)
        if product == "air_pressure":
            return fake_response(json_data=coops_pressure_payload)
        return fake_response(json_data=coops_wind_payload)

    monkeypatch.setattr(tides_mod.requests, "get", fake_get)
    out = tides_mod.fetch_all_coastal()
    assert len(out) == len(tides_mod.COASTAL_STATIONS)
    for c in out:
        assert "water_level_ft" in c
        assert "wind_kt" in c
