from hurricane_asheville import forests as forests_mod
from hurricane_asheville.forests import NC_NATIONAL_FORESTS


def test_four_forests_present():
    shorts = {f.short for f in NC_NATIONAL_FORESTS}
    assert shorts == {"Pisgah", "Nantahala", "Uwharrie", "Croatan"}


def test_regions_are_known():
    for f in NC_NATIONAL_FORESTS:
        assert f.region in {"mountain", "piedmont", "coastal"}


def test_acreage_positive_and_total_in_range():
    total = sum(f.acres for f in NC_NATIONAL_FORESTS)
    # ~1.25M acres total across NC NFs
    assert 1_200_000 < total < 1_400_000


def test_centroids_inside_nc_bounding_box():
    for f in NC_NATIONAL_FORESTS:
        assert 33.5 < f.center_lat < 36.7
        assert -84.5 < f.center_lon < -75.0


def test_fetch_forest_state_uses_mocked_apis(monkeypatch, fake_response,
                                              open_meteo_weather_payload,
                                              open_meteo_soil_payload,
                                              nws_alerts_payload,
                                              usgs_iv_payload,
                                              usgs_history_payload):
    from hurricane_asheville import weather as weather_mod
    from hurricane_asheville import gauge as gauge_mod
    from hurricane_asheville import soil as soil_mod
    from hurricane_asheville import landslide as landslide_mod
    from hurricane_asheville import smoke_air as smoke_mod
    from hurricane_asheville import wildfire as wildfire_mod

    def router(url, *a, **k):
        params = (k.get("params") or {})
        if "air-quality-api.open-meteo" in url:
            return fake_response(json_data={"current": {"us_aqi": 42,
                                                          "pm2_5": 5.0,
                                                          "pm10": 10.0,
                                                          "ozone": 25.0}})
        if "open-meteo" in url:
            current = params.get("current", "")
            if "soil_moisture" in current:
                return fake_response(json_data=open_meteo_soil_payload)
            return fake_response(json_data=open_meteo_weather_payload)
        if "waterservices.usgs.gov" in url:
            if params.get("parameterCd") == "00065":
                return fake_response(json_data=usgs_history_payload)
            return fake_response(json_data=usgs_iv_payload)
        if "api.water.noaa.gov" in url:
            return fake_response(json_data={"data": []})
        if "weather.gov" in url:
            return fake_response(json_data=nws_alerts_payload)
        if "arcgis.com" in url or "FeatureServer" in url:
            return fake_response(json_data={"features": []})
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(weather_mod.requests, "get", router)
    monkeypatch.setattr(gauge_mod.requests, "get", router)
    monkeypatch.setattr(soil_mod.requests, "get", router)
    monkeypatch.setattr(landslide_mod.requests, "get", router)
    monkeypatch.setattr(smoke_mod.requests, "get", router)
    monkeypatch.setattr(wildfire_mod.requests, "get", router)

    pisgah = NC_NATIONAL_FORESTS[0]
    state = forests_mod.fetch_forest_state(pisgah, active_storms=[])
    assert state["short"] == "Pisgah"
    assert state["weather"]["temp_f"] == 72.5
    assert len(state["alerts"]) == 1
    assert state["nearest_storm"] is None
    assert len(state["gauges"]) == 4
    assert all("flood_category" in g for g in state["gauges"])
    assert state["landslide"]["label"] in {"CALM", "ELEVATED", "HIGH", "EXTREME"}
    assert "drivers" in state["landslide"]
    # New: fire weather + air quality + nearby fires
    assert state["fire_weather"]["label"] in {"CALM", "ELEVATED", "HIGH", "EXTREME"}
    assert state["air_quality"]["us_aqi"] == 42
    assert state["fires_summary"]["count"] == 0


def test_fetch_forest_nearest_storm_distance(monkeypatch, fake_response,
                                              open_meteo_weather_payload,
                                              open_meteo_soil_payload,
                                              nws_alerts_payload,
                                              usgs_iv_payload,
                                              usgs_history_payload):
    from hurricane_asheville import weather as weather_mod
    from hurricane_asheville import gauge as gauge_mod
    from hurricane_asheville import soil as soil_mod
    from hurricane_asheville import landslide as landslide_mod
    from hurricane_asheville import smoke_air as smoke_mod
    from hurricane_asheville import wildfire as wildfire_mod
    from hurricane_asheville.active import ActiveStorm

    def router(url, *a, **k):
        params = (k.get("params") or {})
        if "air-quality-api.open-meteo" in url:
            return fake_response(json_data={"current": {"us_aqi": 30}})
        if "open-meteo" in url:
            current = params.get("current", "")
            if "soil_moisture" in current:
                return fake_response(json_data=open_meteo_soil_payload)
            return fake_response(json_data=open_meteo_weather_payload)
        if "waterservices.usgs.gov" in url:
            if params.get("parameterCd") == "00065":
                return fake_response(json_data=usgs_history_payload)
            return fake_response(json_data=usgs_iv_payload)
        if "api.water.noaa.gov" in url:
            return fake_response(json_data={"data": []})
        if "weather.gov" in url:
            return fake_response(json_data=nws_alerts_payload)
        if "arcgis.com" in url or "FeatureServer" in url:
            return fake_response(json_data={"features": []})
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(weather_mod.requests, "get", router)
    monkeypatch.setattr(gauge_mod.requests, "get", router)
    monkeypatch.setattr(soil_mod.requests, "get", router)
    monkeypatch.setattr(landslide_mod.requests, "get", router)
    monkeypatch.setattr(smoke_mod.requests, "get", router)
    monkeypatch.setattr(wildfire_mod.requests, "get", router)

    fake_storm = ActiveStorm(
        id="AL01", name="FAKE", classification="HU", intensity_kt=80,
        lat=34.0, lon=-77.0, distance_mi=200, movement="N", public_advisory_url=None)

    croatan = next(f for f in NC_NATIONAL_FORESTS if f.short == "Croatan")
    state = forests_mod.fetch_forest_state(croatan, active_storms=[fake_storm])
    assert state["nearest_storm"] == "FAKE"
    # Croatan ~ (34.85, -77.0); fake at (34.0, -77.0) -> ~58 mi
    assert state["nearest_storm_mi"] is not None
    assert 50 < state["nearest_storm_mi"] < 70
    # Coastal forest – landslide score is capped low
    assert state["landslide"]["score"] <= 30


def test_per_forest_gauges_are_in_nc():
    """Every gauge associated with a forest sits inside (or very near) NC."""
    from hurricane_asheville.forests import FOREST_GAUGES
    for short, entries in FOREST_GAUGES.items():
        assert entries, f"{short} has no gauges configured"
        for site_id, label, lat, lon, role in entries:
            assert 33.5 < lat < 37.0, f"{site_id} lat out of NC: {lat}"
            assert -85.0 < lon < -75.0, f"{site_id} lon out of NC: {lon}"
            assert role in {"primary", "tributary", "headwaters",
                            "regional", "upstream"}, role
