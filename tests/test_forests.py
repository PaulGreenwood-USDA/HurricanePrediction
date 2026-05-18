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
                                              nws_alerts_payload):
    # Patch both gauge.fetch_nws_alerts (used inside forests.py) and
    # weather.fetch_current_weather indirectly via their requests modules.
    from hurricane_asheville import weather as weather_mod
    from hurricane_asheville import gauge as gauge_mod

    # Both modules share the same `requests` module - route by URL.
    def router(url, *a, **k):
        if "open-meteo" in url:
            return fake_response(json_data=open_meteo_weather_payload)
        if "weather.gov" in url:
            return fake_response(json_data=nws_alerts_payload)
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(weather_mod.requests, "get", router)
    monkeypatch.setattr(gauge_mod.requests, "get", router)

    pisgah = NC_NATIONAL_FORESTS[0]
    state = forests_mod.fetch_forest_state(pisgah, active_storms=[])
    assert state["short"] == "Pisgah"
    assert state["weather"]["temp_f"] == 72.5
    assert len(state["alerts"]) == 1
    assert state["nearest_storm"] is None


def test_fetch_forest_nearest_storm_distance(monkeypatch, fake_response,
                                              open_meteo_weather_payload,
                                              nws_alerts_payload):
    from hurricane_asheville import weather as weather_mod
    from hurricane_asheville import gauge as gauge_mod
    from hurricane_asheville.active import ActiveStorm

    def router(url, *a, **k):
        if "open-meteo" in url:
            return fake_response(json_data=open_meteo_weather_payload)
        if "weather.gov" in url:
            return fake_response(json_data=nws_alerts_payload)
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(weather_mod.requests, "get", router)
    monkeypatch.setattr(gauge_mod.requests, "get", router)

    fake_storm = ActiveStorm(
        id="AL01", name="FAKE", classification="HU", intensity_kt=80,
        lat=34.0, lon=-77.0, distance_mi=200, movement="N", public_advisory_url=None)

    croatan = next(f for f in NC_NATIONAL_FORESTS if f.short == "Croatan")
    state = forests_mod.fetch_forest_state(croatan, active_storms=[fake_storm])
    assert state["nearest_storm"] == "FAKE"
    # Croatan ~ (34.85, -77.0); fake at (34.0, -77.0) -> ~58 mi
    assert state["nearest_storm_mi"] is not None
    assert 50 < state["nearest_storm_mi"] < 70
