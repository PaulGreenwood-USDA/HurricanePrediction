from hurricane_asheville import active as active_mod


def test_active_storm_filters_to_atlantic(monkeypatch, fake_response,
                                           nhc_active_payload):
    monkeypatch.setattr(active_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data=nhc_active_payload))
    out = active_mod.fetch_active_storms()
    assert len(out) == 1
    s = out[0]
    assert s.name == "TESTSTORM"
    assert s.classification == "HU"
    assert s.intensity_kt == 85.0
    assert s.lat == 30.0
    assert s.lon == -82.0
    # rough sanity: from (30, -82) to Asheville (35.6, -82.55) ~390 mi
    assert 350 < s.distance_mi < 450


def test_active_storm_handles_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(active_mod.requests, "get", boom)
    assert active_mod.fetch_active_storms() == []


def test_active_storm_empty_list(monkeypatch, fake_response):
    monkeypatch.setattr(active_mod.requests, "get",
                        lambda *a, **k: fake_response(json_data={"activeStorms": []}))
    assert active_mod.fetch_active_storms() == []
