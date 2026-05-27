from hurricane_asheville.wildfire import (fires_near, summarize_fires,
                                          fetch_active_wildfires)


def test_fires_near_filters_by_radius():
    fires = [
        {"name": "Close",  "lat": 35.78, "lon": -82.30, "acres": 100},
        {"name": "Far",    "lat": 32.0,  "lon": -82.0,  "acres": 500},
        {"name": "NoGeom", "lat": None,  "lon": None,   "acres": 10},
    ]
    near = fires_near(35.78, -82.30, fires, radius_mi=50.0)
    assert [f["name"] for f in near] == ["Close"]
    assert near[0]["distance_mi"] < 1.0


def test_summarize_fires_empty_and_populated():
    assert summarize_fires([]) == {"count": 0, "total_acres": 0.0,
                                    "max_acres": 0.0, "min_contained_pct": None}
    s = summarize_fires([
        {"acres": 100, "contained_pct": 75},
        {"acres": 250, "contained_pct": 30},
        {"acres": None, "contained_pct": None},
    ])
    assert s["count"] == 3
    assert s["total_acres"] == 350.0
    assert s["max_acres"] == 250.0
    assert s["min_contained_pct"] == 30


def test_fetch_active_wildfires_handles_network_failure(monkeypatch):
    from hurricane_asheville import wildfire as wf

    def bad_get(*a, **k):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(wf.requests, "get", bad_get)
    assert fetch_active_wildfires("NC") == []


def test_fetch_active_wildfires_parses_feature_service(monkeypatch, fake_response):
    from hurricane_asheville import wildfire as wf

    payload = {"features": [
        {"attributes": {"IncidentName": "Pisgah Ridge", "DailyAcres": 350.0,
                         "PercentContained": 40, "FireCause": "Lightning",
                         "FireDiscoveryDateTime": "2026-05-25T13:00:00Z",
                         "POOState": "US-NC", "IncidentTypeCategory": "WF",
                         "IrwinID": "abc-123"},
          "geometry": {"x": -82.30, "y": 35.78}},
        {"attributes": {"IncidentName": "Bogus", "DailyAcres": 5.0},
          "geometry": {"x": None, "y": None}},  # should be dropped
    ]}
    monkeypatch.setattr(wf.requests, "get",
                        lambda *a, **k: fake_response(json_data=payload))
    fires = fetch_active_wildfires("NC")
    assert len(fires) == 1
    assert fires[0]["name"] == "Pisgah Ridge"
    assert fires[0]["acres"] == 350.0
