"""Tests for NHC forecast track / cone parsing.

Fixtures mirror the shape of the real products: a TRACK KML holds several
LineStrings (72-hour and 120-hour) plus point placemarks, and a CONE KML holds
one large Polygon. Verified against the archived Helene advisory 18 products.
"""
from __future__ import annotations

import io
import zipfile

import pytest

from hurricane_asheville import storm_track as st

TRACK_KML = """<?xml version="1.0"?><kml><Document>
<Placemark><styleUrl>#72_line</styleUrl><LineString><coordinates>
-85.5,37.5,0 -87.0,37.9,0 -87.2,37.7,0
</coordinates></LineString></Placemark>
<Placemark><styleUrl>#120_line</styleUrl><LineString><coordinates>
-85.5,37.5,0 -87.0,37.9,0 -87.2,37.7,0 -87.3,37.5,0 -87.0,37.3,0
</coordinates></LineString></Placemark>
<Placemark><styleUrl>#initial_point</styleUrl><Point><coordinates>
-85.5,37.5,0
</coordinates></Point></Placemark>
</Document></kml>"""

CONE_KML = """<?xml version="1.0"?><kml><Document>
<Placemark><styleUrl>#cone</styleUrl><Polygon><outerBoundaryIs><LinearRing>
<coordinates>-87.2,36.1,0 -87.1,36.2,0 -86.9,36.4,0 -87.2,36.1,0</coordinates>
</LinearRing></outerBoundaryIs></Polygon></Placemark>
</Document></kml>"""


def _kmz(kml: str, name: str = "storm.kml") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, kml)
    return buf.getvalue()


# ---- coordinate parsing ---------------------------------------------------

def test_parse_track_prefers_the_longest_line():
    """The 120-hour track is the full forecast; the 72-hour line is a subset."""
    pts = st.parse_track(TRACK_KML)
    assert len(pts) == 5
    assert pts[0] == {"lat": 37.5, "lon": -85.5}


def test_parse_cone_reads_the_polygon():
    pts = st.parse_cone(CONE_KML)
    assert len(pts) == 4
    assert pts[0]["lat"] == pytest.approx(36.1)
    assert pts[0]["lon"] == pytest.approx(-87.2)


def test_parse_handles_empty_and_malformed_input():
    for bad in ("", "<kml></kml>", "not xml at all",
                "<coordinates>garbage,,,</coordinates>"):
        assert st.parse_track(bad) == []
        assert st.parse_cone(bad) == []


def test_parse_rejects_out_of_range_coordinates():
    kml = ("<Placemark><LineString><coordinates>"
           "-999,999,0 -85.0,35.0,0 -86.0,36.0,0"
           "</coordinates></LineString></Placemark>")
    pts = st.parse_track(kml)
    assert len(pts) == 2
    assert all(-90 <= p["lat"] <= 90 for p in pts)


def test_thinning_caps_cone_size_and_keeps_endpoints():
    dense = [{"lat": 30 + i * 0.01, "lon": -80.0} for i in range(500)]
    thinned = st._thin(dense, 160)
    assert len(thinned) == 160
    assert thinned[0] == dense[0]
    assert thinned[-1] == dense[-1]


def test_thinning_leaves_short_lists_alone():
    pts = [{"lat": 1, "lon": 2}, {"lat": 3, "lon": 4}]
    assert st._thin(pts, 160) == pts


# ---- fetching -------------------------------------------------------------

def test_fetch_forecast_combines_products(monkeypatch, fake_response):
    def fake_get(url, *a, **k):
        body = _kmz(TRACK_KML if "TRACK" in url else CONE_KML)
        r = fake_response(json_data=None, status_code=200)
        r.content = body
        return r

    monkeypatch.setattr(st.requests, "get", fake_get)
    out = st.fetch_forecast("al092024")
    assert len(out["forecast_track"]) == 5
    assert len(out["cone"]) == 4


def test_fetch_forecast_survives_missing_product(monkeypatch, fake_response):
    """NHC does not publish a cone for every system or every advisory."""
    def fake_get(url, *a, **k):
        r = fake_response(json_data=None, status_code=404)
        r.content = b""
        return r

    monkeypatch.setattr(st.requests, "get", fake_get)
    out = st.fetch_forecast("al092024")
    assert out == {"forecast_track": [], "cone": []}


def test_fetch_forecast_survives_network_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(st.requests, "get", boom)
    assert st.fetch_forecast("al092024") == {"forecast_track": [], "cone": []}


def test_fetch_forecast_survives_corrupt_archive(monkeypatch, fake_response):
    def fake_get(url, *a, **k):
        r = fake_response(json_data=None, status_code=200)
        r.content = b"this is not a zip"
        return r
    monkeypatch.setattr(st.requests, "get", fake_get)
    assert st.fetch_forecast("al092024") == {"forecast_track": [], "cone": []}


def test_fetch_forecast_with_no_storm_id():
    assert st.fetch_forecast("") == {"forecast_track": [], "cone": []}


def test_attach_forecasts_enriches_each_storm(monkeypatch, fake_response):
    def fake_get(url, *a, **k):
        body = _kmz(TRACK_KML if "TRACK" in url else CONE_KML)
        r = fake_response(json_data=None, status_code=200)
        r.content = body
        return r

    monkeypatch.setattr(st.requests, "get", fake_get)
    out = st.attach_forecasts([{"id": "al092024", "name": "Helene"},
                               {"id": "al102024", "name": "Kirk"}])
    assert len(out) == 2
    assert all(len(s["forecast_track"]) == 5 for s in out)
    assert out[0]["name"] == "Helene"       # original fields preserved


def test_attach_forecasts_empty_list():
    assert st.attach_forecasts([]) == []
