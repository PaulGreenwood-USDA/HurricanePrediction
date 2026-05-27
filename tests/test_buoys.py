"""Tests for the NDBC offshore-buoy module."""
from __future__ import annotations


SAMPLE_TXT = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT  m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi  hPa    ft
2026 05 27 12 50  120  8.5 10.5   3.2  10.0   6.2 110  1013.5  22.0  24.5  20.0  MM  MM   MM
2026 05 27 12 40  118  8.0 10.0   3.1   9.8   6.0 110  1013.7  22.0  24.5  20.0  MM  MM   MM
"""

MISSING_TXT = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT  m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi  hPa    ft
2026 05 27 12 50   MM   MM   MM    MM    MM    MM  MM     MM    MM    MM    MM  MM  MM   MM
"""


def test_parse_realtime2_extracts_latest_row():
    from hurricane_asheville.buoys import _parse_realtime2
    row = _parse_realtime2(SAMPLE_TXT)
    assert row is not None
    assert row["wind_dir_deg"] == 120
    # 8.5 m/s -> ~16.5 kt
    assert 16.0 <= row["wind_kt"] <= 17.0
    # 3.2 m -> ~10.5 ft
    assert 10.0 <= row["wave_ht_ft"] <= 11.0
    assert row["dominant_period_s"] == 10.0
    assert row["pressure_mb"] == 1013.5
    # 24.5 C -> ~76 F
    assert 75.0 <= row["water_temp_f"] <= 77.5


def test_parse_realtime2_skips_all_missing_rows():
    from hurricane_asheville.buoys import _parse_realtime2
    assert _parse_realtime2(MISSING_TXT) is None


def test_classify_seas_thresholds():
    from hurricane_asheville.buoys import _classify_seas
    assert _classify_seas(None, None) == "unknown"
    assert _classify_seas(2.0, 10) == "CALM"
    assert _classify_seas(5.0, 15) == "MODERATE"
    assert _classify_seas(10.0, 36) == "ROUGH"
    assert _classify_seas(15.0, 50) == "VERY ROUGH"
    assert _classify_seas(30.0, 70) == "PHENOMENAL"


def test_fetch_buoy_calls_ndbc(monkeypatch, fake_response):
    from hurricane_asheville import buoys as buoys_mod
    monkeypatch.setattr(buoys_mod.requests, "get",
                        lambda *a, **k: fake_response(text=SAMPLE_TXT))
    obs = buoys_mod.fetch_buoy("41025")
    assert obs is not None
    assert obs["pressure_mb"] == 1013.5


def test_fetch_all_buoys_returns_one_per_station(monkeypatch, fake_response):
    from hurricane_asheville import buoys as buoys_mod
    monkeypatch.setattr(buoys_mod.requests, "get",
                        lambda *a, **k: fake_response(text=SAMPLE_TXT))
    out = buoys_mod.fetch_all_buoys()
    assert len(out) == len(buoys_mod.NC_BUOYS)
    assert all("seas" in b and "color" in b for b in out)
    assert all(b["seas"] in {"CALM", "MODERATE", "ROUGH",
                              "VERY ROUGH", "PHENOMENAL", "unknown"}
               for b in out)


def test_nc_buoy_coordinates_are_offshore():
    from hurricane_asheville.buoys import NC_BUOYS
    for station, label, lat, lon, role in NC_BUOYS:
        assert role in {"offshore", "nearshore", "cman"}, role
        # Loose bounding box: SE US offshore / coastal
        assert 30.0 <= lat <= 37.0, f"{station} lat: {lat}"
        assert -80.0 <= lon <= -73.0, f"{station} lon: {lon}"
