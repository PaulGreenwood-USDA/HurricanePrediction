from hurricane_asheville.geo import haversine_mi


def test_zero_distance():
    assert haversine_mi(35.0, -80.0, 35.0, -80.0) == 0.0


def test_known_distance_asheville_to_charlotte():
    # Asheville -> Charlotte ~ 110 mi great-circle
    d = float(haversine_mi(35.5951, -82.5515, 35.2271, -80.8431))
    assert 95 < d < 115


def test_one_degree_lat_is_about_69_mi():
    d = float(haversine_mi(35.0, -80.0, 36.0, -80.0))
    assert 68.0 < d < 70.0


def test_array_input_works():
    import numpy as np
    lats = np.array([35.0, 36.0])
    lons = np.array([-80.0, -80.0])
    d = haversine_mi(35.0, -80.0, lats, lons)
    assert d.shape == (2,)
    assert d[0] == 0.0
    assert 68 < d[1] < 70
