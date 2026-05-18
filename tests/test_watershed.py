import numpy as np

from hurricane_asheville.watershed import (FRENCH_BROAD_POLY,
                                            fraction_track_in_watershed,
                                            in_french_broad,
                                            watershed_distance_mi,
                                            watershed_proximity_factor)


def test_asheville_centroid_inside():
    # ~35.59, -82.55 is right at the polygon's north edge of the Asheville-area
    # watershed. Pick a point clearly inside (Brevard area).
    assert in_french_broad(35.20, -82.50)


def test_far_outside():
    assert not in_french_broad(40.0, -100.0)
    assert not in_french_broad(25.0, -75.0)


def test_polygon_shape():
    assert FRENCH_BROAD_POLY.shape == (9, 2)
    # closed ring
    assert tuple(FRENCH_BROAD_POLY[0]) == tuple(FRENCH_BROAD_POLY[-1])


def test_fraction_in_watershed_full_inside():
    lats = np.array([35.20, 35.25, 35.30])
    lons = np.array([-82.50, -82.50, -82.50])
    assert fraction_track_in_watershed(lats, lons) == 1.0


def test_fraction_in_watershed_full_outside():
    lats = np.array([40.0, 41.0])
    lons = np.array([-100.0, -100.0])
    assert fraction_track_in_watershed(lats, lons) == 0.0


def test_proximity_factor_decreasing_with_distance():
    near = watershed_proximity_factor(0.0)
    mid = watershed_proximity_factor(50.0)
    far = watershed_proximity_factor(500.0)
    assert near == 1.0
    assert 0.0 < far < mid < near


def test_distance_zero_inside():
    assert watershed_distance_mi(35.20, -82.50) == 0.0
