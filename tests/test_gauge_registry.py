"""Guards on the gauge registry itself.

An earlier revision had eight entries whose USGS site ids pointed at entirely
different rivers than their labels claimed -- 02143040 was labelled "Lake
Norman" but is Jacob Fork at Ramsey, 34 miles away. Nothing caught it because
the ids were syntactically fine and USGS returned data for them.

These tests pin each id to a keyword from its real USGS station name, so
swapping an id without updating the label fails here rather than on the
dashboard. Station names come from the USGS site service; refresh them with::

    curl "https://waterservices.usgs.gov/nwis/site/?sites=<ids>&format=rdb&siteOutput=expanded"
"""
from __future__ import annotations

import pytest

from hurricane_asheville.forests import FOREST_GAUGES
from hurricane_asheville.gauge import (RESERVOIR_ROLES, UPSTREAM_GAUGES,
                                       flood_class)

# site_id -> (substring that must appear in the official USGS station name,
#             approximate latitude, approximate longitude)
# Verified against USGS NWIS 2026-08-10.
EXPECTED = {
    "03439000":   ("FRENCH BROAD RIVER AT ROSMAN",     35.14, -82.83),
    "03443000":   ("FRENCH BROAD RIVER AT BLANTYRE",   35.36, -82.62),
    "03446000":   ("MILLS RIVER NEAR MILLS RIVER",     35.39, -82.56),
    "03451000":   ("SWANNANOA RIVER AT BILTMORE",      35.51, -82.54),
    "03451500":   ("FRENCH BROAD RIVER AT ASHEVILLE",  35.61, -82.58),
    "03456991":   ("PIGEON RIVER NEAR CANTON",         35.52, -82.85),
    "02151500":   ("BROAD RIVER NEAR BOILING SPRINGS", 35.21, -81.70),
    "03512000":   ("OCONALUFTEE RIVER AT BIRDTOWN",    35.46, -83.35),
    "0351706800": ("CHEOAH RIVER",                     35.44, -83.92),
    "0214267602": ("CATAWBA RIVER DNSTRM DECK MTN IS", 35.33, -80.99),
    "0214264790": ("COWANS FORD",                      35.43, -80.96),
    "02116500":   ("YADKIN RIVER AT YADKIN COLLEGE",   35.86, -80.39),
    "02129000":   ("PEE DEE R NR ROCKINGHAM",          35.01, -79.87),
    "02102500":   ("CAPE FEAR RIVER AT LILLINGTON",    35.41, -78.81),
    "02105769":   ("CAPE FEAR R AT LOCK 1 NR KELLY",   34.40, -78.30),
    "02083500":   ("TAR RIVER AT TARBORO",             35.89, -77.54),
    "02089500":   ("NEUSE RIVER AT KINSTON",           35.26, -77.59),
    "02134500":   ("LUMBER RIVER AT BOARDMAN",         34.44, -78.96),
    "02087182":   ("FALLS LAKE ABOVE DAM",             35.94, -78.58),
    "02098197":   ("JORDAN LAKE AT DAM",               35.65, -79.07),
}

# Additional sites that appear only in the per-forest table.
EXPECTED_FOREST_ONLY = {
    "03500000":   ("LITTLE TENNESSEE RIVER NEAR PRENTISS", 35.15, -83.38),
    "03513000":   ("TUCKASEGEE RIVER AT BRYSON CITY",  35.43, -83.45),
    "03550000":   ("VALLEY RIVER AT TOMOTLA",          35.14, -83.98),
    "02126000":   ("ROCKY RIVER NEAR NORWOOD",         35.16, -80.17),
    "02123500":   ("UWHARRIE RIVER NEAR ELDORADO",     35.43, -80.02),
    "02092500":   ("TRENT RIVER NEAR TRENTON",         35.06, -77.46),
    "02091814":   ("NEUSE RIVER NEAR FORT BARNWELL",   35.31, -77.30),
    "02093000":   ("NEW RIVER NEAR GUM BRANCH",        34.85, -77.52),
}

ALL_EXPECTED = {**EXPECTED, **EXPECTED_FOREST_ONLY}

FOREST_ENTRIES = [(forest, *entry)
                  for forest, entries in FOREST_GAUGES.items()
                  for entry in entries]


def test_registry_matches_expected_sites():
    assert {g[0] for g in UPSTREAM_GAUGES} == set(EXPECTED), (
        "UPSTREAM_GAUGES changed; re-verify each site id against the USGS "
        "site service and update EXPECTED here"
    )


@pytest.mark.parametrize("site_id,label,lat,lon,role", UPSTREAM_GAUGES)
def test_coordinates_match_the_site_id(site_id, label, lat, lon, role):
    """Coordinates must be the gauge's own, not the place the label names.

    The old registry put coordinates at the labelled location while the id
    pointed elsewhere, so map markers sat up to 34 mi from their data.
    """
    _name, exp_lat, exp_lon = EXPECTED[site_id]
    assert abs(lat - exp_lat) < 0.05, f"{site_id} latitude drifted"
    assert abs(lon - exp_lon) < 0.05, f"{site_id} longitude drifted"


def test_no_duplicate_site_ids():
    ids = [g[0] for g in UPSTREAM_GAUGES]
    assert len(ids) == len(set(ids))


def test_exactly_one_primary_gauge():
    primaries = [g for g in UPSTREAM_GAUGES if g[4] == "primary"]
    assert len(primaries) == 1
    assert primaries[0][0] == "03451500"


def test_reservoir_entries_are_actual_reservoirs():
    """The 'Reservoirs' tab used to contain four river gauges."""
    reservoirs = [g for g in UPSTREAM_GAUGES if g[4] in RESERVOIR_ROLES]
    assert reservoirs, "expected at least one reservoir"
    for site_id, label, _lat, _lon, _role in reservoirs:
        name = EXPECTED[site_id][0]
        assert "LAKE" in name or "RESERVOIR" in name, (
            f"{site_id} ({label}) is tagged reservoir but USGS calls it {name}"
        )


def test_labels_do_not_claim_a_reservoir_that_is_a_river():
    """Guards the specific mistake: a river gauge labelled as a lake."""
    for site_id, label, _lat, _lon, role in UPSTREAM_GAUGES:
        claims_lake = any(w in label.lower()
                          for w in ("lake ", "reservoir", " lake"))
        if claims_lake and "dam" not in label.lower():
            assert role in RESERVOIR_ROLES, (
                f"{label} ({site_id}) reads as a lake but is not tagged one")


@pytest.mark.parametrize("forest,site_id,label,lat,lon,role", FOREST_ENTRIES)
def test_forest_gauge_ids_are_known(forest, site_id, label, lat, lon, role):
    assert site_id in ALL_EXPECTED, (
        f"{forest}: site {site_id} ({label}) is not in the verified registry; "
        f"check it against the USGS site service before adding it"
    )


@pytest.mark.parametrize("forest,site_id,label,lat,lon,role", FOREST_ENTRIES)
def test_forest_gauge_coordinates_match_the_site_id(forest, site_id, label,
                                                     lat, lon, role):
    _name, exp_lat, exp_lon = ALL_EXPECTED[site_id]
    assert abs(lat - exp_lat) < 0.05, f"{forest}/{site_id} latitude drifted"
    assert abs(lon - exp_lon) < 0.05, f"{forest}/{site_id} longitude drifted"


@pytest.mark.parametrize("forest,site_id,label,lat,lon,role", FOREST_ENTRIES)
def test_forest_gauge_is_actually_near_its_forest(forest, site_id, label,
                                                   lat, lon, role):
    """A gauge listed under a forest should be within reach of it.

    02093229 was listed under Croatan but sits in Wilmington, ~60 mi away.
    """
    from hurricane_asheville.forests import NC_NATIONAL_FORESTS
    from hurricane_asheville.geo import haversine_mi

    f = next(x for x in NC_NATIONAL_FORESTS if x.short == forest)
    d = haversine_mi(lat, lon, f.center_lat, f.center_lon)
    assert d < 75, f"{label} is {d:.0f} mi from {forest} NF centre"


def test_every_gauge_gets_a_renderable_css_class():
    for category in ("below action", "pool stage", "unknown", "no thresholds"):
        assert flood_class(category) != ""
        assert " " not in flood_class(category)
