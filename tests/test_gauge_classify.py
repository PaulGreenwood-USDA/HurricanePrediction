import pytest

from hurricane_asheville.gauge import (FLOOD_STAGES_BY_SITE, FLOOD_STAGES_FT,
                                       SITE_FRENCH_BROAD_ASHEVILLE, _classify,
                                       flood_class, format_thresholds,
                                       thresholds_for)

ASH = SITE_FRENCH_BROAD_ASHEVILLE


def test_classify_below_action():
    assert _classify(0.5, ASH) == "below action"
    assert _classify(FLOOD_STAGES_FT["action"] - 0.01, ASH) == "below action"


def test_classify_action():
    assert _classify(FLOOD_STAGES_FT["action"], ASH) == "action stage"
    assert _classify(FLOOD_STAGES_FT["minor"] - 0.01, ASH) == "action stage"


def test_classify_minor():
    assert _classify(FLOOD_STAGES_FT["minor"], ASH) == "MINOR FLOOD"
    assert _classify(FLOOD_STAGES_FT["moderate"] - 0.01, ASH) == "MINOR FLOOD"


def test_classify_moderate():
    assert _classify(FLOOD_STAGES_FT["moderate"], ASH) == "MODERATE FLOOD"


def test_classify_major():
    assert _classify(FLOOD_STAGES_FT["major"], ASH) == "MAJOR FLOOD"
    assert _classify(50.0, ASH) == "MAJOR FLOOD"


def test_classify_none():
    assert _classify(None, ASH) == "unknown"


def test_thresholds_monotone():
    keys = ["action", "minor", "moderate", "major", "record"]
    vals = [FLOOD_STAGES_FT[k] for k in keys]
    assert vals == sorted(vals)


# ---- per-site thresholds --------------------------------------------------

def test_every_site_threshold_set_is_monotone():
    """A non-monotone set would silently mask flood categories."""
    for site_id, t in FLOOD_STAGES_BY_SITE.items():
        vals = [t[k] for k in ("action", "minor", "moderate", "major")
                if t.get(k) is not None]
        assert vals == sorted(vals), f"{site_id} thresholds out of order: {t}"


def test_unknown_site_is_not_classified_against_asheville():
    """The bug this replaces: every gauge measured against Asheville's datum.

    A 14.94 ft reading is MODERATE FLOOD on the French Broad at Asheville and
    an ordinary summer day on the Cape Fear at Lock 1. With no published
    thresholds we must say so, not guess.
    """
    assert _classify(14.94, "99999999") == "no thresholds"
    assert _classify(0.0, "99999999") == "no thresholds"


def test_reservoir_role_without_thresholds_is_pool_stage():
    assert _classify(3.18, "99999999", role="reservoir") == "pool stage"


def test_reservoir_role_still_uses_published_thresholds():
    """The role tag is a fallback label, not an override -- if NWS publishes
    flood stages for a site we must not suppress them."""
    assert _classify(999.0, ASH, role="reservoir") == "MAJOR FLOOD"


def test_thresholds_for_returns_none_for_unknown_site():
    assert thresholds_for("99999999") is None


@pytest.mark.skipif(not FLOOD_STAGES_BY_SITE,
                    reason="threshold store not present")
def test_asheville_thresholds_come_from_store():
    t = thresholds_for(ASH)
    assert t is not None
    assert t["minor"] == FLOOD_STAGES_FT["minor"]


# ---- CSS slugs ------------------------------------------------------------

def test_flood_class_slugs_are_single_tokens():
    """Multi-token classes are what made '.stage-pill.action' match
    'below action' and paint every safe gauge action-stage yellow."""
    for category in ("below action", "action stage", "MINOR FLOOD",
                     "MODERATE FLOOD", "MAJOR FLOOD", "pool stage",
                     "no thresholds", "unknown"):
        slug = flood_class(category)
        assert " " not in slug, f"{category!r} -> {slug!r} has a space"


def test_below_action_and_action_have_distinct_classes():
    assert flood_class("below action") != flood_class("action stage")


def test_flood_class_unknown_category_falls_back():
    assert flood_class("something new") == "unknown"


# ---- threshold tooltip ----------------------------------------------------

def test_format_thresholds_skips_undefined_levels():
    """NWS leaves levels undefined at some gauges; the tooltip must not
    render 'moderate None ft'."""
    label = format_thresholds({"action": None, "minor": 15.0,
                               "moderate": None, "major": None})
    assert "minor 15 ft" in label
    assert "None" not in label
    assert "moderate" not in label


def test_format_thresholds_empty_without_data():
    assert format_thresholds(None) == ""
    assert format_thresholds({}) == ""


def test_format_thresholds_full_set():
    label = format_thresholds({"action": 6.5, "minor": 9.5,
                               "moderate": 13.0, "major": 18.0})
    assert label == ("NWS flood stages here: action 6.5 ft, minor 9.5 ft, "
                     "moderate 13 ft, major 18 ft")
