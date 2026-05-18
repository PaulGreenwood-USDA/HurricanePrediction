from hurricane_asheville.gauge import FLOOD_STAGES_FT, _classify


def test_classify_below_action():
    assert _classify(0.5) == "below action"
    assert _classify(FLOOD_STAGES_FT["action"] - 0.01) == "below action"


def test_classify_action():
    assert _classify(FLOOD_STAGES_FT["action"]) == "action stage"
    assert _classify(FLOOD_STAGES_FT["minor"] - 0.01) == "action stage"


def test_classify_minor():
    assert _classify(FLOOD_STAGES_FT["minor"]) == "MINOR FLOOD"
    assert _classify(FLOOD_STAGES_FT["moderate"] - 0.01) == "MINOR FLOOD"


def test_classify_moderate():
    assert _classify(FLOOD_STAGES_FT["moderate"]) == "MODERATE FLOOD"


def test_classify_major():
    assert _classify(FLOOD_STAGES_FT["major"]) == "MAJOR FLOOD"
    assert _classify(50.0) == "MAJOR FLOOD"


def test_classify_none():
    assert _classify(None) == "unknown"


def test_thresholds_monotone():
    keys = ["action", "minor", "moderate", "major", "record"]
    vals = [FLOOD_STAGES_FT[k] for k in keys]
    assert vals == sorted(vals)
