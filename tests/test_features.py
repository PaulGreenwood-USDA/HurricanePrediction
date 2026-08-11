"""Tests for the Phase-2 feature engineering layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hurricane_asheville import features as F


# ---- fixtures -------------------------------------------------------------

@pytest.fixture
def hist_df():
    """Synthetic long-form history: 5 days hourly for primary + 1 upstream,
    plus daily ERA5-style precip at 'asheville'."""
    ts = pd.date_range("2025-09-25", periods=24 * 5, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    primary = 3.0 + 0.01 * np.arange(len(ts)) + 0.1 * rng.standard_normal(len(ts))
    upstream = 2.0 + 0.008 * np.arange(len(ts)) + 0.1 * rng.standard_normal(len(ts))

    rows: list[dict] = []
    for t, v in zip(ts, primary):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "gauge",
                     "entity_id": "03451500", "metric": "stage_ft",
                     "value": float(v)})
    for t, v in zip(ts, upstream):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "gauge",
                     "entity_id": "03443000", "metric": "stage_ft",
                     "value": float(v)})

    # daily precip series (5 days), one big rain on day 3
    days = pd.date_range("2025-09-25", periods=5, freq="D", tz="UTC")
    precip_daily = [0.1, 0.0, 0.2, 2.5, 0.4]
    for t, p in zip(days, precip_daily):
        rows.append({"ts": t, "source": "open_meteo_archive",
                     "entity_type": "point", "entity_id": "asheville",
                     "metric": "precip_in_24h", "value": float(p)})
    return pd.DataFrame(rows)


# ---- topology -------------------------------------------------------------

def test_default_upstream_for_primary_includes_known_tributary():
    ups = F.default_upstream_for("03451500")
    assert "03443000" in ups
    assert "03446000" in ups
    assert "03451500" not in ups  # never include self


def test_default_upstream_for_unknown_target_is_empty():
    assert F.default_upstream_for("99999999") == []


# ---- series construction --------------------------------------------------

def test_to_series_returns_sorted_unique_index(hist_df):
    s = F.to_series(hist_df, "03451500", "stage_ft")
    assert s.is_monotonic_increasing or s.index.is_monotonic_increasing
    assert s.index.is_unique
    assert s.dtype == "float64"
    assert s.name == "03451500__stage_ft"
    assert len(s) == 24 * 5


def test_to_series_resample_freq(hist_df):
    s = F.to_series(hist_df, "03451500", "stage_ft", freq="h")
    # hourly -> hourly should preserve count
    assert len(s) == 24 * 5


def test_to_series_missing_returns_empty(hist_df):
    s = F.to_series(hist_df, "doesnotexist", "stage_ft")
    assert s.empty


# ---- generic builders ----------------------------------------------------

def test_lag_features_shifts_correctly():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                   index=pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
                   name="x")
    lf = F.lag_features(s, [1, 2])
    assert list(lf.columns) == ["x__lag1", "x__lag2"]
    assert pd.isna(lf["x__lag1"].iloc[0])
    assert lf["x__lag1"].iloc[1] == 1.0
    assert lf["x__lag2"].iloc[2] == 1.0


def test_rolling_features_min_periods_no_head_nan():
    s = pd.Series([1.0, 2.0, 3.0, 4.0],
                   index=pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
                   name="x")
    rf = F.rolling_features(s, [2], aggs=("mean", "max"))
    assert not rf["x__roll2_mean"].isna().any()
    assert rf["x__roll2_max"].iloc[-1] == 4.0
    assert rf["x__roll2_mean"].iloc[1] == 1.5


def test_delta_features():
    s = pd.Series([1.0, 3.0, 6.0, 10.0],
                   index=pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
                   name="x")
    df = F.delta_features(s, [1])
    assert pd.isna(df["x__delta1"].iloc[0])
    assert df["x__delta1"].iloc[1] == 2.0
    assert df["x__delta1"].iloc[2] == 3.0


def test_antecedent_precip_index_decays():
    # impulse precip on day 0, zeros after -- API should decay
    s = pd.Series([10.0, 0.0, 0.0, 0.0, 0.0],
                   index=pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
                   name="precip")
    api = F.antecedent_precip_index(s, decay=0.5, window=4)
    # At t=0: only sample is 10*0.5^0 = 10
    assert api.iloc[0] == pytest.approx(10.0)
    # At t=1: 0 + 10*0.5 = 5
    assert api.iloc[1] == pytest.approx(5.0)
    # At t=2: 0 + 0*0.5 + 10*0.25 = 2.5
    assert api.iloc[2] == pytest.approx(2.5)
    # monotonically decaying after the impulse
    assert api.iloc[1] > api.iloc[2] > api.iloc[3]


# ---- gauge feature frame --------------------------------------------------

def test_build_gauge_features_has_target_and_upstream(hist_df):
    feats = F.build_gauge_features(hist_df, "03451500",
                                    upstream_ids=["03443000"],
                                    precip_entity_id="asheville")
    assert not feats.empty
    assert "self__stage_ft" in feats.columns
    assert any(c.startswith("self__stage_ft__lag") for c in feats.columns)
    assert any(c.startswith("self__stage_ft__roll") for c in feats.columns)
    assert any(c.startswith("self__stage_ft__delta") for c in feats.columns)
    assert any(c.startswith("up_03443000__stage_ft__lag") for c in feats.columns)
    # precip features should be present
    assert any("precip" in c for c in feats.columns)


def test_build_gauge_features_missing_target_returns_empty(hist_df):
    feats = F.build_gauge_features(hist_df, "doesnotexist")
    assert feats.empty


# ---- targets --------------------------------------------------------------

def test_add_targets_future_max_and_value():
    idx = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    s = pd.Series(np.arange(10, dtype=float), index=idx, name="x")
    feats = pd.DataFrame({"f": s.values}, index=idx)
    out = F.add_targets(feats, s, horizons=[3], thresholds=[5.0])
    # at t=0: future_max over (0, 3] = max(1,2,3) = 3
    assert out["y_future_max_3h"].iloc[0] == 3.0
    # at t=0: future_val 3 ahead = 3.0
    assert out["y_future_val_3h"].iloc[0] == 3.0
    # at t=3: future_max over (3,6] = max(4,5,6) = 6, exceeds 5.0 -> 1
    assert out["y_peak_above_5.0_3h"].iloc[3] == 1
    # at t=0: future_max=3, below 5 -> 0
    assert out["y_peak_above_5.0_3h"].iloc[0] == 0


# ---- end to end -----------------------------------------------------------

def test_build_training_frame_end_to_end(hist_df):
    frame = F.build_training_frame(hist_df, "03451500",
                                     horizons=(6, 24),
                                     upstream_ids=["03443000"],
                                     precip_entity_id="asheville",
                                     thresholds=(4.0,))
    assert not frame.empty
    feat_cols = [c for c in frame.columns if not c.startswith("y_")]
    y_cols = [c for c in frame.columns if c.startswith("y_")]
    assert len(feat_cols) >= 10  # plenty of features
    # 2 horizons * 2 (max,val) + 2 horizons * 1 threshold = 6 target cols
    assert set(y_cols) >= {
        "y_future_max_6h", "y_future_val_6h", "y_peak_above_4.0_6h",
        "y_future_max_24h", "y_future_val_24h", "y_peak_above_4.0_24h",
    }


def test_build_training_frame_dropna_features(hist_df):
    frame = F.build_training_frame(hist_df, "03451500",
                                     horizons=(6,),
                                     upstream_ids=["03443000"],
                                     precip_entity_id="asheville",
                                     dropna_features=True)
    feat_cols = [c for c in frame.columns if not c.startswith("y_")]
    assert not frame[feat_cols].isna().any().any()


# ---- exogenous inputs -----------------------------------------------------

def test_precip_features_are_built_from_the_prefixed_metric():
    """Regression: features.py asked for 'precip_in_24h' while the store
    writes 'wx_precip_in_24h', so the block was skipped in silence and every
    model trained on river stage alone -- no rainfall at all."""
    import pandas as pd
    from hurricane_asheville import features as F

    idx = pd.date_range("2024-09-20", periods=200, freq="h", tz="UTC")
    rows = []
    for t in idx:
        rows.append({"ts": t, "source": "usgs_dv", "entity_type": "gauge",
                     "entity_id": "G", "metric": "stage_ft", "value": 2.0})
        rows.append({"ts": t, "source": "open_meteo_archive",
                     "entity_type": "point", "entity_id": "asheville",
                     "metric": "wx_precip_in_24h", "value": 0.1})
    df = pd.DataFrame(rows)
    out = F.build_gauge_features(df, "G", upstream_ids=[],
                                  precip_entity_id="asheville")
    precip_cols = [c for c in out.columns if "precip" in c]
    assert precip_cols, "precipitation features missing"
    assert out[precip_cols].notna().any().any()


def test_soil_features_are_built():
    """Soil moisture is the pre-conditioner that decides whether rain runs
    off; it had five years of history and no feature."""
    import pandas as pd
    from hurricane_asheville import features as F

    idx = pd.date_range("2024-09-20", periods=200, freq="h", tz="UTC")
    rows = []
    for t in idx:
        rows.append({"ts": t, "source": "usgs_dv", "entity_type": "gauge",
                     "entity_id": "G", "metric": "stage_ft", "value": 2.0})
        rows.append({"ts": t, "source": "open_meteo_archive",
                     "entity_type": "point", "entity_id": "asheville",
                     "metric": "soil_era5_0_7cm", "value": 0.42})
    df = pd.DataFrame(rows)
    out = F.build_gauge_features(df, "G", upstream_ids=[],
                                  precip_entity_id="asheville")
    assert [c for c in out.columns if "soil" in c]


def test_first_available_falls_back_across_metric_names():
    import pandas as pd
    from hurricane_asheville import features as F

    idx = pd.date_range("2024-09-20", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame([{"ts": t, "source": "s", "entity_type": "point",
                        "entity_id": "asheville", "metric": "wx_precip_in_24h",
                        "value": 1.0} for t in idx])
    s = F._first_available(df, "asheville",
                            ("precip_in_24h", "wx_precip_in_24h"), "h")
    assert not s.empty


def test_first_available_returns_empty_when_nothing_matches():
    import pandas as pd
    from hurricane_asheville import features as F
    df = pd.DataFrame(columns=["ts", "source", "entity_type",
                                "entity_id", "metric", "value"])
    assert F._first_available(df, "asheville", ("a", "b"), "h").empty


def test_forecast_qpf_is_not_a_feature():
    """wx_next_72h_precip_in covers ~4% of the frame, all of it in the final
    walk-forward fold, so including it teaches the model a time signal
    rather than hydrology."""
    import pandas as pd
    from hurricane_asheville import features as F

    idx = pd.date_range("2024-09-20", periods=200, freq="h", tz="UTC")
    rows = []
    for t in idx:
        rows.append({"ts": t, "source": "usgs_dv", "entity_type": "gauge",
                     "entity_id": "G", "metric": "stage_ft", "value": 2.0})
        rows.append({"ts": t, "source": "snapshot", "entity_type": "point",
                     "entity_id": "asheville",
                     "metric": "wx_next_72h_precip_in", "value": 1.0})
    df = pd.DataFrame(rows)
    out = F.build_gauge_features(df, "G", upstream_ids=[],
                                  precip_entity_id="asheville")
    assert not [c for c in out.columns if "qpf" in c.lower()]
