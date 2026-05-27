"""Tests for the Phase-3 LightGBM modeling layer."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hurricane_asheville import features as F
from hurricane_asheville import models as M


# ---- fixtures -------------------------------------------------------------

@pytest.fixture
def big_hist_df():
    """Larger synthetic history (30 days hourly) so backtest folds make sense."""
    n = 24 * 30
    ts = pd.date_range("2025-08-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    # primary stage with a learnable signal: noisy AR(1) plus precip-driven bumps
    precip_hourly = np.maximum(rng.standard_normal(n) - 1.5, 0.0)
    primary = np.zeros(n)
    primary[0] = 3.0
    for i in range(1, n):
        primary[i] = (0.92 * primary[i - 1] + 0.08 * 3.0
                      + 0.6 * precip_hourly[i - 1]
                      + 0.05 * rng.standard_normal())
    upstream = np.roll(primary, 3) + 0.1 * rng.standard_normal(n)

    rows: list[dict] = []
    for t, v in zip(ts, primary):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "gauge",
                     "entity_id": "03451500", "metric": "stage_ft",
                     "value": float(v)})
    for t, v in zip(ts, upstream):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "gauge",
                     "entity_id": "03443000", "metric": "stage_ft",
                     "value": float(v)})
    for t, p in zip(ts, precip_hourly):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "point",
                     "entity_id": "asheville", "metric": "precip_in_24h",
                     "value": float(p)})
    return pd.DataFrame(rows)


@pytest.fixture
def training_frame(big_hist_df):
    return F.build_training_frame(
        big_hist_df, "03451500",
        horizons=(6,),
        upstream_ids=["03443000"],
        precip_entity_id="asheville",
        thresholds=(3.5,),
        dropna_features=True,
    )


# ---- backtest splits ------------------------------------------------------

def test_walk_forward_splits_basic():
    folds = list(M.walk_forward_splits(n=200, n_folds=5, min_train=50))
    assert len(folds) == 5
    for tr, te in folds:
        assert max(tr) < min(te)  # strictly causal
        assert len(tr) >= 50
    # final fold should cover up to n
    assert folds[-1][1][-1] == 199


def test_walk_forward_splits_too_small_yields_nothing():
    assert list(M.walk_forward_splits(n=30, n_folds=5, min_train=50)) == []


# ---- metrics --------------------------------------------------------------

def test_regression_metrics_perfect_pred():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = M._regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_ignores_nan():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 2.0, 99.0, 4.0])
    m = M._regression_metrics(y, p)
    assert m["n"] == 3
    assert m["mae"] == 0.0


def test_classification_metrics_separable():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = M._classification_metrics(y, s)
    assert m["auc"] == pytest.approx(1.0)
    assert m["positive_rate"] == 0.5


def test_classification_metrics_single_class_returns_none():
    y = np.array([0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    m = M._classification_metrics(y, s)
    assert m["auc"] is None


# ---- training -------------------------------------------------------------

def test_train_regression_produces_bundle(training_frame):
    if training_frame.empty or "y_future_max_6h" not in training_frame.columns:
        pytest.skip("training frame too small")
    b = M.train_with_backtest(training_frame, "03451500", 6,
                               kind="regression", n_folds=3)
    assert b.target_id == "03451500"
    assert b.horizon_h == 6
    assert b.kind == "regression"
    assert b.target_col == "y_future_max_6h"
    assert len(b.feature_cols) > 5
    assert b.metrics["n_folds"] >= 1
    assert b.metrics["overall_mae"] is not None
    # AR(1) + precip signal should be learnable -> MAE under raw std
    raw_std = float(training_frame["y_future_max_6h"].dropna().std())
    assert b.metrics["overall_mae"] < raw_std


def test_train_classification_produces_bundle(training_frame):
    col = "y_peak_above_3.5_6h"
    if col not in training_frame.columns:
        pytest.skip("threshold target missing")
    if training_frame[col].dropna().nunique() < 2:
        pytest.skip("not enough class variation in synthetic data")
    b = M.train_with_backtest(training_frame, "03451500", 6,
                               kind="classification",
                               threshold=3.5, n_folds=3)
    assert b.kind == "classification"
    assert b.threshold == 3.5
    assert b.target_col == col


def test_train_unknown_target_raises(training_frame):
    with pytest.raises(KeyError):
        M.train_with_backtest(training_frame, "03451500", 999,
                               kind="regression")


def test_train_classification_requires_threshold(training_frame):
    with pytest.raises(ValueError):
        M.train_with_backtest(training_frame, "03451500", 6,
                               kind="classification", threshold=None)


# ---- persistence ----------------------------------------------------------

def test_bundle_round_trip(tmp_path, training_frame):
    if "y_future_max_6h" not in training_frame.columns:
        pytest.skip("training frame too small")
    b = M.train_with_backtest(training_frame, "03451500", 6,
                               kind="regression", n_folds=3)
    p = tmp_path / "models" / "reg_h6.joblib"
    b.save(p)
    assert p.exists()
    sidecar = p.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["target_id"] == "03451500"
    assert meta["kind"] == "regression"
    assert "feature_cols" in meta

    loaded = M.ModelBundle.load(p)
    assert loaded.target_id == b.target_id
    assert loaded.feature_cols == b.feature_cols
    # model should still produce predictions
    feat = training_frame[loaded.feature_cols].dropna().iloc[[0]]
    pred1 = loaded.model.predict(feat)
    pred2 = b.model.predict(feat)
    assert pred1 == pytest.approx(pred2)


# ---- inference ------------------------------------------------------------

def test_predict_latest(big_hist_df, training_frame):
    if "y_future_max_6h" not in training_frame.columns:
        pytest.skip("training frame too small")
    b = M.train_with_backtest(training_frame, "03451500", 6,
                               kind="regression", n_folds=3)
    out = M.predict_latest(b, big_hist_df,
                            precip_entity_id="asheville",
                            upstream_ids=["03443000"])
    assert out["ts"] is not None
    assert out["kind"] == "regression"
    assert out["target_id"] == "03451500"
    assert out["horizon_h"] == 6
    assert isinstance(out["prediction"], float)
    # prediction should land in a sane range for synthetic series ~ N(3, 1)
    assert 0.0 < out["prediction"] < 20.0


def test_predict_latest_empty_history(tmp_path):
    empty = pd.DataFrame(columns=["ts", "source", "entity_type",
                                    "entity_id", "metric", "value"])
    b = M.ModelBundle(
        target_id="03451500", horizon_h=6, kind="regression",
        target_col="y_future_max_6h", feature_cols=["self__stage_ft"],
        metrics={}, n_train_rows=0, trained_ts="now", model=None,
    )
    out = M.predict_latest(b, empty)
    assert out == {"ts": None, "prediction": None, "kind": "regression"}


# ---- path helper ----------------------------------------------------------

def test_default_model_path_layout(tmp_path):
    p = M.default_model_path("03451500", "regression", 24, base_dir=tmp_path)
    assert p.name == "regression_h24.joblib"
    assert p.parent.name == "03451500"
