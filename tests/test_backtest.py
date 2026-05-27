"""Tests for the Phase-5 backtest visualization layer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurricane_asheville import backtest as B
from hurricane_asheville import features as F


@pytest.fixture
def hist_df():
    n = 24 * 30
    ts = pd.date_range("2025-08-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(11)
    precip = np.maximum(rng.standard_normal(n) - 1.5, 0.0)
    stage = np.zeros(n)
    stage[0] = 3.0
    for i in range(1, n):
        stage[i] = (0.92 * stage[i - 1] + 0.08 * 3.0
                    + 0.6 * precip[i - 1]
                    + 0.05 * rng.standard_normal())
    rows = []
    for t, v in zip(ts, stage):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "gauge",
                     "entity_id": "03451500", "metric": "stage_ft",
                     "value": float(v)})
    for t, p in zip(ts, precip):
        rows.append({"ts": t, "source": "snapshot", "entity_type": "point",
                     "entity_id": "asheville", "metric": "precip_in_24h",
                     "value": float(p)})
    return pd.DataFrame(rows)


@pytest.fixture
def frame(hist_df):
    return F.build_training_frame(
        hist_df, "03451500",
        horizons=(6,),
        upstream_ids=[],
        precip_entity_id="asheville",
        thresholds=(3.5,),
        dropna_features=True,
    )


# ---- replay ---------------------------------------------------------------

def test_replay_folds_regression_returns_predictions(frame):
    df, col = B.replay_folds(frame, "03451500", 6,
                              kind="regression", n_folds=3)
    assert col == "y_future_max_6h"
    assert len(df) > 0
    assert {"ts", "actual", "pred", "fold"} <= set(df.columns)
    # fold ids start at 0 and are contiguous
    assert df["fold"].min() == 0
    assert df["fold"].max() <= 2


def test_replay_folds_classification_returns_probs(frame):
    col = "y_peak_above_3.5_6h"
    if col not in frame.columns or frame[col].dropna().nunique() < 2:
        pytest.skip("not enough class variation")
    df, _ = B.replay_folds(frame, "03451500", 6,
                            kind="classification", threshold=3.5,
                            n_folds=3)
    assert df["pred"].between(0.0, 1.0).all()


def test_replay_folds_unknown_target_raises(frame):
    with pytest.raises(KeyError):
        B.replay_folds(frame, "03451500", 999, kind="regression")


def test_replay_folds_classification_requires_threshold(frame):
    with pytest.raises(ValueError):
        B.replay_folds(frame, "03451500", 6,
                        kind="classification", threshold=None)


# ---- plotting (smoke -- just that files appear) ---------------------------

def test_plot_regression_scatter_writes_png(frame, tmp_path):
    df, _ = B.replay_folds(frame, "03451500", 6, kind="regression",
                            n_folds=3)
    out = B.plot_regression_scatter(df, target_id="03451500", horizon_h=6,
                                      out_path=tmp_path / "scatter.png")
    assert out.exists()
    assert out.stat().st_size > 1000  # not an empty file


def test_plot_regression_timeseries_writes_png(frame, tmp_path):
    df, _ = B.replay_folds(frame, "03451500", 6, kind="regression",
                            n_folds=3)
    out = B.plot_regression_timeseries(df, target_id="03451500",
                                         horizon_h=6,
                                         out_path=tmp_path / "ts.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_calibration_writes_png(frame, tmp_path):
    col = "y_peak_above_3.5_6h"
    if col not in frame.columns or frame[col].dropna().nunique() < 2:
        pytest.skip("not enough class variation")
    df, _ = B.replay_folds(frame, "03451500", 6, kind="classification",
                            threshold=3.5, n_folds=3)
    out = B.plot_calibration(df, target_id="03451500", horizon_h=6,
                                threshold=3.5,
                                out_path=tmp_path / "cal.png")
    assert out.exists()


# ---- orchestrator ---------------------------------------------------------

def test_backtest_and_plot_writes_summary(frame, tmp_path):
    results = B.backtest_and_plot(
        frame, "03451500",
        horizons=(6,), thresholds=(3.5,),
        out_dir=tmp_path, n_folds=3,
    )
    assert any(r.kind == "regression" for r in results)
    summary = tmp_path / "03451500" / "summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload["target_id"] == "03451500"
    assert payload["results"]
    # at least the regression scatter plot was written
    plots = (tmp_path / "03451500").glob("*.png")
    assert any(plots)


def test_backtest_and_plot_handles_missing_target(frame, tmp_path):
    # frame has no h=999 target column
    results = B.backtest_and_plot(
        frame, "03451500", horizons=(999,), out_dir=tmp_path, n_folds=3,
    )
    assert results == []
    summary = tmp_path / "03451500" / "summary.json"
    assert summary.exists()
    assert json.loads(summary.read_text())["results"] == []
