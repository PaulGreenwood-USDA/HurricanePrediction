"""Tests for the Phase-4 model serving layer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurricane_asheville import features as F
from hurricane_asheville import models as M
from hurricane_asheville import serving as S


@pytest.fixture
def hist_df():
    n = 24 * 30
    ts = pd.date_range("2025-08-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
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
def bundles_dir(tmp_path, hist_df):
    frame = F.build_training_frame(
        hist_df, "03451500",
        horizons=(6, 24),
        upstream_ids=[],
        precip_entity_id="asheville",
        thresholds=(3.5,),
        dropna_features=True,
    )
    base = tmp_path / "models"
    # regression h=6
    b1 = M.train_with_backtest(frame, "03451500", 6,
                                kind="regression", n_folds=3)
    b1.save(M.default_model_path("03451500", "regression", 6, base_dir=base))
    # regression h=24
    b2 = M.train_with_backtest(frame, "03451500", 24,
                                kind="regression", n_folds=3)
    b2.save(M.default_model_path("03451500", "regression", 24, base_dir=base))
    # classification thr=3.5 h=6 (if enough class variation)
    col = "y_peak_above_3.5_6h"
    if col in frame.columns and frame[col].dropna().nunique() == 2:
        b3 = M.train_with_backtest(frame, "03451500", 6,
                                    kind="classification",
                                    threshold=3.5, n_folds=3)
        b3.save(M.default_model_path("03451500",
                                       "classification_thr3.5", 6,
                                       base_dir=base))
    return base


def test_discover_bundles_empty(tmp_path):
    assert S.discover_bundles(tmp_path / "nope") == []


def test_discover_bundles_finds_saved(bundles_dir):
    found = S.discover_bundles(bundles_dir)
    assert len(found) >= 2
    for p in found:
        assert p.suffix == ".joblib"
        assert p.with_suffix(".json").exists()


def test_forecast_all_no_models_returns_empty(hist_df, tmp_path):
    out = S.forecast_all(hist_df, base_dir=tmp_path / "nope")
    assert out == {}


def test_forecast_all_groups_by_target(hist_df, bundles_dir):
    out = S.forecast_all(hist_df, base_dir=bundles_dir)
    assert "03451500" in out
    block = out["03451500"]
    assert block["ts"] is not None
    assert len(block["regression"]) >= 2
    horizons = [r["horizon_h"] for r in block["regression"]]
    assert horizons == sorted(horizons)
    for r in block["regression"]:
        assert isinstance(r["predicted_stage_ft"], float)
        assert 0.0 < r["predicted_stage_ft"] < 50.0


def test_forecast_all_skips_corrupt_bundle(hist_df, bundles_dir, tmp_path):
    # write a fake .joblib + sidecar that will fail to load
    bad = bundles_dir / "03451500" / "regression_h999.joblib"
    bad.write_bytes(b"not a real joblib")
    bad.with_suffix(".json").write_text("{}")
    out = S.forecast_all(hist_df, base_dir=bundles_dir)
    # good bundles still served
    assert "03451500" in out
    assert len(out["03451500"]["regression"]) >= 2


def test_forecast_all_empty_history_returns_empty(bundles_dir):
    empty = pd.DataFrame(columns=["ts", "source", "entity_type",
                                    "entity_id", "metric", "value"])
    # discover_bundles still finds files, but predict_latest returns no
    # prediction for empty history, so result is empty dict.
    out = S.forecast_all(empty, base_dir=bundles_dir)
    assert out == {}
