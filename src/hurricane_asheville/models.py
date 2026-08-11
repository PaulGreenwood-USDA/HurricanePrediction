"""LightGBM models with time-aware backtesting (Phase 3).

Trains one model per (target gauge, horizon, kind) where ``kind`` is one
of ``'regression'`` (predict ``y_future_max_{h}h``) or ``'classification'``
(predict ``y_peak_above_{thr}_{h}h``).

Backtesting
===========
Time series demand causal splits. We use an expanding-window walk-forward:
the data is sorted by timestamp, then split into ``n_folds`` contiguous
test windows. Each fold trains on everything *strictly before* the fold's
test window. This is the simplest defensible scheme for irregular hourly
data and gives realistic out-of-sample error.

Persistence
===========
Trained artifacts are saved to ``data/models/{target}/{kind}_h{horizon}.joblib``
along with a small JSON of metadata (feature list, train cutoff, metrics).
A single ``ModelBundle`` dataclass round-trips both.

Inference
=========
``predict_latest(bundle, history_df)`` builds features for the most recent
timestamp in the history store and returns one prediction per active
target. Used by the dashboard / CLI to attach a "next-6h forecast" to
the live gauge readings.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("data/models")

# LightGBM defaults sized for ~10k-100k row tables (hourly snapshots over
# several years). Conservative to avoid overfit on small histories.
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
}


# ---- bundle ---------------------------------------------------------------

@dataclass
class ModelBundle:
    """Trained model + metadata. Persisted as joblib + sidecar json."""
    target_id: str
    horizon_h: int
    kind: str               # 'regression' | 'classification'
    target_col: str         # column name in training frame
    feature_cols: list[str]
    metrics: dict           # backtest metrics (mae, rmse, auc, ...)
    n_train_rows: int
    trained_ts: str
    threshold: float | None = None     # for classification
    model: object | None = field(default=None, repr=False)

    def save(self, path: Path | str) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        sidecar = path.with_suffix(".json")
        meta = asdict(self)
        meta.pop("model")
        sidecar.write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ModelBundle":
        import joblib

        path = Path(path)
        sidecar = path.with_suffix(".json")
        meta = json.loads(sidecar.read_text())
        model = joblib.load(path)
        return cls(model=model, **meta)


# ---- backtesting ----------------------------------------------------------

def walk_forward_splits(n: int, n_folds: int = 5, min_train: int = 50):
    """Yield (train_idx, test_idx) tuples using expanding windows.

    ``n``         total row count
    ``n_folds``   number of OOS folds (last ``n_folds`` chunks become test)
    ``min_train`` minimum train size before the first fold is yielded
    """
    if n <= min_train + n_folds:
        return
    fold_size = max(1, (n - min_train) // n_folds)
    for k in range(n_folds):
        test_start = min_train + k * fold_size
        test_end = (test_start + fold_size) if k < n_folds - 1 else n
        if test_end <= test_start or test_end > n:
            continue
        train_idx = list(range(0, test_start))
        test_idx = list(range(test_start, test_end))
        yield train_idx, test_idx


def _regression_metrics(y_true, y_pred) -> dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    yt, yp = y_true[mask], y_pred[mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": rmse,
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else None,
    }


def _classification_metrics(y_true, y_score) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    if mask.sum() == 0 or len(set(y_true[mask].tolist())) < 2:
        return {"n": int(mask.sum()), "auc": None, "ap": None,
                "positive_rate": (float(y_true[mask].mean())
                                   if mask.sum() else None)}
    yt, ys = y_true[mask], y_score[mask]
    return {
        "n": int(mask.sum()),
        "auc": float(roc_auc_score(yt, ys)),
        "ap": float(average_precision_score(yt, ys)),
        "positive_rate": float(yt.mean()),
    }


# ---- training -------------------------------------------------------------

def _fit_lgbm(X_train, y_train, kind: str,
               params: dict | None = None):
    """Train a LightGBM model. Returns the fitted estimator."""
    import lightgbm as lgb

    p = dict(DEFAULT_LGBM_PARAMS)
    if params:
        p.update(params)
    if kind == "regression":
        model = lgb.LGBMRegressor(**p)
    elif kind == "classification":
        model = lgb.LGBMClassifier(**p)
    else:
        raise ValueError(f"unknown kind: {kind}")
    model.fit(X_train, y_train)
    return model


def _split_X_y(frame, target_col: str):
    feat_cols = [c for c in frame.columns if not c.startswith("y_")]
    sub = frame.dropna(subset=[target_col])
    X = sub[feat_cols]
    y = sub[target_col]
    return X, y, feat_cols


def train_with_backtest(frame, target_id: str, horizon_h: int,
                          *, kind: str = "regression",
                          threshold: float | None = None,
                          n_folds: int = 5,
                          params: dict | None = None) -> ModelBundle:
    """Train a final model on all data + report walk-forward OOS metrics.

    The returned bundle's ``metrics`` dict has keys
    ``per_fold`` (list of fold dicts) and ``overall`` (mean of per-fold
    primary metric). The model itself is fit on *all* labeled rows so
    inference uses every available example.
    """
    import numpy as np
    from datetime import datetime, timezone

    if kind == "regression":
        target_col = f"y_future_max_{horizon_h}h"
    elif kind == "classification":
        if threshold is None:
            raise ValueError("classification requires a threshold")
        target_col = f"y_peak_above_{threshold}_{horizon_h}h"
    else:
        raise ValueError(f"unknown kind: {kind}")

    if target_col not in frame.columns:
        raise KeyError(f"target column {target_col!r} not in frame")

    X, y, feat_cols = _split_X_y(frame, target_col)
    n = len(X)
    if n < 10:
        raise ValueError(f"need at least 10 labeled rows, got {n}")

    per_fold: list[dict] = []
    for fold_i, (tr, te) in enumerate(walk_forward_splits(n, n_folds=n_folds,
                                                            min_train=max(10, n // 5))):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        model = _fit_lgbm(Xtr, ytr, kind, params=params)
        if kind == "regression":
            pred = model.predict(Xte)
            m = _regression_metrics(yte, pred)
        else:
            pred = model.predict_proba(Xte)[:, 1]
            m = _classification_metrics(yte, pred)
        m["fold"] = fold_i
        per_fold.append(m)

    primary_key = "mae" if kind == "regression" else "auc"
    vals = [f[primary_key] for f in per_fold if f.get(primary_key) is not None]
    overall = (float(np.mean(vals)) if vals else None)

    metrics = {"per_fold": per_fold, "overall_" + primary_key: overall,
               "n_folds": len(per_fold)}

    # Score the naive baseline on the *same* folds. Without this a model can
    # look respectable on absolute error while being several times worse than
    # doing nothing -- which is exactly what happened here: an MAE of 0.55 ft
    # at +72 h reads like a credential until you notice persistence scores
    # 0.26 ft on identical rows.
    baseline = _baseline_metrics(X, y, kind, n_folds=n_folds, threshold=threshold)
    if baseline is not None:
        metrics["baseline"] = baseline
        metrics["beats_baseline"] = _beats_baseline(overall, baseline, kind)

    # Conditional performance: 99.7% of rows are calm, so an overall MAE is
    # arithmetically dominated by days when persistence is unbeatable and
    # says almost nothing about the days the dashboard exists for.
    if kind == "regression":
        event = _event_conditional_mae(X, y, per_fold_threshold=None)
        if event:
            metrics["event"] = event

    # final fit on all labeled rows
    final_model = _fit_lgbm(X, y, kind, params=params)

    return ModelBundle(
        target_id=target_id,
        horizon_h=horizon_h,
        kind=kind,
        target_col=target_col,
        feature_cols=feat_cols,
        metrics=metrics,
        n_train_rows=n,
        trained_ts=datetime.now(timezone.utc).isoformat(),
        threshold=threshold,
        model=final_model,
    )


# ---- naive baselines ------------------------------------------------------

#: Column holding the gauge's current stage -- the persistence prediction.
PERSISTENCE_COL = "self__stage_ft"


def _baseline_metrics(X, y, kind: str, *, n_folds: int = 5,
                       threshold: float | None = None) -> dict | None:
    """Score the trivial predictor on the same walk-forward folds.

    Regression baseline is persistence: the future crest equals the current
    stage. Classification baseline is "current stage already exceeds the
    threshold", which is the decision a person would make without a model.
    """
    import numpy as np

    if PERSISTENCE_COL not in X.columns:
        return None
    current = X[PERSISTENCE_COL]
    scores: list[float] = []
    for tr, te in walk_forward_splits(len(X), n_folds=n_folds,
                                       min_train=max(10, len(X) // 5)):
        cur_te, y_te = current.iloc[te], y.iloc[te]
        if kind == "regression":
            scores.append(float((y_te - cur_te).abs().mean()))
        else:
            if threshold is None or y_te.nunique() < 2:
                continue
            m = _classification_metrics(y_te, (cur_te >= threshold).astype(float))
            if m.get("auc") is not None:
                scores.append(float(m["auc"]))
    if not scores:
        return None
    key = "mae" if kind == "regression" else "auc"
    return {key: float(np.mean(scores)), "kind": "persistence",
            "n_folds": len(scores)}


def _beats_baseline(overall, baseline: dict, kind: str) -> bool | None:
    """True when the model is actually worth shipping over the naive rule."""
    if overall is None or not baseline:
        return None
    if kind == "regression":
        ref = baseline.get("mae")
        return None if ref is None else bool(overall < ref)
    ref = baseline.get("auc")
    return None if ref is None else bool(overall > ref)


def _event_conditional_mae(X, y, *, per_fold_threshold=None,
                            quantile: float = 0.99) -> dict | None:
    """Error restricted to the highest-stage rows.

    Reported alongside the overall figure so a model cannot hide poor flood
    performance behind thousands of easy calm days.
    """
    import numpy as np

    if PERSISTENCE_COL not in X.columns or len(y) < 100:
        return None
    cutoff = float(np.nanquantile(y, quantile))
    mask = y >= cutoff
    if int(mask.sum()) < 10:
        return None
    cur = X.loc[mask, PERSISTENCE_COL]
    return {
        "quantile": quantile,
        "stage_cutoff_ft": cutoff,
        "n_rows": int(mask.sum()),
        "persistence_mae": float((y[mask] - cur).abs().mean()),
    }


# ---- inference ------------------------------------------------------------

def predict_latest(bundle: ModelBundle, history_df,
                    *, precip_entity_id: str = "asheville",
                    upstream_ids: Sequence[str] | None = None):
    """Build features for the latest timestamp in history and predict.

    Returns a dict ``{"ts": ..., "prediction": ..., "kind": ...}``
    (probability for classification, value for regression). Missing
    feature columns are filled with NaN -- LightGBM tolerates NaN
    natively.
    """
    import numpy as np
    import pandas as pd

    from .features import build_gauge_features

    feats = build_gauge_features(history_df, bundle.target_id,
                                   upstream_ids=upstream_ids,
                                   precip_entity_id=precip_entity_id)
    if feats.empty:
        return {"ts": None, "prediction": None, "kind": bundle.kind}

    # align columns to the model's training schema
    for col in bundle.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    X_last = feats[bundle.feature_cols].iloc[[-1]]
    ts = feats.index[-1]

    if bundle.kind == "regression":
        pred = float(bundle.model.predict(X_last)[0])
    else:
        pred = float(bundle.model.predict_proba(X_last)[0, 1])
    return {"ts": str(ts), "prediction": pred, "kind": bundle.kind,
             "target_id": bundle.target_id, "horizon_h": bundle.horizon_h,
             "threshold": bundle.threshold}


# ---- convenience ----------------------------------------------------------

def default_model_path(target_id: str, kind: str, horizon_h: int,
                        *, base_dir: Path | str = DEFAULT_MODELS_DIR) -> Path:
    return Path(base_dir) / target_id / f"{kind}_h{horizon_h}.joblib"
