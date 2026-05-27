"""Backtest visualization for LightGBM gauge models (Phase 5).

Replays walk-forward folds against the parquet history store, collects
per-fold out-of-sample predictions, and writes PNGs + a JSON summary
under ``site/ml/<target>/``. Designed to be called once after
``ml-train`` (or whenever the user wants to refresh the visualizations).

Why a separate pass over the data?
==================================
``train_with_backtest`` fits a *final* model on all rows for serving,
but the per-fold OOS predictions used for the metrics are thrown away.
Visualizing calibration / scatter / time-series requires keeping those
predictions, so we re-run the splits here with the same parameters and
collect predictions per row.

Outputs per (target, kind, horizon, [threshold]):
- ``pred_vs_actual_{kind}_h{h}.png``  -- regression: scatter; classification: reliability curve
- ``timeseries_{kind}_h{h}.png``      -- OOS predictions vs actuals over time (regression only)
- ``summary.json``                    -- metrics + paths so the dashboard can render links
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .models import (DEFAULT_LGBM_PARAMS, _classification_metrics,
                       _fit_lgbm, _regression_metrics, _split_X_y,
                       walk_forward_splits)

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    target_id: str
    horizon_h: int
    kind: str
    target_col: str
    threshold: float | None
    n_rows: int
    metrics: dict
    plots: dict   # {"pred_vs_actual": "path", "timeseries": "path"}


# ---- core replay ----------------------------------------------------------

def replay_folds(frame, target_id: str, horizon_h: int,
                  *, kind: str = "regression",
                  threshold: float | None = None,
                  n_folds: int = 5,
                  params: dict | None = None):
    """Walk-forward replay collecting OOS (ts, pred, actual) per row.

    Returns a pandas DataFrame with columns ``ts``, ``actual``, ``pred``,
    ``fold`` -- only rows that appeared in an OOS fold.
    """
    import pandas as pd

    if kind == "regression":
        target_col = f"y_future_max_{horizon_h}h"
    elif kind == "classification":
        if threshold is None:
            raise ValueError("classification requires a threshold")
        target_col = f"y_peak_above_{threshold}_{horizon_h}h"
    else:
        raise ValueError(f"unknown kind: {kind}")

    if target_col not in frame.columns:
        raise KeyError(target_col)

    X, y, _ = _split_X_y(frame, target_col)
    if len(X) < 20:
        return pd.DataFrame(columns=["ts", "actual", "pred", "fold"]), target_col

    out_rows: list[dict] = []
    for fold_i, (tr, te) in enumerate(
            walk_forward_splits(len(X), n_folds=n_folds,
                                  min_train=max(10, len(X) // 5))):
        model = _fit_lgbm(X.iloc[tr], y.iloc[tr], kind, params=params)
        if kind == "regression":
            pred = model.predict(X.iloc[te])
        else:
            pred = model.predict_proba(X.iloc[te])[:, 1]
        ts_slice = X.index[te]
        for ts_v, a, p in zip(ts_slice, y.iloc[te].to_numpy(), pred):
            out_rows.append({"ts": ts_v, "actual": float(a),
                              "pred": float(p), "fold": fold_i})
    return pd.DataFrame(out_rows), target_col


# ---- plotting -------------------------------------------------------------

def _setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0f1115",
        "axes.facecolor": "#1a1d24",
        "axes.edgecolor": "#9aa0aa",
        "axes.labelcolor": "#e6e6e6",
        "xtick.color": "#9aa0aa",
        "ytick.color": "#9aa0aa",
        "text.color": "#e6e6e6",
        "grid.color": "#333",
        "axes.grid": True,
        "savefig.facecolor": "#0f1115",
        "savefig.bbox": "tight",
    })
    return plt


def plot_regression_scatter(df, *, target_id: str, horizon_h: int,
                              out_path: Path):
    plt = _setup_plot()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["actual"], df["pred"], s=12, alpha=0.45,
               c=df["fold"], cmap="viridis", edgecolors="none")
    lo = float(min(df["actual"].min(), df["pred"].min()))
    hi = float(max(df["actual"].max(), df["pred"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#ef5350", lw=1)
    ax.set_xlabel("Actual stage (ft)")
    ax.set_ylabel("Predicted stage (ft)")
    ax.set_title(f"{target_id} regression — horizon +{horizon_h}h\n"
                 f"walk-forward OOS, n={len(df)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_regression_timeseries(df, *, target_id: str, horizon_h: int,
                                 out_path: Path):
    plt = _setup_plot()
    df = df.sort_values("ts")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["ts"], df["actual"], color="#4fc3f7", lw=1.2, label="actual")
    ax.plot(df["ts"], df["pred"], color="#ffb74d", lw=1.0,
             alpha=0.85, label=f"pred (+{horizon_h}h)")
    ax.set_ylabel("Stage (ft)")
    ax.set_title(f"{target_id} OOS predictions vs actuals (+{horizon_h}h)")
    ax.legend(loc="upper left", framealpha=0.0)
    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_calibration(df, *, target_id: str, horizon_h: int,
                       threshold: float, out_path: Path, n_bins: int = 10):
    """Reliability diagram for classification."""
    import numpy as np
    plt = _setup_plot()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(df["pred"], bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    bin_pred = np.zeros(n_bins)
    bin_obs = np.zeros(n_bins)
    bin_n = np.zeros(n_bins)
    for k in range(n_bins):
        m = (idx == k)
        if m.any():
            bin_pred[k] = df.loc[m, "pred"].mean()
            bin_obs[k] = df.loc[m, "actual"].mean()
            bin_n[k] = m.sum()
    keep = bin_n > 0
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#9aa0aa", lw=1)
    ax.plot(bin_pred[keep], bin_obs[keep], "-o", color="#ef5350", lw=1.5,
             markersize=6, label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{target_id} reliability — P(>{threshold}ft @ +{horizon_h}h)\n"
                 f"n={len(df)}, positives={int(df['actual'].sum())}")
    ax.legend(loc="upper left", framealpha=0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---- top-level orchestration ---------------------------------------------

def backtest_and_plot(frame, target_id: str,
                       *, horizons: Sequence[int],
                       thresholds: Sequence[float] = (),
                       out_dir: Path | str,
                       n_folds: int = 5) -> list[BacktestResult]:
    """Run replay + plot every (horizon, kind) combination for ``target_id``.

    Writes plots under ``<out_dir>/<target_id>/`` and returns the list of
    :class:`BacktestResult`. Also writes a ``summary.json`` next to the
    plots so the dashboard can render links.
    """
    out_base = Path(out_dir) / target_id
    out_base.mkdir(parents=True, exist_ok=True)

    results: list[BacktestResult] = []
    for h in horizons:
        # regression
        try:
            df, tcol = replay_folds(frame, target_id, h, kind="regression",
                                      n_folds=n_folds)
        except KeyError:
            df = None
            tcol = f"y_future_max_{h}h"
        if df is not None and not df.empty:
            scatter = plot_regression_scatter(
                df, target_id=target_id, horizon_h=h,
                out_path=out_base / f"pred_vs_actual_regression_h{h}.png")
            ts_plot = plot_regression_timeseries(
                df, target_id=target_id, horizon_h=h,
                out_path=out_base / f"timeseries_regression_h{h}.png")
            results.append(BacktestResult(
                target_id=target_id, horizon_h=h, kind="regression",
                target_col=tcol, threshold=None, n_rows=len(df),
                metrics=_regression_metrics(df["actual"], df["pred"]),
                plots={"pred_vs_actual": str(scatter.relative_to(out_base.parent)),
                        "timeseries": str(ts_plot.relative_to(out_base.parent))},
            ))

        # classification heads (one per threshold)
        for thr in thresholds:
            try:
                df, tcol = replay_folds(frame, target_id, h,
                                          kind="classification",
                                          threshold=thr, n_folds=n_folds)
            except KeyError:
                continue
            if df.empty:
                continue
            # need both classes to draw a useful reliability diagram
            if df["actual"].nunique() < 2:
                metrics = _classification_metrics(df["actual"], df["pred"])
                results.append(BacktestResult(
                    target_id=target_id, horizon_h=h, kind="classification",
                    target_col=tcol, threshold=thr, n_rows=len(df),
                    metrics=metrics, plots={},
                ))
                continue
            cal = plot_calibration(
                df, target_id=target_id, horizon_h=h, threshold=thr,
                out_path=out_base
                / f"calibration_thr{thr}_h{h}.png")
            results.append(BacktestResult(
                target_id=target_id, horizon_h=h, kind="classification",
                target_col=tcol, threshold=thr, n_rows=len(df),
                metrics=_classification_metrics(df["actual"], df["pred"]),
                plots={"calibration": str(cal.relative_to(out_base.parent))},
            ))

    summary = {
        "target_id": target_id,
        "n_folds": n_folds,
        "results": [
            {
                "horizon_h": r.horizon_h,
                "kind": r.kind,
                "threshold": r.threshold,
                "n_rows": r.n_rows,
                "metrics": r.metrics,
                "plots": r.plots,
            }
            for r in results
        ],
    }
    (out_base / "summary.json").write_text(json.dumps(summary, indent=2))
    return results
