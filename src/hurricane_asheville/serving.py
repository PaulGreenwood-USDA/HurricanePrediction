"""Model serving glue (Phase 4).

Loads every persisted :class:`hurricane_asheville.models.ModelBundle`
under ``data/models/`` and runs ``predict_latest`` for each on demand.

The output is shaped for the dashboard: one block per target gauge
containing a list of forecasts grouped by horizon, e.g.::

    {
      "03451500": {
        "ts": "2026-05-27T14:00:00+00:00",
        "regression": [
          {"horizon_h": 6,  "predicted_stage_ft": 4.12},
          {"horizon_h": 24, "predicted_stage_ft": 4.41},
        ],
        "classification": [
          {"horizon_h": 6, "threshold": 8.0, "probability": 0.03},
        ]
      }
    }

If no models are available the function returns ``{}`` -- the dashboard
silently omits the forecast block. This means the live site keeps
working even before the first ``ml-train`` has been run.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from .models import DEFAULT_MODELS_DIR, ModelBundle, predict_latest

log = logging.getLogger(__name__)


# ---- discovery ------------------------------------------------------------

def discover_bundles(base_dir: Path | str = DEFAULT_MODELS_DIR) -> list[Path]:
    """Return every ``.joblib`` under ``base_dir`` that has a sidecar json."""
    base = Path(base_dir)
    if not base.exists():
        return []
    out: list[Path] = []
    for p in sorted(base.rglob("*.joblib")):
        if p.with_suffix(".json").exists():
            out.append(p)
    return out


# ---- serving --------------------------------------------------------------

def forecast_all(history_df, *, base_dir: Path | str = DEFAULT_MODELS_DIR,
                  precip_entity_id: str = "asheville") -> dict:
    """Run every discovered bundle against the latest history snapshot.

    Returns a dict keyed by target_id. Bundles that fail to load or
    predict are skipped with a logged warning -- one bad model file
    never breaks the dashboard.
    """
    paths = discover_bundles(base_dir)
    if not paths:
        return {}

    grouped: dict[str, dict] = defaultdict(lambda: {
        "ts": None, "regression": [], "classification": []
    })
    for p in paths:
        try:
            bundle = ModelBundle.load(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("serving: failed to load %s: %s", p, exc)
            continue
        try:
            out = predict_latest(bundle, history_df,
                                  precip_entity_id=precip_entity_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("serving: predict failed for %s: %s", p, exc)
            continue
        if out.get("prediction") is None:
            continue
        block = grouped[bundle.target_id]
        block["ts"] = out["ts"]
        if bundle.kind == "regression":
            block["regression"].append({
                "horizon_h": bundle.horizon_h,
                "predicted_stage_ft": float(out["prediction"]),
                "trained_ts": bundle.trained_ts,
            })
        else:
            block["classification"].append({
                "horizon_h": bundle.horizon_h,
                "threshold": bundle.threshold,
                "probability": float(out["prediction"]),
                "trained_ts": bundle.trained_ts,
            })

    # sort horizons ascending for stable UI order
    for block in grouped.values():
        block["regression"].sort(key=lambda r: r["horizon_h"])
        block["classification"].sort(
            key=lambda r: (r["horizon_h"], r["threshold"] or 0.0))
    return dict(grouped)


# ---- accuracy metadata ----------------------------------------------------

# A classifier whose positive class appears in only one walk-forward fold has
# not been validated -- it has been shown one flood. The dashboard needs to say
# so rather than print a confident-looking percentage.
MIN_FOLDS_WITH_EVENTS = 2


def load_model_metrics(target_id: str,
                       *, base_dir: Path | str = DEFAULT_MODELS_DIR) -> dict:
    """Backtest metrics for one gauge's models, keyed for the dashboard.

    Reads the sidecar JSON written next to each ``.joblib`` at training time.
    These ship with the repo, unlike the ``site/ml`` plots, which are only
    produced by an explicit ``ml-backtest`` run.

    Keys look like ``regression_h24`` and ``classification_thr9.5_h24``.
    """
    base = Path(base_dir) / target_id
    if not base.exists():
        return {}

    out: dict[str, dict] = {}
    for path in sorted(base.glob("*.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError) as exc:  # noqa: PERF203
            log.warning("serving: unreadable model sidecar %s: %s", path, exc)
            continue
        kind = meta.get("kind")
        horizon = meta.get("horizon_h")
        metrics = meta.get("metrics") or {}
        folds = metrics.get("per_fold") or []
        common = {
            "kind": kind,
            "horizon_h": horizon,
            "n_folds": metrics.get("n_folds") or len(folds),
            "n_train_rows": meta.get("n_train_rows"),
            "trained_ts": meta.get("trained_ts"),
        }

        if kind == "regression":
            out[f"regression_h{horizon}"] = {
                **common,
                "mae": metrics.get("overall_mae"),
                "trustworthy": metrics.get("overall_mae") is not None,
            }
        elif kind == "classification":
            threshold = meta.get("threshold")
            # Count how much of a positive class the backtest actually saw.
            positives = 0
            folds_with_events = 0
            for f in folds:
                rate = f.get("positive_rate") or 0.0
                n = f.get("n") or 0
                count = round(rate * n)
                positives += count
                if count:
                    folds_with_events += 1
            auc = metrics.get("overall_auc")
            # 0.5 is what the metric code emits when AUC is undefined.
            auc_meaningful = auc is not None and abs(auc - 0.5) > 1e-9
            out[f"classification_thr{threshold}_h{horizon}"] = {
                **common,
                "threshold": threshold,
                "auc": auc if auc_meaningful else None,
                "positive_events": positives,
                "folds_with_events": folds_with_events,
                "trustworthy": bool(
                    auc_meaningful
                    and folds_with_events >= MIN_FOLDS_WITH_EVENTS),
            }
    return out
