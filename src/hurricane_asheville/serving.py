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
