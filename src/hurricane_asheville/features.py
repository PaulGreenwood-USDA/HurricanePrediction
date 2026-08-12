"""Feature engineering over the parquet history store (Phase 2).

This module turns the long-form snapshot table from :mod:`history` into
model-ready wide frames: lag features, rolling aggregates, antecedent
precipitation indices, upstream→downstream signals, and forecast targets.

Design goals
============
- Pure functions on pandas; no I/O except the convenience loader.
- All series ops are timestamp-indexed and tolerate irregular sampling
  (hourly snapshots may have gaps when Pages skips an hour).
- Features are computed *causally* (no leakage from the future).
- The builder for a single gauge target returns a frame indexed by
  timestamp with deterministic column names so the modeler can rely on
  the same schema across train/serve.

Public API
----------
- ``to_series(history_df, entity_id, metric)``  -> pd.Series
- ``lag_features``, ``rolling_features``, ``delta_features``
- ``antecedent_precip_index``
- ``build_gauge_features(history_df, target_id, upstream_ids=None,
                         precip_entity_id=None)``
- ``add_targets(features_df, series, horizons, thresholds=None)``
- ``build_training_frame(...)`` -- one-stop convenience
- ``default_upstream_for(target_id)`` -- WNC-aware default
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .gauge import UPSTREAM_GAUGES

# Roles considered "upstream" of the primary (French Broad @ Asheville)
_UPSTREAM_ROLES = {"headwaters", "upstream", "tributary"}


# ---- topology helpers -----------------------------------------------------

def default_upstream_for(target_id: str) -> list[str]:
    """Return canonical upstream gauge ids for ``target_id``.

    Today we only encode the WNC topology around the primary gauge
    (French Broad @ Asheville, ``03451500``). For any other target we
    return an empty list -- callers should pass ``upstream_ids`` explicitly.
    """
    if target_id != "03451500":
        return []
    return [sid for (sid, _name, _lat, _lon, role) in UPSTREAM_GAUGES
            if role in _UPSTREAM_ROLES and sid != target_id]


# ---- series construction --------------------------------------------------

#: Sources for the same metric, best resolution first. A daily *mean* and an
#: hourly *maximum* are different quantities; blending them into one series
#: would make the model's own history change definition partway through, so
#: the highest-resolution source available for an entity wins outright.
SOURCE_PREFERENCE = ("usgs_iv", "snapshot", "usgs_dv", "open_meteo_archive")


def to_series(history_df, entity_id: str, metric: str,
               *, entity_type: str | None = None,
               freq: str | None = None,
               prefer_sources: Sequence[str] | None = SOURCE_PREFERENCE):
    """Slice the long-form history to a single (entity, metric) series.

    Returns a numeric pd.Series indexed by UTC timestamp, sorted, with
    duplicate timestamps reduced to the last observation. If ``freq``
    is given (e.g. ``"h"`` or ``"D"``), the series is resampled with
    forward-fill (gauge-style step) for instantaneous metrics or with
    sum for accumulation metrics (caller's choice -- we just forward-fill).
    """
    import pandas as pd

    df = history_df
    mask = (df["entity_id"] == entity_id) & (df["metric"] == metric)
    if entity_type is not None:
        mask &= (df["entity_type"] == entity_type)
    sub = df.loc[mask, ["ts", "value", "source"]].copy()
    if sub.empty:
        return pd.Series(dtype="float64", name=f"{entity_id}__{metric}")

    # Take the single best source present rather than concatenating them.
    if prefer_sources and "source" in sub.columns:
        available = set(sub["source"].unique())
        for src in prefer_sources:
            if src in available:
                sub = sub[sub["source"] == src]
                break

    sub = sub[["ts", "value"]]
    sub["ts"] = pd.to_datetime(sub["ts"], utc=True)
    sub = (sub.sort_values("ts")
              .drop_duplicates("ts", keep="last")
              .set_index("ts"))
    s = sub["value"].astype("float64")
    s.name = f"{entity_id}__{metric}"
    if freq:
        s = s.resample(freq).last().ffill()
    return s


# ---- generic feature builders --------------------------------------------

def _first_available(history_df, entity_id: str, metrics: Sequence[str],
                      freq: str | None = None):
    """First of ``metrics`` that actually has data for ``entity_id``.

    Metric names have drifted (the store prefixes weather with ``wx_`` and
    soil with ``soil_``), and a silent miss here is expensive: the caller
    skips the whole feature block and the model trains without ever knowing
    it is missing an input. Trying an explicit list makes the fallback
    visible and keeps old stores working.
    """
    import pandas as pd

    for metric in metrics:
        s = to_series(history_df, entity_id, metric, freq=freq)
        if not s.empty:
            return s
    return pd.Series(dtype="float64")


def lag_features(series, lags: Sequence[int]):
    """One column per lag step. Lags are in *index units* (e.g. hours if
    the series is hourly). No leakage: column ``lag_k`` is ``series.shift(k)``.
    """
    import pandas as pd

    base = series.name or "x"
    return pd.DataFrame({f"{base}__lag{k}": series.shift(k) for k in lags},
                         index=series.index)


def rolling_features(series, windows: Sequence[int],
                      aggs: Sequence[str] = ("mean", "max", "min", "std")):
    """Rolling aggregates with min_periods=1 so the head of the series
    isn't all-NaN. The current observation IS included -- if you need
    strictly-causal features (no current value), shift the series by 1
    before calling.
    """
    import pandas as pd

    base = series.name or "x"
    cols = {}
    for w in windows:
        r = series.rolling(window=w, min_periods=1)
        for a in aggs:
            cols[f"{base}__roll{w}_{a}"] = getattr(r, a)()
    return pd.DataFrame(cols, index=series.index)


def delta_features(series, lags: Sequence[int]):
    """Rise-rate features: value now minus value ``k`` steps ago."""
    import pandas as pd

    base = series.name or "x"
    return pd.DataFrame(
        {f"{base}__delta{k}": series - series.shift(k) for k in lags},
        index=series.index,
    )


def antecedent_precip_index(precip, *, decay: float = 0.85,
                              window: int = 14):
    """Exponentially-weighted antecedent precipitation index (API).

    ``API_t = sum_{i=0..window-1} decay**i * precip_{t-i}``

    Inputs are precipitation per index step (mm or inches -- units are
    pass-through). ``decay`` ~0.85 is a common watershed default.
    Returns a Series aligned to ``precip.index``.
    """
    import numpy as np
    import pandas as pd

    p = precip.fillna(0.0).astype("float64")
    weights = np.array([decay ** i for i in range(window)])
    # rolling dot-product via convolution-style apply
    out = (p.rolling(window=window, min_periods=1)
             .apply(lambda x: float(np.dot(x[::-1][: len(weights)],
                                            weights[: len(x)])),
                     raw=True))
    out.name = f"{precip.name or 'precip'}__api{window}_d{int(decay*100)}"
    return out


# ---- gauge-target feature frame ------------------------------------------

# Default windows (hours) tuned for hourly snapshots.
DEFAULT_LAGS_H = (1, 3, 6, 12, 24)
DEFAULT_ROLL_H = (3, 6, 12, 24, 72)


def build_gauge_features(history_df, target_id: str,
                          *, upstream_ids: Sequence[str] | None = None,
                          precip_entity_id: str | None = None,
                          freq: str = "h",
                          lags=DEFAULT_LAGS_H,
                          rolls=DEFAULT_ROLL_H):
    """Build a feature frame for predicting target gauge stage.

    Columns produced (all causal):
      - ``self__stage_ft`` (current resampled value)
      - lag/rolling/delta features for the target stage
      - lag/rolling for any upstream gauge stage_ft series found in history
      - precipitation rolling sums + API for the optional precip series

    Rows are at ``freq`` cadence (default hourly). Missing values are
    forward-filled within the gauge series before lagging.
    """
    import pandas as pd

    if upstream_ids is None:
        upstream_ids = default_upstream_for(target_id)

    target = to_series(history_df, target_id, "stage_ft",
                        entity_type="gauge", freq=freq)
    target.name = "self__stage_ft"

    blocks: list[pd.DataFrame] = []
    if not target.empty:
        blocks.append(target.to_frame())
        blocks.append(lag_features(target, lags))
        blocks.append(rolling_features(target, rolls))
        blocks.append(delta_features(target, lags))

    for up_id in upstream_ids:
        up = to_series(history_df, up_id, "stage_ft",
                        entity_type="gauge", freq=freq)
        if up.empty:
            continue
        up.name = f"up_{up_id}__stage_ft"
        blocks.append(lag_features(up, lags))
        blocks.append(rolling_features(up, (6, 24)))

    if precip_entity_id is not None:
        # Metric names in the history store carry the prefix that
        # history._weather_rows adds on write ("wx_"), and soil rows carry
        # "soil_". Asking for the unprefixed name returned an empty series
        # and the whole block was skipped in silence, so every model trained
        # before this fix saw river stage and nothing else -- no rainfall at
        # all, in a rainfall-driven flood model.
        precip = _first_available(
            history_df, precip_entity_id,
            ("wx_precip_in_24h", "wx_precip_in", "precip_in_24h"), freq)
        if not precip.empty:
            precip.name = "precip"
            blocks.append(rolling_features(precip, (6, 24, 72),
                                            aggs=("sum", "max")))
            blocks.append(antecedent_precip_index(precip, window=14 * 24,
                                                    decay=0.99)
                          .to_frame())

        # Forecast rainfall (wx_next_72h_precip_in) is deliberately *not* a
        # feature yet. It only exists in snapshot rows, so it covers ~4% of
        # the training frame and all of it sits in the final walk-forward
        # fold. LightGBM would learn "qpf present => recent regime" -- a
        # spurious time signal rather than hydrology. Revisit once the
        # snapshot store has a year or two of it.

        # Antecedent soil moisture: the pre-conditioner that decides whether
        # a given rainfall total runs off or soaks in. ERA5 0-7 cm goes back
        # five years; the live 0-1 cm layer only exists for recent months, so
        # prefer the archive and fall back.
        soil = _first_available(
            history_df, precip_entity_id,
            ("soil_era5_0_7cm", "soil_soil_moisture_top"), freq)
        if not soil.empty:
            soil.name = "soil_top"
            blocks.append(soil.to_frame())
            blocks.append(rolling_features(soil, (24, 72),
                                            aggs=("mean", "max")))
        deep = _first_available(
            history_df, precip_entity_id,
            ("soil_era5_7_28cm", "soil_soil_moisture_root"), freq)
        if not deep.empty:
            deep.name = "soil_deep"
            blocks.append(deep.to_frame())

    if not blocks:
        return pd.DataFrame()

    out = pd.concat(blocks, axis=1)
    # forward-fill the *target* level only (other features are already
    # built off resampled series); leave lag/delta NaNs at the head intact
    # so the modeler can drop or impute deliberately.
    return out.sort_index()


# ---- targets --------------------------------------------------------------

def add_targets(features_df, series, horizons: Sequence[int],
                 *, thresholds: Sequence[float] | None = None):
    """Add forecast targets to a features frame.

    For each horizon ``h`` (in index steps):
      - ``y_future_max_{h}h``   : max(series) over (t, t+h]
      - ``y_future_rise_{h}h``  : that max *minus the current level*
      - ``y_future_val_{h}h``   : series.shift(-h)
      - for each threshold ``thr`` (if given):
            ``y_peak_above_{thr}_{h}h`` : 0/1 if future_max exceeds ``thr``

    Predicting the **rise** rather than the level is what makes the problem
    learnable. On a level target the answer is dominated by the current
    stage, which the model already has as a feature: it spends its capacity
    reproducing its own input and adds noise doing so, which is why the level
    models lost to persistence by 2-4x. Against a rise target, persistence is
    exactly ``0``, so any explained variance in the rise is a real gain, and
    reconstructing a level prediction is just ``current + predicted_rise``.

    Returns a *new* DataFrame (does not mutate input).
    """
    import pandas as pd

    out = features_df.copy()
    s = series.reindex(out.index).astype("float64")
    for h in horizons:
        future_max = (s.shift(-1)
                       .rolling(window=h, min_periods=1).max()
                       .shift(-(h - 1)))
        out[f"y_future_max_{h}h"] = future_max
        # The current level is s itself, which is also the self__stage_ft
        # feature, so this subtraction introduces no leakage.
        out[f"y_future_rise_{h}h"] = future_max - s
        out[f"y_future_val_{h}h"] = s.shift(-h)
        if thresholds:
            for thr in thresholds:
                col = f"y_peak_above_{thr}_{h}h"
                out[col] = (future_max >= thr).astype("Int8")
    return out


# ---- one-stop convenience -------------------------------------------------

def build_training_frame(history_df, target_id: str,
                          *, horizons: Sequence[int] = (6, 24, 72),
                          upstream_ids: Sequence[str] | None = None,
                          precip_entity_id: str | None = None,
                          thresholds: Sequence[float] | None = None,
                          freq: str = "h",
                          dropna_features: bool = False):
    """End-to-end: features + targets for one gauge.

    Returns a DataFrame with feature columns first, then ``y_*`` columns.
    If ``dropna_features`` is True, rows where any feature is NaN are
    dropped (the *targets* may still be NaN at the tail -- those rows
    aren't trainable and should be dropped per-target by the modeler).
    """
    feats = build_gauge_features(history_df, target_id,
                                  upstream_ids=upstream_ids,
                                  precip_entity_id=precip_entity_id,
                                  freq=freq)
    if feats.empty:
        return feats
    target_series = feats["self__stage_ft"]
    full = add_targets(feats, target_series, horizons,
                        thresholds=thresholds)
    if dropna_features:
        feat_cols = [c for c in full.columns if not c.startswith("y_")]
        full = full.dropna(subset=feat_cols)
    return full
