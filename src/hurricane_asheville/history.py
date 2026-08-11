"""Append-only historical snapshot store backed by monthly parquet partitions.

Every hourly dashboard refresh writes a slim *long-form* table to
``data/history/snapshots/YYYY-MM.parquet`` with one row per
(timestamp, entity_type, entity_id, metric, value).

Why long form
=============
The dashboard ships a fat nested JSON every hour (gauges, forests, buoys,
districts, coastal, ...). Flattening it to long form costs almost nothing on
the write side but lets the ML pipeline ask uniform questions like
"every stage_ft for site_id 03451500 in 2024-09" without parsing nested
shapes per record.

Schema
======
ts            : timestamp[us, UTC]
source        : string  -- 'snapshot' | 'usgs_dv' | 'open_meteo_archive'
entity_type   : string  -- 'gauge' | 'forest' | 'district' | 'buoy'
                          | 'coastal' | 'point'
entity_id     : string  -- usgs site id, forest short name, buoy id, ...
metric        : string  -- 'stage_ft', 'discharge_cfs', 'temp_f',
                           'precip_in_72h', 'soil_moisture_top',
                           'wave_ht_ft', 'wind_kt', 'pressure_mb',
                           'flood_index_score', ...
value         : float64

We *intentionally* drop string/categorical fields (flood_category,
fire_weather label, ...) from the snapshot store -- they are pure
functions of the numeric columns, so we recompute on read instead of
duplicating storage. Same goes for static metadata (lat/lon, region) which
lives in the code, not in history.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

DEFAULT_HISTORY_DIR = Path("data/history/snapshots")


# ---- snapshot -> long rows ------------------------------------------------

# Per-metric extractors: each pulls a numeric value from a dashboard
# sub-payload and yields (metric, value) pairs. Missing keys are skipped.
def _num(d: dict, key: str) -> float | None:
    """Coerce d[key] to float, returning None for missing / unparseable."""
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Open-Meteo / USGS sometimes emit sentinel ``-999999``
    if f == -999999:
        return None
    return f


def _gauge_rows(g: dict):
    sid = g.get("site_id")
    if not sid:
        return
    for m in ("stage_ft", "discharge_cfs", "rate_ft_per_hr",
              "pct_to_minor", "pct_to_major"):
        v = _num(g, m)
        if v is not None:
            yield ("gauge", sid, m, v)


def _weather_rows(w: dict, *, entity_type: str, entity_id: str):
    if not isinstance(w, dict) or w.get("error"):
        return
    for m in ("temp_f", "humidity_pct", "wind_mph", "wind_dir_deg",
              "pressure_mb", "precip_in", "next_72h_precip_in"):
        v = _num(w, m)
        if v is not None:
            yield (entity_type, entity_id, f"wx_{m}", v)


def _soil_rows(s: dict, *, entity_type: str, entity_id: str):
    if not isinstance(s, dict) or s.get("error"):
        return
    for m in ("soil_moisture_top", "soil_moisture_shallow",
              "soil_moisture_mid", "soil_moisture_root",
              "past_7d_precip_in"):
        v = _num(s, m)
        if v is not None:
            yield (entity_type, entity_id, f"soil_{m}", v)


def _hazard_rows(payload: dict, *, entity_type: str, entity_id: str):
    """Extract numeric scores out of fire_weather / landslide / air_quality."""
    for sub_key, prefix in (("fire_weather", "fire"),
                              ("landslide", "landslide"),
                              ("air_quality", "aqi")):
        sub = payload.get(sub_key) or {}
        if not isinstance(sub, dict) or sub.get("error"):
            continue
        score = _num(sub, "score")
        if score is not None:
            yield (entity_type, entity_id, f"{prefix}_score", score)
        for extra in ("us_aqi", "pm2_5", "pm10", "ozone"):
            v = _num(sub, extra)
            if v is not None:
                yield (entity_type, entity_id, f"{prefix}_{extra}", v)


def _buoy_rows(b: dict):
    bid = b.get("station_id")
    if not bid:
        return
    for m in ("wind_kt", "wind_gust_kt", "wind_dir_deg",
              "wave_ht_ft", "dominant_period_s", "avg_period_s",
              "pressure_mb", "air_temp_f", "water_temp_f"):
        v = _num(b, m)
        if v is not None:
            yield ("buoy", bid, m, v)


def _coastal_rows(c: dict):
    sid = c.get("station_id")
    if not sid:
        return
    for m in ("water_level_ft", "wind_kt", "wind_gust_kt",
              "wind_dir_deg", "air_pressure_mb"):
        v = _num(c, m)
        if v is not None:
            yield ("coastal", sid, m, v)


def snapshot_to_long(state: dict) -> list[tuple[str, str, str, float]]:
    """Flatten one dashboard ``_collect()`` payload to ``(entity_type,
    entity_id, metric, value)`` tuples. Drops anything unparseable."""
    rows: list[tuple[str, str, str, float]] = []

    # Composite Asheville index — single most useful row to keep
    idx = state.get("index") or {}
    score = _num(idx, "score")
    if score is not None:
        rows.append(("point", "asheville", "flood_index_score", score))
    for k, v in (idx.get("components") or {}).items():
        n = _num({"v": v}, "v")
        if n is not None:
            rows.append(("point", "asheville", f"flood_index_{k}", n))

    # Asheville weather + soil
    rows.extend(_weather_rows(state.get("weather") or {},
                              entity_type="point", entity_id="asheville"))
    rows.extend(_soil_rows(state.get("soil") or {},
                            entity_type="point", entity_id="asheville"))

    for g in state.get("gauges") or []:
        rows.extend(_gauge_rows(g))
    for c in state.get("coastal") or []:
        rows.extend(_coastal_rows(c))
    for b in state.get("buoys") or []:
        rows.extend(_buoy_rows(b))

    for f in state.get("forests") or []:
        short = f.get("short")
        if not short:
            continue
        rows.extend(_weather_rows(f.get("weather") or {},
                                   entity_type="forest", entity_id=short))
        rows.extend(_soil_rows(f.get("soil") or {},
                                entity_type="forest", entity_id=short))
        rows.extend(_hazard_rows(f, entity_type="forest", entity_id=short))
        for d in f.get("districts_data") or []:
            dname = d.get("name")
            if not dname:
                continue
            did = f"{short}/{dname}"
            rows.extend(_weather_rows(d.get("weather") or {},
                                       entity_type="district",
                                       entity_id=did))
            rows.extend(_hazard_rows(d, entity_type="district",
                                      entity_id=did))

    return rows


# ---- parquet append -------------------------------------------------------

def _ts_from_state(state: dict):
    """Prefer ``as_of_epoch`` (already UTC), else now()."""
    ep = state.get("as_of_epoch")
    if isinstance(ep, (int, float)) and ep > 0:
        return datetime.fromtimestamp(int(ep), tz=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _partition_path(ts: datetime, base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ts.year:04d}-{ts.month:02d}.parquet"


def append_snapshot(state: dict, *, base_dir: Path | str = DEFAULT_HISTORY_DIR,
                    source: str = "snapshot") -> Path | None:
    """Append one dashboard snapshot to its month-partition parquet file.

    Returns the partition file path written, or None if no rows were extracted
    (defensive: never let history-writing failures break the dashboard).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        log.warning("pandas not installed; skipping history append: %s", exc)
        return None

    rows = snapshot_to_long(state)
    if not rows:
        return None
    ts = _ts_from_state(state)
    df = pd.DataFrame(rows, columns=["entity_type", "entity_id",
                                       "metric", "value"])
    df.insert(0, "ts", pd.Timestamp(ts))
    df.insert(1, "source", source)

    base = Path(base_dir)
    path = _partition_path(ts, base)
    return _append_parquet(df, path)


def _append_parquet(df, path: Path) -> Path:
    """Append df to ``path``; merge with existing rows if file exists.

    Uses pyarrow + pandas. Deduplicates on (ts, entity_type, entity_id, metric,
    source) so re-running the same hour twice is idempotent.
    """
    import pandas as pd

    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["ts", "source", "entity_type", "entity_id", "metric"],
                keep="last",
            )
        except Exception as exc:  # noqa: BLE001 - corrupted partition
            log.warning("could not read existing partition %s, overwriting: %s",
                        path, exc)
            combined = df
    else:
        combined = df

    combined = combined.sort_values(
        ["ts", "entity_type", "entity_id", "metric"]
    ).reset_index(drop=True)
    combined.to_parquet(path, index=False, compression="zstd")
    return path


# ---- read helpers ---------------------------------------------------------

def list_partitions(base_dir: Path | str = DEFAULT_HISTORY_DIR) -> list[Path]:
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(base.glob("*.parquet"))


def load_history(*, base_dir: Path | str = DEFAULT_HISTORY_DIR,
                  start: str | None = None,
                  end: str | None = None,
                  entity_type: str | None = None,
                  entity_id: str | None = None,
                  metric: str | None = None,
                  source: str | None = None):
    """Load every matching row across partitions into one DataFrame.

    All filters are optional; ``start`` / ``end`` are inclusive ISO timestamps.
    Returns an empty DataFrame if no partitions exist yet -- safe to call on
    fresh checkouts.
    """
    import pandas as pd

    parts = list_partitions(base_dir)
    if not parts:
        return pd.DataFrame(columns=["ts", "source", "entity_type",
                                       "entity_id", "metric", "value"])
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if start:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end:
        df = df[df["ts"] <= pd.Timestamp(end)]
    if source:
        df = df[df["source"] == source]
    if entity_type:
        df = df[df["entity_type"] == entity_type]
    if entity_id:
        df = df[df["entity_id"] == entity_id]
    if metric:
        df = df[df["metric"] == metric]
    return df.reset_index(drop=True)


def pivot_metric(df, *, metric: str, entity_id: str | None = None):
    """Wide-form helper: one row per timestamp, one column per entity for
    a chosen metric. Useful for plotting + as ML feature builder input."""
    import pandas as pd

    sub = df[df["metric"] == metric]
    if entity_id is not None:
        sub = sub[sub["entity_id"] == entity_id]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="ts", columns="entity_id",
                            values="value", aggfunc="last")


def drop_entities(entity_ids: Iterable[str],
                   *, entity_type: str | None = None,
                   base_dir: Path | str = DEFAULT_HISTORY_DIR,
                   dry_run: bool = False) -> dict:
    """Delete every row for the given entity ids, partition by partition.

    Needed when a gauge id turns out to have been wrong: the rows are real
    measurements, but of a different river than the label claimed, so leaving
    them in place would silently mix two watersheds into one series.

    Returns ``{"scanned": n, "removed": n, "partitions_rewritten": n}``.
    """
    import pandas as pd

    targets = {str(e) for e in entity_ids}
    if not targets:
        return {"scanned": 0, "removed": 0, "partitions_rewritten": 0}

    scanned = removed = rewritten = 0
    for part in list_partitions(base_dir):
        df = pd.read_parquet(part)
        scanned += len(df)
        mask = df["entity_id"].isin(targets)
        if entity_type is not None:
            mask &= df["entity_type"] == entity_type
        hits = int(mask.sum())
        if not hits:
            continue
        removed += hits
        rewritten += 1
        if not dry_run:
            keep = df[~mask].reset_index(drop=True)
            if keep.empty:
                part.unlink()
            else:
                keep.to_parquet(part, index=False)
    return {"scanned": scanned, "removed": removed,
            "partitions_rewritten": rewritten}


def history_stats(base_dir: Path | str = DEFAULT_HISTORY_DIR) -> dict:
    """Tiny audit summary, suitable for `cli ml-history-info`."""
    parts = list_partitions(base_dir)
    if not parts:
        return {"partitions": 0, "rows": 0, "first_ts": None, "last_ts": None}
    df = load_history(base_dir=base_dir)
    return {
        "partitions": len(parts),
        "rows": len(df),
        "first_ts": str(df["ts"].min()) if not df.empty else None,
        "last_ts": str(df["ts"].max()) if not df.empty else None,
        "metrics": sorted(df["metric"].unique().tolist()),
        "sources": sorted(df["source"].unique().tolist()),
        "entity_count": (df.groupby("entity_type")["entity_id"]
                          .nunique().to_dict()),
    }


def append_long_rows(rows: Iterable[dict],
                      *, base_dir: Path | str = DEFAULT_HISTORY_DIR) -> list[Path]:
    """Append a batch of already-long rows (used by the bootstrap loaders).

    Each row must have keys ``ts, source, entity_type, entity_id, metric,
    value``. Partitions by month-of-ts. Returns list of touched files.
    """
    import pandas as pd

    df = pd.DataFrame(list(rows))
    if df.empty:
        return []
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.dropna(subset=["value"])
    if df.empty:
        return []
    base = Path(base_dir)
    touched: list[Path] = []
    df["_part"] = df["ts"].dt.strftime("%Y-%m")
    for part, group in df.groupby("_part"):
        year, month = part.split("-")
        path = _partition_path(
            datetime(int(year), int(month), 1, tzinfo=timezone.utc),
            base,
        )
        touched.append(_append_parquet(group.drop(columns=["_part"]), path))
    return touched
