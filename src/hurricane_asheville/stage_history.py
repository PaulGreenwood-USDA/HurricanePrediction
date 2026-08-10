"""Long-run stage history for the primary gauge, and where today sits in it.

The parquet snapshot store goes back to 2021 and contains Helene, but nothing
on the dashboard ever read it -- the only time series on the page were the
24-hour USGS sparklines. This module answers the question the dashboard exists
for: *how does today compare to the thing that happened in September 2024?*

Two products:

``stage_series``
    A downsampled daily-maximum series suitable for a wide sparkline, plus the
    record crest and the Helene peak as reference lines.

``stage_percentile``
    Where the current reading sits among historical readings **for the same
    calendar month**, which is the only comparison that means anything on a
    river with a strong seasonal cycle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Helene's crest at Asheville. The USGS gauge stopped reporting partway up, so
# the recorded daily value understates it; NWS carries the official crest.
HELENE_DATE = "2024-09-27"
HELENE_LABEL = "Helene"


@dataclass
class StageHistory:
    points: list          # [{"t": iso date, "ft": float}, ...] daily maxima
    vmin: float
    vmax: float
    record_ft: float | None
    record_label: str
    helene_ft: float | None
    helene_date: str | None
    current_ft: float | None
    percentile: float | None
    percentile_month: str | None
    n_observations: int
    first_ts: str | None
    last_ts: str | None
    gauge_truncated: bool = False


def _empty() -> StageHistory:
    return StageHistory(points=[], vmin=0.0, vmax=1.0, record_ft=None,
                        record_label="", helene_ft=None, helene_date=None,
                        current_ft=None, percentile=None,
                        percentile_month=None, n_observations=0,
                        first_ts=None, last_ts=None)


def build(*, site_id: str, current_ft: float | None,
          thresholds: dict | None = None,
          history_df=None) -> StageHistory:
    """Assemble the long-run stage picture for one gauge.

    ``history_df`` is injectable so tests do not need the parquet store.
    Returns an empty result rather than raising when history is unavailable --
    the card simply does not render.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dep in practice
        return _empty()

    if history_df is None:
        try:
            from .history import load_history
            history_df = load_history(entity_id=site_id, metric="stage_ft")
        except Exception as exc:  # noqa: BLE001
            log.warning("stage history unavailable: %s", exc)
            return _empty()

    if history_df is None or len(history_df) == 0:
        return _empty()

    df = history_df
    if "value" not in df.columns or "ts" not in df.columns:
        return _empty()

    series = (df.set_index("ts")["value"].sort_index().dropna())
    if series.empty:
        return _empty()

    # Daily maxima keep flood peaks intact while shrinking ~3k points to a
    # size the page can draw. A daily *mean* would flatten Helene away.
    daily = series.resample("1D").max().dropna()
    if daily.empty:
        return _empty()

    points = [{"t": ts.strftime("%Y-%m-%d"), "ft": round(float(v), 2)}
              for ts, v in daily.items()]

    observed_peak = float(daily.max())
    record_ft = (thresholds or {}).get("record")
    record_ft = float(record_ft) if record_ft is not None else None

    helene_ft = None
    helene_date = None
    helene_window = daily.loc[
        (daily.index >= pd.Timestamp(HELENE_DATE, tz="UTC") - pd.Timedelta(days=3)) &
        (daily.index <= pd.Timestamp(HELENE_DATE, tz="UTC") + pd.Timedelta(days=3))
    ] if daily.index.tz is not None else daily.loc[
        (daily.index >= pd.Timestamp(HELENE_DATE) - pd.Timedelta(days=3)) &
        (daily.index <= pd.Timestamp(HELENE_DATE) + pd.Timedelta(days=3))
    ]
    if not helene_window.empty:
        helene_ft = float(helene_window.max())
        helene_date = helene_window.idxmax().strftime("%Y-%m-%d")

    # The axis has to hold the record even though no observation reaches it,
    # otherwise the reference line falls outside the chart.
    vmax = max(observed_peak, record_ft or 0.0, current_ft or 0.0) * 1.05
    vmin = 0.0

    percentile, month_name = _percentile_for_month(series, current_ft)

    return StageHistory(
        points=points,
        vmin=vmin,
        vmax=vmax,
        record_ft=record_ft,
        record_label="NWS record crest",
        helene_ft=helene_ft,
        helene_date=helene_date,
        current_ft=current_ft,
        percentile=percentile,
        percentile_month=month_name,
        n_observations=int(len(series)),
        first_ts=str(series.index.min().date()),
        last_ts=str(series.index.max().date()),
        # Helene overtopped the gauge; the recorded peak is short of the
        # official crest, and saying so is the difference between "we measured
        # 18 ft" and "the river went higher than the gauge could report".
        gauge_truncated=bool(
            helene_ft is not None and record_ft is not None
            and record_ft - helene_ft > 1.0),
    )


def _percentile_for_month(series, current_ft: float | None):
    """Percentile of ``current_ft`` among readings in the same month.

    Month-of-year matters: 1.7 ft is unremarkable in August and would be
    notable in March. Comparing against the whole year hides that.
    """
    if current_ft is None or series.empty:
        return None, None
    try:
        month = series.index[-1].month
        same_month = series[series.index.month == month]
        if len(same_month) < 30:
            return None, None
        pct = float((same_month <= current_ft).sum()) / len(same_month) * 100.0
        import calendar
        return round(pct, 1), calendar.month_name[month]
    except Exception as exc:  # noqa: BLE001
        log.warning("percentile failed: %s", exc)
        return None, None


def chart_points(hist: StageHistory, *, width: float = 1000.0,
                 height: float = 140.0) -> dict:
    """Scale a StageHistory into SVG coordinates for the template."""
    if not hist.points:
        return {}
    vrng = (hist.vmax - hist.vmin) or 1.0

    def y_for(ft: float) -> float:
        return height - ((ft - hist.vmin) / vrng) * height

    n = len(hist.points)
    coords = " ".join(
        f"{(i / (n - 1)) * width:.1f},{y_for(p['ft']):.1f}"
        for i, p in enumerate(hist.points)) if n > 1 else ""

    def marker(ft):
        return None if ft is None else round(y_for(float(ft)), 1)

    # Year boundaries give the eye something to anchor on.
    year_ticks = []
    seen = set()
    for i, p in enumerate(hist.points):
        year = p["t"][:4]
        if year not in seen:
            seen.add(year)
            year_ticks.append({"x": (i / (n - 1)) * width if n > 1 else 0,
                               "label": year})

    helene_x = None
    if hist.helene_date:
        for i, p in enumerate(hist.points):
            if p["t"] == hist.helene_date:
                helene_x = (i / (n - 1)) * width if n > 1 else 0
                break

    return {
        "points": coords,
        "width": width,
        "height": height,
        "y_record": marker(hist.record_ft),
        "y_helene": marker(hist.helene_ft),
        "y_current": marker(hist.current_ft),
        "helene_x": helene_x,
        "year_ticks": year_ticks,
    }
