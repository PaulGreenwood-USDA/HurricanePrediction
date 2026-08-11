"""Replay the Asheville Flood Index across the historical record.

The AFI has only existed since the hourly snapshot job started, and in that
window nothing has happened -- it has never seen a flood. Its six component
weights were chosen by hand and have never been checked against an event. The
question this module answers is the one the whole project exists for:

    Would this dashboard have warned Asheville about Helene?

Reconstruction
==============
:func:`hurricane_asheville.index_score.compute_index` takes plain dicts, so the
replay feeds it historical inputs and calls the *same* scorer the live page
uses. No parallel implementation to drift.

Component            Source                         Fidelity
-------------------  -----------------------------  ------------------------
stage                USGS daily-mean stage          daily mean, not peak
qpf                  ERA5 observed precipitation    perfect-foresight proxy
storm                HURDAT2 best track             exact
rise                 day-over-day stage change      badly understated
soil                 ERA5 soil moisture 0-7 cm      different depth to live
alert                unavailable                    always zero

Three of these bias the replayed score **downward**, and one biases it up:

* Daily-mean stage understates a flash crest. Helene's mean for 27 Sep is
  18.5 ft against a ~24.8 ft crest.
* Rate-of-rise from daily means is a ~24-hour average, so a river that rose
  20 ft in twelve hours reads as ~0.8 ft/hr rather than the multiple ft/hr it
  actually did. This component is effectively dead in the replay.
* NWS alert state is not archived anywhere retrievable, so the 10-point alert
  component is always zero even during an active flood warning.
* QPF uses *observed* rainfall, which is what a perfect forecast would have
  said. The live index uses a real forecast, which is worse. This makes the
  replay optimistic.

The net effect is that a replayed score is a **lower bound in stage/rise/alert
terms and an upper bound in QPF terms**. If the index fails to fire on Helene
even with perfect rainfall foresight, that is a strong negative result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import ASHEVILLE_LAT, ASHEVILLE_LON
from .gauge import SITE_FRENCH_BROAD_ASHEVILLE
from .geo import haversine_mi
from .index_score import compute_index

log = logging.getLogger(__name__)

# Events worth calling out by name when reporting a replay.
KNOWN_EVENTS = {
    "2024-09-27": "Helene",
    "2021-08-17": "Fred",
    "2024-09-26": "Helene (approach)",
}


@dataclass
class ReplayDay:
    date: str
    score: int
    label: str
    components: dict
    triggers: dict
    stage_ft: float | None
    precip_72h_in: float | None
    soil_moisture: float | None
    nearest_storm_mi: float | None
    storm_name: str | None = None


@dataclass
class ReplayResult:
    days: list = field(default_factory=list)
    first_date: str | None = None
    last_date: str | None = None
    peak: ReplayDay | None = None
    #: Days at or above each label, for judging how often the index cries wolf.
    label_counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ---- historical inputs ----------------------------------------------------

def _daily(history_df, entity_id: str, metric: str, source: str | None = None):
    """One value per calendar day for an (entity, metric) pair."""
    import pandas as pd

    df = history_df
    mask = (df["entity_id"] == entity_id) & (df["metric"] == metric)
    if source:
        mask &= df["source"] == source
    sub = df[mask]
    if sub.empty:
        return pd.Series(dtype="float64")
    s = sub.set_index("ts")["value"].sort_index()
    return s.resample("1D").max()


def storm_distances(start: str, end: str, cache_dir="data"):
    """Closest Atlantic TC approach to Asheville per day, from HURDAT2.

    Returns ``{date: (miles, storm_name)}``. Unlike the other inputs this is
    exact: HURDAT2 is the post-season best track, so positions are final.
    """
    import pandas as pd

    from .hurdat import load_hurdat2

    try:
        tracks = load_hurdat2(cache_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("HURDAT2 unavailable, storm component will be zero: %s", exc)
        return {}

    window = tracks[(tracks["datetime"] >= pd.Timestamp(start)) &
                    (tracks["datetime"] <= pd.Timestamp(end))]
    if window.empty:
        return {}

    out: dict[str, tuple[float, str]] = {}
    for row in window.itertuples():
        miles = haversine_mi(ASHEVILLE_LAT, ASHEVILLE_LON, row.lat, row.lon)
        day = row.datetime.strftime("%Y-%m-%d")
        if day not in out or miles < out[day][0]:
            out[day] = (float(miles), str(row.name))
    return out


def _rolling_forward_sum(series, days: int = 3):
    """Observed precipitation over the *next* N days, as a QPF stand-in.

    The live index consumes a 72-hour forecast. Its historical equivalent is
    what actually fell over the following 72 hours -- i.e. a perfect forecast.
    """
    return series[::-1].rolling(days, min_periods=1).sum()[::-1]


# ---- replay ---------------------------------------------------------------

def replay(history_df=None, *, site_id: str = SITE_FRENCH_BROAD_ASHEVILLE,
           weather_entity: str = "asheville",
           start: str | None = None, end: str | None = None,
           cache_dir="data") -> ReplayResult:
    """Recompute the Flood Index for every day we have inputs for."""
    import pandas as pd

    if history_df is None:
        from .history import load_history
        history_df = load_history()
    if history_df is None or len(history_df) == 0:
        return ReplayResult(notes=["no history available"])

    stage = _daily(history_df, site_id, "stage_ft")
    precip = _daily(history_df, weather_entity, "wx_precip_in_24h")
    soil = _daily(history_df, weather_entity, "soil_era5_0_7cm")

    if stage.empty:
        return ReplayResult(notes=[f"no stage history for {site_id}"])

    index = stage.index
    if not precip.empty:
        index = index.union(precip.index)
    index = index.sort_values()
    if start:
        index = index[index >= pd.Timestamp(start, tz="UTC")]
    if end:
        index = index[index <= pd.Timestamp(end, tz="UTC")]
    if len(index) == 0:
        return ReplayResult(notes=["no overlapping days in range"])

    stage = stage.reindex(index)
    precip = precip.reindex(index).fillna(0.0) if not precip.empty else None
    soil = soil.reindex(index) if not soil.empty else None
    qpf = _rolling_forward_sum(precip) if precip is not None else None
    # Past-7-day antecedent rainfall, the other half of the soil component.
    past7 = (precip.rolling(7, min_periods=1).sum()
             if precip is not None else None)

    storms = storm_distances(str(index.min().date()), str(index.max().date()),
                             cache_dir=cache_dir)

    days: list[ReplayDay] = []
    for ts in index:
        day = ts.strftime("%Y-%m-%d")
        stage_v = stage.get(ts)
        stage_v = None if pd.isna(stage_v) else float(stage_v)

        prev = stage.shift(1).get(ts)
        rate = (None if (prev is None or pd.isna(prev) or stage_v is None)
                else (stage_v - float(prev)) / 24.0)

        soil_v = None if soil is None else soil.get(ts)
        soil_v = None if soil_v is None or pd.isna(soil_v) else float(soil_v)
        past_v = None if past7 is None else float(past7.get(ts, 0.0))

        soil_payload = None
        if soil_v is not None or past_v is not None:
            soil_payload = {
                # ERA5 0-7 cm stands in for the live 0-1 cm layer; same
                # quantity and units, coarser depth. Documented in the module
                # docstring rather than silently aliased.
                "soil_moisture_top": soil_v,
                "past_7d_precip_in": past_v,
                "saturated": bool(soil_v is not None and soil_v >= 0.40),
            }

        miles, name = storms.get(day, (None, None))
        storm_payload = ([{"distance_mi": miles}] if miles is not None else [])

        result = compute_index(
            primary_gauge={"stage_ft": stage_v},
            rate_ft_per_hr=rate,
            storms=storm_payload,
            alerts=[],          # never archived; see module docstring
            weather={"next_72h_precip_in":
                     None if qpf is None else float(qpf.get(ts, 0.0))},
            soil=soil_payload,
        )
        days.append(ReplayDay(
            date=day, score=result.score, label=result.label,
            components=result.components, triggers=result.triggers,
            stage_ft=stage_v,
            precip_72h_in=None if qpf is None else float(qpf.get(ts, 0.0)),
            soil_moisture=soil_v,
            nearest_storm_mi=miles,
            storm_name=name,
        ))

    peak = max(days, key=lambda d: d.score) if days else None
    counts: dict[str, int] = {}
    for d in days:
        counts[d.label] = counts.get(d.label, 0) + 1

    notes = [
        "alert component is always 0: NWS alert state is not archived",
        "rate-of-rise is a 24 h average, so flash rises are heavily understated",
        "stage is a daily mean, not the crest",
        "qpf uses observed rainfall, i.e. assumes a perfect 72 h forecast",
    ]
    return ReplayResult(days=days, first_date=days[0].date,
                        last_date=days[-1].date, peak=peak,
                        label_counts=counts, notes=notes)


VALIDATION_PATH = "data/index_validation.json"


def build_validation(result: ReplayResult,
                     *, event_date: str = "2024-09-27",
                     event_name: str = "Helene",
                     alert_threshold: int = 40) -> dict:
    """Condense a replay into the summary the dashboard shows.

    The page needs three numbers to be credible: what the index read during the
    event, how much warning it gave, and how often it fires when nothing is
    happening.
    """
    import datetime as dt

    if not result.days:
        return {}

    by_date = {d.date: d for d in result.days}
    peak_event = max(
        (d for d in event_summary(result, event_date, window_days=2)),
        key=lambda d: d.score, default=None)

    # Consecutive days at or above the alert threshold running up to the event.
    lead_days = 0
    target = dt.date.fromisoformat(event_date)
    while True:
        prev = (target - dt.timedelta(days=lead_days + 1)).isoformat()
        if prev in by_date and by_date[prev].score >= alert_threshold:
            lead_days += 1
        else:
            break

    elevated = [d for d in result.days if d.score >= alert_threshold]
    named = [d for d in elevated if d.storm_name]

    return {
        "generated_from": {"first_date": result.first_date,
                           "last_date": result.last_date,
                           "days": len(result.days)},
        "event": {
            "name": event_name,
            "date": event_date,
            "score": peak_event.score if peak_event else None,
            "label": peak_event.label if peak_event else None,
            "components": peak_event.components if peak_event else {},
            "stage_ft": peak_event.stage_ft if peak_event else None,
            "lead_days": lead_days,
        },
        "base_rate": {
            "alert_or_above_days": len(elevated),
            "total_days": len(result.days),
            "pct": round(len(elevated) / len(result.days) * 100, 2),
            "with_named_storm": len(named),
        },
        "label_counts": result.label_counts,
        "elevated_days": [
            {"date": d.date, "score": d.score, "label": d.label,
             "storm": d.storm_name, "stage_ft": d.stage_ft}
            for d in elevated
        ],
        "caveats": result.notes,
    }


def write_validation(result: ReplayResult, path: str = VALIDATION_PATH,
                     **kwargs) -> str:
    """Persist the validation summary next to the other baked data files.

    Recomputing a five-year replay on every dashboard render would mean
    loading the whole history store and HURDAT2 per request, so it is baked
    like the flood thresholds and refreshed by an explicit CLI run.
    """
    import json
    from pathlib import Path

    payload = build_validation(result, **kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return str(out)


def load_validation(path: str = VALIDATION_PATH) -> dict:
    """Read the baked validation summary; empty dict if absent."""
    import json
    from pathlib import Path

    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def event_summary(result: ReplayResult, date: str,
                  window_days: int = 3) -> list[ReplayDay]:
    """The replayed days around a named event, for reporting."""
    import datetime as dt

    target = dt.date.fromisoformat(date)
    lo = target - dt.timedelta(days=window_days)
    hi = target + dt.timedelta(days=window_days)
    return [d for d in result.days
            if lo <= dt.date.fromisoformat(d.date) <= hi]
