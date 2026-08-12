"""Tests for the parquet history store + bootstrap loaders."""
from __future__ import annotations

import json
from pathlib import Path


def _sample_state():
    return {
        "as_of": "2026-05-27 10:00:00 UTC",
        "as_of_epoch": 1779526800,
        "index": {
            "score": 42, "label": "ALERT", "color": "#c62828",
            "components": {"stage": 10.4, "qpf": 2.6, "storm": 0.0,
                            "rise": 0.0, "alert": 0.0, "soil": 3.1},
            "triggers": {},
        },
        "weather": {"temp_f": 65.9, "humidity_pct": 93,
                     "wind_mph": 5.3, "next_72h_precip_in": 0.64,
                     "pressure_mb": 1017.9, "precip_in": 0.0},
        "soil": {"soil_moisture_top": 0.27, "soil_moisture_shallow": 0.272,
                  "past_7d_precip_in": 2.46},
        "gauges": [
            {"site_id": "03451500", "stage_ft": 4.76,
             "discharge_cfs": 850.0, "rate_ft_per_hr": -0.05},
            {"site_id": "03446000", "stage_ft": 3.03,
             "rate_ft_per_hr": 0.00},
        ],
        "coastal": [
            {"station_id": "8651370", "water_level_ft": 3.17,
             "wind_kt": 12, "air_pressure_mb": 1017.4},
        ],
        "buoys": [
            {"station_id": "41025", "wave_ht_ft": 3.2,
             "wind_kt": 16.5, "pressure_mb": 1013.5, "water_temp_f": 76.1},
        ],
        "forests": [
            {"short": "Pisgah",
             "weather": {"temp_f": 62.5, "next_72h_precip_in": 0.8},
             "soil": {"soil_moisture_top": 0.28, "past_7d_precip_in": 2.1},
             "fire_weather": {"score": 12.3, "label": "CALM"},
             "landslide": {"score": 35, "label": "ELEVATED"},
             "air_quality": {"us_aqi": 18, "pm2_5": 4.1},
             "districts_data": [
                 {"name": "Appalachian",
                  "weather": {"temp_f": 60.1, "next_72h_precip_in": 0.9},
                  "fire_weather": {"score": 8.0, "label": "CALM"},
                  "landslide": {"score": 40, "label": "ELEVATED"},
                  "air_quality": {"us_aqi": 22}},
             ]},
        ],
    }


def test_snapshot_to_long_extracts_expected_metrics():
    from hurricane_asheville.history import snapshot_to_long
    rows = snapshot_to_long(_sample_state())
    # Convert to set of (entity_type, entity_id, metric) for easy assertions
    keys = {(et, eid, m) for (et, eid, m, _v) in rows}
    assert ("point", "asheville", "flood_index_score") in keys
    assert ("point", "asheville", "flood_index_stage") in keys
    assert ("point", "asheville", "wx_temp_f") in keys
    assert ("point", "asheville", "soil_soil_moisture_top") in keys
    assert ("gauge", "03451500", "stage_ft") in keys
    assert ("gauge", "03451500", "discharge_cfs") in keys
    assert ("gauge", "03446000", "stage_ft") in keys
    assert ("coastal", "8651370", "water_level_ft") in keys
    assert ("buoy", "41025", "wave_ht_ft") in keys
    assert ("forest", "Pisgah", "wx_temp_f") in keys
    assert ("forest", "Pisgah", "fire_score") in keys
    assert ("forest", "Pisgah", "landslide_score") in keys
    assert ("forest", "Pisgah", "aqi_us_aqi") in keys
    assert ("district", "Pisgah/Appalachian", "wx_temp_f") in keys
    assert ("district", "Pisgah/Appalachian", "fire_score") in keys
    # No string columns leak through
    assert all(isinstance(v, float) for (_et, _eid, _m, v) in rows)


def test_snapshot_to_long_skips_missing_and_sentinel_values():
    from hurricane_asheville.history import snapshot_to_long
    state = {
        "weather": {"temp_f": None, "humidity_pct": "?"},
        "gauges": [{"site_id": "X", "stage_ft": -999999, "discharge_cfs": "N/A"}],
    }
    rows = snapshot_to_long(state)
    assert rows == []


def test_append_snapshot_writes_monthly_partition(tmp_path: Path):
    from hurricane_asheville.history import (append_snapshot, list_partitions,
                                              load_history)
    state = _sample_state()
    path = append_snapshot(state, base_dir=tmp_path)
    assert path is not None and path.exists()
    assert path.name == "2026-05.parquet"
    assert list_partitions(tmp_path) == [path]

    df = load_history(base_dir=tmp_path)
    assert not df.empty
    assert set(df.columns) == {"ts", "source", "entity_type",
                                 "entity_id", "metric", "value"}
    assert (df["source"] == "snapshot").all()
    # Appending the same state twice must be idempotent (dedup on
    # ts+source+entity+metric).
    append_snapshot(state, base_dir=tmp_path)
    df2 = load_history(base_dir=tmp_path)
    assert len(df2) == len(df)


def test_append_snapshot_merges_separate_hours(tmp_path: Path):
    from hurricane_asheville.history import append_snapshot, load_history
    state_a = _sample_state()
    state_b = json.loads(json.dumps(state_a))  # deep-copy
    state_b["as_of_epoch"] = state_a["as_of_epoch"] + 3600
    state_b["gauges"][0]["stage_ft"] = 5.10  # changed reading

    append_snapshot(state_a, base_dir=tmp_path)
    append_snapshot(state_b, base_dir=tmp_path)
    df = load_history(base_dir=tmp_path, entity_id="03451500",
                       metric="stage_ft")
    assert len(df) == 2
    assert sorted(df["value"].tolist()) == [4.76, 5.10]


def test_append_long_rows_partitions_by_month(tmp_path: Path):
    from hurricane_asheville.history import (append_long_rows, list_partitions,
                                              load_history)
    rows = [
        {"ts": "2025-01-05T12:00:00Z", "source": "usgs_dv",
         "entity_type": "gauge", "entity_id": "03451500",
         "metric": "stage_ft", "value": 2.1},
        {"ts": "2025-02-10T12:00:00Z", "source": "usgs_dv",
         "entity_type": "gauge", "entity_id": "03451500",
         "metric": "stage_ft", "value": 2.4},
        {"ts": "2025-02-20T12:00:00Z", "source": "usgs_dv",
         "entity_type": "gauge", "entity_id": "03451500",
         "metric": "stage_ft", "value": 2.7},
    ]
    written = append_long_rows(rows, base_dir=tmp_path)
    assert len(set(written)) == 2  # Jan + Feb partitions
    df = load_history(base_dir=tmp_path)
    assert len(df) == 3
    assert (df["source"] == "usgs_dv").all()


def test_pivot_metric_wide_form(tmp_path: Path):
    from hurricane_asheville.history import (append_long_rows, load_history,
                                              pivot_metric)
    rows = [
        {"ts": "2025-01-05T12:00:00Z", "source": "usgs_dv",
         "entity_type": "gauge", "entity_id": "03451500",
         "metric": "stage_ft", "value": 2.1},
        {"ts": "2025-01-05T12:00:00Z", "source": "usgs_dv",
         "entity_type": "gauge", "entity_id": "03446000",
         "metric": "stage_ft", "value": 1.4},
    ]
    append_long_rows(rows, base_dir=tmp_path)
    wide = pivot_metric(load_history(base_dir=tmp_path), metric="stage_ft")
    assert sorted(wide.columns.tolist()) == ["03446000", "03451500"]
    assert wide.iloc[0]["03451500"] == 2.1


def test_load_history_returns_empty_df_on_fresh_checkout(tmp_path: Path):
    from hurricane_asheville.history import load_history
    df = load_history(base_dir=tmp_path / "does-not-exist")
    assert df.empty
    assert list(df.columns) == ["ts", "source", "entity_type",
                                 "entity_id", "metric", "value"]


def test_history_stats_counts_partitions(tmp_path: Path):
    from hurricane_asheville.history import append_snapshot, history_stats
    assert history_stats(tmp_path)["partitions"] == 0
    append_snapshot(_sample_state(), base_dir=tmp_path)
    stats = history_stats(tmp_path)
    assert stats["partitions"] == 1
    assert stats["rows"] > 0
    assert stats["sources"] == ["snapshot"]
    assert "gauge" in stats["entity_count"]


# ---- drop_entities --------------------------------------------------------

def _rows(pairs):
    """pairs: [(ts, entity_id, metric, value)]"""
    return [{"ts": t, "source": "usgs_dv", "entity_type": "gauge",
             "entity_id": e, "metric": m, "value": v} for t, e, m, v in pairs]


def test_drop_entities_removes_only_the_targets(tmp_path):
    from hurricane_asheville import history

    history.append_long_rows(_rows([
        ("2024-09-27", "keep_me", "stage_ft", 1.0),
        ("2024-09-27", "drop_me", "stage_ft", 2.0),
        ("2024-09-28", "drop_me", "stage_ft", 3.0),
    ]), base_dir=tmp_path)

    out = history.drop_entities(["drop_me"], base_dir=tmp_path)
    assert out["removed"] == 2
    remaining = history.load_history(base_dir=tmp_path)
    assert set(remaining["entity_id"]) == {"keep_me"}


def test_drop_entities_dry_run_changes_nothing(tmp_path):
    from hurricane_asheville import history

    history.append_long_rows(_rows([
        ("2024-09-27", "drop_me", "stage_ft", 2.0),
    ]), base_dir=tmp_path)

    out = history.drop_entities(["drop_me"], base_dir=tmp_path, dry_run=True)
    assert out["removed"] == 1
    assert len(history.load_history(base_dir=tmp_path)) == 1


def test_drop_entities_respects_entity_type(tmp_path):
    from hurricane_asheville import history

    rows = _rows([("2024-09-27", "shared_id", "stage_ft", 1.0)])
    rows += [{"ts": "2024-09-27", "source": "snapshot",
              "entity_type": "buoy", "entity_id": "shared_id",
              "metric": "wind_kt", "value": 9.0}]
    history.append_long_rows(rows, base_dir=tmp_path)

    history.drop_entities(["shared_id"], entity_type="gauge", base_dir=tmp_path)
    remaining = history.load_history(base_dir=tmp_path)
    assert list(remaining["entity_type"]) == ["buoy"]


def test_drop_entities_no_targets_is_a_noop(tmp_path):
    from hurricane_asheville import history
    assert history.drop_entities([], base_dir=tmp_path)["removed"] == 0


# ---- two stores, one logical history --------------------------------------

def test_load_history_spans_snapshots_and_backfill(tmp_path, monkeypatch):
    """Callers see one store even though writers are separated."""
    from hurricane_asheville import history

    snaps = tmp_path / "snapshots"
    back = tmp_path / "backfill"
    monkeypatch.setattr(history, "HISTORY_DIRS", (snaps, back))

    history.append_long_rows(_rows([("2026-08-01", "G", "stage_ft", 1.0)]),
                              base_dir=snaps)
    history.append_long_rows(_rows([("2021-08-01", "G", "stage_ft", 2.0)]),
                              base_dir=back)

    df = history.load_history()
    assert len(df) == 2
    assert set(df["value"]) == {1.0, 2.0}


def test_backfill_writer_does_not_touch_the_snapshot_store(tmp_path):
    """The whole point: the hourly job and a backfill must never write the
    same file, or every hourly commit conflicts irreconcilably."""
    from hurricane_asheville import history

    assert history.DEFAULT_BACKFILL_DIR != history.DEFAULT_HISTORY_DIR
    # append_long_rows (the backfill writer) defaults to the backfill store.
    import inspect
    sig = inspect.signature(history.append_long_rows)
    assert sig.parameters["base_dir"].default == history.DEFAULT_BACKFILL_DIR
    # append_snapshot (the live writer) defaults to the snapshot store.
    sig = inspect.signature(history.append_snapshot)
    assert sig.parameters["base_dir"].default == history.DEFAULT_HISTORY_DIR


def test_list_partitions_explicit_dir_is_unaffected(tmp_path):
    from hurricane_asheville import history
    history.append_long_rows(_rows([("2026-08-01", "G", "stage_ft", 1.0)]),
                              base_dir=tmp_path)
    assert len(history.list_partitions(tmp_path)) == 1
