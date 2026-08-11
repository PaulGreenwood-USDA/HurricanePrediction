"""Regenerate ``data/nws_flood_thresholds.json`` from the NWS NWPS API.

Every USGS gauge on the dashboard sits on its own datum, so each one needs its
own action / minor / moderate / major stages. NWPS is the authority for those
numbers, but it rate-limits to **10 requests per 5 minutes** -- far too tight
to query at render time for ~25 gauges. So we bake the table into the repo and
refresh it manually when NWS revises a threshold (rare; a few times a year).

Usage::

    uv run python scripts/refresh_flood_thresholds.py           # all gauges
    uv run python scripts/refresh_flood_thresholds.py 03451500  # one site

The script is deliberately slow: it sleeps between calls to stay under the
rate limit. A full refresh takes roughly 15 minutes.

Sites NWPS has no forecast point for are written with ``"thresholds": null``
so the dashboard can say "no thresholds published" rather than silently
measuring them against some other river's flood stage.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from hurricane_asheville.forests import FOREST_GAUGES  # noqa: E402
from hurricane_asheville.gauge import UPSTREAM_GAUGES  # noqa: E402

OUT_PATH = _ROOT / "data" / "nws_flood_thresholds.json"
NWPS_GAUGE_URL = "https://api.water.noaa.gov/nwps/v1/gauges/{ident}"

# 10 requests / 5 minutes => one request per 30 s, plus headroom.
SLEEP_SECONDS = 33
HEADERS = {"User-Agent": "hurricane-asheville/0.1 (threshold refresh)"}


def all_sites() -> dict[str, str]:
    """Every USGS site the dashboard renders, as {site_id: label}."""
    sites: dict[str, str] = {}
    for site_id, label, _lat, _lon, _role in UPSTREAM_GAUGES:
        sites[site_id] = label
    for entries in FOREST_GAUGES.values():
        for site_id, label, _lat, _lon, _role in entries:
            sites.setdefault(site_id, label)
    return sites


def fetch_one(site_id: str) -> dict | None:
    """Ask NWPS for a gauge by USGS id. Returns the raw payload or None."""
    try:
        r = requests.get(NWPS_GAUGE_URL.format(ident=site_id),
                         timeout=30, headers=HEADERS)
    except requests.RequestException as exc:
        print(f"  {site_id}: request failed ({exc})")
        return None
    if r.status_code == 404:
        print(f"  {site_id}: no NWPS forecast point")
        return None
    if r.status_code == 429:
        print(f"  {site_id}: rate limited -- backing off 5 min")
        time.sleep(300)
        return fetch_one(site_id)
    if r.status_code != 200:
        print(f"  {site_id}: HTTP {r.status_code}")
        return None
    try:
        return r.json()
    except ValueError:
        print(f"  {site_id}: unparseable JSON")
        return None


# NWPS encodes "this level is not defined at this gauge" as -9999, not null.
# Taken literally it makes every reading exceed every threshold, so a gauge
# would report MAJOR FLOOD forever.
_NWPS_UNDEFINED = -999.0


def _stage(value) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value <= _NWPS_UNDEFINED:
        return None
    return float(value)


def extract(payload: dict, site_id: str) -> dict | None:
    """Pull the threshold block out of an NWPS gauge payload.

    Returns None if the payload is for a different USGS site than we asked
    for -- attaching one river's flood stages to another is the exact bug
    this table exists to prevent.
    """
    returned = payload.get("usgsId")
    if returned and returned != site_id:
        print(f"  !! NWPS returned usgsId {returned!r} for {site_id!r}; skipping")
        return None

    cats = ((payload.get("flood") or {}).get("categories") or {})
    out: dict = {}
    for level in ("action", "minor", "moderate", "major"):
        out[level] = _stage((cats.get(level) or {}).get("stage"))
    if all(v is None for v in out.values()):
        return None

    # Thresholds must ascend; a non-monotone set would mask categories.
    defined = [v for v in (out["action"], out["minor"],
                           out["moderate"], out["major"]) if v is not None]
    if defined != sorted(defined):
        print(f"  !! non-monotone thresholds {defined}; skipping")
        return None

    # Record crest, for context on the primary gauge.
    crests = ((payload.get("flood") or {}).get("crests") or {}).get("historic") or []
    stages = [s for s in (_stage(c.get("stage")) for c in crests)
              if s is not None]
    out["record"] = max(stages) if stages else None

    out["lid"] = payload.get("lid")
    out["nws_name"] = payload.get("name")
    out["units"] = (payload.get("flood") or {}).get("stageUnits") or "ft"
    return out


def main() -> None:
    wanted = sys.argv[1:]
    sites = all_sites()
    if wanted:
        sites = {k: v for k, v in sites.items() if k in wanted}
        if not sites:
            raise SystemExit(f"no known gauge among {wanted}")

    # Preserve entries we are not refreshing this run.
    existing: dict = {}
    if OUT_PATH.exists():
        try:
            existing = (json.loads(OUT_PATH.read_text()).get("sites") or {})
        except (OSError, ValueError):
            existing = {}

    results: dict[str, dict | None] = dict(existing)
    total = len(sites)
    for i, (site_id, label) in enumerate(sorted(sites.items()), start=1):
        print(f"[{i}/{total}] {site_id}  {label}")
        payload = fetch_one(site_id)
        thresholds = extract(payload, site_id) if payload else None
        if thresholds:
            results[site_id] = {**thresholds, "usgs_label": label}
            print(f"  -> action {thresholds['action']} / minor {thresholds['minor']}"
                  f" / moderate {thresholds['moderate']} / major {thresholds['major']}")
        else:
            results.pop(site_id, None)
            print("  -> no published thresholds; will render as 'no thresholds'")
        if i < total:
            time.sleep(SLEEP_SECONDS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "_source": "NWS National Water Prediction Service, api.water.noaa.gov/nwps/v1",
        "_generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_regenerate_with": "uv run python scripts/refresh_flood_thresholds.py",
        "_note": ("Stages are in feet on each gauge's own datum. Sites absent "
                  "from this table have no NWS forecast point and must not be "
                  "classified against another gauge's thresholds."),
        "sites": dict(sorted(results.items())),
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_PATH} ({len(results)} sites with thresholds)")


if __name__ == "__main__":
    main()
