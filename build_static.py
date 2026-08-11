"""Build a static snapshot of the dashboard for GitHub Pages.

Runs the same data collection the Flask dashboard does, renders the page once,
and writes it (plus the JSON state) to site/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hurricane_asheville.dashboard import app  # noqa: E402


def _copy_assets(out: Path) -> None:
    """Copy templates' static assets next to index.html.

    Flask serves these from /static at runtime, but Pages hosts the site under
    a project subpath where a root-absolute /static/... would 404, so the HTML
    is rewritten to reference them relatively.
    """
    import shutil
    src = Path(app.static_folder)
    dest = out / "static"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"copied {len(list(dest.iterdir()))} static assets -> {dest}")


def main() -> None:
    out = _ROOT / "site"
    out.mkdir(parents=True, exist_ok=True)

    # Tells the dashboard to point its refresh poller at state.json rather
    # than the Flask-only /api/state route.
    app.config["STATIC_BUILD"] = True
    client = app.test_client()

    # Render the dashboard. If a transient upstream API failure causes Flask
    # to return 500, keep any previously deployed index.html instead of
    # failing the Pages build (which would leave the site stale anyway).
    html = client.get("/")
    if html.status_code == 200 and html.data:
        page = html.data.replace(b'"/static/', b'"static/')
        (out / "index.html").write_bytes(page)
        _copy_assets(out)
        print(f"wrote {out / 'index.html'} ({len(page)} bytes)")
    else:
        existing = out / "index.html"
        print(
            f"WARNING: render returned HTTP {html.status_code}; "
            f"{'keeping existing index.html' if existing.exists() else 'no prior index.html to fall back to'}",
            file=sys.stderr,
        )
        if not existing.exists():
            raise SystemExit(f"render failed: HTTP {html.status_code}")

    state = client.get("/api/state")
    if state.status_code == 200:
        try:
            data = json.loads(state.data)
            (out / "state.json").write_text(json.dumps(data, indent=2, default=str))
        except (ValueError, TypeError) as exc:
            print(f"WARNING: could not serialize state.json: {exc}", file=sys.stderr)

    # Append this snapshot to the long-form parquet history store. Never let
    # history errors break the Pages deploy.
    try:
        from hurricane_asheville.history import (DEFAULT_HISTORY_DIR,
                                                  append_snapshot,
                                                  history_stats)
        if state.status_code == 200:
            data = json.loads(state.data)
            path = append_snapshot(data, base_dir=DEFAULT_HISTORY_DIR)
            if path is not None:
                stats = history_stats()
                print(f"history: appended -> {path}  "
                      f"(rows={stats['rows']}, partitions={stats['partitions']})")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: history append skipped: {exc}", file=sys.stderr)

    (out / ".nojekyll").write_text("")


if __name__ == "__main__":
    main()
