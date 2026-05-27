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


def main() -> None:
    out = _ROOT / "site"
    out.mkdir(parents=True, exist_ok=True)

    client = app.test_client()

    # Render the dashboard. If a transient upstream API failure causes Flask
    # to return 500, keep any previously deployed index.html instead of
    # failing the Pages build (which would leave the site stale anyway).
    html = client.get("/")
    if html.status_code == 200 and html.data:
        (out / "index.html").write_bytes(html.data)
        print(f"wrote {out / 'index.html'} ({len(html.data)} bytes)")
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

    (out / ".nojekyll").write_text("")


if __name__ == "__main__":
    main()
