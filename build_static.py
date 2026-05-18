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

    html = client.get("/")
    if html.status_code != 200:
        raise SystemExit(f"render failed: HTTP {html.status_code}")
    (out / "index.html").write_bytes(html.data)

    state = client.get("/api/state")
    if state.status_code == 200:
        data = json.loads(state.data)
        (out / "state.json").write_text(json.dumps(data, indent=2, default=str))

    (out / ".nojekyll").write_text("")
    print(f"wrote {out / 'index.html'} ({len(html.data)} bytes)")


if __name__ == "__main__":
    main()
