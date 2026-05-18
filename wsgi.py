"""WSGI entry point for Azure App Service.

Azure App Service (Oryx) deploys the repo to /home/site/wwwroot and runs:
    gunicorn --bind=0.0.0.0:8000 --timeout 600 wsgi:app

The hurricane_asheville package lives under src/, so we add it to sys.path
before importing the Flask app.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hurricane_asheville.dashboard import app  # noqa: E402

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
