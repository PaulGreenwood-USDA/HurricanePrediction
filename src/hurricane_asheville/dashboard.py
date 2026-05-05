"""Real-time dashboard for Asheville hurricane risk.

Run with:
    uv run hurricane-asheville dashboard
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import time
from dataclasses import asdict

from flask import Flask, jsonify, render_template_string

from .active import fetch_active_storms
from .config import ASHEVILLE_LAT, ASHEVILLE_LON, CSU_2026_FORECAST
from .gauge import FLOOD_STAGES_FT, fetch_gauge, fetch_nws_alerts
from .weather import fetch_current_weather

app = Flask(__name__)

# Simple in-process cache so refreshes don't hammer the upstream APIs.
_CACHE: dict = {"data": None, "ts": 0.0}
_TTL_SECONDS = 60.0


def _collect():
    now = time.time()
    if _CACHE["data"] is not None and now - _CACHE["ts"] < _TTL_SECONDS:
        return _CACHE["data"]

    gauge = fetch_gauge()
    storms = fetch_active_storms()
    alerts = fetch_nws_alerts(ASHEVILLE_LAT, ASHEVILLE_LON)
    weather = fetch_current_weather(ASHEVILLE_LAT, ASHEVILLE_LON)

    # Risk level: combine signals into a 0-4 score
    level = 0
    reasons = []
    if gauge and gauge.stage_ft is not None:
        if gauge.stage_ft >= FLOOD_STAGES_FT["major"]:
            level = max(level, 4); reasons.append("French Broad in MAJOR flood")
        elif gauge.stage_ft >= FLOOD_STAGES_FT["moderate"]:
            level = max(level, 3); reasons.append("French Broad moderate flood")
        elif gauge.stage_ft >= FLOOD_STAGES_FT["minor"]:
            level = max(level, 2); reasons.append("French Broad minor flood")
        elif gauge.stage_ft >= FLOOD_STAGES_FT["action"]:
            level = max(level, 1); reasons.append("River at action stage")
    nearest = storms[0].distance_mi if storms else float("inf")
    if storms:
        if nearest < 300:
            level = max(level, 3); reasons.append(f"Active TC {storms[0].name} {nearest:.0f} mi")
        elif nearest < 600:
            level = max(level, 2); reasons.append(f"Active TC {storms[0].name} {nearest:.0f} mi")
        elif nearest < 1500:
            level = max(level, 1); reasons.append(f"Atlantic TC {storms[0].name} active")
    if any("Tropical" in (a.get("event") or "") or "Flood" in (a.get("event") or "")
           for a in alerts):
        level = max(level, 2); reasons.append("Tropical/flood NWS alert active")
    if weather.get("next_72h_precip_in", 0) and weather["next_72h_precip_in"] >= 3.0:
        level = max(level, 2); reasons.append(f"Heavy 72h rain forecast ({weather['next_72h_precip_in']}\")")
    elif weather.get("next_72h_precip_in", 0) and weather["next_72h_precip_in"] >= 1.0:
        level = max(level, 1); reasons.append(f"Notable 72h rain forecast ({weather['next_72h_precip_in']}\")")

    label = ["CALM", "ELEVATED", "ALERT", "WARNING", "EMERGENCY"][level]
    color = ["#2e7d32", "#9e9d24", "#ef6c00", "#c62828", "#6a1b9a"][level]

    data = {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "level": level,
        "label": label,
        "color": color,
        "reasons": reasons or ["No elevated signals"],
        "gauge": asdict(gauge) if gauge else None,
        "flood_stages": FLOOD_STAGES_FT,
        "storms": [asdict(s) for s in storms],
        "alerts": alerts,
        "weather": weather,
        "season": CSU_2026_FORECAST,
        "asheville": {"lat": ASHEVILLE_LAT, "lon": ASHEVILLE_LON},
    }
    _CACHE["data"] = data
    _CACHE["ts"] = now
    return data


PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Asheville Hurricane Risk - Live</title>
<meta http-equiv="refresh" content="60" />
<style>
  :root {
    --bg: #0f1115;
    --panel: #1a1d24;
    --panel2: #232732;
    --text: #e6e6e6;
    --dim: #9aa0aa;
    --accent: #4fc3f7;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background: var(--bg); color: var(--text); }
  header { padding: 1.5rem 2rem; background: var(--panel);
           display: flex; align-items: center; justify-content: space-between;
           border-bottom: 1px solid #000; }
  header h1 { margin:0; font-size: 1.4rem; font-weight: 600; }
  .level-pill { padding: .5rem 1.2rem; border-radius: 999px;
                font-weight: 700; letter-spacing: .05em; color: white; }
  .grid { display: grid; gap: 1rem; padding: 1rem;
          grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
  .card { background: var(--panel); border-radius: 10px; padding: 1rem 1.2rem;
          box-shadow: 0 2px 6px rgba(0,0,0,.4); }
  .card h2 { margin: 0 0 .6rem 0; font-size: 1rem; color: var(--accent);
             text-transform: uppercase; letter-spacing: .08em; }
  .big { font-size: 2.2rem; font-weight: 700; }
  .dim { color: var(--dim); font-size: .85rem; }
  .row { display:flex; justify-content: space-between; padding: .25rem 0;
         border-bottom: 1px solid #2a2e36; }
  .row:last-child { border-bottom: 0; }
  .bar { height: 18px; background: var(--panel2); border-radius: 4px;
         overflow:hidden; margin: .4rem 0; position: relative; }
  .bar > span { display:block; height:100%; background: linear-gradient(90deg,#2196f3,#ef5350); }
  .bar > .marker { position:absolute; top:0; bottom:0; width:2px; background:#fff8; }
  .reasons { font-size: .9rem; color: var(--dim); }
  .reasons li { margin: .2rem 0; }
  .alert { background: #b71c1c; padding:.6rem; border-radius: 6px; margin:.4rem 0; }
  footer { text-align:center; padding: 1rem; color: var(--dim); font-size:.8rem; }
  table { width: 100%; border-collapse: collapse; font-size:.9rem;}
  th, td { padding: .35rem .5rem; text-align:left; border-bottom: 1px solid #2a2e36;}
  th { color: var(--dim); font-weight: 500; }
</style>
</head>
<body>
<header>
  <h1>&#127786; Asheville Hurricane Risk - Live</h1>
  <span class="level-pill" style="background: {{ data.color }}">{{ data.label }}</span>
</header>

<div class="grid">

  <div class="card" style="grid-column: 1 / -1;">
    <h2>Active signals</h2>
    <ul class="reasons">
    {% for r in data.reasons %}<li>{{ r }}</li>{% endfor %}
    </ul>
    <div class="dim">As of {{ data.as_of }} | auto-refresh 60s</div>
  </div>

  <div class="card">
    <h2>French Broad @ Asheville</h2>
    {% if data.gauge and data.gauge.stage_ft is not none %}
      <div class="big">{{ "%.2f"|format(data.gauge.stage_ft) }} ft</div>
      <div class="dim">{{ data.gauge.flood_category }}
        &nbsp;|&nbsp; {{ data.gauge.discharge_cfs|round(0)|int if data.gauge.discharge_cfs else "?" }} cfs</div>
      {% set pct = (data.gauge.stage_ft / data.flood_stages.major * 100)|round(0)|int %}
      <div class="bar">
        <span style="width: {{ [pct, 100]|min }}%"></span>
        <div class="marker" style="left: {{ (data.flood_stages.minor / data.flood_stages.major * 100)|round(0)|int }}%"></div>
        <div class="marker" style="left: {{ (data.flood_stages.moderate / data.flood_stages.major * 100)|round(0)|int }}%"></div>
      </div>
      <div class="dim">action {{ data.flood_stages.action }} | minor {{ data.flood_stages.minor }} |
                       moderate {{ data.flood_stages.moderate }} | major {{ data.flood_stages.major }} ft
                       (record {{ data.flood_stages.record }})</div>
      <div class="dim">USGS 03451500 | {{ data.gauge.timestamp }}</div>
    {% else %}
      <div class="dim">USGS data unavailable</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>Asheville weather</h2>
    {% if data.weather and not data.weather.error %}
      <div class="big">{{ data.weather.temp_f }}&deg;F</div>
      <div class="row"><span>Humidity</span><span>{{ data.weather.humidity_pct }}%</span></div>
      <div class="row"><span>Wind</span><span>{{ data.weather.wind_mph }} mph @ {{ data.weather.wind_dir_deg }}&deg;</span></div>
      <div class="row"><span>Pressure</span><span>{{ data.weather.pressure_mb }} mb</span></div>
      <div class="row"><span>Current precip</span><span>{{ data.weather.precip_in }} in</span></div>
      <div class="row"><span>Next 72h precip (forecast)</span><span><b>{{ data.weather.next_72h_precip_in }} in</b></span></div>
      <div class="dim">Open-Meteo | {{ data.weather.as_of }}</div>
    {% else %}
      <div class="dim">Weather feed unavailable</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>Active Atlantic storms</h2>
    {% if data.storms %}
      <table>
        <tr><th>Name</th><th>Class</th><th>kt</th><th>Dist (mi)</th><th>Move</th></tr>
        {% for s in data.storms %}
        <tr>
          <td><b>{{ s.name }}</b></td>
          <td>{{ s.classification }}</td>
          <td>{{ s.intensity_kt|round(0)|int if s.intensity_kt else "?" }}</td>
          <td>{{ s.distance_mi|round(0)|int }}</td>
          <td>{{ s.movement }}</td>
        </tr>
        {% endfor %}
      </table>
    {% else %}
      <div class="dim">No active Atlantic storms.</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>NWS alerts (Asheville)</h2>
    {% if data.alerts %}
      {% for a in data.alerts %}
        <div class="alert">
          <b>{{ a.event }}</b> [{{ a.severity }}]<br>
          <small>{{ a.headline }}</small>
        </div>
      {% endfor %}
    {% else %}
      <div class="dim">No active watches/warnings/advisories.</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>2026 CSU Atlantic outlook</h2>
    <div class="row"><span>Named storms</span><span>{{ data.season.named_storms }} (climo 14.4)</span></div>
    <div class="row"><span>Hurricanes</span><span>{{ data.season.hurricanes }} (climo 7.2)</span></div>
    <div class="row"><span>Major hurricanes</span><span>{{ data.season.major_hurricanes }} (climo 3.2)</span></div>
    <div class="row"><span>ACE</span><span>{{ data.season.ace }} (climo 123)</span></div>
    <div class="row"><span>P(major US landfall)</span><span>{{ (data.season.p_us_major_landfall * 100)|round(0)|int }}%</span></div>
    <div class="dim">Klotzbach et al., CSU, {{ data.season.issued }}</div>
  </div>

</div>

<footer>
  Data: USGS NWIS, NWS api.weather.gov, NHC CurrentStorms, Open-Meteo, CSU Tropical Met Project.<br>
  Asheville {{ data.asheville.lat }}, {{ data.asheville.lon }}.
</footer>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, data=_collect())


@app.route("/api/state")
def api_state():
    return jsonify(_collect())


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    print(f"Asheville hurricane dashboard -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
