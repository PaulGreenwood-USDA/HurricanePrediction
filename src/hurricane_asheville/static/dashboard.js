/* Dashboard client script.
 *
 * Reads the map payload from an inline <script type="application/json">
 * rather than a `tojson` dump of the whole state: the full payload is ~260 KB,
 * most of it gauge history the map never touches, and it was being shipped
 * twice (inline and as state.json).
 */
'use strict';

const STATE = JSON.parse(document.getElementById('map-state').textContent);
const ASH = [STATE.asheville.lat, STATE.asheville.lon];

/* ── Map ─────────────────────────────────────────────────────────── */
const map = L.map('map', { zoomControl: true }).setView([35.4, -78.5], 7);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 18,
}).addTo(map);

L.circleMarker(ASH, {
  radius: 8, color: '#4fc3f7', weight: 2, fillColor: '#4fc3f7', fillOpacity: 0.6,
}).addTo(map).bindPopup('<b>Asheville</b>');

const fmt = (v, digits, suffix) =>
  v == null ? 'n/a' : v.toFixed(digits) + (suffix || '');

/* Coastal NOAA stations (storm-surge early warning) */
(STATE.coastal || []).forEach((c) => {
  const wind = c.wind_kt || 0;
  let color = '#1e88e5';
  if (wind >= 34) color = '#fb8c00';
  if (wind >= 64) color = '#c62828';
  L.marker([c.lat, c.lon], {
    icon: L.divIcon({
      className: 'coast-icon',
      html: '<div style="background:' + color + ';color:#fff;font-weight:600;' +
            'padding:1px 5px;border-radius:3px;font-size:10px;border:1px solid #000;">' +
            '&#127754; ' + (c.water_level_ft != null ? c.water_level_ft.toFixed(1) + "'" : '?') + '</div>',
      iconSize: null,
    }),
  }).addTo(map).bindPopup(
    '<b>' + c.label + '</b><br>' +
    'Water: ' + fmt(c.water_level_ft, 2, ' ft MLLW') + '<br>' +
    'Wind: ' + fmt(c.wind_kt, 0, ' kt') +
      (c.wind_gust_kt ? ' (g ' + c.wind_gust_kt.toFixed(0) + ')' : '') + '<br>' +
    'Pressure: ' + fmt(c.air_pressure_mb, 1, ' mb') + '<br>' +
    '<small>NOAA CO-OPS ' + c.station_id + '</small>');
});

/* NDBC offshore buoys — wave height is the 12-24 h TC leading indicator */
(STATE.buoys || []).forEach((b) => {
  const wave = b.wave_ht_ft || 0;
  L.circleMarker([b.lat, b.lon], {
    radius: Math.max(5, Math.min(16, 5 + wave * 0.8)),
    color: b.color || '#1976d2', fillColor: b.color || '#1976d2',
    fillOpacity: 0.55, weight: 2,
  }).addTo(map).bindPopup(
    '<b>&#9883; ' + b.label + '</b><br>' +
    'Seas: <b>' + b.seas + '</b><br>' +
    'Wave: ' + fmt(b.wave_ht_ft, 1, ' ft') +
      (b.dominant_period_s ? ' / ' + b.dominant_period_s.toFixed(0) + ' s' : '') + '<br>' +
    'Wind: ' + fmt(b.wind_kt, 0, ' kt') + '<br>' +
    'Pressure: ' + fmt(b.pressure_mb, 1, ' mb') + '<br>' +
    '<small>NDBC ' + b.station_id + '</small>');
});

/* National forests */
const FOREST_COLOR = { mountain: '#4caf50', piedmont: '#fdd835', coastal: '#ef5350' };
(STATE.forests || []).forEach((f) => {
  const c = FOREST_COLOR[f.region] || '#4caf50';
  L.circle([f.center_lat, f.center_lon], {
    radius: Math.max(10, Math.sqrt(f.acres / 1000) * 0.6) * 1000,
    color: c, fillColor: c, fillOpacity: 0.12, weight: 1.5,
  }).addTo(map);
  L.marker([f.center_lat, f.center_lon], {
    icon: L.divIcon({
      className: 'nf-icon',
      html: '<div style="background:' + c + ';color:#000;font-weight:700;' +
            'padding:2px 6px;border-radius:4px;font-size:11px;border:1px solid #000;">' +
            '&#127794; ' + f.short + '</div>',
      iconSize: null,
    }),
  }).addTo(map).bindPopup(
    '<b>' + f.name + '</b><br>' + f.acres.toLocaleString() + ' ac<br>' +
    '<small>' + (f.notes || '') + '</small>');
});

/* Gauges — keyed off the explicit flood_class slug, never a label substring */
const GAUGE_COLOR = {
  'below-action': '#4caf50', action: '#fdd835', minor: '#ef6c00',
  moderate: '#c62828', major: '#6a1b9a', pool: '#1976d2',
};
(STATE.gauges || []).forEach((g) => {
  const color = GAUGE_COLOR[g.flood_class] || '#607d8b';
  const r = g.rate_ft_per_hr;
  const rateStr = r == null ? 'n/a'
    : (r > 0.05 ? '&uarr; ' : (r < -0.05 ? '&darr; ' : '')) + Math.abs(r).toFixed(2) + ' ft/hr';
  L.circleMarker([g.lat, g.lon], {
    radius: g.site_id === STATE.primary_site ? 9 : 6,
    color, fillColor: color, fillOpacity: 0.7, weight: 2,
  }).addTo(map).bindPopup(
    '<b>' + g.label + '</b><br>' +
    (g.display_ft != null
      ? (g.pool_elevation_ft != null && g.stage_ft == null ? 'Pool: ' : 'Stage: ') +
        g.display_ft.toFixed(2) + ' ' + (g.display_units || 'ft') + '<br>'
      : 'Stage: n/a<br>') +
    'Rate: ' + rateStr + '<br>' +
    'Cat: ' + g.flood_category + '<br>' +
    '<small>USGS ' + g.site_id + '</small>');
});

/* Active storms: current position, NHC forecast track, and the cone.
 * A dot plus a line to Asheville was never what people look at during a
 * storm — the forecast track is. */
(STATE.storms || []).forEach((s) => {
  if (s.lat == null || s.lon == null) return;
  const wind = s.intensity_kt || 0;
  let color = '#90caf9';
  if (wind >= 64) color = '#ef5350';
  if (wind >= 96) color = '#c62828';
  if (wind >= 113) color = '#6a1b9a';

  if (s.forecast_track && s.forecast_track.length > 1) {
    const pts = s.forecast_track.map((p) => [p.lat, p.lon]);
    L.polyline(pts, {
      color, weight: 2.5, opacity: 0.9, dashArray: '6,4',
    }).addTo(map).bindPopup('<b>' + s.name + '</b> — NHC forecast track');
    s.forecast_track.forEach((p) => {
      L.circleMarker([p.lat, p.lon], {
        radius: 3, color: '#fff', weight: 1, fillColor: color, fillOpacity: 0.9,
      }).addTo(map).bindPopup(
        '<b>' + s.name + '</b><br>' + (p.valid_time || '') + '<br>' +
        (p.intensity_kt != null ? p.intensity_kt + ' kt' : '') +
        (p.classification ? ' ' + p.classification : ''));
    });
  }
  if (s.cone && s.cone.length > 2) {
    L.polygon(s.cone.map((p) => [p.lat, p.lon]), {
      color, weight: 1, opacity: 0.5, fillColor: color, fillOpacity: 0.12,
    }).addTo(map).bindPopup('<b>' + s.name + '</b> — NHC forecast cone');
  }

  L.circleMarker([s.lat, s.lon], {
    radius: 10, color, fillColor: color, fillOpacity: 0.5, weight: 2,
  }).addTo(map).bindPopup(
    '<b>' + s.name + '</b><br>' + s.classification + ' ' + wind + ' kt<br>' +
    (s.distance_mi || 0).toFixed(0) + ' mi from Asheville<br>' + (s.movement || ''));
  L.polyline([[s.lat, s.lon], ASH], {
    color: '#888', weight: 1, dashArray: '4,4',
  }).addTo(map);
});

/* Active wildfires, deduplicated across forests */
const seenFires = new Set();
(STATE.forests || []).forEach((f) => {
  (f.fires_nearby || []).forEach((fire) => {
    const key = fire.irwin_id || (fire.name + ':' + fire.lat + ',' + fire.lon);
    if (seenFires.has(key)) return;
    seenFires.add(key);
    const acres = fire.acres || 0;
    L.circleMarker([fire.lat, fire.lon], {
      radius: Math.max(6, Math.min(18, Math.sqrt(acres) * 0.6)),
      color: '#ff6f00', fillColor: '#ff3d00', fillOpacity: 0.7, weight: 2,
    }).addTo(map).bindPopup(
      '<b>&#128293; ' + fire.name + '</b><br>' +
      (acres ? acres.toLocaleString() + ' ac' : 'size unknown') + '<br>' +
      '<small>NIFC WFIGS</small>');
  });
});

/* ── Freshness ───────────────────────────────────────────────────────
 * The old page reloaded itself every 15 minutes regardless of whether
 * anything had changed, re-serving identical HTML on a page rebuilt hourly.
 * Now the age ticks locally, and we only reload when the published snapshot
 * is actually newer than the one being displayed. */
(function freshness() {
  const el = document.getElementById('freshness');
  if (!el) return;
  const asOf = parseInt(el.dataset.asOf, 10);
  const intervalMin = parseInt(el.dataset.interval, 10) || 60;
  if (!asOf) return;

  const ageEl = document.getElementById('age');
  const labelEl = document.getElementById('fresh-label');

  function describe(minutes) {
    if (minutes < 75) return ['fresh', 'Live'];
    if (minutes < 150) return ['aging', 'One rebuild missed'];
    if (minutes < 360) return ['stale', 'Several rebuilds missed'];
    return ['frozen', 'Stale — build may have failed'];
  }

  function tick() {
    const minutes = Math.max(0, (Date.now() / 1000 - asOf) / 60);
    ageEl.textContent = minutes < 1 ? 'just now'
      : minutes < 60 ? Math.floor(minutes) + ' min old'
      : Math.floor(minutes / 60) + ' h ' + Math.floor(minutes % 60) + ' min old';
    const [level, label] = describe(minutes);
    el.className = 'freshness ' + level;
    labelEl.textContent = label;
  }
  tick();
  setInterval(tick, 30000);

  // Poll for a genuinely newer snapshot. state.json sits next to index.html
  // on Pages; on the Flask app /api/state serves the same shape.
  const endpoint = STATE.state_url || 'state.json';
  async function checkForUpdate() {
    try {
      const res = await fetch(endpoint, { cache: 'no-store' });
      if (!res.ok) return;
      const fresh = await res.json();
      if (fresh && fresh.as_of_epoch && fresh.as_of_epoch > asOf) {
        location.reload();
      }
    } catch (e) {
      /* offline or blocked — keep showing what we have and keep ageing it */
    }
  }
  setInterval(checkForUpdate, Math.max(5, intervalMin / 6) * 60 * 1000);
})();

/* ── Gauge network filter ────────────────────────────────────────── */
(function gaugeFilter() {
  const tabs = document.querySelectorAll('#net-tabs .net-tab');
  const rows = document.querySelectorAll('.gauge-row[data-net]');
  function apply(net) {
    rows.forEach((r) => {
      r.classList.toggle('hidden', !(net === 'all' || r.dataset.net === net));
    });
    tabs.forEach((t) => {
      const on = t.dataset.net === net;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    try { localStorage.setItem('gaugeNet', net); } catch (e) { /* private mode */ }
  }
  tabs.forEach((t) => {
    t.addEventListener('click', () => apply(t.dataset.net));
    t.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); apply(t.dataset.net); }
    });
  });
  let saved = 'all';
  try { saved = localStorage.getItem('gaugeNet') || 'all'; } catch (e) { /* ignore */ }
  if ([...tabs].some((t) => t.dataset.net === saved)) apply(saved);
})();

/* ── Jargon definitions, reachable by tap and keyboard ───────────── */
(function jargon() {
  document.querySelectorAll('.jargon[data-term]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const note = document.getElementById(btn.getAttribute('aria-controls'));
      if (!note) return;
      const open = note.classList.toggle('open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  });
})();
