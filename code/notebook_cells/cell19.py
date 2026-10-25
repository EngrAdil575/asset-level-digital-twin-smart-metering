# ============================================================
# CELL 19: Digital Twin interface
# Builds a self-contained HTML Digital Twin of the 4 assets from the
# results of Cells 15-18 and saves it as 'digital_twin.html'.
# The twin replays the held-out TEST period hour by hour and shows, for
# the selected asset:
#   * a plant-floor view with an industrial symbol for each asset (press,
#     AHU fan, compressor, conveyor) coloured by state: running / idle / anomaly
#   * machine state (running / idle) and how long it has been in it
#   * measured demand and the next-hour forecast (Enhanced XGBoost),
#     next to the persistence baseline
#   * day health from the context-aware kNN anomaly detector (Cell 18
#     method), with the top-5% alert threshold
#   * today's load curve against the typical weekday / weekend curve
#   * LF, PBR, VI of the day, forecast accuracy, and an alert log
# Open the downloaded file in any web browser (Chrome / Edge).
# Requires: Cells 1-17 (uses final_df, scale, ON_THRESHOLD, hourly,
# daily_indices_ctx, SELECTED_ASSETS, POWER_COL).
# ============================================================

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROLE = {'p_c': 'Press drive (production)', 'ahu_a': 'Air handling unit (HVAC)',
        'comp_b': 'Air compressor', 'mh_a': 'Material handling'}
K_NN, ALERT_Q = 5, 0.95

twin = {'assets': {},
        'note': ('Replay of the chronological test period (last 20% of the timeline, never used for '
                 'training). Forecast = Enhanced XGBoost (hourly + 15-min + ON/OFF + cross-asset '
                 'features). Day health = mean distance to the 5 most similar real days of the same '
                 'day type; alert when above the 95th percentile of all real days of that asset. '
                 'Data: open industrial asset-level energy dataset, 15-min aggregates, Demand_kW.')}

prof_all = hourly.pivot_table(index=['AssetId', 'date'], columns='hour', values=POWER_COL).dropna()
ctx_idx = daily_indices_ctx.copy()
ctx_idx['date'] = pd.to_datetime(ctx_idx['date'])

for a in SELECTED_ASSETS:
    amax = float(scale[a])
    sub = final_df[final_df['AssetId'] == a].set_index('ts').sort_index()
    # continuous hourly axis over the test period (gaps stay empty)
    rng_t = pd.date_range(sub.index.min(), sub.index.max(), freq='h')
    sub = sub.reindex(rng_t)
    r2 = lambda s: [None if pd.isna(v) else round(float(v), 3) for v in s]

    # context-aware kNN day score for every complete real day of this asset
    P = prof_all.loc[a]
    dates = pd.to_datetime(P.index)
    L = P.values / amax
    wk = np.asarray(dates.dayofweek >= 5).astype(int)
    score = np.zeros(len(L))
    typical = {}
    for t in [0, 1]:
        m = wk == t
        if m.sum() > K_NN + 1:
            d, _ = NearestNeighbors(n_neighbors=K_NN + 1).fit(L[m]).kneighbors(L[m])
            score[m] = d[:, 1:].mean(axis=1)          # skip the day itself
            typical[str(t)] = np.round(np.median(P.values[m], axis=0), 3).tolist()
        elif m.any():
            d, _ = NearestNeighbors(n_neighbors=K_NN).fit(L).kneighbors(L[m])
            score[m] = d.mean(axis=1)
    for t in ['0', '1']:
        typical.setdefault(t, np.round(np.median(P.values, axis=0), 3).tolist())
    thr = float(np.quantile(score, ALERT_Q))

    idx_a = ctx_idx[ctx_idx['AssetId'] == a].set_index('date')
    first_day = rng_t[0].normalize() - pd.Timedelta(days=2)
    days = {}
    for d, s, w in zip(dates, score, wk):
        if d < first_day:
            continue
        rec = {'s': round(float(s), 4), 'w': int(w)}
        if d in idx_a.index:
            row = idx_a.loc[d]
            rec['LF'] = None if pd.isna(row['LF']) else round(float(row['LF']), 4)
            rec['PBR'] = None if pd.isna(row['PBR']) else round(float(row['PBR']), 3)
            rec['VI'] = None if pd.isna(row['VI']) else round(float(row['VI']), 3)
        days[d.strftime('%Y-%m-%d')] = rec

    t_str = [t.strftime('%Y-%m-%d %H') for t in rng_t]
    start = next((k for k, t in enumerate(t_str) if t >= '2025-10-21 10'), 48)
    start = min(max(start, 48), len(t_str) - 2)
    ok = sub['Actual'].notna()
    twin['assets'][a] = {
        'role': ROLE.get(a, a), 'max': round(amax, 3), 'onKw': round(ON_THRESHOLD * amax, 3),
        't': t_str, 'a': r2(sub['Actual']), 'f': r2(sub['Enhanced XGBoost']), 'p': r2(sub['Persistence']),
        'rmse_enh': float(np.sqrt(((sub.loc[ok, 'Actual'] - sub.loc[ok, 'Enhanced XGBoost']) ** 2).mean())),
        'rmse_pers': float(np.sqrt(((sub.loc[ok, 'Actual'] - sub.loc[ok, 'Persistence']) ** 2).mean())),
        'thr': round(thr, 4), 'days': days, 'typical': typical, 'start': start}
    n_alert = sum(1 for dd, r in days.items() if r['s'] > thr and dd >= t_str[0][:10])
    print(f"{a}: {len(t_str)} test hours, {n_alert} anomalous test days, alert threshold {thr:.4f}")

TWIN_HEAD = r"""<title>Industrial Energy Digital Twin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&display=swap">
<style>
/* Layout: control-room console. Header + asset tabs + time scrubber, then a 4-tile
   status row, then trend + day-profile panels, then indices / accuracy / alerts. */
:root {
  color-scheme: dark;
  --bg: #0d1520;          /* console background, blue-biased */
  --panel: #142031;       /* instrument panel */
  --panel-2: #1a293d;     /* raised element / track */
  --line: #25374f;        /* hairlines, grid */
  --fg: #dfe8f3;          /* primary text */
  --muted: #8195ad;       /* secondary text */
  --actual: #eef3f9;      /* measured load trace */
  --forecast: #4cc3e6;    /* model forecast trace (cyan instrument) */
  --persist: #7d8ea4;     /* persistence baseline */
  --ok: #3ecf8e;          /* state: normal / ON */
  --warn: #f2b33d;        /* state: warning */
  --crit: #f0625a;        /* state: anomaly */
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --f-body: "IBM Plex Sans", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --f-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
body { background: var(--bg); color: var(--fg); font-family: var(--f-body); font-size: 14px; }
.wrap { max-width: 1280px; margin: 0 auto; padding-inline: 20px; padding-block: 18px 28px; display: grid; gap: 14px; }
header.top { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px; }
.brand h1 { font-family: var(--f-display); font-weight: 600; font-size: 1.55rem; letter-spacing: .01em; margin: 0; text-wrap: balance; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: .82rem; }
.live { font-family: var(--f-data); font-size: .8rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.live .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 25%, transparent); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; }
.tab { font: 500 .85rem var(--f-body); color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; cursor: pointer; display: grid; text-align: left; gap: 1px; }
.tab small { font-size: .7rem; color: var(--muted); font-weight: 400; }
.tab[aria-selected="true"] { color: var(--fg); border-color: var(--forecast); background: var(--panel-2); }
.tab:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--forecast); outline-offset: 2px; }
.scrub { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
.scrub button { font: 600 .8rem var(--f-body); background: var(--forecast); color: #06202b; border: 0; border-radius: 5px; padding: 7px 12px; cursor: pointer; min-width: 64px; }
.scrub button.ghost { background: var(--panel-2); color: var(--fg); border: 1px solid var(--line); min-width: 0; }
.scrub input[type=range] { flex: 1 1 260px; accent-color: var(--forecast); min-width: 0; }
.scrub .when { font-family: var(--f-data); font-size: .92rem; min-width: 190px; font-variant-numeric: tabular-nums; }
.grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.grid-main { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(0, 1fr); gap: 12px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-width: 0; display: grid; gap: 8px; align-content: start; }
.label { font: 600 .7rem var(--f-body); text-transform: uppercase; letter-spacing: .09em; color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
.big { font-family: var(--f-data); font-size: 1.9rem; font-weight: 500; font-variant-numeric: tabular-nums; line-height: 1.1; }
.big small { font-size: .9rem; color: var(--muted); margin-left: 4px; }
.sub { color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: 6px; font: 600 .78rem var(--f-body); letter-spacing: .04em; padding: 4px 10px; border-radius: 999px; width: fit-content; }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.ok { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
.pill.off { color: var(--muted); background: var(--panel-2); }
.pill.warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 14%, transparent); }
.pill.crit { color: var(--crit); background: color-mix(in srgb, var(--crit) 16%, transparent); }
.meter { height: 8px; background: var(--panel-2); border-radius: 4px; position: relative; overflow: hidden; }
.meter > i { position: absolute; inset: 0 auto 0 0; background: var(--forecast); border-radius: 4px; }
.meter.state > i { background: var(--ok); }
.meter .mark { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--warn); }
.card.alert-crit { border-color: color-mix(in srgb, var(--crit) 60%, var(--line)); }
.chart { width: 100%; height: auto; display: block; }
.chart text { fill: var(--muted); font: 10px var(--f-data); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .75rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
dl.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; font-variant-numeric: tabular-nums; }
dl.kv dt { color: var(--muted); font-size: .8rem; }
dl.kv dd { margin: 0; font-family: var(--f-data); text-align: right; }
.log { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; max-height: 190px; overflow-y: auto; }
.log li { display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; font-size: .8rem; padding: 6px 8px; background: var(--panel-2); border-radius: 5px; cursor: pointer; border: 1px solid transparent; }
.log li:hover { border-color: var(--line); }
.log li b { font-family: var(--f-data); font-weight: 500; }
.log .sev { width: 8px; height: 8px; border-radius: 50%; background: var(--crit); }
.foot { color: var(--muted); font-size: .75rem; line-height: 1.5; max-width: 110ch; }
@media (max-width: 980px) { .grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid-main, .grid3 { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 520px) { .grid4 { grid-template-columns: minmax(0, 1fr); } .wrap { padding-inline: 16px; } .big { font-size: 1.6rem; } }
@media (prefers-reduced-motion: no-preference) { .meter > i { transition: width .25s ease; } }

/* plant floor: machine symbols tinted by state */
.floor { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.mach { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; cursor: pointer;
  display: grid; grid-template-columns: 58px minmax(0, 1fr); gap: 4px 12px; align-items: center; text-align: left; color: var(--fg); font: inherit; }
.mach[aria-selected="true"] { border-color: var(--forecast); background: var(--panel-2); box-shadow: inset 0 0 0 1px var(--forecast); }
.mach .ico { grid-row: span 3; width: 58px; height: 58px; color: var(--muted); }
.mach.s-run .ico { color: var(--ok); } .mach.s-anom .ico { color: var(--crit); }
.mach.s-anom { border-color: color-mix(in srgb, var(--crit) 55%, var(--line)); }
.mach b { font: 600 .95rem var(--f-display); letter-spacing: .02em; }
.mach small { color: var(--muted); font-size: .72rem; }
.mach .st { font: 500 .78rem var(--f-data); font-variant-numeric: tabular-nums; }
.mach.s-run .st { color: var(--ok); } .mach.s-anom .st { color: var(--crit); }
.bigico { width: 84px; height: 84px; color: var(--muted); }
.bigico.s-run { color: var(--ok); } .bigico.s-anom { color: var(--crit); }
.statebox { display: flex; align-items: center; gap: 14px; }
.ico svg, .bigico svg { width: 100%; height: 100%; display: block; overflow: visible; }
@media (prefers-reduced-motion: no-preference) {
  .s-run .spin { animation: spin 1.1s linear infinite; transform-origin: 32px 32px; transform-box: view-box; }
  .s-run .ram { animation: ram 1.2s ease-in-out infinite; }
  .s-run .item { animation: belt 2.4s linear infinite; }
  .s-run .needle { animation: needle 1.6s ease-in-out infinite alternate; transform-origin: 20px 18px; transform-box: view-box; }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes ram { 50% { transform: translateY(12px); } }
@keyframes belt { from { transform: translateX(-14px); } to { transform: translateX(14px); } }
@keyframes needle { from { transform: rotate(-35deg); } to { transform: rotate(35deg); } }
@media (max-width: 980px) { .floor { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .floor { grid-template-columns: minmax(0, 1fr); } }
</style>"""

TWIN_BODY = r"""
<div class="wrap">
  <header class="top">
    <div class="brand">
      <h1>Industrial Energy Digital Twin</h1>
      <p>Asset-level load profiling, next-hour forecasting and anomaly alerts &middot; replaying the held-out test period</p>
    </div>
    <div class="live"><span class="dot" id="liveDot"></span><span id="liveTxt">Replay paused</span></div>
  </header>

  <nav class="floor" id="tabs" role="tablist" aria-label="Plant floor: select an asset"></nav>

  <section class="scrub" aria-label="Time control">
    <button id="play" type="button">Play</button>
    <button id="stepB" class="ghost" type="button" aria-label="Back one hour">&minus;1 h</button>
    <button id="stepF" class="ghost" type="button" aria-label="Forward one hour">+1 h</button>
    <input id="slider" type="range" min="0" max="1" value="0" aria-label="Replay time">
    <div class="when" id="when">&ndash;</div>
  </section>

  <section class="grid4">
    <div class="card" id="cState">
      <div class="label"><span>Machine state</span><span id="stateName"></span></div>
      <div class="statebox"><div class="bigico" id="bigIco"></div>
        <div style="display:grid;gap:6px;min-width:0"><div id="statePill" class="pill ok">ON</div>
        <div class="sub" id="stateSub">&ndash;</div></div></div>
    </div>
    <div class="card">
      <div class="label"><span>Measured demand</span><span id="maxTxt"></span></div>
      <div class="big" id="nowKw">&ndash;<small>kW</small></div>
      <div class="meter"><i id="nowBar"></i></div>
    </div>
    <div class="card">
      <div class="label"><span>Next-hour forecast</span><span>Enhanced XGBoost</span></div>
      <div class="big" id="fcKw">&ndash;<small>kW</small></div>
      <div class="sub" id="fcSub">&ndash;</div>
    </div>
    <div class="card" id="cHealth">
      <div class="label"><span>Day health</span><span>context kNN</span></div>
      <div id="healthPill" class="pill ok">Normal</div>
      <div class="meter"><i id="healthBar"></i><span class="mark" id="healthMark"></span></div>
      <div class="sub" id="healthSub">&ndash;</div>
    </div>
  </section>

  <section class="grid-main">
    <div class="card">
      <div class="label"><span>Last 48 hours and next-hour forecast</span><span id="trendNote"></span></div>
      <svg class="chart" id="trend" viewBox="0 0 720 250" role="img" aria-label="Actual and forecast demand"></svg>
      <div class="legend">
        <span><i style="border-color:var(--actual)"></i>Measured</span>
        <span><i style="border-color:var(--forecast)"></i>Forecast (Enhanced XGBoost)</span>
        <span><i style="border-color:var(--persist);border-top-style:dashed"></i>Persistence baseline</span>
      </div>
    </div>
    <div class="card">
      <div class="label"><span>Today vs typical day</span><span id="dayType"></span></div>
      <svg class="chart" id="dayChart" viewBox="0 0 420 250" role="img" aria-label="Today's profile against the typical profile"></svg>
      <div class="legend">
        <span><i style="border-color:var(--actual)"></i>Today so far</span>
        <span><i style="border-color:var(--warn);border-top-style:dashed"></i>Typical (median)</span>
      </div>
    </div>
  </section>

  <section class="grid3">
    <div class="card">
      <div class="label"><span>Load-profile indices</span><span>today</span></div>
      <dl class="kv">
        <dt>Load factor (LF)</dt><dd id="iLF">&ndash;</dd>
        <dt>Peak-to-base ratio (PBR)</dt><dd id="iPBR">&ndash;</dd>
        <dt>Variability index (VI, kW&sup2;)</dt><dd id="iVI">&ndash;</dd>
        <dt>Energy so far</dt><dd id="iE">&ndash;</dd>
      </dl>
    </div>
    <div class="card">
      <div class="label"><span>Forecast accuracy</span><span>test period</span></div>
      <dl class="kv">
        <dt>RMSE, Enhanced XGBoost</dt><dd id="aEnh">&ndash;</dd>
        <dt>RMSE, persistence</dt><dd id="aPers">&ndash;</dd>
        <dt>Skill vs persistence</dt><dd id="aSkill">&ndash;</dd>
        <dt>Rolling 24 h MAE</dt><dd id="aRoll">&ndash;</dd>
      </dl>
    </div>
    <div class="card">
      <div class="label"><span>Anomaly alerts</span><span id="alertCount"></span></div>
      <ul class="log" id="log"></ul>
    </div>
  </section>

  <p class="foot" id="foot"></p>
</div>

<script>
const TWIN = /*__DATA__*/null;
(function () {
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (x, d = 2) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);

  const ICON = {
    press: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="9" y="5" width="46" height="8" rx="1.5"/><line x1="14" y1="13" x2="14" y2="54"/><line x1="50" y1="13" x2="50" y2="54"/>
      <g class="ram"><line x1="32" y1="13" x2="32" y2="19"/><rect x="21" y="19" width="22" height="11" rx="1" fill="currentColor" fill-opacity=".25"/></g>
      <rect x="21" y="45" width="22" height="9" rx="1"/><rect x="6" y="54" width="52" height="6" rx="1.5" fill="currentColor" fill-opacity=".25"/></svg>`,
    fan: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="5" y="5" width="54" height="54" rx="6"/><circle cx="32" cy="32" r="22" stroke-opacity=".45"/>
      <g class="spin" fill="currentColor" fill-opacity=".3">
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z"/>
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z" transform="rotate(120 32 32)"/>
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z" transform="rotate(240 32 32)"/>
        <circle cx="32" cy="32" r="3.5" fill-opacity="1"/></g></svg>`,
    compressor: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="4" y="30" width="42" height="20" rx="10" fill="currentColor" fill-opacity=".2"/>
      <line x1="12" y1="50" x2="10" y2="58"/><line x1="38" y1="50" x2="40" y2="58"/>
      <rect x="44" y="20" width="16" height="16" rx="2"/><line x1="46" y1="25" x2="58" y2="25"/><line x1="46" y1="30" x2="58" y2="30"/>
      <line x1="20" y1="26" x2="20" y2="30"/><circle cx="20" cy="18" r="8"/>
      <line class="needle" x1="20" y1="18" x2="20" y2="12" stroke-linecap="round"/></svg>`,
    conveyor: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="36" width="58" height="12" rx="6"/>
      <circle cx="9" cy="42" r="2.6"/><circle cx="24" cy="42" r="2.6"/><circle cx="40" cy="42" r="2.6"/><circle cx="55" cy="42" r="2.6"/>
      <line x1="12" y1="48" x2="10" y2="60"/><line x1="52" y1="48" x2="54" y2="60"/>
      <rect class="item" x="23" y="20" width="18" height="16" rx="1.5" fill="currentColor" fill-opacity=".25"/></svg>`
  };
  const ICON_OF = { p_c: "press", ahu_a: "fan", comp_b: "compressor", mh_a: "conveyor" };
  const iconFor = (a) => ICON[ICON_OF[a] || "press"];
  const TIDX = {}; Object.keys(TWIN.assets).forEach((a) => { const m = {}; TWIN.assets[a].t.forEach((x, k) => m[x] = k); TIDX[a] = m; });
  const assets = Object.keys(TWIN.assets);
  let cur = assets.includes("p_c") ? "p_c" : assets[0];
  let i = 0, timer = null;

  // tabs
  assets.forEach((a) => {
    const b = document.createElement("button");
    b.className = "mach"; b.type = "button"; b.setAttribute("role", "tab"); b.id = "tab-" + a;
    b.innerHTML = `<span class="ico">${iconFor(a)}</span><b>${a}</b><small>${TWIN.assets[a].role}</small><span class="st" id="st-${a}">&ndash;</span>`;
    b.onclick = () => { const t = TWIN.assets[cur].t[i]; cur = a; const k = TIDX[a][t]; if (k != null) i = k; setup(); };
    $("tabs").appendChild(b);
  });

  function setup() {
    const A = TWIN.assets[cur];
    assets.forEach((a) => $("tab-" + a).setAttribute("aria-selected", a === cur ? "true" : "false"));
    $("slider").max = A.t.length - 2;
    i = Math.min(Math.max(i, 48), A.t.length - 2);
    if (A.start != null && i === 48) i = A.start;
    $("slider").value = i;
    $("maxTxt").textContent = `max ${fmt(A.max, 1)} kW`;
    $("aEnh").textContent = fmt(A.rmse_enh, 3) + " kW";
    $("aPers").textContent = fmt(A.rmse_pers, 3) + " kW";
    $("aSkill").textContent = fmt(100 * (1 - A.rmse_enh / A.rmse_pers), 1) + " %";
    // alert log: anomalous days inside the test period
    const days = Object.entries(A.days).filter(([d, r]) => r.s > A.thr && d >= A.t[0].slice(0, 10))
      .sort((x, y) => y[1].s - x[1].s);
    $("alertCount").textContent = `${days.length} days above threshold`;
    const log = $("log"); log.innerHTML = "";
    days.slice(0, 40).forEach(([d, r]) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="sev"></span><span>${d} (${r.w ? "weekend" : "weekday"})</span><b>${fmt(r.s / A.thr, 2)}&times;</b>`;
      li.title = "Jump to this day";
      li.onclick = () => { const k = A.t.findIndex((t) => t.startsWith(d)); if (k >= 0) { i = Math.min(k + 18, A.t.length - 2); render(); } };
      log.appendChild(li);
    });
    if (!days.length) log.innerHTML = `<li><span></span><span>No anomalous days in the test period</span><b></b></li>`;
    $("foot").textContent = TWIN.note;
    render();
  }

  function hoursSame(A, k) { // hours the machine has stayed in its current state
    const on = (x) => x > A.onKw;
    let n = 0; const s = on(A.a[k]);
    for (let j = k; j >= 0 && on(A.a[j]) === s && n < 999; j--) n++;
    return n;
  }

  function render() {
    const A = TWIN.assets[cur];
    const t = A.t[i], day = t.slice(0, 10), hr = +t.slice(11, 13);
    $("when").textContent = `${t}:00 UTC`;
    $("slider").value = i;
    // state
    const on = A.a[i] > A.onKw, hs = hoursSame(A, i);
    $("statePill").className = "pill " + (on ? "ok" : "off");
    $("statePill").textContent = on ? "RUNNING" : "IDLE";
    $("stateName").textContent = A.role;
    const anomToday = A.days[day] && A.days[day].s > A.thr;
    $("bigIco").innerHTML = iconFor(cur);
    $("bigIco").className = "bigico " + (anomToday ? "s-anom" : on ? "s-run" : "");
    assets.forEach((a) => {
      const B = TWIN.assets[a], k = TIDX[a][t], el = $("tab-" + a);
      if (k == null || B.a[k] == null) { el.className = "mach"; $("st-" + a).textContent = "no data"; return; }
      const r = B.days[t.slice(0, 10)], an = r && r.s > B.thr, run = B.a[k] > B.onKw;
      el.className = "mach " + (an ? "s-anom" : run ? "s-run" : "");
      $("st-" + a).textContent = `${an ? "ANOMALY" : run ? "RUNNING" : "IDLE"} · ${fmt(B.a[k], 1)} kW`;
    });
    $("stateSub").textContent = `${on ? "Running" : "Idle"} for ${hs >= 999 ? "999+" : hs} h · ON above ${fmt(A.onKw, 1)} kW`;
    // demand
    $("nowKw").innerHTML = `${fmt(A.a[i])}<small>kW</small>`;
    $("nowBar").style.width = Math.min(100, 100 * A.a[i] / A.max) + "%";
    // forecast for next hour (index i+1)
    const nt = A.t[i + 1];
    $("fcKw").innerHTML = `${fmt(A.f[i + 1])}<small>kW</small>`;
    const err = A.a[i] - A.f[i];
    $("fcSub").textContent = `for ${nt.slice(11, 13)}:00 · persistence ${fmt(A.p[i + 1])} kW · last-hour error ${err >= 0 ? "+" : ""}${fmt(err)} kW`;
    // day health
    const r = A.days[day];
    if (r) {
      const ratio = r.s / A.thr, bad = r.s > A.thr, near = ratio > 0.8;
      $("healthPill").className = "pill " + (bad ? "crit" : near ? "warn" : "ok");
      $("healthPill").textContent = bad ? "ANOMALY" : near ? "WATCH" : "NORMAL";
      $("healthBar").style.width = Math.min(100, 100 * ratio / 2) + "%";
      $("healthBar").style.background = css(bad ? "--crit" : near ? "--warn" : "--ok");
      $("healthMark").style.left = "50%";
      $("healthSub").textContent = `score ${fmt(r.s, 3)} · alert threshold ${fmt(A.thr, 3)} (top 5 %)`;
      $("cHealth").classList.toggle("alert-crit", bad);
      $("iLF").textContent = fmt(r.LF, 3); $("iPBR").textContent = r.PBR == null ? "n/a (base < 0.5 kW)" : fmt(r.PBR, 2);
      $("iVI").textContent = fmt(r.VI, 2);
    } else {
      $("healthPill").className = "pill off"; $("healthPill").textContent = "INCOMPLETE DAY";
      $("healthBar").style.width = "0%"; $("healthSub").textContent = "Day has missing hours; not scored";
      $("cHealth").classList.remove("alert-crit");
      ["iLF", "iPBR", "iVI"].forEach((k) => $(k).textContent = "–");
    }
    // energy so far today and rolling MAE
    let e = 0, k = i; while (k >= 0 && A.t[k].startsWith(day)) { e += A.a[k]; k--; }
    $("iE").textContent = fmt(e, 1) + " kWh";
    let s = 0, n = 0; for (let j = Math.max(0, i - 23); j <= i; j++) { s += Math.abs(A.a[j] - A.f[j]); n++; }
    $("aRoll").textContent = fmt(s / n, 3) + " kW";
    drawTrend(A); drawDay(A, day);
    $("trendNote").textContent = A.a[i + 1] != null ? `actual next hour ${fmt(A.a[i + 1])} kW` : "";
  }

  function axis(svg, x0, x1, y0, y1, ymax, ticks) {
    let g = "";
    for (let k = 0; k <= ticks; k++) {
      const v = ymax * k / ticks, y = y1 - (y1 - y0) * k / ticks;
      g += `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" stroke="${css("--line")}" stroke-width="1"/>`;
      g += `<text x="${x0 - 6}" y="${y + 3}" text-anchor="end">${String(+v.toFixed(2))}</text>`;
    }
    return g;
  }
  const niceMax = (m) => { const p = Math.pow(10, Math.floor(Math.log10(m || 1))); for (const s of [1, 2, 2.5, 5, 10]) if (s * p >= m) return s * p; return 10 * p; };

  function path(xs, ys) {
    let d = "", pen = false;
    xs.forEach((x, k) => { const y = ys[k]; if (y == null || isNaN(y)) { pen = false; return; } d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; });
    return d;
  }

  function drawTrend(A) {
    const W = 720, H = 250, x0 = 44, x1 = W - 14, y0 = 12, y1 = H - 28;
    const a0 = Math.max(0, i - 47), idx = []; for (let k = a0; k <= i + 1; k++) idx.push(k);
    const vals = idx.flatMap((k) => [A.a[k], A.f[k], A.p[k]]).filter((v) => v != null);
    const ymax = niceMax(Math.max(...vals, A.onKw) * 1.05);
    const X = (k) => x0 + (x1 - x0) * (k - a0) / 48, Y = (v) => y1 - (y1 - y0) * v / ymax;
    let g = axis(null, x0, x1, y0, y1, ymax, 4);
    // time labels every 12 h
    idx.forEach((k) => { if (+A.t[k].slice(11, 13) % 12 === 0) g += `<text x="${X(k)}" y="${H - 10}" text-anchor="middle">${A.t[k].slice(5, 10)} ${A.t[k].slice(11, 13)}h</text>`; });
    // ON threshold
    g += `<line x1="${x0}" x2="${x1}" y1="${Y(A.onKw)}" y2="${Y(A.onKw)}" stroke="${css("--ok")}" stroke-opacity=".35" stroke-dasharray="2 4"/>`;
    // now marker
    g += `<line x1="${X(i)}" x2="${X(i)}" y1="${y0}" y2="${y1}" stroke="${css("--muted")}" stroke-dasharray="3 3"/>`;
    g += `<text x="${X(i) - 4}" y="${y0 + 10}" text-anchor="end">now</text>`;
    const past = idx.filter((k) => k <= i);
    g += `<path d="${path(idx.map(X), idx.map((k) => A.p[k] == null ? null : Y(A.p[k])))}" fill="none" stroke="${css("--persist")}" stroke-width="1.3" stroke-dasharray="5 4"/>`;
    g += `<path d="${path(idx.map(X), idx.map((k) => A.f[k] == null ? null : Y(A.f[k])))}" fill="none" stroke="${css("--forecast")}" stroke-width="1.8"/>`;
    g += `<path d="${path(past.map(X), past.map((k) => A.a[k] == null ? null : Y(A.a[k])))}" fill="none" stroke="${css("--actual")}" stroke-width="2.2"/>`;
    // next-hour forecast point
    g += `<circle cx="${X(i + 1)}" cy="${Y(A.f[i + 1])}" r="5" fill="${css("--forecast")}" stroke="${css("--bg")}" stroke-width="2"/>`;
    $("trend").innerHTML = g;
  }

  function drawDay(A, day) {
    const W = 420, H = 250, x0 = 40, x1 = W - 10, y0 = 12, y1 = H - 28;
    const wk = A.days[day] ? A.days[day].w : ((new Date(day + "T00:00:00Z").getUTCDay() % 6 === 0) ? 1 : 0);
    const typ = A.typical[String(wk)];
    $("dayType").textContent = wk ? "weekend reference" : "weekday reference";
    const today = new Array(24).fill(null);
    for (let k = i; k >= 0 && A.t[k].startsWith(day); k--) today[+A.t[k].slice(11, 13)] = A.a[k];
    const ymax = niceMax(Math.max(...typ, ...today.filter((v) => v != null), 0.1) * 1.08);
    const X = (h) => x0 + (x1 - x0) * h / 23, Y = (v) => y1 - (y1 - y0) * v / ymax;
    let g = axis(null, x0, x1, y0, y1, ymax, 4);
    [0, 6, 12, 18, 23].forEach((h) => g += `<text x="${X(h)}" y="${H - 10}" text-anchor="middle">${String(h).padStart(2, "0")}h</text>`);
    const hrs = [...Array(24).keys()];
    g += `<path d="${path(hrs.map(X), typ.map(Y))}" fill="none" stroke="${css("--warn")}" stroke-width="1.6" stroke-dasharray="6 4"/>`;
    const area = path(hrs.map(X), today.map((v) => v == null ? null : Y(v)));
    g += `<path d="${area}" fill="none" stroke="${css("--actual")}" stroke-width="2.2"/>`;
    $("dayChart").innerHTML = g;
  }

  function play() {
    if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; $("liveTxt").textContent = "Replay paused"; return; }
    $("play").textContent = "Pause"; $("liveTxt").textContent = "Replaying test data";
    timer = setInterval(() => { const A = TWIN.assets[cur]; i = i >= A.t.length - 2 ? 48 : i + 1; render(); }, 350);
  }
  $("play").onclick = play;
  $("stepF").onclick = () => { i = Math.min(i + 1, TWIN.assets[cur].t.length - 2); render(); };
  $("stepB").onclick = () => { i = Math.max(i - 1, 0); render(); };
  $("slider").oninput = (e) => { i = +e.target.value; render(); };
  setup();
})();
</script>
"""

html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + TWIN_HEAD + '\n</head>\n<body>\n'
        + TWIN_BODY.replace('/*__DATA__*/null', json.dumps(twin, separators=(',', ':')))
        + '\n</body>\n</html>\n')
with open('digital_twin.html', 'w', encoding='utf-8') as fh:
    fh.write(html)
print(f"\nSaved digital_twin.html ({len(html)/1e6:.1f} MB).")

# Download it from Colab and open in a browser:
try:
    from google.colab import files
    files.download('digital_twin.html')
except Exception:
    print("Not running in Colab -- open digital_twin.html from the working folder.")
print("Cell 19 complete.")
