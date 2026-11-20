# =====================================================================================================
#  ASSET-LEVEL SMART-METERING DIGITAL TWIN, COMPLETE PIPELINE IN ONE CELL (Google Colab)
#  8 assets (p_c, ahu_a, comp_b, mh_a, p_h, u_b, ex_a, mix_c) | next-hour forecasting | 7 injected fault types + unseen fault mixtures
#  metering view (daily energy, 15-min peak demand, excess energy of alert days) | digital twin | 600-dpi figures
#
#  HOW TO RUN
#   1. Open a new Colab notebook, paste this whole cell, Runtime -> Run all (GPU optional).
#   2. The dataset (reliable 15-min windows of the eight assets, CC BY 4.0, Flynn et al. 2026) is downloaded
#      automatically from the project repository. Without internet, upload eight_assets_15min.parquet
#      when asked, or set DATA_FILE to its path.
#   3. At the end results.zip is downloaded: figures (PNG + PDF, 600 dpi), tables (CSV), LaTeX parts,
#      the offline digital twin (HTML) and this code. QUICK=True runs one seed for a fast check.
# =====================================================================================================
QUICK       = False
USE_DRIVE   = False
DATA_FILE   = 'eight_assets_15min.parquet'
DATA_URL    = 'https://raw.githubusercontent.com/EngrAdil575/asset-level-digital-twin-smart-metering/main/data/eight_assets_15min.parquet'
DATA_ZIP    = ''
RUN_NN      = True            # LSTM, CNN-LSTM, TCN (+ LSTM+XGBoost hybrid and weighted ensemble)
ASSETS      = ['p_c', 'ahu_a', 'comp_b', 'mh_a', 'p_h', 'u_b', 'ex_a', 'mix_c']
ROLE        = {'p_c': 'Press drive', 'ahu_a': 'Air-handling unit', 'comp_b': 'Air compressor', 'mh_a': 'Material handling',
               'p_h': 'Press (main)', 'u_b': 'Utility circuit', 'ex_a': 'Extraction fan', 'mix_c': 'Process mixer'}
MODEL_NAME  = 'SA-XGBoost'    # two-stage switching-aware XGBoost (the proposed forecaster)
import os, sys, io, re, json, glob, time, math, random, shutil, zipfile, subprocess, warnings, pickle
warnings.filterwarnings('ignore')
TEST = os.environ.get('PIPELINE_TEST') == '1'          # developer switch: mechanical test with stand-in libraries
if TEST: QUICK = True; USE_DRIVE = False
SEEDS = [42] if QUICK else [42, 7, 123, 2024, 31]
OUT = os.environ.get('PIPELINE_OUT', 'pipeline_outputs') if not TEST else os.environ.get('PIPELINE_OUT', 'pipeline_outputs_test')
for sub in ['figures', 'tables', 'latex_parts', 'twin', 'code', 'cache']: os.makedirs(f'{OUT}/{sub}', exist_ok=True)
T0 = time.time()
def say(msg): print(f'[{(time.time()-T0)/60:5.1f} min] {msg}', flush=True)
def need(mod, pip=None):
    try: __import__(mod); return True
    except Exception:
        if TEST: return False
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pip or mod], check=False)
        try: __import__(mod); return True
        except Exception: return False
if not TEST:
    for m, p in [('xgboost', 'xgboost'), ('statsmodels', 'statsmodels'), ('pyarrow', 'pyarrow')]: need(m, p)
if os.environ.get('PIPELINE_ZIP'): DATA_ZIP = os.environ['PIPELINE_ZIP']
if USE_DRIVE and not TEST and DATA_ZIP:
    from google.colab import drive; drive.mount('/content/drive')
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---- libraries (the cell still runs if one of them is missing; missing ones are reported and skipped)
HAVE_XGB = HAVE_SM = HAVE_TF = False
if not TEST:
    try: import xgboost as xgb; HAVE_XGB = True
    except Exception as ex_: print('WARNING: xgboost could not be imported:', ex_)
    try: import statsmodels.api; HAVE_SM = True
    except Exception as ex_: print('WARNING: statsmodels missing, ARIMA and VAR will be skipped:', ex_)
    try: import tensorflow as tf; HAVE_TF = RUN_NN
    except Exception as ex_: print('WARNING: tensorflow missing, LSTM / CNN-LSTM / TCN will be skipped:', ex_)
STANDIN = not HAVE_XGB                 # True only if xgboost is unavailable (then scikit-learn gradient boosting is used and the outputs say so)
if STANDIN:
    from sklearn.ensemble import HistGradientBoostingRegressor as _HR, HistGradientBoostingClassifier as _HC
    class _XR:
        def __init__(s, n_estimators=200, learning_rate=0.1, max_depth=6, random_state=0, **k):
            s.m = _HR(max_iter=min(n_estimators, 300), learning_rate=max(learning_rate, 0.06), max_depth=max_depth, random_state=random_state, early_stopping=False); s.best_iteration = 299
        def fit(s, X, y, sample_weight=None, eval_set=None, verbose=False): s.m.fit(X, y, sample_weight=sample_weight); return s
        def predict(s, X): return s.m.predict(X)
    class _XC:
        def __init__(s, n_estimators=200, learning_rate=0.1, max_depth=5, random_state=0, **k):
            s.m = _HC(max_iter=min(n_estimators, 300), learning_rate=max(learning_rate, 0.06), max_depth=max_depth, random_state=random_state, early_stopping=False)
        def fit(s, X, y, sample_weight=None, eval_set=None, verbose=False): s.m.fit(X, y, sample_weight=sample_weight); return s
        def predict_proba(s, X): return s.m.predict_proba(X)
    class xgb: XGBRegressor = _XR; XGBClassifier = _XC
    print('NOTE: XGBoost is not available here, scikit-learn gradient boosting is used instead (marked as stand-in in the outputs).')
say(f'assets: {ASSETS} | seeds: {SEEDS} | xgboost: {HAVE_XGB} | statsmodels: {HAVE_SM} | tensorflow: {HAVE_TF} | test-mode: {TEST}')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'TeX Gyre Termes', 'Nimbus Roman', 'STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
def savefig(fig, name):
    fig.savefig(f'{OUT}/figures/{name}.pdf', bbox_inches='tight', dpi=600); fig.savefig(f'{OUT}/figures/{name}.png', bbox_inches='tight', dpi=600); plt.close(fig)
NUM = {}                                           # numbers that the paper uses (written to numbers.tex)
def fmt_tex(v):
    return str(v)
# ------------------------------------------------------------------ 1. data
say('STEP 1  reading the dataset')
def find_zip():
    if os.path.exists(DATA_ZIP): return DATA_ZIP
    cands = []
    for root in ['/content/drive/MyDrive', '/content']:
        for pat in ['*.zip', '*/*.zip', '*/*/*.zip']:
            cands += glob.glob(os.path.join(root, pat))
    cands = [c for c in cands if re.search(r'industrial|energy|asset', os.path.basename(c), re.I) and os.path.getsize(c) > 5e7]
    if cands:
        print('dataset ZIP found automatically:', cands[0]); return cands[0]
    try:
        from google.colab import files
        print('Dataset ZIP not found. Please select the dataset ZIP now (the large file with data_csv / data_parquet):')
        up = files.upload(); return [k for k in up if k.endswith('.zip')][0]
    except Exception:
        raise FileNotFoundError('Dataset ZIP not found. Put it in Google Drive (MyDrive) or set DATA_ZIP at the top of the cell.')

COLS = ['AssetId', 'window_start_utc', 'Phase', 'IsReliableWindow', 'Demand_kW']
def load_raw():
    if os.path.exists(f'{OUT}/cache/raw.pkl'): return pd.read_pickle(f'{OUT}/cache/raw.pkl')
    if TEST and os.path.exists(os.environ.get('PIPELINE_RAW', '')): df = pd.read_pickle(os.environ['PIPELINE_RAW'])
    else:
        z = zipfile.ZipFile(find_zip()); names = z.namelist(); parts = []
        rel = [n for n in names if n.endswith('all_assets_rel.parquet')]
        if rel:
            df = pd.read_parquet(io.BytesIO(z.read(rel[0]))); df['IsReliableWindow'] = True
        else:
            files = [n for n in names if n.endswith('.csv') and '/data_csv/' in n] or [n for n in names if n.endswith('.parquet') and '/data_parquet/' in n]
            assert files, 'no data_csv / data_parquet folder in the ZIP'
            for n in files:
                m = re.search(r'asset_id=([^/]+)/', n)
                if m and m.group(1) not in ASSETS: continue
                d = pd.read_csv(io.BytesIO(z.read(n)), usecols=COLS) if n.endswith('.csv') else pd.read_parquet(io.BytesIO(z.read(n)), columns=COLS)
                parts.append(d[(d.Phase == 'ALL') & d.AssetId.astype(str).isin(ASSETS)])
            assert parts, 'none of the assets %s was found in the ZIP' % ASSETS
            df = pd.concat(parts, ignore_index=True)
    df['AssetId'] = df.AssetId.astype(str); df = df[df.AssetId.isin(ASSETS)]
    df['rel'] = df.IsReliableWindow.astype(str).str.lower().isin(['true', '1', '1.0'])
    df.to_pickle(f'{OUT}/cache/raw.pkl'); return df
def load_file():
    if os.path.exists(f'{OUT}/cache/raw.pkl'): return pd.read_pickle(f'{OUT}/cache/raw.pkl')
    fn = os.environ.get('PIPELINE_DATA', DATA_FILE)
    if not os.path.exists(fn):
        try:
            import urllib.request; urllib.request.urlretrieve(DATA_URL, fn); print('dataset downloaded from', DATA_URL)
        except Exception as ex_:
            print('download failed (%s)' % ex_)
            try:
                from google.colab import files
                print('Please upload eight_assets_15min.parquet'); up = files.upload(); fn = list(up)[0]
            except Exception: raise FileNotFoundError('dataset file not found: set DATA_FILE')
    df = pd.read_parquet(fn) if fn.endswith('.parquet') else pd.read_csv(fn)
    df = df[(df.Phase.astype(str) == 'ALL') & df.AssetId.astype(str).isin(ASSETS)].copy(); df['AssetId'] = df.AssetId.astype(str)
    df['rel'] = df.IsReliableWindow.astype(str).str.lower().isin(['true', '1', '1.0']); df.to_pickle(f'{OUT}/cache/raw.pkl'); return df
raw = load_file() if not DATA_ZIP else load_raw(); raw = raw[raw.rel].copy()
raw['Demand_kW'] = pd.to_numeric(raw.Demand_kW, errors='coerce').astype(float).round(6); raw['window_start_utc'] = pd.to_datetime(raw.window_start_utc)
raw = raw.dropna(subset=['Demand_kW']).sort_values(['AssetId', 'window_start_utc', 'Demand_kW'], kind='mergesort').reset_index(drop=True); raw['t'] = pd.to_datetime(raw.window_start_utc)
q = raw.groupby(['AssetId', 't']).Demand_kW.mean().reset_index(); q['hts'] = q.t.dt.floor('h')
hourly = q.groupby(['AssetId', 'hts']).Demand_kW.mean().reset_index().rename(columns={'hts': 'ts', 'Demand_kW': 'load'})
frames = []
for a in ASSETS:
    s = hourly[hourly.AssetId == a].set_index('ts')['load'].sort_index(); s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
    frames.append(pd.DataFrame({'AssetId': a, 'ts': s.index, 'load': s.values}))
base = pd.concat(frames, ignore_index=True)
NUM['n_asset_hours'] = int(base.load.notna().sum())
all_ts = np.sort(base.loc[base.load.notna(), 'ts'].unique())
t_val = pd.Timestamp(all_ts[int(len(all_ts) * 0.70)]).normalize(); t_test = pd.Timestamp(all_ts[int(len(all_ts) * 0.80)]).normalize()
scale = base[base.ts < t_val].groupby('AssetId')['load'].max().replace(0, 1.0)
base['load_s'] = base.load / base.AssetId.map(scale); q['q_s'] = q.Demand_kW / q.AssetId.map(scale)
q_lookup = q.set_index(['AssetId', 't'])['q_s']
THETA, N_LAGS, N_Q = 0.2, 24, 8
say(f'train < {t_val.date()} <= validation < {t_test.date()} <= test; max demand per asset (kW): {scale.round(1).to_dict()}')
# ------------------------------------------------------------------ 2. daily profiles, indices, clustering
say('STEP 2  daily profiles, load-profile indices, K-means and DBSCAN')
base['date'] = base.ts.dt.normalize(); base['hr'] = base.ts.dt.hour
day = base.pivot_table(index=['AssetId', 'date'], columns='hr', values='load', fill_value=0)
day_norm = day.div(day.max(axis=1), axis=0).fillna(0); Xd = day_norm.values
NUM['n_asset_days'] = len(Xd)
best = (-1, None)
for eps in np.round(np.arange(0.02, 0.41, 0.01), 3):
    lab = DBSCAN(eps=eps, min_samples=5).fit(Xd).labels_; n = len(set(lab)) - (1 if -1 in lab else 0); m = lab != -1
    if n >= 2 and m.sum() > n:
        s_ = silhouette_score(Xd[m], lab[m])
        if s_ > best[0]: best = (s_, eps)
EPS = float(best[1]); lab_db = DBSCAN(eps=EPS, min_samples=5).fit(Xd).labels_; keep = lab_db != -1
res_k = []
for k in range(2, 8):
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xd); res_k.append((k, silhouette_score(Xd, l), davies_bouldin_score(Xd, l)))
kb = max(res_k, key=lambda r: r[1]); K_BEST = kb[0]
rows_cl = [('K-means, all days', len(Xd), K_BEST, kb[1], kb[2]),
           (f'DBSCAN, eps={EPS:g}, kept days', int(keep.sum()), len(set(lab_db[keep])), silhouette_score(Xd[keep], lab_db[keep]), davies_bouldin_score(Xd[keep], lab_db[keep]))]
for k in [len(set(lab_db[keep])), 2]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xd[keep]); rows_cl.append((f'K-means, k={k}, same days', int(keep.sum()), k, silhouette_score(Xd[keep], l), davies_bouldin_score(Xd[keep], l)))
CL = pd.DataFrame(rows_cl, columns=['Method', 'Days', 'Clusters', 'Silh', 'DBI']); CL.to_csv(f'{OUT}/tables/clustering.csv', index=False)
NUM.update(kmeans_k=K_BEST, sil_all=kb[1], dbi_all=kb[2], eps=EPS, n_noise=int((~keep).sum()), noise_share=100 * (~keep).mean(), n_kept=int(keep.sum()), n_dbscan_clusters=len(set(lab_db[keep])),
           sil_db=rows_cl[1][3], dbi_db=rows_cl[1][4], sil_k2=rows_cl[3][3], dbi_k2=rows_cl[3][4], sil_k3=rows_cl[2][3])
for e in [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2]:
    l = DBSCAN(eps=e, min_samples=5).fit(Xd).labels_; ns = (l == -1).mean()
    if ns < 0.20: NUM['eps_below20'] = e; NUM['clusters_below20'] = len(set(l)) - (1 if -1 in l else 0); break
NUM.setdefault('eps_below20', 'n/a'); NUM.setdefault('clusters_below20', 'n/a')
# day indices
didx = day_norm.index.to_frame(index=False); trd = (didx.date < t_val).values
db = DBSCAN(eps=EPS, min_samples=5).fit(Xd[trd]); core = Xd[trd][db.core_sample_indices_] if len(db.core_sample_indices_) else Xd[trd]
from sklearn.metrics import pairwise_distances
didx['day_dist'] = pairwise_distances(Xd, core).min(axis=1); didx['day_anom'] = (didx.day_dist > EPS).astype(int)
def _idx(g):
    p = g['load'].dropna(); mx, mn = p.max(), p.min()
    return pd.Series({'LF': p.mean() / mx if mx > 0 else np.nan, 'PBR': mx / mn if mn >= 0.5 else np.nan, 'VI': p.var()})
ind = base.groupby(['AssetId', 'date']).apply(_idx).reset_index(); didx = didx.merge(ind, on=['AssetId', 'date'], how='left')
didx['day_mean'] = base.groupby(['AssetId', 'date'])['load_s'].mean().reindex(pd.MultiIndex.from_frame(didx[['AssetId', 'date']])).values
didx['VIs'] = didx['VI'] / didx.AssetId.map(scale) ** 2
tab1 = []
for a in ASSETS:
    x = didx[didx.AssetId == a]; tab1.append(dict(asset=a, role=ROLE[a], days=len(x), LF=x.LF.mean(), PBR=x.PBR.mean(), PBR_n=x.PBR.notna().mean(), VI=x.VI.mean(), mean_kw=base[base.AssetId == a].load.mean(), max_kw=scale[a]))
TAB1 = pd.DataFrame(tab1); TAB1.to_csv(f'{OUT}/tables/assets.csv', index=False)
# ------------------------------------------------------------------ 3. features (next hour)
say('STEP 3  features for the next-hour horizon')
on_prob = (base[(base.ts < t_val) & base.load_s.notna()].assign(on=lambda d: (d.load_s > THETA).astype(int), dow=lambda d: d.ts.dt.dayofweek).groupby(['AssetId', 'dow', 'hr'])['on'].mean())
PRESS = [a for a in ASSETS if a.startswith('p_')]
def build(H):
    d = base[['AssetId', 'ts', 'load', 'load_s']].copy(); g = d.groupby('AssetId')['load_s']
    for k in range(1, N_LAGS + 1): d[f'lag_{k}'] = g.shift(H + k - 1)
    d['lag_168'] = g.shift(168); sh = g.shift(H)
    d['roll_mean_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).mean()); d['roll_std_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).std())
    hr, dow = d.ts.dt.hour, d.ts.dt.dayofweek
    d['hour_sin'], d['hour_cos'] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24); d['dow_sin'], d['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7); d['is_weekend'] = (dow >= 5).astype(int)
    for a in ASSETS: d[f'asset_{a}'] = (d.AssetId == a).astype(int)
    oe = d.ts - pd.Timedelta(hours=H - 1)
    for k in range(1, N_Q + 1): d[f'q_lag_{k}'] = q_lookup.reindex(pd.MultiIndex.from_arrays([d.AssetId, oe - pd.Timedelta(minutes=15 * k)])).values
    d['q_slope'] = d['q_lag_1'] - d['q_lag_4']
    d['on_lag1'] = (d['lag_1'] > THETA).astype(int); d['on_yday'] = (g.shift(24) > THETA).astype(int)
    onm = d[[f'lag_{k}' for k in range(1, 25)]].values > THETA; same = onm == onm[:, [0]]; d['hours_in_state'] = np.where(same.all(axis=1), 24, np.argmin(same, axis=1))
    d['usual_on_prob'] = on_prob.reindex(pd.MultiIndex.from_arrays([d.AssetId, dow, hr])).values
    hp = d.ts - pd.Timedelta(hours=1); d['usual_on_prob_prev'] = on_prob.reindex(pd.MultiIndex.from_arrays([d.AssetId, hp.dt.dayofweek, hp.dt.hour])).values
    wl = d.pivot_table(index='ts', columns='AssetId', values='lag_1'); wq = d.pivot_table(index='ts', columns='AssetId', values='q_lag_1')
    on_l, on_q = (wl > THETA).astype(float).where(wl.notna()), (wq > THETA).astype(float).where(wq.notna())
    for name, W in {'x_mean_lag1': wl, 'x_mean_q1': wq, 'x_on_frac': on_l, 'x_on_frac_q1': on_q}.items():
        tot, cnt = W.sum(axis=1), W.notna().sum(axis=1); sv = W.stack().rename('own').reset_index(); m = d[['AssetId', 'ts']].merge(sv, on=['ts', 'AssetId'], how='left')
        T_, C_ = d.ts.map(tot).values, d.ts.map(cnt).values; own = m['own'].values; has = ~np.isnan(own)
        d[name] = np.where(C_ - has > 0, (T_ - np.nan_to_num(own)) / np.maximum(C_ - has, 1), np.nan)
    Wp = on_l[[c for c in on_l.columns if c in PRESS]]; tot, cnt = Wp.sum(axis=1), Wp.notna().sum(axis=1); sv = Wp.stack().rename('own').reset_index()
    m = d[['AssetId', 'ts']].merge(sv, on=['ts', 'AssetId'], how='left'); own = m['own'].values; has = ~np.isnan(own); T_, C_ = d.ts.map(tot).values, d.ts.map(cnt).values
    d['x_press_on'] = np.where(C_ - has > 0, (T_ - np.nan_to_num(own)) / np.maximum(C_ - has, 1), np.nan)
    od = (d.ts - pd.Timedelta(hours=H)).dt.normalize() - pd.Timedelta(days=1); dl = didx.set_index(['AssetId', 'date']); k2 = pd.MultiIndex.from_arrays([d.AssetId, od])
    d['prev_day_mean'] = dl['day_mean'].reindex(k2).values
    for c, n in [('day_dist', 'prev_day_dist'), ('day_anom', 'prev_day_anom'), ('LF', 'prev_LF'), ('PBR', 'prev_PBR'), ('VIs', 'prev_VI')]: d[n] = dl[c].reindex(k2).values
    d['pers'] = d['lag_1']; d['snaive'] = g.shift(24)
    n_before = len(d); d = d.dropna(subset=['load_s'] + [f'lag_{k}' for k in range(1, N_LAGS + 1)]).reset_index(drop=True)
    d['split'] = np.where(d.ts < t_val, 'train', np.where(d.ts < t_test, 'val', 'test')); d.attrs['dropped'] = int(base.load.notna().sum() - len(d))
    return d
GROUPS = {'lags': [f'lag_{k}' for k in range(1, N_LAGS + 1)] + ['lag_168', 'roll_mean_24', 'roll_std_24'],
          'calendar_asset': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'] + [f'asset_{a}' for a in ASSETS],
          'q15': [f'q_lag_{k}' for k in range(1, N_Q + 1)] + ['q_slope'], 'onoff': ['on_lag1', 'on_yday', 'hours_in_state', 'usual_on_prob', 'usual_on_prob_prev'],
          'cross': ['x_mean_lag1', 'x_mean_q1', 'x_on_frac', 'x_on_frac_q1', 'x_press_on'], 'prevday': ['prev_day_mean'], 'dbscan': ['prev_day_dist', 'prev_day_anom', 'prev_LF', 'prev_PBR', 'prev_VI']}
BASE = GROUPS['lags'] + GROUPS['calendar_asset']; ENH = BASE + GROUPS['q15'] + GROUPS['onoff'] + GROUPS['cross'] + GROUPS['prevday']
D = build(1); tr, va, te = (D.split == 'train').values, (D.split == 'val').values, (D.split == 'test').values
NUM.update(n_dropped=D.attrs['dropped'], n_remaining=len(D), n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()), test_start=str(D.loc[te, 'ts'].min().date()),
           n_base=len(BASE), n_enh=len(ENH), n_added=len(ENH) - len(BASE), n_cal=len(GROUPS['calendar_asset']))
y = D.load_s.values; sc = D.AssetId.map(scale).values; yk = y * sc
say(f'rows: train {tr.sum()}, validation {va.sum()}, test {te.sum()}; features: base {len(BASE)}, proposed {len(ENH)}')
# ------------------------------------------------------------------ 4. reference forecasts
say('STEP 4  reference forecasts: persistence, seasonal naive, K-means profile, ARIMA, VAR')
REF = {'Persistence': D.pers.values, 'Seasonal naive': D.snaive.fillna(D.pers).values}
dd = base[['AssetId', 'ts', 'load_s']].copy(); dd['day'] = dd.ts.dt.normalize(); dd['hr'] = dd.ts.dt.hour
shp = dd.pivot_table(index=['AssetId', 'day'], columns='hr', values='load_s'); shp = shp[shp.notna().sum(axis=1) >= 20].ffill(axis=1).bfill(axis=1)
peak = shp.max(axis=1).replace(0, np.nan); norm = shp.div(peak, axis=0).fillna(0); trm = norm.index.get_level_values('day') < t_val
bk = (2, -1)
for k in range(2, 8):
    s_ = silhouette_score(norm[trm], KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm[trm]))
    if s_ > bk[1]: bk = (k, s_)
kmp = KMeans(n_clusters=bk[0], random_state=42, n_init=10).fit(norm[trm]); cent = kmp.cluster_centers_
tabk = pd.DataFrame({'cl': kmp.predict(norm), 'pk': peak.values}, index=norm.index)
od = (D.ts - pd.Timedelta(hours=1)).dt.normalize() - pd.Timedelta(days=1); t2 = tabk.reindex(pd.MultiIndex.from_arrays([D.AssetId, od]))
cl_, pk_, hr_ = t2['cl'].values, t2['pk'].values, D.ts.dt.hour.values; okk = ~np.isnan(cl_); pp = np.full(len(D), np.nan); pp[okk] = cent[cl_[okk].astype(int), hr_[okk]] * pk_[okk]
REF['K-means profile'] = np.where(np.isnan(pp), D.snaive.fillna(D.pers).values, pp)
def _arima_var():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.api import VAR
    arima = np.full(len(D), np.nan)
    for a in ASSETS:
        s = base[base.AssetId == a].set_index('ts')['load_s']; s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
        res = SARIMAX(s[s.index < t_val], order=(2, 0, 1), seasonal_order=(1, 0, 0, 24), trend='c', enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
        ra = res.apply(s); Z = ra.model['design'][:, :, 0] if ra.model['design'].ndim == 3 else ra.model['design']
        dobs = float(np.ravel(ra.model['obs_intercept'])[0]); PS = ra.predicted_state; pos = pd.Series(np.arange(len(s)), index=s.index)
        rows = (D.AssetId == a).values; j = pos.reindex(D.loc[rows, 'ts']).values.astype(int); arima[rows] = (Z @ PS[:, j]).ravel() + dobs
    REF['ARIMA'] = np.where(np.isnan(arima), D.pers.values, arima)
    wide = base.pivot_table(index='ts', columns='AssetId', values='load_s')[ASSETS]; wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='h'))
    vr = VAR(wide[wide.index < t_val].dropna()).fit(maxlags=24, ic='aic'); pl = vr.k_ar; arr = wide.ffill().fillna(0).values; tpos = pd.Series(np.arange(len(wide)), index=wide.index)
    jj = tpos.reindex(D.ts).values.astype(int); oo = jj - 1; hist = np.stack([arr[np.clip(oo - L, 0, None)] for L in range(pl)], axis=1)
    yh = vr.intercept + sum(hist[:, L] @ vr.coefs[L].T for L in range(pl)); col = pd.Series(range(len(ASSETS)), index=ASSETS).reindex(D.AssetId).values
    REF['VAR'] = yh[np.arange(len(D)), col]; NUM['var_lag'] = int(pl)

if HAVE_SM:
    try: _arima_var()
    except Exception as ex_: print('WARNING: ARIMA / VAR skipped because of an error:', repr(ex_)); REF.pop('ARIMA', None); REF.pop('VAR', None)
# ------------------------------------------------------------------ 5. learned models
say('STEP 5  learned models (XGBoost family, random forest, neural networks, hybrids)')
y_tr, y_va = y[tr], y[va]
def XR(seed, **kw):
    p = dict(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, early_stopping_rounds=100, random_state=seed, n_jobs=2, tree_method='hist'); p.update(kw)
    return xgb.XGBRegressor(**p)
def fit_xgb(feats, seed, target=None):
    t_tr = y_tr if target is None else target[0]; t_va = y_va if target is None else target[1]
    m = XR(seed); m.fit(D.loc[tr, feats], t_tr, eval_set=[(D.loc[va, feats], t_va)], verbose=False); return m
def run_seed(sd):
    fn = f'{OUT}/cache/learned_s{sd}.pkl'
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    random.seed(sd); np.random.seed(sd); P = {}; t1 = time.time()
    def put(k, pv, pt): P[k + '|va'] = np.asarray(pv, float); P[k] = np.asarray(pt, float)
    m = fit_xgb(BASE, sd); put('XGBoost', m.predict(D.loc[va, BASE]), m.predict(D.loc[te, BASE]))
    rf = RandomForestRegressor(n_estimators=60 if TEST else 300, min_samples_leaf=3, max_features=0.5, random_state=sd, n_jobs=2).fit(D.loc[tr, BASE].fillna(-1), y_tr)
    put('Random forest', rf.predict(D.loc[va, BASE].fillna(-1)), rf.predict(D.loc[te, BASE].fillna(-1)))
    f = BASE + GROUPS['dbscan']; m = fit_xgb(f, sd); put('DBSCAN + XGBoost', m.predict(D.loc[va, f]), m.predict(D.loc[te, f]))
    m = fit_xgb(ENH, sd); put(MODEL_NAME, m.predict(D.loc[va, ENH]), m.predict(D.loc[te, ENH]))
    for gname in ['q15', 'onoff', 'cross']:
        f = [c for c in ENH if c not in GROUPS[gname]]; m = fit_xgb(f, sd); put('Ablation -' + gname, m.predict(D.loc[va, f]), m.predict(D.loc[te, f]))
    Xtr, Xva, Xte = D.loc[tr, ENH], D.loc[va, ENH], D.loc[te, ENH]; on_tr, on_va = (y_tr > THETA).astype(int), (y_va > THETA).astype(int)
    clf = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, early_stopping_rounds=100, eval_metric='logloss', random_state=sd, n_jobs=2) if not TEST else xgb.XGBClassifier(random_state=sd)
    clf.fit(Xtr, on_tr, eval_set=[(Xva, on_va)], verbose=False); pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]; P['auc'] = float(roc_auc_score(on_va, pva))
    def reg(mt, mv):
        r = XR(sd); r.fit(Xtr[mt], y_tr[mt], eval_set=[(Xva[mv], y_va[mv])], verbose=False); return r
    ron, roff = reg(on_tr == 1, on_va == 1), reg(on_tr == 0, on_va == 0)
    put('Two-stage hybrid', pva * ron.predict(Xva) + (1 - pva) * roff.predict(Xva), pte * ron.predict(Xte) + (1 - pte) * roff.predict(Xte))
    say(f'seed {sd}: tree models done ({time.time()-t1:.0f} s)')
    if HAVE_TF:
      try:
          from tensorflow import keras
          from tensorflow.keras import layers
          tf.random.set_seed(sd); seq = [f'lag_{k}' for k in range(24, 0, -1)]; aux = GROUPS['calendar_asset']
          def inp(mk): return [D.loc[mk, seq].values[..., None].astype('float32'), D.loc[mk, aux].values.astype('float32')]
          Xtr_n, Xva_n, Xte_n = inp(tr), inp(va), inp(te)
          def build_nn(kind):
              s_in, a_in = keras.Input((24, 1)), keras.Input((len(aux),)); x = s_in
              if kind == 'LSTM': x = layers.LSTM(64)(x)
              elif kind == 'CNN-LSTM':
                  x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x); x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x); x = layers.LSTM(64)(x)
              elif kind == 'TCN':
                  for dl in [1, 2, 4, 8]:
                      h = layers.Conv1D(32, 3, padding='causal', dilation_rate=dl, activation='relu')(x); h = layers.Conv1D(32, 3, padding='causal', dilation_rate=dl, activation='relu')(h)
                      x = layers.Add()([layers.Conv1D(32, 1)(x) if x.shape[-1] != 32 else x, h])
                  x = layers.Lambda(lambda z: z[:, -1, :])(x)
              x = layers.Concatenate()([x, a_in]); x = layers.Dense(32, activation='relu')(x); o = layers.Dense(1)(x)
              m = keras.Model([s_in, a_in], o); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse'); return m
          for k in ['LSTM', 'CNN-LSTM', 'TCN']:
              tf.random.set_seed(sd); m = build_nn(k)
              m.fit(Xtr_n, y_tr, validation_data=(Xva_n, y_va), epochs=30, batch_size=512, verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
              put(k, m.predict(Xva_n, verbose=0, batch_size=2048).ravel(), m.predict(Xte_n, verbose=0, batch_size=2048).ravel())
              if k == 'LSTM':   # hybrid: LSTM whose residual is corrected by XGBoost
                  ltr = m.predict(Xtr_n, verbose=0, batch_size=2048).ravel(); mr = fit_xgb(BASE, sd, target=(y_tr - ltr, y_va - P['LSTM|va']))
                  put('LSTM + XGBoost', P['LSTM|va'] + mr.predict(D.loc[va, BASE]), P['LSTM'] + mr.predict(D.loc[te, BASE]))
              say(f'seed {sd}: {k} done')
          comp = ['XGBoost', 'LSTM', 'CNN-LSTM', 'DBSCAN + XGBoost']; w = np.array([1 / np.mean((y_va - np.clip(P[c + '|va'], 0, None)) ** 2) for c in comp]); w = w / w.sum(); P['ens_w'] = w
          put('Weighted ensemble', sum(wi * P[c + '|va'] for wi, c in zip(w, comp)), sum(wi * P[c] for wi, c in zip(w, comp)))
      except Exception as ex_: print('WARNING: neural networks skipped because of an error:', repr(ex_))
    pickle.dump(P, open(fn, 'wb')); return P
PRED = {sd: run_seed(sd) for sd in SEEDS}
# The proposed SA-XGBoost is the two-stage switching-aware model (ON/OFF classifier + ON-hour and OFF-hour regressors on the
# same switching-aware inputs); it was selected because it has the lowest validation RMSE. The single-stage variant is kept for comparison.
for sd in SEEDS:
    P_ = PRED[sd]
    for suf in ('', '|va'):
        one, two = P_.pop('SA-XGBoost' + suf), P_.pop('Two-stage hybrid' + suf)
        P_['Single-stage SA-XGBoost' + suf] = one; P_[MODEL_NAME + suf] = two
# ------------------------------------------------------------------ 6. metrics, Diebold-Mariano, ablation
say('STEP 6  metrics, statistical test and ablation')
prev = D.lag_1.values[te]; swt = np.abs(y[te] - prev) > 0.2; yt = yk[te]; sct = sc[te]
def metr(p):
    pr = np.clip(np.nan_to_num(p), 0, None) * sct; e = yt - pr; mk = yt >= 0.5
    return dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e[mk] / yt[mk])), R2=1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2), SW=np.sqrt(np.mean(e[swt] ** 2)), ST=np.sqrt(np.mean(e[~swt] ** 2)))
RMSE_PERS = metr(REF['Persistence'][te])['RMSE']; M = {}
for k, v in REF.items(): M.setdefault(k, []).append(metr(v[te]))
for sd, P in PRED.items():
    for k in [k for k in P if '|' not in k and k not in ('auc', 'ens_w') and not k.startswith('Ablation')]: M.setdefault(k, []).append(metr(P[k]))
SUM = {}
for k, L in M.items():
    df = pd.DataFrame(L); SUM[k] = {c: (df[c].mean(), df[c].std(ddof=1) if len(df) > 1 else 0.0) for c in df.columns}; SUM[k]['Skill'] = (100 * (1 - SUM[k]['RMSE'][0] / RMSE_PERS), 0.0); SUM[k]['n'] = len(df)
pd.DataFrame({k: {c: v[c][0] for c in ['RMSE', 'MAE', 'MAPE', 'R2', 'Skill', 'SW', 'ST']} for k, v in SUM.items()}).T.to_csv(f'{OUT}/tables/forecast_summary.csv')
def hourly(ts_, l): return pd.Series(l).groupby(pd.Series(ts_)).mean().sort_index().values
def dm(d, h=1, lags=None):
    d = np.asarray(d, float); n = len(d); db = d.mean(); L = h - 1 if lags is None else lags
    g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(L + 1)]; var = (g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))) / n
    st = db / np.sqrt(var); return st, 2 * (1 - stats.t.cdf(abs(st), n - 1))
B2 = [m for m in ['ARIMA', 'VAR', 'Random forest', 'XGBoost', 'LSTM', 'TCN'] if m in SUM]
def val_rmse(k): return np.mean([np.sqrt(np.mean((y_va - np.clip(np.nan_to_num(PRED[sd][k + '|va']), 0, None)) ** 2)) for sd in SEEDS]) if k + '|va' in PRED[SEEDS[0]] else np.nan
vr_ = {k: (val_rmse(k) if k in PRED[SEEDS[0]] or k + '|va' in PRED[SEEDS[0]] else np.nan) for k in B2}
sel = min([k for k in B2 if not np.isnan(vr_[k])], key=lambda k: vr_[k]) if any(not np.isnan(v) for v in vr_.values()) else 'XGBoost'
ts_te = D.ts.values[te]; DMR = []
for sd in SEEDS:
    pb = PRED[sd][sel] if sel in PRED[sd] else REF[sel][te]; pp_ = PRED[sd][MODEL_NAME]
    e1 = yt - np.clip(np.nan_to_num(pb), 0, None) * sct; e2 = yt - np.clip(pp_, 0, None) * sct; d_ = hourly(ts_te, e1 ** 2 - e2 ** 2); DMR.append(dict(seed=sd, vs=sel, DM=dm(d_)[0], p=dm(d_)[1], p_nw24=dm(d_, lags=24)[1]))
DMR = pd.DataFrame(DMR); DMR.to_csv(f'{OUT}/tables/dm_tests.csv', index=False)
NUM.update(best_baseline=sel, dm_pmax=DMR.p.max(), dm_pmax_nw24=DMR.p_nw24.max(), dm_min=DMR.DM.min())
pr_ = SUM[MODEL_NAME]; NUM.update(rmse_prop=pr_['RMSE'][0], rmse_prop_sd=pr_['RMSE'][1], mae_prop=pr_['MAE'][0], mape_prop=pr_['MAPE'][0], r2_prop=pr_['R2'][0], skill_prop=pr_['Skill'][0], sw_prop=pr_['SW'][0], sw_prop_sd=pr_['SW'][1],
           rmse_pers=RMSE_PERS, sw_pers=SUM['Persistence']['SW'][0], st_pers=SUM['Persistence']['ST'][0], rmse_xgb=SUM['XGBoost']['RMSE'][0], imp_xgb=100 * (1 - pr_['RMSE'][0] / SUM['XGBoost']['RMSE'][0]),
           imp_pers=100 * (1 - pr_['RMSE'][0] / RMSE_PERS), sw_gain=100 * (1 - pr_['SW'][0] / SUM['Persistence']['SW'][0]), sw_share=100 * swt.mean(), n_switch=int(swt.sum()))
a_q = np.mean([metr(PRED[sd]['Ablation -q15'])['RMSE'] for sd in SEEDS]); NUM['share_q15'] = 100 * (a_q - pr_['RMSE'][0]) / max(SUM['XGBoost']['RMSE'][0] - pr_['RMSE'][0], 1e-9)
NUM['auc_two_stage'] = float(np.mean([PRED[sd]['auc'] for sd in SEEDS]))
# squared-error share of switching hours in validation, base XGBoost
e_va = (y_va - np.clip(PRED[SEEDS[0]]['XGBoost|va'], 0, None)) ** 2 * (sc[va] ** 2); sw_va = np.abs(y_va - D.lag_1.values[va]) > 0.2
NUM['val_sw_rows'] = 100 * sw_va.mean(); NUM['val_sw_err'] = 100 * e_va[sw_va].sum() / e_va.sum()
# per-asset comparison (seed mean), persistence vs XGBoost vs proposed
PA = []
for a in ASSETS:
    m_ = (D.AssetId.values[te] == a)
    r = lambda p: np.sqrt(np.mean((yt[m_] - np.clip(p[m_], 0, None) * sct[m_]) ** 2))
    PA.append(dict(asset=a, pers=r(REF['Persistence'][te]), xgb=np.mean([r(PRED[sd]['XGBoost']) for sd in SEEDS]), prop=np.mean([r(PRED[sd][MODEL_NAME]) for sd in SEEDS])))
PA = pd.DataFrame(PA); PA.to_csv(f'{OUT}/tables/per_asset_rmse.csv', index=False)
NUM['assets_prop_best'] = ', '.join(PA[(PA.prop < PA.pers) & (PA.prop < PA.xgb)].asset.tolist()); NUM['assets_pers_best'] = ', '.join(PA[PA.pers <= PA.prop].asset.tolist()) or 'none'
# ------------------------------------------------------------------ 7. anomaly detection (7 injected faults + unseen mixtures)
say('STEP 7  day-level anomaly detection')
prof = base.pivot_table(index=['AssetId', 'date'], columns='hr', values='load').dropna()
NUM['n_anom_days'] = len(prof)
FAULTS = ['Left ON overnight', 'Missed shift', 'Load spike', 'Schedule shift', 'Base-load increase', 'Early shutdown', 'Partial load']
SHORT = ['ON', 'Miss', 'Spike', 'Shift', 'Base', 'Early', 'Part']
def inject(p, kind, c, rng):
    q = p.copy(); lo, hi = p.min(), p.max(); off, on = p <= lo + 0.2 * (hi - lo), p >= lo + 0.5 * (hi - lo)
    if kind == 'Left ON overnight': q[off] = 0.8 * hi
    elif kind == 'Missed shift': q[on] = lo
    elif kind == 'Load spike': s = rng.integers(6, 20); q[s:s + 2] = 2.5 * np.maximum(p[s:s + 2], 0.1 * c)
    elif kind == 'Schedule shift': q = np.roll(p, 4)
    elif kind == 'Base-load increase': q = p + 0.3 * c
    elif kind == 'Early shutdown':
        idx = np.where(on)[0]
        if len(idx): q[idx[-4:]] = lo
    elif kind == 'Partial load': q[on] = lo + 0.6 * (p[on] - lo)
    return q
def feats_of(X, c):
    mx = X.max(1, keepdims=True); shape = np.divide(X, mx, out=np.zeros_like(X), where=mx > 0); lev = X / c
    return shape, lev, np.c_[shape, lev.mean(1), lev.max(1), lev.min(1), lev.std(1)]
def knn_dist(Z, src, dty, nreal, K, typed=True):
    """sorted K nearest distances to real days (rows 0..nreal-1), excluding the source day itself"""
    out = np.zeros((len(Z), K))
    for t in ([0, 1] if typed else [None]):
        ri = np.arange(nreal) if t is None else np.where(dty[:nreal] == t)[0]
        if len(ri) < K + 2: ri = np.arange(nreal)
        m = np.arange(len(Z)) if t is None else np.where(dty == t)[0]
        if len(m) == 0: continue
        dd, jj = NearestNeighbors(n_neighbors=K + 1).fit(Z[ri]).kneighbors(Z[m]); dd = np.where(ri[jj] == src[m][:, None], np.inf, dd); out[m] = np.sort(dd, 1)[:, :K]
    return out
N_INJ, N_MIX = 40, 40
METHODS = ['DBSCAN (fixed eps)', 'K-means distance', 'DBSCAN density', 'Isolation Forest', 'Proposed kNN']
def anomaly_run(sd):
    rng = np.random.default_rng(sd); rows = []; tm = []
    for a in ASSETS:
        c = scale[a]; P = prof.loc[a]; dates = pd.to_datetime(P.index); P = P.values; n = len(P); wk = np.asarray(dates.dayofweek >= 5).astype(int)
        iP, ik, iw, isrc = [], [], [], []
        def draw(fn, label, N):
            got = tries = 0
            while got < N and tries < 3000:
                tries += 1; i = rng.integers(n); q_ = fn(i)
                if np.abs(q_ - P[i]).mean() > 0.05 * c: iP.append(q_); ik.append(label); iw.append(wk[i]); isrc.append(i); got += 1
        for kind in FAULTS: draw(lambda i, kind=kind: inject(P[i], kind, c, rng), kind, N_INJ)
        for m_ in (2, 3):
            def mix(i, m_=m_):
                ks = rng.choice(len(FAULTS), size=m_, replace=False); return np.mean([inject(P[i], FAULTS[k], c, rng) for k in ks], axis=0)
            draw(mix, f'Mixture of {m_}', N_MIX)
        X = np.vstack([P, np.array(iP)]); dty = np.r_[wk, iw]; src = np.r_[np.arange(n), isrc]; kinds = np.array(['real'] * n + ik)
        shape, lev, F = feats_of(X, c); real = np.arange(n); S = {}
        S['Proposed kNN'] = knn_dist(lev, src, dty, n, 5).mean(1)
        S['DBSCAN density'] = knn_dist(shape, src, dty, n, 5, typed=False)[:, 4]
        S['K-means distance'] = KMeans(n_clusters=2, random_state=sd, n_init=10).fit(F[real]).transform(F).min(1)
        S['Isolation Forest'] = -IsolationForest(n_estimators=300, random_state=sd).fit(F[real]).score_samples(F)
        dbm = DBSCAN(eps=EPS, min_samples=5).fit(shape[real]); cr = shape[real][dbm.core_sample_indices_]
        noise = (NearestNeighbors(n_neighbors=1).fit(cr).kneighbors(shape)[0][:, 0] > EPS) if len(cr) else np.ones(len(X), bool)
        for me in METHODS:
            f = noise if me == 'DBSCAN (fixed eps)' else S[me] > np.quantile(S[me][real], 0.95)
            r = dict(seed=sd, asset=a, method=me, FA=100 * f[real].mean())
            for kd in FAULTS + ['Mixture of 2', 'Mixture of 3']: r[kd] = 100 * f[kinds == kd].mean()
            rows.append(r)
        if sd == SEEDS[0]:
            thr = np.quantile(S['Proposed kNN'][real], 0.95); tm.append(pd.DataFrame({'AssetId': a, 'date': dates, 'score': S['Proposed kNN'][real], 'thr': thr, 'wk': wk}))
    return pd.DataFrame(rows), (pd.concat(tm) if tm else None)
fn_an = f'{OUT}/cache/anomaly.pkl'
if os.path.exists(fn_an): AN, DAYSC = pickle.load(open(fn_an, 'rb'))
else:
    outs = [anomaly_run(sd) for sd in SEEDS]; AN = pd.concat([o[0] for o in outs], ignore_index=True); DAYSC = outs[0][1]; pickle.dump((AN, DAYSC), open(fn_an, 'wb'))
AN['All'] = AN[FAULTS].mean(axis=1); AN.to_csv(f'{OUT}/tables/anomaly_all_rows.csv', index=False)
AG = AN.groupby(['method', 'seed'])[['FA', 'All'] + FAULTS + ['Mixture of 2', 'Mixture of 3']].mean().groupby('method').agg(['mean', 'std']); AG.columns = [f'{a}|{b}' for a, b in AG.columns]
AG = AG.reindex(METHODS); AG.to_csv(f'{OUT}/tables/anomaly_summary.csv')
for me, key in [('Proposed kNN', 'prop'), ('DBSCAN (fixed eps)', 'dbfix'), ('K-means distance', 'kmd'), ('DBSCAN density', 'dens'), ('Isolation Forest', 'iso')]:
    NUM[f'det_{key}'] = AG.loc[me, 'All|mean']; NUM[f'det_{key}_sd'] = AG.loc[me, 'All|std'] if len(SEEDS) > 1 else 0.0; NUM[f'fa_{key}'] = AG.loc[me, 'FA|mean']
    NUM[f'mix2_{key}'] = AG.loc[me, 'Mixture of 2|mean']; NUM[f'mix3_{key}'] = AG.loc[me, 'Mixture of 3|mean']
for kd, sh in zip(FAULTS, SHORT): NUM[f'prop_{sh.lower()}'] = AG.loc['Proposed kNN', f'{kd}|mean']
NUM['mix_prop'] = 0.5 * (NUM['mix2_prop'] + NUM['mix3_prop']); NUM['mix_kmd'] = 0.5 * (NUM['mix2_kmd'] + NUM['mix3_kmd']); NUM['mix_iso'] = 0.5 * (NUM['mix2_iso'] + NUM['mix3_iso'])
NUM['n_alert_days'] = DAYSC.assign(al=DAYSC.score > DAYSC.thr).query('date >= @t_test').groupby('AssetId').al.sum().to_dict()
# ------------------------------------------------------------------ 7b. metering view: daily energy, 15-min peak demand, excess energy of alert days
say('STEP 7b metering view (energy and peak demand)')
q['date'] = q.t.dt.normalize()
pk15 = q.groupby(['AssetId', 'date']).Demand_kW.max()
EN = prof.sum(axis=1).rename('kwh').reset_index(); EN['date'] = pd.to_datetime(EN.date); EN['wk'] = (EN.date.dt.dayofweek >= 5).astype(int)
EN['pk15'] = pk15.reindex(pd.MultiIndex.from_frame(EN[['AssetId', 'date']])).values
EN = EN.merge(DAYSC[['AssetId', 'date', 'score', 'thr']].assign(date=lambda d: pd.to_datetime(d.date)), on=['AssetId', 'date'], how='left'); EN['alert'] = EN.score > EN.thr
EN['active'] = EN.kwh > 0.2 * EN.groupby('AssetId').kwh.transform('max')
ref_ = EN[(EN.date < t_val) & (~EN.alert) & EN.active].groupby(['AssetId', 'wk']).kwh.agg(['median', lambda s: s.quantile(.25), lambda s: s.quantile(.75)]); ref_.columns = ['med', 'q25', 'q75']
EN = EN.join(ref_, on=['AssetId', 'wk']); EN['excess'] = EN.kwh - EN.med
EN.to_csv(f'{OUT}/tables/daily_energy.csv', index=False)
MT = []
for a in ASSETS:
    e = EN[EN.AssetId == a]; tst = e[e.date >= t_test]; al_ = tst[tst.alert]
    MT.append(dict(asset=a, kwh_day=e.kwh.mean(), pk15=e.pk15.mean(), pk15_max=e.pk15.max(), test_days=len(tst), alert_days=len(al_), excess_pos=al_.excess.clip(lower=0).sum(), excess_abs=al_.excess.abs().sum(),
                   test_kwh=tst.kwh.sum(), share=100 * al_.excess.abs().sum() / max(tst.kwh.sum(), 1e-9)))
MT = pd.DataFrame(MT); MT.to_csv(f'{OUT}/tables/metering_view.csv', index=False)
TAB1 = TAB1.merge(MT[['asset', 'kwh_day', 'pk15']], on='asset', how='left'); TAB1.to_csv(f'{OUT}/tables/assets.csv', index=False)
NUM.update(mt_alert_total=int(MT.alert_days.sum()), mt_excess_total=float(MT.excess_abs.sum()), mt_test_kwh=float(MT.test_kwh.sum()))
for r in MT.itertuples(): NUM[f'mt_{r.asset}_alerts'] = int(r.alert_days); NUM[f'mt_{r.asset}_excess'] = float(r.excess_abs); NUM[f'mt_{r.asset}_kwhday'] = float(r.kwh_day)
# energy-weighted detection: share of the energy deviation of injected days that each detector catches is not needed; keep the plain rates
# ablation table (seed means)
ABL = [(f'Single-stage SA-XGBoost (all {len(ENH)} inputs)', SUM['Single-stage SA-XGBoost']['RMSE'][0], SUM['Single-stage SA-XGBoost']['SW'][0])]
for gname, lab in [('q15', 'without 15-min readings'), ('onoff', 'without ON/OFF descriptors'), ('cross', 'without cross-asset inputs')]:
    rr = [metr(PRED[sd]['Ablation -' + gname]) for sd in SEEDS]; ABL.append((lab, np.mean([x['RMSE'] for x in rr]), np.mean([x['SW'] for x in rr])))
ABL.append(('XGBoost, base inputs only', SUM['XGBoost']['RMSE'][0], SUM['XGBoost']['SW'][0]))
ABL = pd.DataFrame(ABL, columns=['variant', 'RMSE', 'SW']); ABL.to_csv(f'{OUT}/tables/ablation.csv', index=False)
for i_, k_ in zip(range(1, 4), ['q15', 'onoff', 'cross']): NUM[f'abl_{k_}'] = float(ABL.RMSE[i_]); NUM[f'abl_{k_}_d'] = 100 * (ABL.RMSE[i_] / ABL.RMSE[0] - 1)
# ------------------------------------------------------------------ 8. figures
say('STEP 8  figures (600 dpi, PDF and PNG)')
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon, Wedge
from matplotlib.lines import Line2D
CLR = dict(blue='#2b6cb0', green='#2f855a', amber='#c27c0e', rose='#b83280', violet='#553c9a', grey='#6b7280', ink='#1f2933', sky='#38a8e0')
def day_stats(a):
    P = prof.loc[a]; wkd = P[pd.to_datetime(P.index).dayofweek < 5]; return P, wkd
Pp, wkp = day_stats('p_c'); act = wkp[wkp.max(axis=1) > 0.4 * wkp.values.max()]; en = act.sum(axis=1)
rep = act.loc[(en - en.median()).abs().idxmin()].values; mid = act[(en >= en.quantile(.35)) & (en <= en.quantile(.65))]; band_lo, band_hi = mid.min().values, mid.max().values
cp = scale['p_c']
# forecast example window (p_c, seed 42): 48 h in the first test weeks with the most switching hours
mp = (D.AssetId.values[te] == 'p_c'); tsp = pd.to_datetime(D.ts.values[te][mp]); idxp = np.where(mp)[0]; best_w = (-1, 0)
for s0 in range(0, max(len(idxp) - 48, 1), 6):
    w_ = idxp[s0:s0 + 48]
    if len(w_) == 48 and (tsp[s0 + 47] - tsp[s0]) == pd.Timedelta(hours=47):
        sc_ = swt[w_].sum() if swt[w_].sum() <= 4 else 0
        if sc_ > best_w[0]: best_w = (sc_, s0)
W_ = idxp[best_w[1]:best_w[1] + 48]; FWIN = D.loc[te].reset_index(drop=True).iloc[W_]; P0 = PRED[SEEDS[0]]
def kw(pv): return np.clip(pv[W_], 0, None) * sct[W_]
FC = {'Recorded': yt[W_], 'Persistence': REF['Persistence'][te][W_] * sct[W_]}
for k in ['XGBoost', 'LSTM', 'Single-stage SA-XGBoost', MODEL_NAME]:
    if k in P0: FC[k] = kw(P0[k])
STY = {'Persistence': (CLR['grey'], ':'), 'XGBoost': (CLR['blue'], '-'), 'LSTM': (CLR['green'], '-.'), 'Single-stage SA-XGBoost': (CLR['amber'], '--'), MODEL_NAME: (CLR['rose'], '-')}
# ---- Fig. 1: architecture
def icon(ax, kind, x, y, s=1.0, c='#334155'):
    if kind == 'press':
        ax.add_patch(Rectangle((x - .5 * s, y + .35 * s), 1.0 * s, .18 * s, fc=c, ec='none')); ax.add_patch(Rectangle((x - .34 * s, y - .1 * s), .68 * s, .4 * s, fc='none', ec=c, lw=1)); ax.add_patch(Rectangle((x - .55 * s, y - .5 * s), 1.1 * s, .15 * s, fc=c, ec='none'))
    if kind == 'fan':
        ax.add_patch(Circle((x, y), .5 * s, fc='none', ec=c, lw=1.2))
        for a_ in (90, 210, 330): ax.add_patch(Wedge((x, y), .42 * s, a_ - 22, a_ + 22, fc=c, ec='none'))
    if kind == 'comp':
        ax.add_patch(FancyBboxPatch((x - .55 * s, y - .28 * s), .95 * s, .4 * s, boxstyle='round,pad=0,rounding_size=0.18', fc=c, ec='none')); ax.add_patch(Circle((x + .28 * s, y + .38 * s), .2 * s, fc='none', ec=c, lw=1))
    if kind == 'conv':
        ax.add_patch(FancyBboxPatch((x - .6 * s, y - .2 * s), 1.2 * s, .3 * s, boxstyle='round,pad=0,rounding_size=0.15', fc='none', ec=c, lw=1.1))
        for dx in (-.42, -.14, .14, .42): ax.add_patch(Circle((x + dx * s, y - .05 * s), .07 * s, fc=c))
        ax.add_patch(Rectangle((x - .2 * s, y + .12 * s), .4 * s, .28 * s, fc=c))
FW, FH = 716, 372
fig = plt.figure(figsize=(7.16, 3.72)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
def card(x, y, w, h, dark, light, num, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=7', fc=light, ec=dark, lw=0.9)); ax.add_patch(FancyBboxPatch((x, y + h - 26), w, 26, boxstyle='round,pad=0,rounding_size=7', fc=dark, ec=dark, lw=0))
    ax.add_patch(Rectangle((x, y + h - 26), w, 10, fc=dark, ec='none')); ax.add_patch(FancyBboxPatch((x + 5, y + h - 22), 18, 18, boxstyle='round,pad=0,rounding_size=4', fc='white', ec='none'))
    ax.text(x + 14, y + h - 13, str(num), ha='center', va='center', fontsize=8.5, fontweight='bold', color=dark); ax.text(x + 28, y + h - 13, title, ha='left', va='center', fontsize=6.3, fontweight='bold', color='white', linespacing=1.05)
def mini(rect, fn):
    a_ = fig.add_axes([rect[0] / FW, rect[1] / FH, rect[2] / FW, rect[3] / FH]); a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False); a_.tick_params(labelsize=4.6, length=1.2, pad=1); fn(a_); return a_
def arrow_r(x, y): ax.add_patch(Polygon([[x, y - 7], [x + 14, y], [x, y + 7]], closed=True, fc='#64748b', ec='none'))
TOPY, TOPH = 250, 118
cards = [(6, CLR['blue'], '#e8f0fa', 1, 'Data acquisition\n& preprocessing'), (201, CLR['green'], '#e9f6ee', 2, 'Load profiling\n(baseline indices)'), (396, CLR['amber'], '#fdf3e3', 3, 'Next-hour load\nforecasting')]
for x, d_, l_, n_, t_ in cards: card(x, TOPY, 170, TOPH, d_, l_, n_, t_)
arrow_r(178, TOPY + 46); arrow_r(373, TOPY + 46)
for k, (kind, lab) in enumerate([('press', 'Press\ndrive'), ('fan', 'Air-handling\nunit'), ('comp', 'Air\ncompressor'), ('conv', 'Material\nhandling')]):
    xx = 6 + 22 + k * 40; icon(ax, kind, xx, TOPY + 64, 9); ax.text(xx, TOPY + 44, lab, ha='center', va='center', fontsize=4.6, color=CLR['ink'], linespacing=1.05)
ax.text(91, TOPY + 28, '+ press, utility, extraction fan, mixer:\n8 sub-metered assets', ha='center', va='center', fontsize=4.9, color=CLR['ink'], linespacing=1.15); ax.text(91, TOPY + 11, '15-min readings, one year, reliable windows', ha='center', fontsize=4.9, style='italic', color=CLR['ink'])
def d2(a_): h = np.arange(24); a_.fill_between(h, band_lo, band_hi, color=CLR['green'], alpha=.18, lw=0); a_.plot(h, rep, color=CLR['green'], lw=1.1); a_.set_xticks([0, 12, 24]); a_.set_ylim(0, None)
mini([224, TOPY + 24, 140, 56], d2); ax.text(286, TOPY + 11, 'LF, PBR, VI  |  K-means / DBSCAN shapes', ha='center', fontsize=4.9, color=CLR['ink'])
def d3(a_):
    h = np.arange(24); a_.plot(h, FC['Recorded'][:24], color='k', lw=1.1); k_ = MODEL_NAME if MODEL_NAME in FC else 'Persistence'; a_.plot(h, FC[k_][:24], color=CLR['amber'], lw=1.0, ls='--'); a_.set_xticks([0, 12, 24])
mini([419, TOPY + 24, 140, 56], d3); ax.text(481, TOPY + 11, f'{MODEL_NAME}: 15-min, ON/OFF, cross-asset inputs', ha='center', fontsize=4.9, color=CLR['ink'])
ax.add_patch(FancyArrowPatch((481, TOPY - 0.5), (481, 240), arrowstyle='-|>', mutation_scale=7, lw=1.2, color='#64748b'))
ax.add_patch(FancyBboxPatch((6, 6), 560, 232, boxstyle='round,pad=0,rounding_size=8', fc='#fff7fb', ec=CLR['rose'], lw=1.1, ls=(0, (4, 3)))); ax.text(16, 226, '4   Anomaly detection phase: seven injected fault types', ha='left', va='center', fontsize=6.6, fontweight='bold', color=CLR['rose'])
WW, WH = 118, 90; XS = [16 + i * 140 for i in range(4)]; YA, YB = 118, 16
pos = [(XS[i], YA) for i in range(4)] + [(XS[i], YB) for i in range(3)]
for i, (x, yv) in enumerate(pos):
    ax.add_patch(FancyBboxPatch((x, yv), WW, WH, boxstyle='round,pad=0,rounding_size=5', fc='white', ec='#efc3da', lw=.8))
    def dfault(a_, i=i): h = np.arange(24); a_.fill_between(h, 0, rep, color='#cbd5e1', alpha=.8, lw=0, step='mid'); a_.step(h, inject(rep, FAULTS[i], cp, np.random.default_rng(3)), where='mid', color=CLR['rose'], lw=1.0); a_.set_xticks([0, 12, 24]); a_.set_ylim(0, None)
    mini([x + 16, yv + 24, WW - 24, 56], dfault); ax.text(x + WW / 2, yv + 10, f'{i + 1}. {FAULTS[i]}', ha='center', va='center', fontsize=5.0, fontweight='bold', color=CLR['ink'])
    if i < 3 or (3 < i < 6): arrow_r(x + WW + 2, yv + WH / 2)
ax.plot([XS[3] + WW / 2, XS[3] + WW / 2], [YA, 111], color='#64748b', lw=1.2); ax.plot([XS[3] + WW / 2, XS[0] + WW / 2], [111, 111], color='#64748b', lw=1.2)
ax.add_patch(FancyArrowPatch((XS[0] + WW / 2, 111), (XS[0] + WW / 2, YB + WH + 1), arrowstyle='-|>', mutation_scale=7, lw=1.2, color='#64748b'))
ax.add_patch(FancyBboxPatch((XS[3], YB), WW, WH, boxstyle='round,pad=0,rounding_size=5', fc='#fdeaf4', ec='#efc3da', lw=.8)); ax.text(XS[3] + WW / 2, YB + 62, 'Day score', ha='center', fontsize=5.6, fontweight='bold', color=CLR['rose'])
ax.text(XS[3] + WW / 2, YB + 34, 'mean distance to the\nK = 5 most similar days\n(same day type);\nalert above the 95th\npercentile of normal days', ha='center', va='center', fontsize=4.8, color=CLR['ink'], linespacing=1.2)
arrow_r(568, 122)
card(576, 6, 134, FH - 12, CLR['violet'], '#eeeafa', 5, 'Digital twin\ninterface')
ax.add_patch(FancyBboxPatch((584, 150), 118, 170, boxstyle='round,pad=0,rounding_size=6', fc='#0b5d9e', ec='none')); ax.add_patch(FancyBboxPatch((588, 154), 110, 162, boxstyle='round,pad=0,rounding_size=4', fc='#e6f4fd', ec='none'))
ax.text(643, 305, 'Asset p_c: running', ha='center', fontsize=5.0, color='#0b3a5b', fontweight='bold')
g_cx, g_cy, g_r = 643, 255, 30
ax.add_patch(Wedge((g_cx, g_cy), g_r, 20, 160, width=7, fc='#bfe3f8', ec='none')); ax.add_patch(Wedge((g_cx, g_cy), g_r, 20, 118, width=7, fc='#38a8e0', ec='none')); ax.add_patch(Wedge((g_cx, g_cy), g_r, 118, 160, width=7, fc='#f2b33d', ec='none'))
ang = math.radians(118); ax.plot([g_cx, g_cx + 25 * math.cos(ang)], [g_cy, g_cy + 25 * math.sin(ang)], color='#0b3a5b', lw=1.3); ax.add_patch(Circle((g_cx, g_cy), 2.6, fc='#0b3a5b', ec='none'))
ax.text(g_cx, g_cy - 12, '14.2 kW', ha='center', fontsize=6.4, fontweight='bold', color='#0b3a5b'); ax.text(g_cx, g_cy - 21, 'measured load', ha='center', fontsize=4.2, color='#0b3a5b')
for k, (lab, val) in enumerate([('Next hour', '13.9 kW'), ('Day score', 'normal')]):
    bx = 592 + k * 54; ax.add_patch(FancyBboxPatch((bx, 166), 50, 34, boxstyle='round,pad=0,rounding_size=3', fc='#ffffff', ec='#9fcbe8', lw=.6)); ax.text(bx + 25, 190, lab, ha='center', fontsize=4.4, color='#527089'); ax.text(bx + 25, 175, val, ha='center', fontsize=5.4, fontweight='bold', color='#0b3a5b')
ax.text(643, 126, 'Replay of the test period:\nstate, forecast, day score,\nkWh, 15-min peak, alert log', ha='center', va='center', fontsize=5.0, color=CLR['ink'], linespacing=1.25, style='italic'); ax.text(643, 80, 'Monitoring level only:\nno control of the assets', ha='center', va='center', fontsize=5.0, color=CLR['ink'], linespacing=1.25)
savefig(fig, 'fig1_framework')
# ---- Fig. 2: injected faults
fig, axs = plt.subplots(2, 4, figsize=(7.16, 2.7), sharey=True); axf = axs.ravel()
for i, a_ in enumerate(axf[:7]):
    qq = inject(rep, FAULTS[i], cp, np.random.default_rng(3)); h = np.arange(24); a_.fill_between(h, 0, rep, step='mid', color='#cbd5e1', alpha=.8, lw=0); a_.step(h, qq, where='mid', color=CLR['rose'], lw=1.2)
    a_.set_title(f'{i + 1}. {FAULTS[i]}', fontsize=7); a_.set_xticks([0, 6, 12, 18]); a_.grid(alpha=.25, lw=.4)
for a_ in axs[1]: a_.set_xlabel('Hour (UTC)', fontsize=7)
for a_ in axs[:, 0]: a_.set_ylabel('Demand (kW)')
axf[7].axis('off'); axf[7].legend([Rectangle((0, 0), 1, 1, fc='#cbd5e1'), Line2D([0], [0], color=CLR['rose'], lw=1.2)], ['Recorded day', 'Day with injected fault'], loc='center', frameon=False, fontsize=7)
fig.tight_layout(w_pad=0.4, h_pad=0.6); savefig(fig, 'fig2_faults')
# ---- Fig. 3: mean weekday profile of the eight assets
PROFILE_ASSETS = ASSETS
H8 = base.rename(columns={'load': 'Demand_kW'}).copy(); H8['ts'] = pd.to_datetime(H8.ts)
fig, ax_ = plt.subplots(figsize=(3.45, 2.5)); h = np.arange(24)
cols_ = [CLR['blue'], CLR['green'], CLR['amber'], CLR['rose'], CLR['violet'], CLR['sky'], CLR['grey'], '#a0522d']; mk_ = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
for n_, ak in enumerate(PROFILE_ASSETS):
    s_ = H8[H8.AssetId == ak]
    if s_.empty: continue
    pv = s_.assign(date=s_.ts.dt.date, hr=s_.ts.dt.hour).pivot_table(index='date', columns='hr', values='Demand_kW').dropna()
    pv = pv[pd.to_datetime(pv.index).dayofweek < 5]; pv = pv[pv.max(axis=1) > 0.4 * pv.values.max()]
    m_ = pv.mean(); ax_.plot(h, m_ / m_.max(), color=cols_[n_], lw=1.1, marker=mk_[n_], ms=2.4, mfc='white', mew=.6, label=ak)
ax_.set_xlabel('Hour of day (UTC)'); ax_.set_ylabel('Normalized demand'); ax_.set_xticks([0, 3, 6, 9, 12, 15, 18, 21]); ax_.set_ylim(0, 1.32); ax_.grid(alpha=.25, lw=.4)
ax_.legend(ncol=4, frameon=False, fontsize=6.2, loc='upper center', columnspacing=0.8, handlelength=1.6)
fig.tight_layout(); savefig(fig, 'fig3_profiles')
# ---- Fig. 4: forecast example
fig, axs = plt.subplots(2, 1, figsize=(7.16, 2.25), sharex=True, gridspec_kw={'height_ratios': [2.0, 1]}); hh = np.arange(48); a_ = axs[0]; a_.plot(hh, FC['Recorded'], color='k', lw=1.7, label='Recorded')
for k, (c_, ls) in STY.items():
    if k in FC: a_.plot(hh, FC[k], color=c_, ls=ls, lw=1.2, label=k)
a_.set_ylabel('Demand (kW)'); a_.legend(ncol=6, frameon=False, fontsize=6.4, loc='upper center', bbox_to_anchor=(0.5, 1.27), columnspacing=0.9, handlelength=1.8); a_.grid(alpha=.25, lw=.4)
b_ = axs[1]
for k in ['Persistence', 'XGBoost', MODEL_NAME]:
    if k in FC: c_, ls = STY[k]; b_.plot(hh, np.abs(FC[k] - FC['Recorded']), color=c_, ls=ls, lw=1.1)
b_.set_ylabel('|Error| (kW)'); b_.set_xlabel(f'Hour from {pd.Timestamp(FWIN.ts.iloc[0]):%d %b %Y %H:%M} (UTC)'); b_.set_xticks(range(0, 48, 6)); b_.grid(alpha=.25, lw=.4); fig.tight_layout(h_pad=0.3); savefig(fig, 'fig4_forecast')
# ---- Fig. 5: steady vs switching
order_ = [k for k in ['Persistence', 'LSTM', 'CNN-LSTM', 'XGBoost', 'Weighted ensemble', 'Single-stage SA-XGBoost', MODEL_NAME] if k in SUM]
fig, ax_ = plt.subplots(figsize=(3.45, 1.95)); yy = np.arange(len(order_))[::-1]
ax_.barh(yy + .19, [SUM[k]['SW'][0] for k in order_], .36, color=CLR['rose'], label='Switching hours'); ax_.barh(yy - .19, [SUM[k]['ST'][0] for k in order_], .36, color=CLR['blue'], label='Steady hours')
for yi, k in zip(yy, order_): ax_.text(SUM[k]['SW'][0] + .12, yi + .19, f"{SUM[k]['SW'][0]:.2f}", va='center', fontsize=6); ax_.text(SUM[k]['ST'][0] + .12, yi - .19, f"{SUM[k]['ST'][0]:.2f}", va='center', fontsize=6)
ax_.set_yticks(yy); ax_.set_yticklabels(order_, fontsize=6.8); ax_.set_xlabel('Test RMSE (kW)'); ax_.set_xlim(0, max(SUM[k]['SW'][0] for k in order_) * 1.25); ax_.legend(frameon=False, fontsize=6.6, loc='lower right'); ax_.grid(axis='x', alpha=.25, lw=.4); fig.tight_layout(); savefig(fig, 'fig5_switching')
# ---- Fig. 6: detection rate by fault type and mixtures
fig, ax_ = plt.subplots(figsize=(7.16, 2.1)); cats = FAULTS + ['Mixture of 2', 'Mixture of 3']; xx = np.arange(len(cats)); mcol = [CLR['grey'], CLR['sky'], CLR['amber'], CLR['green'], CLR['rose']]
for i, me in enumerate(METHODS[1:]): ax_.bar(xx + (i - 1.5) * 0.2, [AG.loc[me, f'{c}|mean'] for c in cats], 0.2, color=mcol[i + 1] if False else [CLR['sky'], CLR['amber'], CLR['green'], CLR['rose']][i], edgecolor='white', lw=.4, label=me)
ax_.set_xticks(xx); ax_.set_xticklabels([c.replace(' ', '\n', 1) for c in cats], fontsize=6.4); ax_.set_ylabel('Detected (%)'); ax_.set_ylim(0, 112); ax_.legend(ncol=4, frameon=False, fontsize=6.4, loc='upper center'); ax_.grid(axis='y', alpha=.25, lw=.4); fig.tight_layout(); savefig(fig, 'fig6_detection')
# ---- Fig. 8: metering view of the alert days (daily energy against the expected range)
MT_sorted = MT.sort_values('alert_days', ascending=False); a_e = MT_sorted.asset.iloc[0]; NUM['ev_asset'] = a_e
e_ = EN[(EN.AssetId == a_e) & (EN.date >= t_test)].sort_values('date')
fig, ax_ = plt.subplots(figsize=(3.45, 1.9)); xd = np.arange(len(e_))
ax_.fill_between(xd, e_.q25.values, e_.q75.values, step='mid', color=CLR['green'], alpha=.16, lw=0, label='Typical range (IQR, same day type)')
ax_.step(xd, e_.med.values, where='mid', color=CLR['green'], lw=1.0, ls='--', label='Typical median')
ax_.bar(xd, e_.kwh.values, 0.8, color=np.where(e_.alert.values, CLR['rose'], '#9fb3c8'), lw=0)
ax_.bar([np.nan], [np.nan], color='#9fb3c8', label='Normal day'); ax_.bar([np.nan], [np.nan], color=CLR['rose'], label='Alert day')
tk = np.arange(0, len(e_), max(len(e_) // 6, 1)); ax_.set_xticks(tk); ax_.set_xticklabels([d.strftime('%d %b') for d in e_.date.iloc[tk]], fontsize=6.4)
ax_.set_ylabel('Daily energy (kWh)'); ax_.grid(axis='y', alpha=.25, lw=.4); ax_.legend(ncol=2, frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.5, 1.30), columnspacing=0.8, handlelength=1.4)
ax_.set_xlim(-0.8, len(e_) - 0.2); fig.tight_layout(); savefig(fig, 'fig8_energy_view')
NUM['ev_alerts'] = int(e_.alert.sum()); NUM['ev_excess'] = float(e_[e_.alert].excess.abs().sum()); NUM['ev_days'] = len(e_)
# ------------------------------------------------------------------ 9. LaTeX tables
say('STEP 9  tables and numbers for the paper')
def pm(v, d=3): return f'{v[0]:.{d}f}$\\pm${v[1]:.{d}f}' if len(SEEDS) > 1 and v[1] > 0 else f'{v[0]:.{d}f}'
def esc(s): return s.replace('_', '\\_')
def wr(name, txt): open(f'{OUT}/latex_parts/{name}', 'w').write(txt)
r_ = []
for rr in TAB1.itertuples():
    r_.append(f"{esc(rr.asset)} & {rr.role} & {rr.days} & {rr.kwh_day:.1f} & {rr.pk15:.1f} & {rr.LF:.3f} & {('--' if rr.PBR_n < 0.05 else f'{rr.PBR:.2f}')} & {rr.VI:.2f} \\\\")
wr('tab_assets.tex', "\\begin{table}[!t]\n\\caption{Selected Assets, Metering Indicators, and Mean Daily Profile Indices}\n\\label{tab:assets}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llcccccc}\n\\toprule\n\\textbf{Asset} & \\textbf{Role} & \\textbf{Days} & \\textbf{kWh/d} & \\textbf{$\\hat{P}_{15}$} & \\textbf{LF} & \\textbf{PBR} & \\textbf{VI} \\\\\n\\midrule\n" + "\n".join(r_) + "\n\\bottomrule\n\\end{tabular}\n\\par\\vspace{2pt}\\begin{minipage}{0.97\\columnwidth}\\footnotesize kWh/d: mean daily energy; $\\hat{P}_{15}$: mean daily peak of the 15-min demand (kW); VI in kW$^2$.\\end{minipage}\n\\end{table}\n")
G = GROUPS; fr = [('Hourly lags', len(G['lags']), '$P_{h-1..h-24}$, $P_{h-168}$, 24-h mean, std'), ('Calendar, asset', len(G['calendar_asset']), 'sin/cos of hour, weekday; weekend; asset'), None,
      ('15-min readings', len(G['q15']), '$q_{h,1..8}$ and slope $\\Delta q_h$'), ('ON/OFF state', len(G['onoff']), '$s_h$, $s_{h-24}$, hours in state, $\\rho(d,t)$, $\\rho(d,t{-}1)$'), ('Cross-asset', len(G['cross']), 'mean demand and ON share of the other assets'), ('Previous day', len(G['prevday']), 'mean demand of the previous day')]
tx = "\n".join('\\midrule' if f is None else f'{f[0]} & {f[1]} & {f[2]} \\\\' for f in fr)
wr('tab_features.tex', f"\\begin{{table}}[!t]\n\\caption{{Input Features of the Proposed {MODEL_NAME}}}\n\\label{{tab:features}}\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{2.4pt}}\n\\begin{{tabular}}{{lcl}}\n\\toprule\n\\textbf{{Group}} & \\textbf{{No.}} & \\textbf{{Content}} \\\\\n\\midrule\n{tx}\n\\midrule\n\\textbf{{Total}} & \\textbf{{{len(ENH)}}} & {len(BASE)} base + {len(ENH) - len(BASE)} added \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
CLt = CL.copy()
# Table III: four rows (K-means all days; DBSCAN kept days; K-means with the same k on the same days; K-means k=2 on the same days)
cr = [f"{m} & {int(d_):,} & {c_} & {s_:.4f} & {b_:.4f} \\\\".replace(',', '{,}') for m, d_, c_, s_, b_ in CLt.itertuples(index=False)]
cr = [c.replace('eps=', '$\\varepsilon{=}$').replace('k=', '$k{=}$').replace(')*', ')$^{*}$') for c in cr]
wr('tab_cluster.tex', "\\begin{table}[!t]\n\\caption{Clustering Quality on All Days and on the Same Days}\n\\label{tab:cluster}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Method} & \\textbf{Days} & \\textbf{Clust.} & \\textbf{Silh.} $\\uparrow$ & \\textbf{DBI} $\\downarrow$ \\\\\n\\midrule\n" + "\n".join(cr) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
FORD = [('K-means profile', 'K-means profile$^\\dagger$', 'Baseline-1', 'No'), ('ARIMA', 'ARIMA', 'Baseline-2', 'Yes'), ('VAR', 'VAR', 'Baseline-2', 'Yes'), ('Random forest', 'Random forest', 'Baseline-2', 'Yes'), ('XGBoost', 'XGBoost', 'Baseline-2', 'Yes'),
        ('LSTM', 'LSTM', 'Baseline-2', 'Yes'), ('TCN', 'TCN', 'Baseline-2', 'Yes'), None, ('CNN-LSTM', 'CNN-LSTM', 'Hybrid', 'Yes'), ('LSTM + XGBoost', 'LSTM + XGBoost', 'Hybrid', 'Yes'), ('DBSCAN + XGBoost', 'DBSCAN + XGBoost', 'Hybrid', 'Yes'),
        ('Weighted ensemble', 'Weighted ensemble', 'Hybrid', 'Yes'), ('Single-stage SA-XGBoost', 'Single-stage SA-XGBoost', 'Variant', 'Yes'), None, (MODEL_NAME, f'\\textbf{{Proposed ({MODEL_NAME})}}', '\\textbf{Proposed}', 'Yes')]
lines = []
for f in FORD:
    if f is None:
        if lines and lines[-1] != '\\midrule': lines.append('\\midrule')
        continue
    k, lab, ty, fs = f
    if k not in SUM: continue
    s = SUM[k]; lines.append(f"{lab} & {ty} & {fs} & {pm(s['RMSE'])} & {s['MAE'][0]:.3f} & {s['MAPE'][0]:.2f} & {s['R2'][0]:.3f} & {s['Skill'][0]:.1f} & {pm(s['SW'])} \\\\")
while lines and lines[-1] == '\\midrule': lines.pop()
NUM['n_compared'] = sum(1 for f in FORD if f and f[0] in SUM and f[0] not in ('K-means profile', MODEL_NAME))
wr('tab_forecast.tex', "\\begin{table*}[!t]\n\\caption{Next-Hour Forecast Accuracy on the Test Period (" + f"{len(ASSETS)} Assets, {len(SEEDS)} Seeds, Mean $\\pm$ SD; Errors in kW)" + "}\n\\label{tab:forecast}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3.4pt}\n\\renewcommand{\\arraystretch}{0.95}\n\\begin{tabular}{llcccccccc}\n\\toprule\n\\textbf{Algorithm} & \\textbf{Type} & \\begin{tabular}[c]{@{}c@{}}\\textbf{Forecasting}\\\\\\textbf{in source}\\end{tabular} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{MAPE (\\%)} & $\\mathbf{R^2}$ & \\begin{tabular}[c]{@{}c@{}}\\textbf{Skill}\\\\\\textbf{(\\%)}\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}\\textbf{Switching-}\\\\\\textbf{hour RMSE}\\end{tabular} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\par\\vspace{2pt}\\begin{minipage}{0.97\\textwidth}\\footnotesize Baseline-1 is the profiling method, which has no forecaster of its own ($^\\dagger$its K-means day profiles serve as the forecast); Baseline-2 holds the methods of the same authors, implemented here on the same data. Skill is measured against persistence " + f"({RMSE_PERS:.3f}~kW overall, {SUM['Persistence']['SW'][0]:.3f}~kW in switching hours). ARIMA, VAR, and Baseline-1 have no random seed." + "\\end{minipage}\n\\end{table*}\n")
lab_m = {'DBSCAN (fixed eps)': 'DBSCAN ($\\varepsilon$ fixed)', 'K-means distance': 'K-means distance', 'DBSCAN density': 'DBSCAN density', 'Isolation Forest': 'Isolation Forest', 'Proposed kNN': '\\textbf{Proposed (context-aware kNN)}'}
al = []
for me in METHODS:
    v = lambda c, d=1: f"{AG.loc[me, c + '|mean']:.{d}f}"
    allc = f"{AG.loc[me, 'All|mean']:.1f}" + (f"$\\pm${AG.loc[me, 'All|std']:.1f}" if len(SEEDS) > 1 else '')
    al.append(f"{lab_m[me]} & {v('FA')} & {allc} & " + " & ".join(v(f) for f in FAULTS) + f" & {v('Mixture of 2')} & {v('Mixture of 3')} \\\\")
    if me == 'DBSCAN (fixed eps)': al.append('\\midrule')
wr('tab_anomaly.tex', "\\begin{table*}[!t]\n\\caption{Detection Rate (\\%) of Injected Faults and of Unseen Fault Mixtures (" + f"{len(SEEDS)} Seeds)" + "}\n\\label{tab:anomaly}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{lccccccccccc}\n\\toprule\n\\textbf{Method} & \\textbf{FA} & \\textbf{All} & \\textbf{ON} & \\textbf{Miss} & \\textbf{Spike} & \\textbf{Shift} & \\textbf{Base} & \\textbf{Early} & \\textbf{Part} & \\textbf{Mix-2} & \\textbf{Mix-3} \\\\\n\\midrule\n" + "\n".join(al) + "\n\\bottomrule\n\\end{tabular}\n\\par\\vspace{2pt}\\begin{minipage}{0.97\\textwidth}\\footnotesize FA: share of recorded days flagged. All: mean of the seven fault types. ON to Part: equipment left on at night, skipped shift, load spike, shifted schedule, raised base load, early stop, and partial load. Mix-2, Mix-3: unseen averages of two or three faults.\\end{minipage}\n\\end{table*}\n")
ab_l = [f"{r.variant} & {r.RMSE:.3f} & {100 * (r.RMSE / ABL.RMSE[0] - 1):+.1f} & {r.SW:.3f} \\\\" for r in ABL.itertuples()]
wr('tab_ablation.tex', "\\begin{table}[!t]\n\\caption{Contribution of the Added Input Groups (Mean of " + f"{len(SEEDS)} Seeds)" + "}\n\\label{tab:ablation}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccc}\n\\toprule\n\\textbf{Variant} & \\textbf{RMSE} & \\textbf{$\\Delta$ (\\%)} & \\textbf{Switch. RMSE} \\\\\n\\midrule\n" + "\n".join(ab_l).replace('+0.0', '0.0') + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
mt_l = [f"{esc(r.asset)} & {r.test_days} & {r.alert_days} & {r.test_kwh:,.0f} & {r.excess_abs:,.1f} & {r.share:.2f} \\\\".replace(',', '{,}') for r in MT.itertuples()]
wr('tab_metering.tex', "\\begin{table}[!t]\n\\caption{Metering View of the Alerts Raised in the Test Period}\n\\label{tab:metering}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Asset} & \\textbf{Days} & \\textbf{Alerts} & \\textbf{Energy (kWh)} & \\textbf{$|\\Delta E|$ (kWh)} & \\textbf{Share (\\%)} \\\\\n\\midrule\n" + "\n".join(mt_l) + "\n\\bottomrule\n\\end{tabular}\n\\par\\vspace{2pt}\\begin{minipage}{0.97\\columnwidth}\\footnotesize Days: test-period days with all 24 hours; Alerts: days flagged by the proposed detector; $|\\Delta E|$: summed absolute difference between the energy of each alert day and the median energy of normal running days of the same type; Share: $|\\Delta E|$ relative to the test-period energy.\\end{minipage}\n\\end{table}\n")
def tmac(k, v):
    name = 'Num' + ''.join(ch for ch in re.sub(r'[0-9]', lambda m_: ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'][int(m_.group())], ''.join(w.capitalize() for w in k.split('_'))) if ch.isalpha())
    if isinstance(v, (int, np.integer)): s = f'{int(v):,}'.replace(',', '{,}')
    elif isinstance(v, (float, np.floating)):
        d = 4 if any(t_ in k for t_ in ['sil', 'dbi']) else (3 if any(t_ in k for t_ in ['rmse', 'mae', 'sw_prop', 'sw_pers', 'st_', 'r2', 'auc', 'twin_score', 'twin_thr']) else (2 if (k.startswith('dm_') or k.startswith('twin_')) else 1)); s = f'{float(v):.{d}f}' if abs(v) >= 1e-3 or v == 0 else f'{float(v):.1e}'
    else: s = str(v).replace('_', '\\_')
    return name, s
def write_numbers():
    mac = {}
    for k, v in NUM.items():
        if isinstance(v, dict):
            for a_, n_ in v.items(): mac[tmac(f'alerts_{a_}', int(n_))[0]] = tmac(f'alerts_{a_}', int(n_))[1]
        else: n_, s_ = tmac(k, v); mac[n_] = s_
    wr('numbers.tex', '% generated by the pipeline cell\n' + '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(mac.items())) + '\n')
    json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in NUM.items()}, open(f'{OUT}/tables/numbers.json', 'w'), indent=1, default=str)
write_numbers()
# ------------------------------------------------------------------ 10. claims used in the text
CK = []
def chk(txt, cond): CK.append((txt, bool(cond)))
others = [k for k in SUM if k != MODEL_NAME]
chk(f'{MODEL_NAME} has the lowest mean RMSE of all compared models', all(SUM[MODEL_NAME]['RMSE'][0] < SUM[k]['RMSE'][0] for k in others))
chk(f'{MODEL_NAME} is the most accurate model for every seed', all(metr(PRED[sd][MODEL_NAME])['RMSE'] < min(metr(PRED[sd][k])['RMSE'] for k in PRED[sd] if '|' not in k and k not in ('auc', 'ens_w', MODEL_NAME) and not k.startswith('Ablation')) for sd in SEEDS))
chk('Diebold-Mariano p < 0.01 for every seed (primary test)', NUM['dm_pmax'] < 0.01)
chk('best baseline algorithm on validation data is XGBoost', NUM['best_baseline'] == 'XGBoost')
chk('persistence is hard to beat in steady hours (steady RMSE of all models within 0.5 kW of persistence)', all(abs(SUM[k]['ST'][0] - SUM['Persistence']['ST'][0]) < 0.5 for k in others if k in ('XGBoost', 'LSTM', MODEL_NAME)))
chk('the proposed detector has the highest All rate among the budgeted detectors', AG.loc['Proposed kNN', 'All|mean'] == AG.loc[METHODS[1:], 'All|mean'].max())
chk('DBSCAN with fixed eps flags more than 50% of normal days', AG.loc['DBSCAN (fixed eps)', 'FA|mean'] > 50)
chk('proposed detector finds more than 80% of load spikes', AG.loc['Proposed kNN', 'Load spike|mean'] > 80)
chk('missed shifts are detected in at most 20% of cases by every budgeted method', all(AG.loc[m, 'Missed shift|mean'] <= 20 for m in METHODS[1:]))
chk('proposed detector has the highest Mixture-3 rate among budgeted detectors', AG.loc['Proposed kNN', 'Mixture of 3|mean'] == AG.loc[METHODS[1:], 'Mixture of 3|mean'].max())
chk('DBSCAN reaches a higher silhouette than K-means on all days only because it discards days (noise share > 50%)', NUM['noise_share'] > 50)
chk('K-means on the kept days scores higher than DBSCAN', NUM['sil_k2'] > NUM['sil_db'])
open(f'{OUT}/tables/claims_check.txt', 'w').write('Statements used in the paper text and whether this run supports them (edit the text where False):\n\n' + '\n'.join(f"[{'OK   ' if c else 'FALSE'}] {t}" for t, c in CK) + f"\n\nassets where the proposed model beats persistence and XGBoost: {NUM['assets_prop_best']}\nassets where persistence is at least as good: {NUM['assets_pers_best']}\n")
for t, c in CK: print(('  OK   ' if c else '  FALSE'), t)
# ------------------------------------------------------------------ 11. digital twin (meter style)
say('STEP 10  digital twin')
TW = {'assets': {}, 'note': f'Offline replay of the test period (from {NUM["test_start"]}), which was not used for training. Next-hour forecast: {MODEL_NAME}' + (' (PREVIEW: scikit-learn gradient boosting stand-in, not XGBoost).' if STANDIN else '.') + f' Day score: mean distance to the 5 most similar days of the same type; alert above the 95th percentile of the scores of normal days. {len(ASSETS)} metered assets. Display only; no control of the assets.'}
Dte = D.loc[te].reset_index(drop=True); Dte['fc'] = np.clip(PRED[SEEDS[0]][MODEL_NAME], 0, None) * sct; Dte['kw'] = yt; Dte['pe'] = Dte.pers * sct
for a in ASSETS:
    sub = Dte[Dte.AssetId == a].set_index('ts').sort_index(); rng_t = pd.date_range(sub.index.min(), sub.index.max(), freq='h'); sub = sub.reindex(rng_t); r2 = lambda s: [None if pd.isna(v) else round(float(v), 3) for v in s]
    P_, _ = day_stats(a); dts = pd.to_datetime(P_.index); wk_ = (dts.dayofweek >= 5).astype(int); trn = dts < t_val; typ = {}
    for t in (0, 1):
        Xs = P_.values[trn & (wk_ == t)] if (trn & (wk_ == t)).sum() else P_.values; Xa = Xs[Xs.max(axis=1) > 0.4 * P_.values.max()]; typ[str(t)] = np.round(np.median(Xa if len(Xa) else Xs, axis=0), 3).tolist()
    ds = DAYSC[DAYSC.AssetId == a]; thr = float(ds.thr.iloc[0]); dl = didx[didx.AssetId == a].set_index('date'); days = {}
    for dte, s_, w_ in zip(ds.date, ds.score, ds.wk):
        if dte < rng_t[0].normalize() - pd.Timedelta(days=2): continue
        rec = {'s': round(float(s_), 4), 'w': int(w_)}
        if dte in dl.index: rw = dl.loc[dte]; rec.update(LF=None if pd.isna(rw.LF) else round(float(rw.LF), 3), PBR=None if pd.isna(rw.PBR) else round(float(rw.PBR), 2), VI=None if pd.isna(rw.VI) else round(float(rw.VI), 2))
        ee = EN[(EN.AssetId == a) & (EN.date == pd.Timestamp(dte))]
        if len(ee): rec.update(E=round(float(ee.kwh.iloc[0]), 1), X=None if pd.isna(ee.med.iloc[0]) else round(float(ee.med.iloc[0]), 1), Pk=None if pd.isna(ee.pk15.iloc[0]) else round(float(ee.pk15.iloc[0]), 2))
        days[dte.strftime('%Y-%m-%d')] = rec
    t_str = [t.strftime('%Y-%m-%d %H') for t in rng_t]; ok = sub.kw.notna()
    TW['assets'][a] = {'role': ROLE[a], 'max': round(float(scale[a]), 2), 'onKw': round(THETA * float(scale[a]), 2), 't': t_str, 'a': r2(sub.kw), 'f': r2(sub.fc), 'p': r2(sub.pe), 'thr': round(thr, 4), 'days': days, 'typical': typ,
        'rmse_f': float(np.sqrt(((sub.loc[ok, 'kw'] - sub.loc[ok, 'fc']) ** 2).mean())), 'rmse_p': float(np.sqrt(((sub.loc[ok, 'kw'] - sub.loc[ok, 'pe']) ** 2).mean())), 'start': 60}
bad = [(a, d_) for a in ASSETS for d_, r in TW['assets'][a]['days'].items() if r['s'] > TW['assets'][a]['thr'] and d_ >= TW['assets'][a]['t'][0][:10]]
TW['start'] = (bad[0][0], bad[0][1]) if bad else (ASSETS[0], TW['assets'][ASSETS[0]]['t'][100][:10])
S0 = TW['assets'][TW['start'][0]]; k0 = next(k for k, t in enumerate(S0['t']) if t.startswith(TW['start'][1])); i0 = min(max(k0 + 14, 48), len(S0['t']) - 2); r0 = S0['days'].get(TW['start'][1], {'s': float('nan')})
NUM.update(twin_asset=TW['start'][0], twin_when=S0['t'][i0] + ':00', twin_meas=S0['a'][i0] or 0.0, twin_fc=S0['f'][i0 + 1] or 0.0, twin_score=r0['s'], twin_thr=S0['thr']); write_numbers()
TWIN_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Asset-Level Smart-Metering Digital Twin</title>
<style>:root{color-scheme:light;--bg:#e9f3fb;--card:#fff;--ink:#12324a;--mut:#527089;--line:#bcd7ea;--sky:#38a8e0;--deep:#0b5d9e;--ok:#1f9d6b;--warn:#e0a21b;--crit:#d6452f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 "Segoe UI",Arial,sans-serif}.wrap{display:grid;grid-template-columns:210px 1fr;min-height:100vh}
aside{background:var(--deep);color:#e6f4fd;padding:14px 10px;display:flex;flex-direction:column;gap:5px}aside h1{margin:0 0 2px;font:600 15px Georgia,serif}aside p{margin:0 0 8px;color:#b6d8f0;font-size:11.5px}
.asset{display:grid;grid-template-columns:12px 1fr auto;gap:8px;align-items:center;border:0;background:transparent;color:inherit;font:inherit;text-align:left;padding:7px 8px;border-radius:7px;cursor:pointer}.asset:hover{background:#1f78b8}.asset[aria-selected=true]{background:#2a8ad0;box-shadow:inset 3px 0 #ffd166}
.dot{width:10px;height:10px;border-radius:50%;background:#7aa9c8}.dot.on{background:#5fe0a8}.dot.al{background:#ff8a72;box-shadow:0 0 0 3px rgba(255,138,114,.35)}.asset small{display:block;color:#b6d8f0;font-size:11px}.asset .kw{font:12px ui-monospace,Consolas,monospace}
main{padding:16px 20px;display:grid;gap:12px;align-content:start}header{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:6px}header h2{margin:0;font:600 21px Georgia,serif}header span{color:var(--mut);font-size:12.5px}
.ctrl{display:flex;gap:10px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}.ctrl button{font:600 12.5px inherit;border:1px solid var(--line);background:#fff;border-radius:5px;padding:6px 11px;cursor:pointer}.ctrl button.p{background:var(--sky);color:#fff;border-color:var(--sky)}.ctrl input{flex:1;accent-color:var(--sky)}.when{font:13px ui-monospace,Consolas,monospace;min-width:160px}
.meters{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.card h3{margin:0 0 2px;font:600 11px sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.pill{display:inline-block;font:600 13px sans-serif;padding:5px 14px;border-radius:999px;margin-top:8px}.pill.ok{background:#d9f3e9;color:var(--ok)}.pill.off{background:#e4edf4;color:var(--mut)}.pill.crit{background:#fde3dd;color:var(--crit)}.pill.warn{background:#fcefcf;color:#9a6b00}
.sub{color:var(--mut);font-size:12px}.g2{display:grid;grid-template-columns:1.7fr 1fr;gap:12px}.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}svg{width:100%;height:auto;display:block}svg text{font:10.5px "Segoe UI",sans-serif;fill:#527089}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:4px}.legend i{display:inline-block;width:14px;border-top:2px solid;margin-right:5px;vertical-align:middle}
dl{display:grid;grid-template-columns:auto 1fr;gap:5px 12px;margin:6px 0 0}dt{color:var(--mut);font-size:12.5px}dd{margin:0;text-align:right;font-family:ui-monospace,Consolas,monospace}.log{list-style:none;margin:6px 0 0;padding:0;display:grid;gap:5px;max-height:150px;overflow:auto}.log li{display:flex;justify-content:space-between;background:#eef6fc;border-radius:5px;padding:5px 8px;font-size:12.5px;cursor:pointer}
.foot{color:var(--mut);font-size:11.5px;max-width:110ch}@media(max-width:1000px){.wrap{grid-template-columns:1fr}.meters{grid-template-columns:1fr 1fr}.g2,.g3{grid-template-columns:1fr}aside{flex-direction:row;flex-wrap:wrap}}</style></head><body><div class="wrap">
<aside><h1>Smart-Metering Digital Twin</h1><p>Asset-level meters, replay of the test period</p><div id="list"></div></aside><main>
<header><div><h2 id="title">-</h2><span id="sub"></span></div><span>Monitoring level only</span></header>
<div class="ctrl"><button class="p" id="play">Play</button><button id="b1">-1 h</button><button id="f1">+1 h</button><input id="sl" type="range" min="0" max="1" value="0"><div class="when" id="when"></div></div>
<div class="meters"><div class="card"><h3>Measured load</h3><svg id="g1" viewBox="0 0 220 150"></svg></div><div class="card"><h3>Next-hour forecast</h3><svg id="g2" viewBox="0 0 220 150"></svg></div><div class="card"><h3>Day deviation score</h3><svg id="g3" viewBox="0 0 220 150"></svg></div>
<div class="card"><h3>Asset state</h3><div id="pill" class="pill ok">RUNNING</div><div class="sub" id="stsub"></div><dl><dt>Load factor</dt><dd id="iLF">-</dd><dt>Peak-to-base</dt><dd id="iPBR">-</dd><dt>Variability</dt><dd id="iVI">-</dd><dt>Energy today</dt><dd id="iE">-</dd><dt>Typical day</dt><dd id="iX">-</dd><dt>Peak 15-min</dt><dd id="iPk">-</dd></dl></div></div>
<div class="g2"><div class="card"><h3>Last 48 hours and next-hour forecast</h3><svg id="trend" viewBox="0 0 720 230"></svg><div class="legend"><span><i style="border-color:#12324a"></i>Measured</span><span><i style="border-color:#38a8e0"></i>Forecast</span><span><i style="border-color:#9aa9b6;border-top-style:dashed"></i>Persistence</span></div></div>
<div class="card"><h3>Today vs typical day</h3><svg id="dayc" viewBox="0 0 420 230"></svg><div class="legend"><span><i style="border-color:#12324a"></i>Today so far</span><span><i style="border-color:#e0a21b;border-top-style:dashed"></i>Typical</span></div></div></div>
<div class="g3"><div class="card"><h3>Forecast accuracy (test period)</h3><dl><dt>RMSE, forecast</dt><dd id="aF">-</dd><dt>RMSE, persistence</dt><dd id="aP">-</dd><dt>Skill</dt><dd id="aS">-</dd><dt>Rolling 24 h MAE</dt><dd id="aR">-</dd></dl></div>
<div class="card" style="grid-column:span 2"><h3>Anomaly alerts <span id="ac" class="sub"></span></h3><ul class="log" id="log"></ul></div></div><p class="foot" id="foot"></p></main></div>
<script>const TWIN=/*__DATA__*/null;(function(){const $=id=>document.getElementById(id),fmt=(x,d=2)=>x==null||isNaN(x)?"-":Number(x).toFixed(d),A=Object.keys(TWIN.assets);let cur=TWIN.start[0],i=0,timer=null;
const TI={};A.forEach(a=>{const m={};TWIN.assets[a].t.forEach((x,k)=>m[x]=k);TI[a]=m});
function pol(cx,cy,r,deg){const t=(deg-90)*Math.PI/180;return[cx+r*Math.cos(t),cy+r*Math.sin(t)]}
function arc(cx,cy,r,a0,a1){const[p,q]=pol(cx,cy,r,a0),[u,v]=pol(cx,cy,r,a1);return`M${p} ${q}A${r} ${r} 0 ${a1-a0>180?1:0} 1 ${u} ${v}`}
function gauge(id,val,max,zones,label,ticks,mark){const cx=110,cy=100,r=78,a0=-120,a1=120,fr=Math.max(0,Math.min(1,val/max));let g="";zones.forEach(z=>{g+=`<path d="${arc(cx,cy,r,a0+(a1-a0)*z[0],a0+(a1-a0)*z[1])}" stroke="${z[2]}" stroke-width="12" fill="none"/>`});
 for(let k=0;k<=4;k++){const f=k/4,a=a0+(a1-a0)*f,[x1,y1]=pol(cx,cy,r-9,a),[x2,y2]=pol(cx,cy,r+8,a),[tx,ty]=pol(cx,cy,r+19,a);g+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#527089" stroke-width="1"/><text x="${tx}" y="${ty+3}" text-anchor="middle">${(max*f).toFixed(max<10?1:0)}</text>`}
 if(mark!=null){const a=a0+(a1-a0)*Math.min(1,mark/max),[x1,y1]=pol(cx,cy,r-12,a),[x2,y2]=pol(cx,cy,r+10,a);g+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#7b8a97" stroke-width="2.4" stroke-dasharray="2 2"/>`}
 const[nx,ny]=pol(cx,cy,r-16,a0+(a1-a0)*fr);g+=`<line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#0b3a5b" stroke-width="3" stroke-linecap="round"/><circle cx="${cx}" cy="${cy}" r="6" fill="#0b3a5b"/><text x="${cx}" y="${cy+30}" text-anchor="middle" style="font-size:17px;font-weight:700;fill:#12324a">${label}</text>`;$(id).innerHTML=g}
A.forEach(a=>{const b=document.createElement("button");b.className="asset";b.id="a-"+a;b.innerHTML=`<span class="dot" id="d-${a}"></span><span><b>${a}</b><small>${TWIN.assets[a].role}</small></span><span class="kw" id="w-${a}">-</span>`;b.onclick=()=>{const t=TWIN.assets[cur].t[i];cur=a;const k=TI[a][t];if(k!=null)i=k;setup()};$("list").appendChild(b)});
function setup(){const S=TWIN.assets[cur];$("sl").max=S.t.length-2;i=Math.min(Math.max(i,48),S.t.length-2);const k0=S.t.findIndex(t=>t.startsWith(TWIN.start[1]));if(i==48&&cur==TWIN.start[0]&&k0>=0)i=Math.min(k0+14,S.t.length-2);$("sl").value=i;
 const days=Object.entries(S.days).filter(([d,r])=>r.s>S.thr&&d>=S.t[0].slice(0,10)).sort((x,y)=>y[1].s-x[1].s);$("ac").textContent=`(${days.length} days above threshold)`;const L=$("log");L.innerHTML="";days.slice(0,40).forEach(([d,r])=>{const li=document.createElement("li");li.innerHTML=`<span>${d} (${r.w?"weekend":"weekday"})</span><span>${r.E!=null&&r.X!=null?(r.E-r.X>=0?"+":"")+fmt(r.E-r.X,1)+" kWh":""}</span><b>${fmt(r.s/S.thr,2)}x</b>`;li.onclick=()=>{const k=S.t.findIndex(t=>t.startsWith(d));if(k>=0){i=Math.min(k+14,S.t.length-2);render()}};L.appendChild(li)});
 if(!days.length)L.innerHTML="<li><span>No anomalous days in the test period</span></li>";$("aF").textContent=fmt(S.rmse_f,3)+" kW";$("aP").textContent=fmt(S.rmse_p,3)+" kW";$("aS").textContent=fmt(100*(1-S.rmse_f/S.rmse_p),1)+" %";$("foot").textContent=TWIN.note;render()}
function line(xs,ys,c,w,d){let p="",pen=false;xs.forEach((x,k)=>{const y=ys[k];if(y==null){pen=false;return}p+=(pen?"L":"M")+x.toFixed(1)+" "+y.toFixed(1);pen=true});return`<path d="${p}" fill="none" stroke="${c}" stroke-width="${w}" ${d?'stroke-dasharray="'+d+'"':''}/>`}
const nm=m=>{const p=Math.pow(10,Math.floor(Math.log10(m||1)));for(const s of[1,2,2.5,5,10])if(s*p>=m)return s*p;return 10*p};
function render(){const S=TWIN.assets[cur],t=S.t[i],day=t.slice(0,10);$("title").textContent=cur;$("sub").textContent=S.role+" - maximum "+S.max+" kW";$("when").textContent=t+":00 UTC";$("sl").value=i;
 const on=S.a[i]>S.onKw,r=S.days[day],bad=r&&r.s>S.thr;A.forEach(a=>{const B=TWIN.assets[a],k=TI[a][t],e=$("d-"+a);$("a-"+a).setAttribute("aria-selected",a==cur);if(k==null||B.a[k]==null){e.className="dot";$("w-"+a).textContent="n/a";return}const q=B.days[day],bb=q&&q.s>B.thr;e.className="dot"+(bb?" al":B.a[k]>B.onKw?" on":"");$("w-"+a).textContent=fmt(B.a[k],1)+" kW"});
 $("pill").className="pill "+(bad?"crit":on?"ok":"off");$("pill").textContent=bad?"ANOMALY DAY":on?"RUNNING":"IDLE";let n=0;for(let j=i;j>=0&&(S.a[j]>S.onKw)===on&&n<999;j--)n++;$("stsub").textContent=`${on?"Running":"Idle"} for ${n>=999?"999+":n} h`;
 gauge("g1",S.a[i],S.max,[[0,S.onKw/S.max,"#bcd7ea"],[S.onKw/S.max,1,"#38a8e0"]],fmt(S.a[i])+" kW",4,S.onKw);gauge("g2",S.f[i+1],S.max,[[0,S.onKw/S.max,"#bcd7ea"],[S.onKw/S.max,1,"#0b7bd0"]],fmt(S.f[i+1])+" kW",4,S.p[i+1]);
 const sm=2*S.thr;gauge("g3",r?r.s:0,sm,[[0,.4,"#8fd9b8"],[.4,.5,"#f2d27a"],[.5,1,"#f08a73"]],r?fmt(r.s,3):"n/a",4,S.thr);
 if(r){$("iLF").textContent=fmt(r.LF,3);$("iPBR").textContent=r.PBR==null?"n/a":fmt(r.PBR,2);$("iVI").textContent=fmt(r.VI,2);$("iX").textContent=r.X==null?"-":fmt(r.X,1)+" kWh";$("iPk").textContent=r.Pk==null?"-":fmt(r.Pk,1)+" kW"}else{["iLF","iPBR","iVI","iX","iPk"].forEach(k=>$(k).textContent="-")}
 let e=0,k=i;while(k>=0&&S.t[k].startsWith(day)){e+=S.a[k]||0;k--}$("iE").textContent=fmt(e,1)+" kWh";let s=0,c=0;for(let j=Math.max(0,i-23);j<=i;j++)if(S.a[j]!=null&&S.f[j]!=null){s+=Math.abs(S.a[j]-S.f[j]);c++}$("aR").textContent=fmt(c?s/c:null,3)+" kW";
 // trend
 {const W=720,H=230,x0=42,x1=W-12,y0=10,y1=H-28,N=48,a0=Math.max(0,i-N+1),idx=[];for(let q=a0;q<=i+1;q++)idx.push(q);const v=idx.flatMap(q=>[S.a[q],S.f[q],S.p[q]]).filter(z=>z!=null),mx=nm(Math.max(...v,S.onKw)*1.06),X=q=>x0+(x1-x0)*(q-a0)/N,Y=z=>y1-(y1-y0)*z/mx;let g="";
  for(let q=0;q<=4;q++){const z=mx*q/4;g+=`<line x1="${x0}" x2="${x1}" y1="${Y(z)}" y2="${Y(z)}" stroke="#d6e6f2"/><text x="${x0-6}" y="${Y(z)+3}" text-anchor="end">${+z.toFixed(1)}</text>`}
  idx.forEach(q=>{if(+S.t[q].slice(11,13)%12==0)g+=`<text x="${X(q)}" y="${H-9}" text-anchor="middle">${S.t[q].slice(5,10)} ${S.t[q].slice(11,13)}h</text>`});g+=`<line x1="${X(i)}" x2="${X(i)}" y1="${y0}" y2="${y1}" stroke="#7b8a97" stroke-dasharray="3 3"/><text x="${X(i)-4}" y="${y0+10}" text-anchor="end">now</text>`;
  const xs=idx.map(X),pa=idx.filter(q=>q<=i);g+=line(xs,idx.map(q=>S.p[q]==null?null:Y(S.p[q])),"#9aa9b6",1.3,"5 4")+line(xs,idx.map(q=>S.f[q]==null?null:Y(S.f[q])),"#38a8e0",2)+line(pa.map(X),pa.map(q=>S.a[q]==null?null:Y(S.a[q])),"#12324a",2.2);if(S.f[i+1]!=null)g+=`<circle cx="${X(i+1)}" cy="${Y(S.f[i+1])}" r="5" fill="#38a8e0" stroke="#fff" stroke-width="2"/>`;$("trend").innerHTML=g}
 // day curve
 {const W=420,H=230,x0=40,x1=W-10,y0=10,y1=H-28,wk=r?r.w:0,typ=S.typical[String(wk)],td=new Array(24).fill(null);for(let q=i;q>=0&&S.t[q].startsWith(day);q--)td[+S.t[q].slice(11,13)]=S.a[q];const mx=nm(Math.max(...typ,...td.filter(z=>z!=null),.1)*1.08),X=h=>x0+(x1-x0)*h/23,Y=z=>y1-(y1-y0)*z/mx;let g="";
  for(let q=0;q<=4;q++){const z=mx*q/4;g+=`<line x1="${x0}" x2="${x1}" y1="${Y(z)}" y2="${Y(z)}" stroke="#d6e6f2"/><text x="${x0-6}" y="${Y(z)+3}" text-anchor="end">${+z.toFixed(1)}</text>`}[0,6,12,18,23].forEach(h=>g+=`<text x="${X(h)}" y="${H-9}" text-anchor="middle">${String(h).padStart(2,"0")}h</text>`);
  const hs=[...Array(24).keys()];g+=line(hs.map(X),typ.map(Y),"#e0a21b",1.6,"6 4")+line(hs.map(X),td.map(z=>z==null?null:Y(z)),"#12324a",2.2);$("dayc").innerHTML=g}}
$("play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("play").textContent="Play";return}$("play").textContent="Pause";timer=setInterval(()=>{const S=TWIN.assets[cur];i=i>=S.t.length-3?48:i+1;render()},350)};$("f1").onclick=()=>{i=Math.min(i+1,TWIN.assets[cur].t.length-2);render()};$("b1").onclick=()=>{i=Math.max(i-1,0);render()};$("sl").oninput=e=>{i=+e.target.value;render()};setup()})();</script></body></html>'''
open(f'{OUT}/twin/digital_twin.html', 'w', encoding='utf-8').write(TWIN_HTML.replace('/*__DATA__*/null', json.dumps(TW, separators=(',', ':'))))
try:
    try: from playwright.sync_api import sync_playwright
    except ImportError:          # Colab: install a headless browser once to take the twin screenshot (about 1-2 min)
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'playwright'], check=False)
        subprocess.run([sys.executable, '-m', 'playwright', 'install', '--with-deps', 'chromium'], check=False, capture_output=True)
        from playwright.sync_api import sync_playwright
    with sync_playwright() as p_:
        b_ = p_.chromium.launch(); pg_ = b_.new_page(viewport={'width': 1400, 'height': 900}, device_scale_factor=2); pg_.goto('file://' + os.path.abspath(f'{OUT}/twin/digital_twin.html')); pg_.wait_for_timeout(700)
        pg_.screenshot(path=f'{OUT}/figures/fig7_twin.png', clip={'x': 0, 'y': 0, 'width': 1400, 'height': 720}); b_.close()
    say('twin screenshot saved (figures/fig7_twin.png)')
except Exception as ex:
    say('twin screenshot skipped (%s). Open twin/digital_twin.html in Chrome and save a screenshot as figures/fig7_twin.png' % type(ex).__name__)
# ------------------------------------------------------------------ 12. package
say('STEP 11  package')
try:
    src = globals().get('_ih', [''])[-1]
    if 'COMPLETE PIPELINE IN ONE CELL' in src: open(f'{OUT}/code/pipeline_cell.py', 'w').write(src)
except Exception: pass
with zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root_, _, fl_ in os.walk(OUT):
        if os.sep + 'cache' in root_ + os.sep or root_.endswith('cache'): continue
        for f_ in fl_: zf.write(os.path.join(root_, f_), os.path.relpath(os.path.join(root_, f_), OUT))
say(f'DONE. {len(glob.glob(OUT + "/figures/*.png"))} figures, tables in {OUT}/tables and {OUT}/latex_parts. results.zip written.')
try:
    from google.colab import files; files.download('results.zip')
except Exception: print('Download results.zip from the file panel.')