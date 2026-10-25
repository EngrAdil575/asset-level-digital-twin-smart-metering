# ============================================================
# CELL 6: Forecasting setup -- continuous hourly series, train/val/test
# split and all input features (lags, calendar, DBSCAN day features).
# Predicts the NEXT HOUR's Demand_kW for each of the 4 selected assets.
# Requires: 'hourly', 'pivot_norm', 'X', 'SELECTED_ASSETS', 'POWER_COL',
# 'chosen_eps', 'min_samples', 'daily_indices_ctx' (run Cells 1-5 first).
# ============================================================

import os, random, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import tensorflow as tf

warnings.filterwarnings('ignore')
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- Settings ----------------
N_LAGS      = 24      # hours of history each model sees
TRAIN_FRAC  = 0.70    # first 70% of the timeline -> training
VAL_FRAC    = 0.10    # next 10% -> validation (early stopping, ensemble weights)
                      # last 20% -> test (all methods scored here)
MAPE_MIN_KW = 0.5     # MAPE only on hours with actual load >= 0.5 kW (avoids /0)
PLOT_ASSET  = 'p_c'   # asset shown in the comparison graph
EPOCHS      = 40

# ============================================================
# STEP 1: Build a continuous hourly series per asset
# (missing/unreliable hours stay NaN so lags never jump across gaps)
# ============================================================
h = hourly.copy()
h['ts'] = pd.to_datetime(h['date']) + pd.to_timedelta(h['hour'], unit='h')

frames = []
for a in SELECTED_ASSETS:
    s = h[h['AssetId'] == a].set_index('ts')[POWER_COL].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
    frames.append(pd.DataFrame({'AssetId': a, 'ts': s.index, 'load': s.values}))
data = pd.concat(frames, ignore_index=True)

# Chronological split points (same calendar cut-off for every asset)
all_ts = np.sort(data.loc[data['load'].notna(), 'ts'].unique())
t_val  = pd.Timestamp(all_ts[int(len(all_ts) * TRAIN_FRAC)])
t_test = pd.Timestamp(all_ts[int(len(all_ts) * (TRAIN_FRAC + VAL_FRAC))])
print(f"Train: before {t_val:%Y-%m-%d} | Validation: {t_val:%Y-%m-%d} to "
      f"{t_test:%Y-%m-%d} | Test: from {t_test:%Y-%m-%d}")

# Per-asset scaling using TRAINING data only (no information from the future)
scale = (data[data['ts'] < t_val].groupby('AssetId')['load'].max()
         .replace(0, 1.0))
data['load_s'] = data['load'] / data['AssetId'].map(scale)

# ============================================================
# STEP 2: Features
# ============================================================
g = data.groupby('AssetId')['load_s']
for k in range(1, N_LAGS + 1):
    data[f'lag_{k}'] = g.shift(k)
data['lag_168']   = g.shift(168)                      # same hour last week
data['roll_mean_24'] = g.transform(lambda x: x.shift(1).rolling(24, min_periods=18).mean())
data['roll_std_24']  = g.transform(lambda x: x.shift(1).rolling(24, min_periods=18).std())

hr, dow = data['ts'].dt.hour, data['ts'].dt.dayofweek
data['hour_sin'], data['hour_cos'] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
data['dow_sin'],  data['dow_cos']  = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
data['is_weekend'] = (dow >= 5).astype(int)
for a in SELECTED_ASSETS:
    data[f'asset_{a}'] = (data['AssetId'] == a).astype(int)

# ---- DBSCAN day-level features (for the DBSCAN + XGBoost hybrid) ----
# DBSCAN is RE-FITTED on training days only (same eps/min_samples as Cell 3),
# so no test-period load shape influences the model. For every day we take
# its distance to the nearest "normal" (core) training day -- exactly the
# is_anomalous_day() rule from Cell 5 -- and give the model YESTERDAY's value,
# which is known at forecast time.
day_idx = pivot_norm.index.to_frame(index=False)[['AssetId', 'date']]
day_idx['date'] = pd.to_datetime(day_idx['date'])
train_days = (day_idx['date'] < t_val.normalize()).values
db_train = DBSCAN(eps=chosen_eps, min_samples=min_samples).fit(X[train_days])
core_pts = X[train_days][db_train.core_sample_indices_]
if len(core_pts) == 0:                      # safety fallback
    core_pts = X[train_days]
day_idx['day_dist'] = pairwise_distances(X, core_pts).min(axis=1)
day_idx['day_anom'] = (day_idx['day_dist'] > chosen_eps).astype(int)

ctx = daily_indices_ctx.copy()
ctx['date'] = pd.to_datetime(ctx['date'])
day_feat = day_idx.merge(ctx, on=['AssetId', 'date'], how='left')
day_feat['date'] = day_feat['date'] + pd.Timedelta(days=1)       # -> "yesterday"
day_feat = day_feat.rename(columns={'day_dist': 'prev_day_dist', 'day_anom': 'prev_day_anom',
                                    'LF': 'prev_LF', 'PBR': 'prev_PBR', 'VI': 'prev_VI'})
data['date'] = data['ts'].dt.normalize()
data = data.merge(day_feat, on=['AssetId', 'date'], how='left')
# VI is in kW^2 -> scale it per asset like the load
data['prev_VI'] = data['prev_VI'] / data['AssetId'].map(scale) ** 2

# Keep only rows where the target and the full 24-h history exist
lag_cols = [f'lag_{k}' for k in range(1, N_LAGS + 1)]
data = data.dropna(subset=['load_s'] + lag_cols).reset_index(drop=True)

asset_cols = [f'asset_{a}' for a in SELECTED_ASSETS]
cal_cols   = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
base_feats = lag_cols + ['lag_168', 'roll_mean_24', 'roll_std_24'] + cal_cols + asset_cols
dbscan_feats = base_feats + ['prev_day_dist', 'prev_day_anom', 'prev_LF', 'prev_PBR', 'prev_VI']

tr = data['ts'] < t_val
va = (data['ts'] >= t_val) & (data['ts'] < t_test)
te = data['ts'] >= t_test
y_tr, y_va, y_te = data.loc[tr, 'load_s'].values, data.loc[va, 'load_s'].values, data.loc[te, 'load_s'].values
print(f"Samples -> train: {tr.sum():,} | validation: {va.sum():,} | test: {te.sum():,}")

preds_va, preds_te = {}, {}   # scaled predictions of every method (filled by Cells 7-12)
print("\nCell 6 complete. Features and train/validation/test split are ready.")
