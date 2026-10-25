# ============================================================
# CELL 20: Baseline-paper algorithms in the forecasting comparison
# Adds, on the SAME chronological split and test hours as Cells 6-17:
#   * K-means profile forecast -- the algorithm of the baseline paper
#     (Fesseha et al., IEEE Access 2025): K-means (k chosen by silhouette,
#     fitted on TRAINING days) groups daily load shapes; the forecast for
#     hour h is the centroid value of yesterday's cluster at hour h,
#     scaled by yesterday's peak.
#   * ARIMA, VAR, Random Forest and TCN -- the models compared by the
#     same authors in their forecasting study (Fesseha et al., IEEE
#     Access 2026). XGBoost and LSTM from that study are already in
#     Cells 8-9.
# All forecasts are one-step-ahead (next hour) using only past data.
# Requires: Cells 6-17 (data, tr, va, te, y_*, scale, preds_te, fit_nn
# inputs, final_metrics). Takes about 3-5 minutes.
# ============================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display
warnings.filterwarnings('ignore')

# ---------- 1) K-means profile forecast (baseline paper) ----------
dd = data[['AssetId', 'ts', 'load_s']].copy()
dd['day'] = dd['ts'].dt.normalize(); dd['hr'] = dd['ts'].dt.hour
shape_tbl = dd.pivot_table(index=['AssetId', 'day'], columns='hr', values='load_s')
shape_tbl = shape_tbl[shape_tbl.notna().sum(axis=1) >= 20].ffill(axis=1).bfill(axis=1)
peak = shape_tbl.max(axis=1).replace(0, np.nan)
norm = shape_tbl.div(peak, axis=0).fillna(0)
train_mask = norm.index.get_level_values('day') < t_val.normalize()
best_k, best_s = 2, -1
for k in range(2, 8):
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm[train_mask])
    s_ = silhouette_score(norm[train_mask], lab)
    if s_ > best_s: best_k, best_s = k, s_
km_base = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(norm[train_mask])
cent = km_base.cluster_centers_
day_cluster = pd.Series(km_base.predict(norm), index=norm.index)
prev = pd.DataFrame({'cl': day_cluster, 'pk': peak}).reset_index()
prev['day'] = prev['day'] + pd.Timedelta(days=1)                # yesterday -> today
m = data[['AssetId', 'ts']].assign(day=data['ts'].dt.normalize(), hr=data['ts'].dt.hour)
m = m.merge(prev, on=['AssetId', 'day'], how='left')
km_pred = np.where(m['cl'].notna(),
                   [cent[int(c)][h] * p if pd.notna(c) else np.nan for c, h, p in zip(m['cl'], m['hr'], m['pk'])],
                   np.nan)
km_pred = pd.Series(km_pred).fillna(data['lag_24'].reset_index(drop=True)).values  # no yesterday -> same hour yesterday
preds_te['K-means profile'] = km_pred[te.values]
print(f"K-means profile forecast ready (k = {best_k}, fitted on training days).")

# ---------- 2) ARIMA (per asset, seasonal, one-step-ahead) ----------
arima_pred = pd.Series(np.nan, index=data.index)
for a in SELECTED_ASSETS:
    rows = data['AssetId'] == a
    full = data.loc[rows].set_index('ts')['load_s']
    idx = pd.date_range(full.index.min(), full.index.max(), freq='h')
    series = full.reindex(idx)                                   # gaps stay NaN (Kalman filter handles them)
    y_train = series[series.index < t_val]
    mod = SARIMAX(y_train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 24), trend='c',
                  enforce_stationarity=True, enforce_invertibility=True)
    res = mod.fit(disp=False, maxiter=200)
    res_all = res.apply(series)                                  # same parameters, filtered over full series
    one_step = res_all.get_prediction().predicted_mean           # one-step-ahead predictions
    arima_pred[rows] = one_step.reindex(data.loc[rows, 'ts']).values
arima_pred = arima_pred.fillna(data['lag_1'])                   # safety: any unfilled hour -> last hour
preds_te['ARIMA'] = arima_pred[te].values
print("ARIMA ready: SARIMA(2,0,1)(1,0,0)24 per asset, trained on the training period.")

# ---------- 3) VAR (all four assets jointly) ----------
wide = data.pivot_table(index='ts', columns='AssetId', values='load_s')[SELECTED_ASSETS]
wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='h'))
wide_in = wide.ffill().fillna(0)                                 # inputs only (lags); targets stay real values
train_block = wide[(wide.index < t_val)].dropna()
var_res = VAR(train_block).fit(maxlags=24, ic='aic')
p = var_res.k_ar
coefs, const = var_res.coefs, var_res.intercept                  # coefs: (p, n, n)
arr = wide_in.values
var_hat = np.full(arr.shape, np.nan)
for t_i in range(p, len(arr)):
    yhat = const.copy()
    for L in range(1, p + 1):
        yhat = yhat + coefs[L - 1] @ arr[t_i - L]
    var_hat[t_i] = yhat
var_df = pd.DataFrame(var_hat, index=wide.index, columns=SELECTED_ASSETS)
var_long = var_df.stack().rename('var').reset_index().rename(columns={'level_0': 'ts', 'level_1': 'AssetId'})
var_long.columns = ['ts', 'AssetId', 'var']
vm = data[['AssetId', 'ts']].merge(var_long, on=['AssetId', 'ts'], how='left')
preds_te['VAR'] = vm.loc[te.values, 'var'].values
print(f"VAR ready (lag order {p} chosen by AIC).")

# ---------- 4) Random Forest (same base features as XGBoost) ----------
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                           random_state=SEED, n_jobs=-1)
rf.fit(data.loc[tr, base_feats].fillna(-1), y_tr)
preds_te['Random Forest'] = rf.predict(data.loc[te, base_feats].fillna(-1))
print("Random Forest ready.")

# ---------- 5) TCN (dilated causal convolutions, same inputs as LSTM) ----------
def build_tcn():
    seq_in = keras.Input(shape=(N_LAGS, 1)); aux_in = keras.Input(shape=(len(aux_cols),))
    x = seq_in
    for d in [1, 2, 4, 8]:
        h = layers.Conv1D(32, 3, padding='causal', dilation_rate=d, activation='relu')(x)
        h = layers.Conv1D(32, 3, padding='causal', dilation_rate=d, activation='relu')(h)
        x = layers.Add()([layers.Conv1D(32, 1)(x) if x.shape[-1] != 32 else x, h])   # residual block
    x = layers.Lambda(lambda z: z[:, -1, :])(x)
    x = layers.Concatenate()([x, aux_in]); x = layers.Dense(32, activation='relu')(x)
    mdl = keras.Model([seq_in, aux_in], layers.Dense(1)(x))
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return mdl
tf.random.set_seed(SEED)
tcn = build_tcn()
tcn.fit(Xtr_nn, y_tr, validation_data=(Xva_nn, y_va), epochs=EPOCHS, batch_size=128, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
preds_te['TCN'] = tcn.predict(Xte_nn, verbose=0).ravel()
print("TCN ready.")

# ---------- Final table with every method on the same test hours ----------
SOURCE = {'Seasonal Naive': 'Reference', 'Persistence': 'Reference',
          'K-means profile': 'Baseline paper [1]',
          'ARIMA': 'Same authors [2]', 'VAR': 'Same authors [2]', 'Random Forest': 'Same authors [2]',
          'TCN': 'Same authors [2]', 'XGBoost': 'Same authors [2]', 'LSTM': 'Same authors [2]',
          'CNN-LSTM': 'Hybrid', 'LSTM + XGBoost': 'Hybrid', 'DBSCAN + XGBoost': 'Hybrid',
          'Weighted Ensemble': 'Hybrid', 'Two-Stage Hybrid': 'Proposed', 'Enhanced XGBoost': 'Proposed'}
fin = data.loc[te, ['AssetId', 'ts']].reset_index(drop=True)
sc = fin['AssetId'].map(scale).values
y = y_te * sc
y_pers = np.asarray(preds_te['Persistence']) * sc
switching = np.abs(y - y_pers) > 0.2 * sc
rm = lambda a, b: np.sqrt(mean_squared_error(a, b))
rows_out = {}
for mth in SOURCE:
    if mth not in preds_te: continue
    pr = np.clip(np.nan_to_num(np.asarray(preds_te[mth], dtype=float), nan=0.0), 0, None) * sc
    mk = y >= MAPE_MIN_KW
    rows_out[mth] = {'Source': SOURCE[mth], 'RMSE (kW)': rm(y, pr), 'MAE (kW)': mean_absolute_error(y, pr),
                     'MAPE (%)': 100 * np.mean(np.abs((y[mk] - pr[mk]) / y[mk])), 'R2': r2_score(y, pr),
                     'Skill vs Persistence (%)': 100 * (1 - rm(y, pr) / rm(y, y_pers)),
                     'Switching-hour RMSE (kW)': rm(y[switching], pr[switching]),
                     'Steady-hour RMSE (kW)': rm(y[~switching], pr[~switching])}
full_table = pd.DataFrame(rows_out).T.sort_values('RMSE (kW)', ascending=False)
nc = full_table.columns.drop('Source'); full_table[nc] = full_table[nc].astype(float).round(3)
print("\n" + "=" * 70)
print("TABLE: All forecasting methods incl. baseline-paper algorithms (test period)")
print("=" * 70)
display(full_table)
full_table.to_csv('forecast_with_baselines.csv')
print("\nCell 20 complete. Saved: forecast_with_baselines.csv")
