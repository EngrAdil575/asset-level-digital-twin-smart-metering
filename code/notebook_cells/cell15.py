# ============================================================
# CELL 15: Persistence baseline + Enhanced features + Enhanced XGBoost
# Why: most forecast error (about 2/3) comes from the few hours when a
# machine switches ON or OFF. The hourly average hides the first sign of
# a switch, so this cell adds:
#   (a) 15-minute features  -- the last eight 15-min readings before the
#       forecast hour (the freshest signal of a ramp up/down)
#   (b) ON/OFF state features -- is the machine on, for how many hours,
#       and how often it is usually on at this hour & weekday (learned
#       from TRAINING data only)
#   (c) Cross-asset features -- the other 3 assets' latest load
#       (e.g. production running -> compressor running)
#   (d) Yesterday's average load (catches shutdown / holiday periods)
# Also adds the Persistence baseline (next hour = last hour).
# Requires: 'reliable_df' (Cell 1) and Cells 6-8.
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from IPython.display import display

ON_THRESHOLD = 0.2   # machine counted as ON when load > 20% of its training max

# ---- Persistence baseline: next hour = last hour ----
preds_va['Persistence'] = data.loc[va, 'lag_1'].values
preds_te['Persistence'] = data.loc[te, 'lag_1'].values

# ---- (a) 15-minute lag features ----
q = reliable_df[(reliable_df['Phase'] == 'ALL') &
                (reliable_df['AssetId'].isin(SELECTED_ASSETS))][['AssetId', 'window_start_utc', POWER_COL]].copy()
q['window_start_utc'] = pd.to_datetime(q['window_start_utc'])
q = q.groupby(['AssetId', 'window_start_utc'])[POWER_COL].mean().reset_index()
q['q_s'] = q[POWER_COL] / q['AssetId'].map(scale)
q_lookup = q.set_index(['AssetId', 'window_start_utc'])['q_s']

N_Q = 8   # last 8 x 15 min = last 2 hours
for k in range(1, N_Q + 1):
    key = pd.MultiIndex.from_arrays([data['AssetId'], data['ts'] - pd.Timedelta(minutes=15 * k)])
    data[f'q_lag_{k}'] = q_lookup.reindex(key).values
data['q_slope'] = data['q_lag_1'] - data['q_lag_4']        # rising or falling in the last hour

# ---- (b) ON/OFF state features ----
data['on_lag1'] = (data['lag_1'] > ON_THRESHOLD).astype(int)
data['on_lag24'] = (data['lag_24'] > ON_THRESHOLD).astype(int)
# hours the machine has been in its current state (up to 24)
on_mat = (data[[f'lag_{k}' for k in range(1, 25)]].values > ON_THRESHOLD)
same = on_mat == on_mat[:, [0]]
data['hours_in_state'] = np.where(same.all(axis=1), 24, np.argmin(same, axis=1))
# usual ON probability at this asset / weekday / hour (from TRAINING rows only)
data['dow'], data['hr'] = data['ts'].dt.dayofweek, data['ts'].dt.hour
on_prob = (data.loc[tr].assign(on=lambda d: (d['load_s'] > ON_THRESHOLD).astype(int))
           .groupby(['AssetId', 'dow', 'hr'])['on'].mean().rename('usual_on_prob'))
data = data.drop(columns=['usual_on_prob'], errors='ignore').merge(
    on_prob.reset_index(), on=['AssetId', 'dow', 'hr'], how='left')
# same weekday & hour: will the machine usually switch on/off in the NEXT hour?
data['usual_on_prob_prev'] = data.groupby('AssetId')['usual_on_prob'].shift(1)

# ---- (c) Cross-asset features: other assets' latest hourly and 15-min load ----
wide_h = data.pivot_table(index='ts', columns='AssetId', values='lag_1')
wide_q = data.pivot_table(index='ts', columns='AssetId', values='q_lag_1')
for a in SELECTED_ASSETS:
    data[f'x_lag1_{a}'] = data['ts'].map(wide_h[a]) if a in wide_h else np.nan
    data[f'x_q1_{a}'] = data['ts'].map(wide_q[a]) if a in wide_q else np.nan

# ---- (d) Yesterday's average load (shutdown / holiday indicator) ----
day_mean = data.groupby(['AssetId', 'date'])['load_s'].mean().rename('prev_day_mean').reset_index()
day_mean['date'] = day_mean['date'] + pd.Timedelta(days=1)
data = data.drop(columns=['prev_day_mean'], errors='ignore').merge(day_mean, on=['AssetId', 'date'], how='left')

# the merges above keep row order, but re-create the masks to be safe
tr = data['ts'] < t_val
va = (data['ts'] >= t_val) & (data['ts'] < t_test)
te = data['ts'] >= t_test

enh_feats = (base_feats
             + [f'q_lag_{k}' for k in range(1, N_Q + 1)] + ['q_slope']
             + ['on_lag1', 'on_lag24', 'hours_in_state', 'usual_on_prob', 'usual_on_prob_prev']
             + [f'x_lag1_{a}' for a in SELECTED_ASSETS] + [f'x_q1_{a}' for a in SELECTED_ASSETS]
             + ['prev_day_mean'])

# ---- Enhanced XGBoost (same settings as Cell 8, more features) ----
enh_xgb_model = fit_xgb(enh_feats)
preds_va['Enhanced XGBoost'] = enh_xgb_model.predict(data.loc[va, enh_feats])
preds_te['Enhanced XGBoost'] = enh_xgb_model.predict(data.loc[te, enh_feats])
print(f"Enhanced XGBoost trained with {len(enh_feats)} features "
      f"(was {len(base_feats)} in Cell 8).")

# Top 15 most useful features (shows which new features matter)
imp = pd.Series(enh_xgb_model.feature_importances_, index=enh_feats).sort_values(ascending=False)
print("\nTop 15 features by importance:")
display(imp.head(15).round(4).to_frame('importance'))
print("\nCell 15 complete.")
