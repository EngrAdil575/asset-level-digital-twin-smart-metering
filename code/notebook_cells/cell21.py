# ============================================================
# CELL 21: Five-seed evaluation (mean +- SD) and Diebold-Mariano test
# Re-trains every learned forecasting model with 5 seeds, scores all of
# them on the same test hours, and tests whether the proposed Enhanced
# XGBoost is significantly better than the best Baseline-2 algorithm.
# Requires: Cells 1-20 have been run once in this Colab session; this cell
# re-uses their code from the notebook history. Runtime: about 5-7 min per seed.
# ============================================================
import numpy as np, pandas as pd, random, os
from scipy import stats
from IPython.display import display
import tensorflow as tf

def cell_src(tag):                          # latest executed code of a cell, from the notebook history
    found = [c for c in In if tag in c]
    if not found:
        raise RuntimeError(f"Run the cell containing '{tag}' first.")
    return found[-1]
c20 = cell_src('# CELL 20:')
CELL_CODE = [cell_src(f'# CELL {n}:') for n in [8, 9, 10, 11, 12, 13, 15, 16]] + \
            [c20[c20.index('# ---------- 4) Random Forest'):c20.index('# ---------- Final table')]]

SEEDS = [42, 7, 123, 2024, 31]
FIXED = ['K-means profile', 'ARIMA', 'VAR', 'Persistence', 'Seasonal Naive']   # deterministic
fixed_preds = {k: np.asarray(preds_te[k]).copy() for k in FIXED if k in preds_te}
all_runs = {}
for sd in SEEDS:
    SEED = sd
    random.seed(sd); np.random.seed(sd); tf.random.set_seed(sd); os.environ['PYTHONHASHSEED'] = str(sd)
    for cell in CELL_CODE:
        exec(cell)
    preds_te.update(fixed_preds)
    all_runs[sd] = {k: np.asarray(v, dtype=float).copy() for k, v in preds_te.items()}
    print(f"seed {sd} done")

sc = data.loc[te, 'AssetId'].map(scale).values
y = y_te * sc
def scores(pr, y, sw):
    return dict(RMSE=np.sqrt(np.mean((y - pr) ** 2)), MAE=np.mean(np.abs(y - pr)),
                SW=np.sqrt(np.mean((y[sw] - pr[sw]) ** 2)), ST=np.sqrt(np.mean((y[~sw] - pr[~sw]) ** 2)))
rows = []
for sd, P in all_runs.items():
    pers = P['Persistence'] * sc; sw = np.abs(y - pers) > 0.2 * sc
    for k, v in P.items():
        rows.append(dict(model=k, seed=sd, **scores(np.clip(np.nan_to_num(v), 0, None) * sc, y, sw)))
R = pd.DataFrame(rows)
summary = R.groupby('model')[['RMSE', 'MAE', 'SW', 'ST']].agg(['mean', 'std']).round(3).sort_values(('RMSE', 'mean'))
print("TABLE: five-seed mean and standard deviation (kW)"); display(summary)

def dm_test(e1, e2):             # H0: equal squared-error loss (one-step forecasts)
    d = e1 ** 2 - e2 ** 2
    stat = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / len(d))
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))
b2 = ['ARIMA', 'VAR', 'Random Forest', 'XGBoost', 'LSTM', 'TCN']
best = summary.loc[b2, ('RMSE', 'mean')].idxmin()
out = []
for sd, P in all_runs.items():
    st, p = dm_test(y - P[best] * sc, y - P['Enhanced XGBoost'] * sc)
    out.append(dict(seed=sd, baseline=best, DM_statistic=round(st, 2), p_value=p))
print(f"Diebold-Mariano test: Enhanced XGBoost vs {best}"); display(pd.DataFrame(out))
