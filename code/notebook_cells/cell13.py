# ============================================================
# CELL 13: Model 7 -- Weighted Ensemble (HYBRID) + accuracy tables
# Ensemble weights = 1 / (validation MSE), chosen on VALIDATION data only,
# then every method is scored on the SAME test hours (in kW).
# Requires: Cells 6-12.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import display

ens_members = ['XGBoost', 'LSTM', 'CNN-LSTM', 'DBSCAN + XGBoost']
w = np.array([1 / mean_squared_error(y_va, preds_va[m]) for m in ens_members])
w = w / w.sum()
preds_te['Weighted Ensemble'] = sum(wi * preds_te[m] for wi, m in zip(w, ens_members))
print("Ensemble weights: " + ", ".join(f"{m} = {wi:.2f}" for m, wi in zip(ens_members, w)))

# ============================================================
# STEP 3: Evaluate every method on the SAME test hours (in kW)
# ============================================================
test_df = data.loc[te, ['AssetId', 'ts']].reset_index(drop=True)
asset_scale = test_df['AssetId'].map(scale).values
test_df['Actual'] = y_te * asset_scale

METHODS = ['Seasonal Naive', 'XGBoost', 'LSTM', 'CNN-LSTM',
           'LSTM + XGBoost', 'DBSCAN + XGBoost', 'Weighted Ensemble']
METHOD_TYPE = {'Seasonal Naive': 'Baseline', 'XGBoost': 'Single', 'LSTM': 'Single',
               'CNN-LSTM': 'Hybrid', 'LSTM + XGBoost': 'Hybrid',
               'DBSCAN + XGBoost': 'Hybrid', 'Weighted Ensemble': 'Hybrid'}
for m in METHODS:
    test_df[m] = np.clip(preds_te[m], 0, None) * asset_scale

def metrics(y, p):
    mk = y >= MAPE_MIN_KW
    return {'RMSE (kW)': np.sqrt(mean_squared_error(y, p)),
            'MAE (kW)': mean_absolute_error(y, p),
            'MAPE (%)': 100 * np.mean(np.abs((y[mk] - p[mk]) / y[mk])),
            'R²': r2_score(y, p)}

# Overall table (all 4 assets together). RMSE/MAE in kW are dominated by the
# larger assets, so the per-asset table below is shown as well.
metrics_df = pd.DataFrame({m: metrics(test_df['Actual'].values, test_df[m].values)
                           for m in METHODS}).T
metrics_df.insert(0, 'Type', [METHOD_TYPE[m] for m in METHODS])
metrics_df = metrics_df.sort_values('RMSE (kW)')
print("\n" + "=" * 70)
print("TABLE: Next-hour forecast accuracy on the test period (all 4 assets)")
print("=" * 70)
display(metrics_df.round(4))

per_asset = pd.DataFrame({a: {m: np.sqrt(mean_squared_error(
                              test_df.loc[test_df.AssetId == a, 'Actual'],
                              test_df.loc[test_df.AssetId == a, m])) for m in METHODS}
                          for a in SELECTED_ASSETS})
print("\nTABLE: Test RMSE (kW) per asset (lowest per column = best for that asset)")
display(per_asset.round(4))

best = metrics_df.index[0]
naive_rmse = metrics_df.loc['Seasonal Naive', 'RMSE (kW)']
print(f"\nBest method overall: {best} (RMSE {metrics_df.loc[best, 'RMSE (kW)']:.4f} kW, "
      f"{100*(1 - metrics_df.loc[best, 'RMSE (kW)']/naive_rmse):.1f}% lower than the "
      f"Seasonal Naive baseline).")
