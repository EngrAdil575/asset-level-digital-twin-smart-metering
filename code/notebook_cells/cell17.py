# ============================================================
# CELL 17: Final comparison of ALL methods (old + new)
# Adds two paper-quality metrics on top of RMSE / MAE / MAPE / R²:
#   * Skill vs Persistence (%) = how much better than "next hour = last
#     hour". Positive = better, negative = worse.
#   * Switching-hour RMSE = error only in the hours when load changes by
#     more than 20% of the asset's max (machine switching ON/OFF), where
#     most of the error comes from.
# Also draws an updated black-and-white friendly comparison graph.
# Requires: Cells 6-16.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import display

ALL_METHODS = ['Seasonal Naive', 'Persistence', 'XGBoost', 'LSTM', 'CNN-LSTM',
               'LSTM + XGBoost', 'DBSCAN + XGBoost', 'Weighted Ensemble',
               'Enhanced XGBoost', 'Two-Stage Hybrid']
ALL_TYPE = {'Seasonal Naive': 'Baseline', 'Persistence': 'Baseline',
            'XGBoost': 'Single', 'LSTM': 'Single', 'CNN-LSTM': 'Hybrid',
            'LSTM + XGBoost': 'Hybrid', 'DBSCAN + XGBoost': 'Hybrid',
            'Weighted Ensemble': 'Hybrid', 'Enhanced XGBoost': 'Proposed',
            'Two-Stage Hybrid': 'Proposed'}
ALL_METHODS = [m for m in ALL_METHODS if m in preds_te]    # skip any cell not run

final_df = data.loc[te, ['AssetId', 'ts']].reset_index(drop=True)
sc = final_df['AssetId'].map(scale).values
final_df['Actual'] = y_te * sc
for m in ALL_METHODS:
    final_df[m] = np.clip(preds_te[m], 0, None) * sc

y = final_df['Actual'].values
y_pers = final_df['Persistence'].values
switching = np.abs(y - y_pers) > 0.2 * sc          # ON/OFF switching hours
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
pers_rmse = rmse(y, y_pers)

rows = {}
for m in ALL_METHODS:
    p = final_df[m].values
    mk = y >= MAPE_MIN_KW
    rows[m] = {'Type': ALL_TYPE[m],
               'RMSE (kW)': rmse(y, p),
               'MAE (kW)': mean_absolute_error(y, p),
               'MAPE (%)': 100 * np.mean(np.abs((y[mk] - p[mk]) / y[mk])),
               'R²': r2_score(y, p),
               'Skill vs Persistence (%)': 100 * (1 - rmse(y, p) / pers_rmse),
               'Switching-hour RMSE (kW)': rmse(y[switching], p[switching]),
               'Steady-hour RMSE (kW)': rmse(y[~switching], p[~switching])}
final_metrics = pd.DataFrame(rows).T.sort_values('RMSE (kW)')
num_cols = final_metrics.columns.drop('Type')
final_metrics[num_cols] = final_metrics[num_cols].astype(float).round(3)

print("=" * 70)
print("TABLE: Final next-hour forecast accuracy on the test period (all 4 assets)")
print(f"Switching hours = {100*switching.mean():.1f}% of test hours")
print("=" * 70)
display(final_metrics)

final_per_asset = pd.DataFrame({a: {m: rmse(final_df.loc[final_df.AssetId == a, 'Actual'],
                                             final_df.loc[final_df.AssetId == a, m])
                                    for m in ALL_METHODS} for a in SELECTED_ASSETS}).round(3)
print("\nTABLE: Test RMSE (kW) per asset")
display(final_per_asset)

best = final_metrics.index[0]
print(f"\nBest method: {best} -- RMSE {final_metrics.loc[best, 'RMSE (kW)']:.3f} kW, "
      f"{final_metrics.loc[best, 'Skill vs Persistence (%)']:.1f}% better than Persistence, "
      f"switching-hour RMSE {final_metrics.loc[best, 'Switching-hour RMSE (kW)']:.3f} kW "
      f"(Persistence: {final_metrics.loc['Persistence', 'Switching-hour RMSE (kW)']:.3f} kW).")

# ============================================================
# Updated comparison graph (black & white friendly)
# Fewer lines than Cell 14 so the proposed methods are easy to follow.
# ============================================================
PLOT_METHODS = [m for m in ['Persistence', 'XGBoost', 'LSTM', 'Weighted Ensemble',
                            'Enhanced XGBoost', 'Two-Stage Hybrid'] if m in ALL_METHODS]
STYLE2 = {
    'Persistence':       dict(color='0.60', ls=(0, (1, 1.5)),       marker='x'),
    'XGBoost':           dict(color='0.40', ls=(0, (6, 2)),         marker='s'),
    'LSTM':              dict(color='0.50', ls=(0, (6, 2, 1, 2)),   marker='^'),
    'Weighted Ensemble': dict(color='0.30', ls=(0, (3, 1.5)),       marker='o'),
    'Enhanced XGBoost':  dict(color='0.00', ls=(0, (12, 3)),        marker='D'),
    'Two-Stage Hybrid':  dict(color='0.15', ls=(0, (1, 1, 4, 1)),   marker='v'),
}

def plot_final(asset, n_hours, fname, title_extra=''):
    sub = final_df[final_df['AssetId'] == asset].set_index('ts').sort_index()
    counts = [(d, sub.loc[d:d + pd.Timedelta(hours=n_hours - 1)].shape[0])
              for d in sub.index.normalize().unique()]
    start = max(counts, key=lambda c: c[1])[0]
    win = sub.loc[start:start + pd.Timedelta(hours=n_hours - 1)]
    win = win.reindex(pd.date_range(start, periods=n_hours, freq='h'))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(win.index, win['Actual'], color='black', lw=2.8, ls='-', label='Actual load', zorder=10)
    step = max(n_hours // 14, 3)
    for i, m in enumerate(PLOT_METHODS):
        st = STYLE2[m]
        ax.plot(win.index, win[m], color=st['color'], ls=st['ls'], lw=1.5,
                marker=st['marker'], markersize=6, markerfacecolor='white',
                markeredgecolor=st['color'], markeredgewidth=1.2,
                markevery=(i % step, step), label=m)
    ax.set_title(f"Next-hour load forecast vs actual load: asset {asset}{title_extra}",
                 fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{POWER_COL} (kW)")
    ax.grid(alpha=0.3, color='0.7')
    fig.autofmt_xdate()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

plot_final(PLOT_ASSET, 48, 'final_forecast_48h.png', ' (48-hour zoom)')
print("Figure: Proposed methods vs baselines over 48 test hours.")

# ---- Bar chart: overall RMSE vs switching-hour RMSE (hatched, B&W friendly) ----
order = final_metrics.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, col, hatch in zip(axes, ['RMSE (kW)', 'Switching-hour RMSE (kW)'], ['//', 'xx']):
    vals = final_metrics.loc[order, col].values
    ax.bar(range(len(order)), vals, color='white', edgecolor='black', hatch=hatch, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha='right')
    ax.set_title(f"Test {col} (lower = better)", fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('final_error_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure: Overall RMSE and switching-hour RMSE of every method.")

final_metrics.to_csv('final_forecast_metrics.csv')
print("\nCell 17 complete. Saved: final_forecast_48h.png, final_error_comparison.png, "
      "final_forecast_metrics.csv")
