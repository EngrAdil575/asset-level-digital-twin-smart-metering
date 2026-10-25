# ============================================================
# CELL 14: Comparison graphs -- black & white friendly
# Actual load = thick solid black. Every method has its own line style,
# marker shape and grey shade, so all lines stay distinguishable when
# printed in black and white. Figures are saved as 300-dpi PNGs.
# Requires: Cell 13.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STYLE = {
    'Seasonal Naive':    dict(color='0.60', ls=(0, (1, 1.5)),          marker='x'),
    'XGBoost':           dict(color='0.30', ls=(0, (6, 2)),            marker='s'),
    'LSTM':              dict(color='0.45', ls=(0, (6, 2, 1, 2)),      marker='^'),
    'CNN-LSTM':          dict(color='0.20', ls=(0, (3, 1.5)),          marker='o'),
    'LSTM + XGBoost':    dict(color='0.50', ls=(0, (10, 2, 2, 2, 2, 2)), marker='D'),
    'DBSCAN + XGBoost':  dict(color='0.10', ls=(0, (1, 1, 4, 1)),      marker='v'),
    'Weighted Ensemble': dict(color='0.00', ls=(0, (12, 3)),           marker='P'),
}

def plot_comparison(asset, n_hours, fname, title_extra=''):
    sub = test_df[test_df['AssetId'] == asset].set_index('ts').sort_index()
    # pick the window of n_hours with the most available (reliable) test data
    counts = [(d, sub.loc[d:d + pd.Timedelta(hours=n_hours - 1)].shape[0])
              for d in sub.index.normalize().unique()]
    start = max(counts, key=lambda c: c[1])[0]
    win = sub.loc[start:start + pd.Timedelta(hours=n_hours - 1)]
    win = win.reindex(pd.date_range(start, periods=n_hours, freq='h'))   # gaps -> breaks in the line

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(win.index, win['Actual'], color='black', lw=2.8, ls='-', label='Actual load', zorder=10)
    step = max(n_hours // 14, 3)
    for i, m in enumerate(METHODS):
        st = STYLE[m]
        ax.plot(win.index, win[m], color=st['color'], ls=st['ls'], lw=1.4,
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

plot_comparison(PLOT_ASSET, 24 * 7, 'forecast_comparison_week.png', ' (one test week)')
print("Figure: Actual vs predicted load for all 7 methods over one test week.")
plot_comparison(PLOT_ASSET, 48, 'forecast_comparison_48h.png', ' (48-hour zoom)')
print("Figure: 48-hour zoom of the same comparison, where differences between methods are easier to see.")

# ---- Error bar chart (hatch patterns instead of colours -> B&W friendly) ----
HATCH = ['xx', '//', '\\\\', '..', 'oo', '++', '--']
order = metrics_df.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ['RMSE (kW)', 'MAE (kW)']):
    vals = metrics_df.loc[order, col].values
    bars = ax.bar(range(len(order)), vals, color='white', edgecolor='black', lw=1.2)
    for b, m in zip(bars, order):
        b.set_hatch(HATCH[METHODS.index(m)])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_title(f"Test {col.split(' ')[0]} by method (lower = better)", fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_error_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure: Test-set RMSE and MAE of every method, sorted best to worst.")

metrics_df.round(4).to_csv('forecast_metrics.csv')
print("\nCell 14 complete. Saved: forecast_comparison_week.png, forecast_comparison_48h.png, "
      "forecast_error_comparison.png, forecast_metrics.csv")
print("Variables available for later cells: test_df, metrics_df, per_asset, preds_te")
