# Builds the CSV files of Table IV (forecasting) and Table V (anomaly) of the paper from the five-seed results.
import numpy as np, pandas as pd, json, re, os
S = os.environ.get('OUT_DIR', 'seeds')
agg = pd.read_csv(f'{S}/summary.csv', header=[0, 1], index_col=0)
g = lambda m, c: float(agg.loc[m, (c, 'mean')])
sdv = lambda m, c: float(agg.loc[m, (c, 'std')]) if not np.isnan(agg.loc[m, (c, 'std')]) else 0.0

F1 = r"-[\citen{fesseha2025realtime}]"; F2 = r"-[\citen{fesseha2026multistep}]"
rows = [('K-means profile', r"Fesseha's model (K-means profile)" + F1 + r"$^\dagger$", 'Baseline-1', 'No'),
        ('ARIMA', r"Fesseha's model (ARIMA)" + F2, 'Baseline-2', 'Yes'),
        ('VAR', r"Fesseha's model (VAR)" + F2, 'Baseline-2', 'Yes'),
        ('Random Forest', r"Fesseha's model (Random forest)" + F2, 'Baseline-2', 'Yes'),
        ('XGBoost', r"Fesseha's model (XGBoost)" + F2, 'Baseline-2', 'Yes'),
        ('LSTM', r"Fesseha's model (LSTM)" + F2, 'Baseline-2', 'Yes'),
        ('TCN', r"Fesseha's model (TCN)" + F2, 'Baseline-2', 'Yes'),
        None,
        ('CNN-LSTM', 'CNN-LSTM', 'Hybrid', 'Yes'),
        ('LSTM + XGBoost', 'LSTM + XGBoost', 'Hybrid', 'Yes'),
        ('DBSCAN + XGBoost', 'DBSCAN + XGBoost', 'Hybrid', 'Yes'),
        ('Weighted Ensemble', 'Weighted ensemble', 'Hybrid', 'Yes'),
        ('Two-Stage Hybrid', 'Two-stage hybrid', 'Hybrid', 'Yes'),
        None,
        ('Enhanced XGBoost', r"\textbf{Proposed (Enhanced XGBoost)}", r"\textbf{Proposed}", 'Yes')]
def pm(m, c, d=3):
    s = sdv(m, c)
    return f"{g(m, c):.{d}f}" if s < 0.5 * 10 ** (-d) else f"{g(m, c):.{d}f}$\\pm${s:.{d}f}"
body, csv3 = [], []
for r in rows:
    if r is None: body.append(r"\midrule"); continue
    k, name, typ, fin = r
    cells = [pm(k, 'RMSE'), f"{g(k,'MAE'):.3f}", f"{g(k,'MAPE'):.2f}", f"{g(k,'R2'):.3f}", (f"${g(k,'Skill'):.1f}$" if g(k,'Skill') < 0 else f"{g(k,'Skill'):.1f}"),
             pm(k, 'SW')]
    if k == 'Enhanced XGBoost':
        cells = [r"\textbf{" + c + "}" for c in cells]
    body.append(f"{name} & {typ} & {fin} & " + " & ".join(cells) + r" \\")
    csv3.append(dict(Algorithm=re.sub(r'\\textbf\{|\}|\$\^\\dagger\$|-\[\\citen\{fesseha2025realtime|-\[\\citen\{fesseha2026multistep', '', name)
                     .replace(']', '').strip(), Type=typ.replace(r'\textbf{', '').replace('}', ''), **{
        'Forecasting algorithm in source': fin,
        'RMSE mean (kW)': round(g(k, 'RMSE'), 3), 'RMSE SD': round(sdv(k, 'RMSE'), 3),
        'MAE mean (kW)': round(g(k, 'MAE'), 3), 'MAE SD': round(sdv(k, 'MAE'), 3),
        'MAPE (%)': round(g(k, 'MAPE'), 2), 'R2': round(g(k, 'R2'), 3), 'Skill vs persistence (%)': round(g(k, 'Skill'), 1),
        'Switching-hour RMSE (kW)': round(g(k, 'SW'), 3), 'Steady-hour RMSE (kW)': round(g(k, 'ST'), 3)}))

# ---------- anomaly (Table V) ----------
A = pd.read_csv(f'{S}/anomaly_seeds.csv')
per_seed = A.groupby(['Method', 'seed']).mean(numeric_only=True)
am = per_seed.groupby('Method').mean(); asd = per_seed.groupby('Method').std()
cols = ['False alarms on real days (%)', 'Detection rate, all faults (%)', 'Left ON overnight', 'Missed shift', 'Load spike', 'Schedule shift', 'Base-load increase']
order = [('DBSCAN', r'DBSCAN ($\varepsilon$ fixed)'), None,
         ('K-means distance', r"Fesseha's model (K-means dist.)-[\citen{fesseha2025realtime}]"),
         ('DBSCAN density score', 'DBSCAN density'), ('Isolation Forest', 'Isolation Forest'),
         ('Context-aware kNN (proposed)', r'\textbf{Proposed (Context-aware kNN)}')]
best_col = {c: am.loc[[o[0] for o in order if o and o[0] != 'DBSCAN'], c].idxmax() for c in cols[1:]}
b4, csv4 = [], []
for o in order:
    if o is None: b4.append(r"\midrule"); continue
    k, name = o
    cells = [f"{am.loc[k, cols[0]]:.1f}", f"{am.loc[k, cols[1]]:.1f}$\\pm${asd.loc[k, cols[1]]:.1f}"] + [f"{am.loc[k, c]:.1f}" for c in cols[2:]]
    if k != 'DBSCAN':
        cells = [r"\textbf{" + v + "}" if (i >= 1 and best_col[cols[i]] == k) else v for i, v in enumerate(cells)]
    b4.append(f"{name} & " + " & ".join(cells) + r" \\")
    csv4.append({'Method': re.sub(r'\\textbf\{|\}|\$\\varepsilon\$', '', name).replace(r'-[\citen{fesseha2025realtime', '-[1').strip(),
                 'Normal days flagged (%)': round(am.loc[k, cols[0]], 1), 'All faults mean (%)': round(am.loc[k, cols[1]], 1),
                 'All faults SD (%)': round(asd.loc[k, cols[1]], 1), **{f'{c} (%)': round(am.loc[k, c], 1) for c in cols[2:]}})
pd.DataFrame(csv3).to_csv(f'{S}/Table3.csv', index=False)
pd.DataFrame(csv4).to_csv(f'{S}/Table4.csv', index=False)
print(pd.DataFrame(csv3).to_string(index=False)); print(pd.DataFrame(csv4).to_string(index=False))
