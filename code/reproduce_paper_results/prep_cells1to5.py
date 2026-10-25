# Local reproduction of notebook Cells 1-5 (only the variables Cell 6 needs),
# loading just the 4 selected assets to save time.
import glob, os, numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

BASE = os.environ.get("DATASET_DIR", "raw_data/Industrial Asset-Level Electrical Energy Dataset from a Manufacturing Facility (15-Minute Aggregates)/data_csv")
SELECTED_ASSETS = ['p_c', 'ahu_a', 'comp_b', 'mh_a']
POWER_COL = 'Demand_kW'
files = []
for a in SELECTED_ASSETS:
    files += glob.glob(os.path.join(BASE, f"asset_id={a}", "**", "*.csv"), recursive=True)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
reliable_df = df[df['IsReliableWindow'] == True]
df_sel = reliable_df[reliable_df['Phase'] == 'ALL'].copy()
df_sel['date'] = df_sel['window_start_utc'].dt.date
df_sel['hour'] = df_sel['window_start_utc'].dt.hour
hourly = df_sel.groupby(['AssetId', 'date', 'hour'])[POWER_COL].mean().reset_index()
pivot = hourly.pivot_table(index=['AssetId', 'date'], columns='hour', values=POWER_COL, fill_value=0)
pivot_norm = pivot.div(pivot.max(axis=1), axis=0).fillna(0)
X = pivot_norm.values
min_samples = 5
best = (-1, None)
for eps in np.round(np.arange(0.02, 0.41, 0.01), 3):
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
    n = len(set(lab)) - (1 if -1 in lab else 0)
    m = lab != -1
    if n >= 2 and m.sum() > n:
        s = silhouette_score(X[m], lab[m])
        if s > best[0]:
            best = (s, eps)
chosen_eps = best[1]
EPSILON_KW = 0.5
def compute_indices(group):
    p = group[POWER_COL]
    mean_p, max_p, min_p = p.mean(), p.max(), p.min()
    return pd.Series({'LF': mean_p / max_p if max_p != 0 else np.nan,
                      'PBR': (max_p / min_p) if min_p >= EPSILON_KW else np.nan,
                      'VI': p.var()})
daily_indices_ctx = hourly.groupby(['AssetId', 'date']).apply(compute_indices).reset_index()
print("rows", len(hourly), "days", X.shape[0], "chosen_eps", chosen_eps)
