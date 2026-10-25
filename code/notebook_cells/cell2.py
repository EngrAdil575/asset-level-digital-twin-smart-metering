# ============================================================
# CELL 2: Proposed Algorithm Comparison -- DBSCAN vs K-means
# (Baseline paper used K-means. This cell tests DBSCAN as a
# proposed alternative and compares performance using the same
# validation metrics: Silhouette Score and Davies-Bouldin Index.)
# Everything displays inline below this cell. Nothing is downloaded.
# Requires: 'reliable_df' from Cell 1 (run Cell 1 first).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from IPython.display import display

# ---- Use data already in memory from Cell 1 ----
try:
    df = reliable_df.copy()
    print(f"Using 'reliable_df' from Cell 1 ({len(df):,} rows).")
except NameError:
    print("'reliable_df' not found -- loading from CSV instead.")
    df = pd.read_csv("combined_dataset_reliable_only.csv")

df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
df_all = df[df['Phase'] == 'ALL'].copy()

SELECTED_ASSETS = ['p_c', 'ahu_a', 'comp_b', 'mh_a']
df_sel = df_all[df_all['AssetId'].isin(SELECTED_ASSETS)].copy()
POWER_COL = 'Demand_kW'
df_sel['date'] = df_sel['window_start_utc'].dt.date
df_sel['hour'] = df_sel['window_start_utc'].dt.hour

hourly = df_sel.groupby(['AssetId', 'date', 'hour'])[POWER_COL].mean().reset_index()

# ---- Build normalized daily load-shape vectors (same as baseline) ----
pivot = hourly.pivot_table(index=['AssetId', 'date'], columns='hour',
                            values=POWER_COL, fill_value=0)
pivot_norm = pivot.div(pivot.max(axis=1), axis=0).fillna(0)
X = pivot_norm.values

print(f"\nTotal daily load-shape samples: {X.shape[0]} (across {len(SELECTED_ASSETS)} assets)")

# ============================================================
# BASELINE METHOD: K-means (as used in the baseline paper)
# ============================================================
print("=" * 70)
print("BASELINE METHOD: K-means")
print("=" * 70)

k_range = range(2, 8)
km_silhouette, km_dbi = {}, {}
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    km_silhouette[k] = silhouette_score(X, km.labels_)
    km_dbi[k] = davies_bouldin_score(X, km.labels_)

best_k = max(km_silhouette, key=km_silhouette.get)
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
km_final_silhouette = silhouette_score(X, kmeans_final.labels_)
km_final_dbi = davies_bouldin_score(X, kmeans_final.labels_)
km_n_clusters = best_k
km_noise_pct = 0.0  # K-means assigns every point to a cluster

print(f"Best k: {best_k}")
print(f"Silhouette Score: {km_final_silhouette:.4f}")
print(f"Davies-Bouldin Index: {km_final_dbi:.4f}")

# ============================================================
# PROPOSED METHOD: DBSCAN (density-based, also flags anomalies)
# ============================================================
print("\n" + "=" * 70)
print("PROPOSED METHOD: DBSCAN")
print("=" * 70)

# Estimate a good eps using the k-distance elbow (standard DBSCAN practice)
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, _ = neighbors.kneighbors(X)
k_distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_distances)
ax.set_title(f"K-Distance Plot (min_samples={min_samples}) \u2013 used to choose eps", fontweight='bold')
ax.set_xlabel("Points sorted by distance")
ax.set_ylabel(f"Distance to {min_samples}th nearest neighbor")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print("Figure: K-distance plot used to select the DBSCAN eps parameter "
      "(the 'elbow' of this curve is a good eps value).\n")

# Try a small range of eps values around the visual elbow and pick the
# one with the best silhouette score (excluding noise points from scoring)
# Try a range of eps values based on the actual data's distance distribution
# (adaptive, so it works regardless of dataset scale)
eps_candidates = np.percentile(k_distances, np.linspace(10, 90, 12))
eps_candidates = np.unique(np.round(eps_candidates, 3))
eps_candidates = eps_candidates[eps_candidates > 0]
dbscan_results = []
for eps in eps_candidates:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * np.sum(labels == -1) / len(labels)
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > n_clusters:
            sil = silhouette_score(X[mask], labels[mask])
            dbi = davies_bouldin_score(X[mask], labels[mask])
        else:
            sil, dbi = np.nan, np.nan
    else:
        sil, dbi = np.nan, np.nan
    dbscan_results.append({'eps': round(eps, 2), 'n_clusters': n_clusters,
                            'noise_pct': round(noise_pct, 1),
                            'silhouette': sil, 'dbi': dbi})

dbscan_table = pd.DataFrame(dbscan_results)
print("TABLE: DBSCAN results across candidate eps values")
display(dbscan_table)

# Pick the eps with the best (highest) silhouette score among valid runs
valid = dbscan_table.dropna(subset=['silhouette'])
if len(valid) > 0:
    best_row = valid.loc[valid['silhouette'].idxmax()]
    best_eps = best_row['eps']
else:
    best_eps = 0.3  # fallback
    print("\nNo eps value produced 2+ valid clusters; using fallback eps=0.3")

db_final = DBSCAN(eps=best_eps, min_samples=min_samples).fit(X)
db_labels = db_final.labels_
db_n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
db_noise_pct = 100 * np.sum(db_labels == -1) / len(db_labels)

mask = db_labels != -1
db_final_silhouette = silhouette_score(X[mask], db_labels[mask]) if db_n_clusters >= 2 else np.nan
db_final_dbi = davies_bouldin_score(X[mask], db_labels[mask]) if db_n_clusters >= 2 else np.nan

print(f"\nChosen eps: {best_eps}")
print(f"Clusters found: {db_n_clusters}")
print(f"Points flagged as noise/anomalies: {db_noise_pct:.1f}%")
print(f"Silhouette Score (excluding noise): {db_final_silhouette:.4f}")
print(f"Davies-Bouldin Index (excluding noise): {db_final_dbi:.4f}")

# ============================================================
# COMPARISON: Baseline (K-means) vs Proposed (DBSCAN)
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: Baseline (K-means) vs Proposed (DBSCAN)")
print("=" * 70)

comparison_table = pd.DataFrame({
    'K-means (baseline)': {
        'Clusters found': km_n_clusters,
        'Silhouette Score (higher=better)': round(km_final_silhouette, 4),
        'Davies-Bouldin Index (lower=better)': round(km_final_dbi, 4),
        'Noise/anomaly points flagged (%)': km_noise_pct,
    },
    'DBSCAN (proposed)': {
        'Clusters found': db_n_clusters,
        'Silhouette Score (higher=better)': round(db_final_silhouette, 4),
        'Davies-Bouldin Index (lower=better)': round(db_final_dbi, 4),
        'Noise/anomaly points flagged (%)': round(db_noise_pct, 1),
    },
})
display(comparison_table)

if db_final_silhouette > km_final_silhouette:
    print("\nResult: DBSCAN achieves a higher Silhouette Score than K-means on this "
          "dataset, and additionally flags anomalous/irregular days automatically "
          "(noise points) -- something K-means cannot do. DBSCAN can be proposed "
          "as the improved clustering method for this project.")
else:
    print("\nResult: K-means still achieves a higher Silhouette Score on this "
          "dataset. DBSCAN's main advantage here is automatic anomaly flagging "
          "(noise points), which may still be useful for the anomaly-alert "
          "module even if raw clustering quality is similar.")

# ---- Visual comparison: cluster shapes side by side (using first 2 hours as example axes) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

scatter1 = axes[0].scatter(X[:, 9], X[:, 15], c=kmeans_final.labels_, cmap='tab10', s=15)
axes[0].set_title(f"K-means (k={km_n_clusters})", fontweight='bold')
axes[0].set_xlabel("Normalized Power @ Hour 9")
axes[0].set_ylabel("Normalized Power @ Hour 15")

colors = ['red' if l == -1 else l for l in db_labels]
scatter2 = axes[1].scatter(X[:, 9], X[:, 15], c=db_labels, cmap='tab10', s=15)
axes[1].set_title(f"DBSCAN ({db_n_clusters} clusters, {db_noise_pct:.1f}% flagged as anomalies)", fontweight='bold')
axes[1].set_xlabel("Normalized Power @ Hour 9")
axes[1].set_ylabel("Normalized Power @ Hour 15")

plt.tight_layout()
plt.show()
print("Figure: Cluster assignments compared between K-means and DBSCAN "
      "(shown on two example hourly dimensions for visualization purposes).\n")

print("Cell 2 complete.")