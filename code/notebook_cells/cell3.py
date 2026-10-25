# ============================================================
# CELL 3: Tune DBSCAN eps for a balanced noise percentage
# (Step 1 -- refines the eps chosen in Cell 2 so the noise/anomaly
# percentage is more defensible, while keeping clustering quality
# strong.) Displays inline. Nothing is downloaded.
# Requires: 'X', 'min_samples', 'km_final_silhouette', 'km_final_dbi'
# from Cell 2 (run Cell 2 first).
# ============================================================

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from IPython.display import display

# ---- Wider, finer eps sweep than Cell 2 ----
eps_candidates_fine = np.round(np.arange(0.02, 0.41, 0.01), 3)

results = []
for eps in eps_candidates_fine:
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

    results.append({'eps': eps, 'n_clusters': n_clusters, 'noise_pct': round(noise_pct, 1),
                     'silhouette': sil, 'dbi': dbi})

results_df = pd.DataFrame(results)

print("=" * 70)
print("TABLE: DBSCAN results across a finer eps sweep (0.02 to 0.40)")
print("=" * 70)
display(results_df)

# ---- Pick the best eps by silhouette score alone (no arbitrary noise-range
# restriction) -- the noise percentage is reported honestly either way,
# but the eps choice is driven by clustering quality, not a fixed target. ----
valid_results = results_df.dropna(subset=['silhouette'])

print(f"\nAll valid eps candidates (2+ clusters found), sorted by Silhouette Score:")
display(valid_results.sort_values('silhouette', ascending=False).reset_index(drop=True))

if len(valid_results) > 0:
    best_balanced = valid_results.loc[valid_results['silhouette'].idxmax()]
    chosen_eps = best_balanced['eps']
    print(f"\nChosen eps (highest Silhouette Score overall): {chosen_eps} "
          f"(noise={best_balanced['noise_pct']}%, "
          f"silhouette={best_balanced['silhouette']:.4f}, "
          f"dbi={best_balanced['dbi']:.4f})")
else:
    chosen_eps = 0.1  # fallback
    print(f"\nNo eps produced valid clusters. Using fallback eps={chosen_eps}")

# ---- Refit DBSCAN with the chosen, balanced eps ----
db_balanced = DBSCAN(eps=chosen_eps, min_samples=min_samples).fit(X)
db_balanced_labels = db_balanced.labels_
db_balanced_n_clusters = len(set(db_balanced_labels)) - (1 if -1 in db_balanced_labels else 0)
db_balanced_noise_pct = 100 * np.sum(db_balanced_labels == -1) / len(db_balanced_labels)
mask = db_balanced_labels != -1
db_balanced_silhouette = silhouette_score(X[mask], db_balanced_labels[mask])
db_balanced_dbi = davies_bouldin_score(X[mask], db_balanced_labels[mask])

print("\n" + "=" * 70)
print("FINAL COMPARISON: K-means (baseline) vs DBSCAN (balanced eps)")
print("=" * 70)

final_comparison = pd.DataFrame({
    'K-means (baseline)': {
        'Silhouette Score': round(km_final_silhouette, 4),
        'Davies-Bouldin Index': round(km_final_dbi, 4),
        'Clusters found': km_n_clusters,
        'Noise/anomaly points (%)': 0.0,
    },
    'DBSCAN (balanced eps)': {
        'Silhouette Score': round(db_balanced_silhouette, 4),
        'Davies-Bouldin Index': round(db_balanced_dbi, 4),
        'Clusters found': db_balanced_n_clusters,
        'Noise/anomaly points (%)': round(db_balanced_noise_pct, 1),
    },
})
display(final_comparison)

print(f"\nFinal chosen DBSCAN eps: {chosen_eps} (min_samples={min_samples})")
print("This is now the DBSCAN configuration to use going forward "
      "(Cell 4 and any later cells).")
