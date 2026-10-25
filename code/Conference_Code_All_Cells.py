# Conference paper code (Google Colab): Cells 1-21 of Conference_Notebook.ipynb in one file.
# In Colab, run the cells one by one from the notebook; this file is the same code for reading and archiving.

# ============================================================
# CELL 1 (RUN THIS FIRST): Upload dataset ZIP, combine into one
# clean dataset, and KEEP IT IN MEMORY for all cells below.
# No download/re-upload needed for later steps in this session.
# ============================================================

import pandas as pd
import glob
import os
import zipfile
from collections import Counter
from google.colab import files

# ---- STEP 1: Upload dataset ----
print("Please select your dataset file(s) to upload (ZIP or CSV)...")
uploaded = files.upload()

RAW_FOLDER = "raw_data"
os.makedirs(RAW_FOLDER, exist_ok=True)

for fname in uploaded.keys():
    if fname.endswith(".zip"):
        print(f"Extracting {fname} ...")
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(RAW_FOLDER)
    elif fname.endswith(".csv"):
        os.rename(fname, os.path.join(RAW_FOLDER, fname))

# ---- STEP 2: Find CSVs, exclude metadata/log files by name ----
all_csv_files = glob.glob(os.path.join(RAW_FOLDER, "**", "*.csv"), recursive=True)
print(f"\nFound {len(all_csv_files)} total CSV files.")

EXCLUDE_KEYWORDS = ["dictionary", "metadata", "readme", "issue", "log", "description"]
candidate_files = [
    f for f in all_csv_files
    if not any(k in os.path.basename(f).lower() for k in EXCLUDE_KEYWORDS)
]
print(f"{len(candidate_files)} candidate data files after removing metadata/log files.")

# ---- STEP 3: Detect the DOMINANT schema across files ----
schema_counter = Counter()
file_schema_map = {}

for f in candidate_files:
    try:
        cols = tuple(pd.read_csv(f, nrows=0).columns)
        schema_counter[cols] += 1
        file_schema_map[f] = cols
    except Exception:
        file_schema_map[f] = None

main_schema = schema_counter.most_common(1)[0][0]
print(f"\nUsing target schema ({schema_counter[main_schema]} files match):")
print(main_schema)

data_files = [f for f, cols in file_schema_map.items() if cols == main_schema]
skipped_files = [f for f, cols in file_schema_map.items() if cols != main_schema]
print(f"\nWill combine {len(data_files)} files. Skipping {len(skipped_files)} mismatched files.")

# ---- STEP 4: Combine matching files directly into ONE in-memory DataFrame ----
# (Reads all matching files and concatenates in memory. If this ever runs out
# of RAM on a very large dataset, fall back to the disk-based version instead.)
df_list = []
for i, file in enumerate(data_files, 1):
    try:
        temp_df = pd.read_csv(file)[list(main_schema)]
        df_list.append(temp_df)
    except Exception as e:
        print(f"  Skipping {file} due to error: {e}")
    if i % 500 == 0 or i == len(data_files):
        print(f"  Loaded {i}/{len(data_files)} files...")

combined_df = pd.concat(df_list, ignore_index=True)
del df_list  # free memory

# ---- STEP 5: Clean (timestamps, duplicates, reliability filter) ----
timestamp_col = "window_start_utc"
if timestamp_col in combined_df.columns:
    combined_df[timestamp_col] = pd.to_datetime(combined_df[timestamp_col])
    sort_cols = [c for c in ["AssetId", "Phase", timestamp_col] if c in combined_df.columns]
    combined_df = combined_df.sort_values(by=sort_cols)

before = len(combined_df)
combined_df = combined_df.drop_duplicates()
print(f"\nRemoved {before - len(combined_df)} duplicate rows.")

if "IsReliableWindow" in combined_df.columns:
    reliable_df = combined_df[combined_df["IsReliableWindow"] == True].reset_index(drop=True)
else:
    reliable_df = combined_df.reset_index(drop=True)

print(f"Final reliable dataset: {len(reliable_df):,} rows, {reliable_df.shape[1]} columns")

# ---- STEP 6: Save a copy to disk as backup, but DO NOT force-download ----
# (Kept on disk only as a safety net in case the session disconnects;
# every later cell in this notebook should just use `reliable_df` directly.)
reliable_df.to_csv("combined_dataset_reliable_only.csv", index=False)
print("\nBackup copy saved to disk as 'combined_dataset_reliable_only.csv' "
      "(not downloaded -- stays in this session).")

print("\n'reliable_df' is now ready in memory. Use this variable directly in "
      "all the cells below -- no need to upload or download anything again "
      "in this session.")

reliable_df.head()


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


# ============================================================
# CELL 4: Inspect which specific days DBSCAN flagged as anomalies
# (Step 2 -- pulls out the exact asset-days marked as noise by the
# balanced DBSCAN run in Cell 3, and shows their Load Factor, PBR,
# and Variability Index for context.) Displays inline.
# Requires: 'db_balanced_labels', 'pivot_norm', 'hourly', 'POWER_COL'
# from Cells 2 and 3 (run Cell 2 then Cell 3 first).
# ============================================================

import pandas as pd
import numpy as np
from IPython.display import display

# ---- Map DBSCAN noise labels back to their (AssetId, date) identity ----
index_df = pivot_norm.index.to_frame(index=False)[['AssetId', 'date']]
index_df['dbscan_label'] = db_balanced_labels
index_df['is_anomaly'] = index_df['dbscan_label'] == -1

flagged_days = index_df[index_df['is_anomaly']].copy()

print("=" * 70)
print(f"TOTAL DAYS FLAGGED AS ANOMALIES BY DBSCAN: {len(flagged_days)} "
      f"out of {len(index_df)} ({100*len(flagged_days)/len(index_df):.1f}%)")
print("=" * 70)

print("\nBreakdown by asset:")
display(flagged_days.groupby('AssetId').size().rename("Flagged Days").to_frame())

# ---- Compute LF/PBR/VI for the flagged days for context ----
EPSILON_KW = 0.5

def compute_indices(group):
    p = group[POWER_COL]
    mean_p, max_p, min_p = p.mean(), p.max(), p.min()
    LF = mean_p / max_p if max_p != 0 else np.nan
    PBR = (max_p / min_p) if min_p >= EPSILON_KW else np.nan
    VI = p.var()
    return pd.Series({'LF': LF, 'PBR': PBR, 'VI': VI})

daily_indices_ctx = hourly.groupby(['AssetId', 'date']).apply(compute_indices).reset_index()

flagged_with_context = flagged_days.merge(daily_indices_ctx, on=['AssetId', 'date'], how='left')

print("\n" + "=" * 70)
print("TABLE: Flagged anomaly days with their Load Factor / PBR / VI")
print("=" * 70)
display(flagged_with_context[['AssetId', 'date', 'LF', 'PBR', 'VI']]
        .sort_values(['AssetId', 'date']).reset_index(drop=True))

# ---- Compare: do these overlap with the statistically extreme PBR days? ----
overall_pbr_q3 = daily_indices_ctx['PBR'].quantile(0.75)
overall_pbr_iqr = overall_pbr_q3 - daily_indices_ctx['PBR'].quantile(0.25)
high_pbr_threshold = overall_pbr_q3 + 1.5 * overall_pbr_iqr

flagged_also_high_pbr = flagged_with_context[flagged_with_context['PBR'] > high_pbr_threshold]

print("\n" + "=" * 70)
print("OVERLAP CHECK: flagged anomaly days that also have a statistically high PBR")
print("=" * 70)
print(f"Of the {len(flagged_with_context)} DBSCAN-flagged days, "
      f"{len(flagged_also_high_pbr)} ({100*len(flagged_also_high_pbr)/max(len(flagged_with_context),1):.1f}%) "
      f"also have a PBR above the overall 1.5×IQR threshold ({high_pbr_threshold:.2f}).")

if len(flagged_also_high_pbr) > 0:
    display(flagged_also_high_pbr[['AssetId', 'date', 'LF', 'PBR', 'VI']].reset_index(drop=True))
    # Wording matches the progress report (Section 8.4): a small overlap does
    # NOT confirm the flagged days are genuine faults; it shows DBSCAN
    # captures a different kind of irregularity than PBR.
    print("\nThe overlap with high-PBR days is small, so it does not by itself "
          "confirm that the flagged days are genuine faults. It indicates that "
          "DBSCAN captures a different kind of irregularity (the shape of the "
          "whole day) than PBR (peak vs minimum only), which makes it a useful "
          "complementary signal for anomaly detection.")
else:
    print("\nNo direct overlap with high-PBR days. This suggests DBSCAN is "
          "flagging a different kind of irregularity (shape-based, across "
          "the full day) than PBR (which only looks at peak vs minimum), "
          "which can be a useful complementary signal for anomaly detection.")

print("\nCell 4 complete. Variable available for later cells: flagged_with_context")


# ============================================================
# CELL 5: Anomaly-Detection Module (seeded from DBSCAN)
# (Step 5 -- turns the DBSCAN clustering result from Cells 2-3
# into a reusable anomaly-detection function: given any day's
# normalized load shape, it flags whether that day is anomalous.)
# Displays inline. Nothing is downloaded.
# Requires: 'db_balanced', 'X', 'chosen_eps', 'pivot_norm',
# 'flagged_with_context' from Cells 2-4 (run those first).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from IPython.display import display

# ---- Identify DBSCAN's "core points" (the dense cluster cores) ----
core_mask = np.zeros_like(db_balanced.labels_, dtype=bool)
core_mask[db_balanced.core_sample_indices_] = True
core_points = X[core_mask]

print(f"Number of core points defining normal behavior: {len(core_points)} "
      f"out of {len(X)} total days.\n")

def is_anomalous_day(daily_profile, core_points=core_points, eps=chosen_eps):
    """
    Given a single day's normalized 24-hour load profile (array of 24
    values, each 0-1), returns True if it is anomalous (i.e. it does
    not fall within `eps` distance of any known normal/core day),
    and the distance to the nearest known-normal day.

    This directly reuses the DBSCAN density rule: a point is only
    'normal' if it lies within eps of an existing core point.
    """
    daily_profile = np.array(daily_profile).reshape(1, -1)
    distances = pairwise_distances(daily_profile, core_points)[0]
    nearest_distance = distances.min()
    return nearest_distance > eps, nearest_distance

# ---- Demonstrate the function on a few known days (mix of normal + flagged) ----
print("=" * 70)
print("DEMO: Applying the anomaly-detection function to sample days")
print("=" * 70)

sample_index = pivot_norm.drop(columns=['cluster'], errors='ignore').index
sample_rows = pd.DataFrame({'AssetId': [i[0] for i in sample_index],
                             'date': [i[1] for i in sample_index]})
sample_rows['row_position'] = range(len(sample_rows))

# Take 3 known-flagged days and 3 known-normal days as a demo set
flagged_keys = set(zip(flagged_with_context['AssetId'], flagged_with_context['date']))
sample_rows['was_flagged_by_dbscan'] = sample_rows.apply(
    lambda r: (r['AssetId'], r['date']) in flagged_keys, axis=1)

demo_flagged = sample_rows[sample_rows['was_flagged_by_dbscan']].head(3)
demo_normal = sample_rows[~sample_rows['was_flagged_by_dbscan']].head(3)
demo_set = pd.concat([demo_flagged, demo_normal])

results = []
for _, row in demo_set.iterrows():
    profile = X[row['row_position']]
    anomaly, dist = is_anomalous_day(profile)
    results.append({
        'AssetId': row['AssetId'], 'date': row['date'],
        'originally_flagged_by_dbscan': row['was_flagged_by_dbscan'],
        'function_says_anomalous': anomaly,
        'distance_to_nearest_normal_day': round(dist, 4),
    })

demo_results = pd.DataFrame(results)
display(demo_results)

agreement = (demo_results['originally_flagged_by_dbscan'] ==
             demo_results['function_says_anomalous']).mean() * 100
print(f"\nAgreement between the reusable function and the original DBSCAN "
      f"labels on this demo set: {agreement:.0f}%")

# ---- Visualize: distance-to-normal for every day, with the anomaly threshold ----
all_distances = []
for i in range(len(X)):
    _, dist = is_anomalous_day(X[i])
    all_distances.append(dist)

dist_df = sample_rows.copy()
dist_df['distance_to_nearest_normal'] = all_distances

fig, ax = plt.subplots(figsize=(9, 4))
colors = ['red' if d > chosen_eps else 'steelblue' for d in all_distances]
ax.scatter(range(len(all_distances)), sorted(all_distances), c=sorted(colors, reverse=True), s=10)
ax.axhline(y=chosen_eps, color='black', linestyle='--', label=f'Anomaly threshold (eps={chosen_eps})')
ax.set_xlabel("Day index (sorted by distance)")
ax.set_ylabel("Distance to nearest normal day")
ax.set_title("Anomaly-Detection Function: Distance-to-Normal for Every Day", fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()
print("Figure: Every day's distance to the nearest known-normal pattern. "
      "Points above the dashed line are flagged as anomalies.\n")

print("Cell 5 complete.")
print("\nThis 'is_anomalous_day()' function is the starting point for the "
      "project's anomaly-alert module: going forward, any new day's load "
      "profile can be passed to it to get an immediate anomaly flag, "
      "without re-running the full clustering each time.")


# ============================================================
# CELL 6: Forecasting setup -- continuous hourly series, train/val/test
# split and all input features (lags, calendar, DBSCAN day features).
# Predicts the NEXT HOUR's Demand_kW for each of the 4 selected assets.
# Requires: 'hourly', 'pivot_norm', 'X', 'SELECTED_ASSETS', 'POWER_COL',
# 'chosen_eps', 'min_samples', 'daily_indices_ctx' (run Cells 1-5 first).
# ============================================================

import os, random, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import tensorflow as tf

warnings.filterwarnings('ignore')
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- Settings ----------------
N_LAGS      = 24      # hours of history each model sees
TRAIN_FRAC  = 0.70    # first 70% of the timeline -> training
VAL_FRAC    = 0.10    # next 10% -> validation (early stopping, ensemble weights)
                      # last 20% -> test (all methods scored here)
MAPE_MIN_KW = 0.5     # MAPE only on hours with actual load >= 0.5 kW (avoids /0)
PLOT_ASSET  = 'p_c'   # asset shown in the comparison graph
EPOCHS      = 40

# ============================================================
# STEP 1: Build a continuous hourly series per asset
# (missing/unreliable hours stay NaN so lags never jump across gaps)
# ============================================================
h = hourly.copy()
h['ts'] = pd.to_datetime(h['date']) + pd.to_timedelta(h['hour'], unit='h')

frames = []
for a in SELECTED_ASSETS:
    s = h[h['AssetId'] == a].set_index('ts')[POWER_COL].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
    frames.append(pd.DataFrame({'AssetId': a, 'ts': s.index, 'load': s.values}))
data = pd.concat(frames, ignore_index=True)

# Chronological split points (same calendar cut-off for every asset)
all_ts = np.sort(data.loc[data['load'].notna(), 'ts'].unique())
t_val  = pd.Timestamp(all_ts[int(len(all_ts) * TRAIN_FRAC)])
t_test = pd.Timestamp(all_ts[int(len(all_ts) * (TRAIN_FRAC + VAL_FRAC))])
print(f"Train: before {t_val:%Y-%m-%d} | Validation: {t_val:%Y-%m-%d} to "
      f"{t_test:%Y-%m-%d} | Test: from {t_test:%Y-%m-%d}")

# Per-asset scaling using TRAINING data only (no information from the future)
scale = (data[data['ts'] < t_val].groupby('AssetId')['load'].max()
         .replace(0, 1.0))
data['load_s'] = data['load'] / data['AssetId'].map(scale)

# ============================================================
# STEP 2: Features
# ============================================================
g = data.groupby('AssetId')['load_s']
for k in range(1, N_LAGS + 1):
    data[f'lag_{k}'] = g.shift(k)
data['lag_168']   = g.shift(168)                      # same hour last week
data['roll_mean_24'] = g.transform(lambda x: x.shift(1).rolling(24, min_periods=18).mean())
data['roll_std_24']  = g.transform(lambda x: x.shift(1).rolling(24, min_periods=18).std())

hr, dow = data['ts'].dt.hour, data['ts'].dt.dayofweek
data['hour_sin'], data['hour_cos'] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
data['dow_sin'],  data['dow_cos']  = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
data['is_weekend'] = (dow >= 5).astype(int)
for a in SELECTED_ASSETS:
    data[f'asset_{a}'] = (data['AssetId'] == a).astype(int)

# ---- DBSCAN day-level features (for the DBSCAN + XGBoost hybrid) ----
# DBSCAN is RE-FITTED on training days only (same eps/min_samples as Cell 3),
# so no test-period load shape influences the model. For every day we take
# its distance to the nearest "normal" (core) training day -- exactly the
# is_anomalous_day() rule from Cell 5 -- and give the model YESTERDAY's value,
# which is known at forecast time.
day_idx = pivot_norm.index.to_frame(index=False)[['AssetId', 'date']]
day_idx['date'] = pd.to_datetime(day_idx['date'])
train_days = (day_idx['date'] < t_val.normalize()).values
db_train = DBSCAN(eps=chosen_eps, min_samples=min_samples).fit(X[train_days])
core_pts = X[train_days][db_train.core_sample_indices_]
if len(core_pts) == 0:                      # safety fallback
    core_pts = X[train_days]
day_idx['day_dist'] = pairwise_distances(X, core_pts).min(axis=1)
day_idx['day_anom'] = (day_idx['day_dist'] > chosen_eps).astype(int)

ctx = daily_indices_ctx.copy()
ctx['date'] = pd.to_datetime(ctx['date'])
day_feat = day_idx.merge(ctx, on=['AssetId', 'date'], how='left')
day_feat['date'] = day_feat['date'] + pd.Timedelta(days=1)       # -> "yesterday"
day_feat = day_feat.rename(columns={'day_dist': 'prev_day_dist', 'day_anom': 'prev_day_anom',
                                    'LF': 'prev_LF', 'PBR': 'prev_PBR', 'VI': 'prev_VI'})
data['date'] = data['ts'].dt.normalize()
data = data.merge(day_feat, on=['AssetId', 'date'], how='left')
# VI is in kW^2 -> scale it per asset like the load
data['prev_VI'] = data['prev_VI'] / data['AssetId'].map(scale) ** 2

# Keep only rows where the target and the full 24-h history exist
lag_cols = [f'lag_{k}' for k in range(1, N_LAGS + 1)]
data = data.dropna(subset=['load_s'] + lag_cols).reset_index(drop=True)

asset_cols = [f'asset_{a}' for a in SELECTED_ASSETS]
cal_cols   = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
base_feats = lag_cols + ['lag_168', 'roll_mean_24', 'roll_std_24'] + cal_cols + asset_cols
dbscan_feats = base_feats + ['prev_day_dist', 'prev_day_anom', 'prev_LF', 'prev_PBR', 'prev_VI']

tr = data['ts'] < t_val
va = (data['ts'] >= t_val) & (data['ts'] < t_test)
te = data['ts'] >= t_test
y_tr, y_va, y_te = data.loc[tr, 'load_s'].values, data.loc[va, 'load_s'].values, data.loc[te, 'load_s'].values
print(f"Samples -> train: {tr.sum():,} | validation: {va.sum():,} | test: {te.sum():,}")

preds_va, preds_te = {}, {}   # scaled predictions of every method (filled by Cells 7-12)
print("\nCell 6 complete. Features and train/validation/test split are ready.")


# ============================================================
# CELL 7: Model 1 -- Seasonal Naive (BASELINE)
# Prediction = load at the same hour last week
# (falls back to the same hour yesterday if last week is missing).
# Requires: Cell 6.
# ============================================================

naive = data['lag_168'].fillna(data['lag_24'])
preds_va['Seasonal Naive'] = naive[va].values
preds_te['Seasonal Naive'] = naive[te].values
print("Seasonal Naive baseline ready.")


# ============================================================
# CELL 8: Model 2 -- XGBoost (single ML model)
# Features: last 24 hours, same hour last week, 24-h rolling mean/std,
# hour/day-of-week (sin/cos), weekend flag, asset one-hot.
# Requires: Cell 6.
# ============================================================

import xgboost as xgb

def fit_xgb(feats):
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    m.fit(data.loc[tr, feats], y_tr, eval_set=[(data.loc[va, feats], y_va)], verbose=False)
    return m

xgb_model = fit_xgb(base_feats)
preds_va['XGBoost'] = xgb_model.predict(data.loc[va, base_feats])
preds_te['XGBoost'] = xgb_model.predict(data.loc[te, base_feats])
print("XGBoost trained.")


# ============================================================
# CELL 9: Model 3 -- LSTM (single deep-learning model)
# Input 1: last 24 hours of load (oldest -> newest)
# Input 2: calendar + asset features of the hour being predicted
# Requires: Cell 6.
# ============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

seq_cols = lag_cols[::-1]                # lag_24 ... lag_1
aux_cols = cal_cols + asset_cols
def nn_inputs(mask):
    return [data.loc[mask, seq_cols].values[..., None].astype('float32'),
            data.loc[mask, aux_cols].values.astype('float32')]
Xtr_nn, Xva_nn, Xte_nn = nn_inputs(tr), nn_inputs(va), nn_inputs(te)

def build_nn(kind):
    seq_in = keras.Input(shape=(N_LAGS, 1))
    aux_in = keras.Input(shape=(len(aux_cols),))
    x = seq_in
    if kind == 'cnn_lstm':
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
    x = layers.LSTM(64)(x)
    x = layers.Concatenate()([x, aux_in])
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    m = keras.Model([seq_in, aux_in], out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m

def fit_nn(kind):
    tf.random.set_seed(SEED)
    m = build_nn(kind)
    m.fit(Xtr_nn, y_tr, validation_data=(Xva_nn, y_va), epochs=EPOCHS, batch_size=128,
          verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    return m

lstm_model = fit_nn('lstm')
preds_va['LSTM'] = lstm_model.predict(Xva_nn, verbose=0).ravel()
preds_te['LSTM'] = lstm_model.predict(Xte_nn, verbose=0).ravel()
print("LSTM trained.")


# ============================================================
# CELL 10: Model 4 -- CNN-LSTM (HYBRID)
# Two 1-D convolution layers extract short local patterns (ramps, on/off
# steps) from the last 24 hours, then an LSTM learns the time dependency.
# Same inputs and training settings as the LSTM in Cell 9.
# Requires: Cells 6 and 9 (uses build_nn / fit_nn defined in Cell 9).
# ============================================================

cnn_lstm_model = fit_nn('cnn_lstm')
preds_va['CNN-LSTM'] = cnn_lstm_model.predict(Xva_nn, verbose=0).ravel()
preds_te['CNN-LSTM'] = cnn_lstm_model.predict(Xte_nn, verbose=0).ravel()
print("CNN-LSTM trained.")


# ============================================================
# CELL 11: Model 5 -- LSTM + XGBoost (HYBRID, residual correction)
# LSTM captures the temporal pattern; a small XGBoost learns what the
# LSTM gets wrong (its residual) from the tabular features, and the two
# are added: final = LSTM prediction + predicted LSTM error.
# Requires: Cells 6, 8 and 9.
# ============================================================

res_tr = y_tr - lstm_model.predict(Xtr_nn, verbose=0).ravel()
res_va = y_va - preds_va['LSTM']
res_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8,
                             early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
res_model.fit(data.loc[tr, base_feats], res_tr,
              eval_set=[(data.loc[va, base_feats], res_va)], verbose=False)
preds_va['LSTM + XGBoost'] = preds_va['LSTM'] + res_model.predict(data.loc[va, base_feats])
preds_te['LSTM + XGBoost'] = preds_te['LSTM'] + res_model.predict(data.loc[te, base_feats])
print("LSTM + XGBoost trained.")


# ============================================================
# CELL 12: Model 6 -- DBSCAN + XGBoost (HYBRID, links to our DBSCAN work)
# Same XGBoost as Cell 8, plus YESTERDAY's DBSCAN distance-to-normal,
# anomaly flag, Load Factor, PBR and VI. DBSCAN was re-fitted on training
# days only in Cell 6, so no test data leaks into these features.
# Requires: Cells 6 and 8 (uses fit_xgb from Cell 8).
# ============================================================

dbx_model = fit_xgb(dbscan_feats)
preds_va['DBSCAN + XGBoost'] = dbx_model.predict(data.loc[va, dbscan_feats])
preds_te['DBSCAN + XGBoost'] = dbx_model.predict(data.loc[te, dbscan_feats])
print("DBSCAN + XGBoost trained.")


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


# ============================================================
# CELL 16: Two-Stage Hybrid (ON/OFF classifier + load regressors)
# Stage 1: an XGBoost CLASSIFIER predicts the probability that the
#          machine will be ON in the next hour.
# Stage 2: two XGBoost REGRESSORS predict the load IF the machine is on,
#          and IF it is off.
# Final forecast = P(on) x load_if_on + (1 - P(on)) x load_if_off
# This targets the ON/OFF switching hours, where most error comes from.
# Uses the enhanced features from Cell 15.
# Requires: Cell 15.
# ============================================================

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

on_tr = (y_tr > ON_THRESHOLD).astype(int)
on_va = (y_va > ON_THRESHOLD).astype(int)
Xtr_e, Xva_e, Xte_e = data.loc[tr, enh_feats], data.loc[va, enh_feats], data.loc[te, enh_feats]

# ---- Stage 1: ON/OFF classifier ----
clf = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        early_stopping_rounds=100, eval_metric='logloss',
                        random_state=SEED, n_jobs=-1)
clf.fit(Xtr_e, on_tr, eval_set=[(Xva_e, on_va)], verbose=False)
p_on_va, p_on_te = clf.predict_proba(Xva_e)[:, 1], clf.predict_proba(Xte_e)[:, 1]
print(f"Stage 1 (ON/OFF classifier) validation AUC: {roc_auc_score(on_va, p_on_va):.4f}")

# ---- Stage 2: separate regressors for ON hours and OFF hours ----
def fit_reg(mask_tr, mask_va):
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    m.fit(Xtr_e[mask_tr], y_tr[mask_tr], eval_set=[(Xva_e[mask_va], y_va[mask_va])], verbose=False)
    return m

reg_on = fit_reg(on_tr == 1, on_va == 1)
reg_off = fit_reg(on_tr == 0, on_va == 0)

def two_stage(X, p_on):
    return p_on * reg_on.predict(X) + (1 - p_on) * reg_off.predict(X)

preds_va['Two-Stage Hybrid'] = two_stage(Xva_e, p_on_va)
preds_te['Two-Stage Hybrid'] = two_stage(Xte_e, p_on_te)
print("Two-Stage Hybrid trained.")
print("\nCell 16 complete.")


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


# ============================================================
# CELL 18: Anomaly detection -- DBSCAN vs better alternatives,
# evaluated with SYNTHETIC ANOMALY INJECTION.
# Problem: DBSCAN flags ~78% of days, far too many for a practical
# alert system, and the dataset has no labelled anomalies to test with.
# Standard solution: copy real days, inject known faults into the copies,
# and measure how many injected faults each method catches (recall)
# while flagging few normal days (false-alarm rate).
# Injected fault types (realistic for industrial assets):
#   1. Left ON overnight     -- the day's OFF hours run at 80% of the day's peak
#   2. Missed shift          -- the day's running hours drop to its base level
#   3. Load spike            -- 2 consecutive hours at 2.5x the normal load
#   4. Schedule shift        -- the whole day's pattern moves 4 hours later
#   5. Base-load increase    -- +30% of the asset's max added to every hour
# Methods compared (all per asset):
#   * DBSCAN (fixed eps, as in Cells 3-5)  -- flags whatever DBSCAN calls noise
#   * DBSCAN density score  -- same DBSCAN rule (distance to the 5th nearest
#                              day), but only the top 5% most unusual days flagged
#   * K-means distance      -- distance to nearest K-means centroid (baseline paper's method)
#   * Isolation Forest
#   * Context-aware kNN (PROPOSED) -- compares each day only with REAL days of
#     the same type (weekday vs weekend), using the actual load level (not
#     just the shape), and scores it by the average distance to its 5
#     most similar days.
# All scoring methods use the SAME 5% alert budget, so the comparison is fair.
# Requires: 'hourly', 'SELECTED_ASSETS', 'POWER_COL', 'chosen_eps',
# 'min_samples' (Cells 2-3).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from IPython.display import display

rng = np.random.default_rng(42)
ALERT_BUDGET = 0.05          # scoring methods flag the top 5% most unusual days
N_INJECT_PER_TYPE = 40       # injected copies per fault type per asset

# ---- Raw 24-hour profiles (kW) per asset-day; only complete days ----
prof = hourly.pivot_table(index=['AssetId', 'date'], columns='hour', values=POWER_COL)
prof = prof.dropna()
print(f"Complete asset-days used: {len(prof)}")

def inject(p, kind, asset_max):
    """Return a modified copy of a 24-h profile p (kW) with one fault."""
    q = p.copy()
    lo, hi = p.min(), p.max()
    off_hours = p <= lo + 0.2 * (hi - lo)      # hours the machine is (nearly) off on THIS day
    on_hours = p >= lo + 0.5 * (hi - lo)       # hours the machine is running on THIS day
    if kind == 'Left ON overnight':            # left running during its off-hours
        q[off_hours] = 0.8 * hi
    elif kind == 'Missed shift':               # running hours drop to the off/base level
        q[on_hours] = lo
    elif kind == 'Load spike':
        s = rng.integers(6, 20)
        q[s:s + 2] = 2.5 * np.maximum(p[s:s + 2], 0.1 * asset_max)
    elif kind == 'Schedule shift':
        q = np.roll(p, 4)
    elif kind == 'Base-load increase':
        q = p + 0.3 * asset_max
    return q

FAULTS = ['Left ON overnight', 'Missed shift', 'Load spike', 'Schedule shift', 'Base-load increase']

def day_features(P, asset_max):
    """Features for each day: normalized shape (as in Cell 2) + level/variation."""
    P = np.asarray(P, dtype=float)
    mx = P.max(axis=1, keepdims=True)
    shape = np.divide(P, mx, out=np.zeros_like(P), where=mx > 0)
    level = P / asset_max
    extra = np.c_[level.mean(axis=1), level.max(axis=1), level.min(axis=1), level.std(axis=1)]
    return shape, np.c_[shape, extra]

results, examples = [], {}
for a in SELECTED_ASSETS:
    P = prof.loc[a].values
    asset_max = P.max()
    wk = np.asarray(pd.to_datetime(prof.loc[a].index).dayofweek >= 5).astype(int)   # 1 = weekend
    # build injected copies (only keep injections that really change the day)
    inj_P, inj_kind, inj_wk, inj_src = [], [], [], []
    for kind in FAULTS:
        tries = 0
        while sum(k == kind for k in inj_kind) < N_INJECT_PER_TYPE and tries < 2000:
            tries += 1
            i = rng.integers(len(P))
            p = P[i]
            q = inject(p, kind, asset_max)
            if np.abs(q - p).mean() > 0.05 * asset_max:
                inj_P.append(q); inj_kind.append(kind); inj_wk.append(wk[i]); inj_src.append(i)
                examples.setdefault((a, kind), (p, q))
    inj_P, inj_kind = np.array(inj_P), np.array(inj_kind)

    all_P = np.vstack([P, inj_P])
    is_inj = np.r_[np.zeros(len(P), bool), np.ones(len(inj_P), bool)]
    kinds = np.r_[np.array(['Normal'] * len(P)), inj_kind]
    day_type = np.r_[wk, inj_wk]
    src_day = np.r_[np.arange(len(P)), inj_src]      # real day each row comes from (itself for real days)
    shape, feats = day_features(all_P, asset_max)

    flags = {}
    # DBSCAN, same eps / min_samples as Cell 3, on the normalized shape
    flags['DBSCAN'] = DBSCAN(eps=chosen_eps, min_samples=min_samples).fit(shape).labels_ == -1

    def top_budget(score):                     # threshold from the REAL days only
        thr = np.quantile(score[~is_inj], 1 - ALERT_BUDGET)
        return score > thr

    def knn_excl(Xall, ref_idx, k):
        # distance to the k nearest REAL days (rows ref_idx), always excluding the day itself
        # (real days) or the real day the fault was injected into (injected days)
        nn_ = NearestNeighbors(n_neighbors=min(k + 1, len(ref_idx))).fit(Xall[ref_idx])
        d_, j_ = nn_.kneighbors(Xall)
        out = np.zeros((len(Xall), k))
        for r in range(len(Xall)):
            keep = ref_idx[j_[r]] != src_day[r]
            out[r] = d_[r][keep][:k]
        return out

    # DBSCAN density score: distance to the 5th nearest REAL day (DBSCAN's own rule)
    real_idx = np.where(~is_inj)[0]
    dens = knn_excl(shape, real_idx, min_samples)[:, -1]
    flags['DBSCAN density score'] = top_budget(dens)

    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(feats[~is_inj])
    flags['K-means distance'] = top_budget(km.transform(feats).min(axis=1))

    iso = IsolationForest(n_estimators=300, random_state=42).fit(feats[~is_inj])
    flags['Isolation Forest'] = top_budget(-iso.score_samples(feats))

    # PROPOSED: context-aware kNN on the load level, within the same day type
    level = all_P / asset_max
    score = np.zeros(len(all_P))
    K = 5
    for t in [0, 1]:
        ref_idx = np.where(~is_inj & (day_type == t))[0]
        if len(ref_idx) <= K + 1:               # too few days of this type -> use all real days
            ref_idx = real_idx
        m_t = np.where(day_type == t)[0]
        d_all = knn_excl(level, ref_idx, K)    # excludes the day itself / its source day
        score[m_t] = d_all[m_t].mean(axis=1)
    flags['Context-aware kNN (proposed)'] = top_budget(score)

    for m, f in flags.items():
        row = {'Asset': a, 'Method': m,
               'False alarms on real days (%)': 100 * f[~is_inj].mean(),
               'Detection rate, all faults (%)': 100 * f[is_inj].mean()}
        for kind in FAULTS:
            row[kind] = 100 * f[kinds == kind].mean() if (kinds == kind).any() else np.nan
        results.append(row)

res = pd.DataFrame(results)
summary = res.groupby('Method', sort=False).mean(numeric_only=True).round(1)
summary = summary.sort_values('Detection rate, all faults (%)', ascending=False)

print("=" * 70)
print("TABLE: Anomaly detection with injected faults (average over 4 assets)")
print(f"Scoring methods use a {int(ALERT_BUDGET*100)}% alert budget; DBSCAN uses eps={chosen_eps}")
print("=" * 70)
display(summary)

print("\nTABLE: Detection rate (%) per asset")
display(res.pivot(index='Method', columns='Asset', values='Detection rate, all faults (%)').round(1))

best_m = summary.drop(index='DBSCAN', errors='ignore').index[0]
db_s = summary.loc['DBSCAN density score', 'Detection rate, all faults (%)']
print(f"\nBest method at a {int(ALERT_BUDGET*100)}% alert budget: {best_m} -- catches "
      f"{summary.loc[best_m, 'Detection rate, all faults (%)']:.1f}% of injected faults while flagging "
      f"{summary.loc[best_m, 'False alarms on real days (%)']:.1f}% of real days. "
      f"At the same budget, the DBSCAN density rule catches only {db_s:.1f}%. "
      f"Plain DBSCAN (fixed eps) flags {summary.loc['DBSCAN', 'False alarms on real days (%)']:.1f}% of real days.")

# ---- Figure: example of each injected fault (asset p_c or first available) ----
ex_asset = 'p_c' if any(k[0] == 'p_c' for k in examples) else SELECTED_ASSETS[0]
fig, axes = plt.subplots(1, len(FAULTS), figsize=(18, 3.6), sharey=True)
for ax, kind in zip(axes, FAULTS):
    if (ex_asset, kind) in examples:
        p, q = examples[(ex_asset, kind)]
        ax.plot(range(24), p, color='black', lw=2.2, ls='-', label='Original day')
        ax.plot(range(24), q, color='0.45', lw=1.8, ls=(0, (5, 2)), marker='o',
                markersize=4, markerfacecolor='white', label='Injected fault')
    ax.set_title(kind, fontweight='bold', fontsize=10)
    ax.set_xlabel("Hour of day")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(f"{POWER_COL} (kW)")
axes[0].legend(fontsize=8, frameon=False)
plt.suptitle(f"Injected fault types (example days, asset {ex_asset})", fontweight='bold')
plt.tight_layout()
plt.savefig('anomaly_fault_examples.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- Figure: detection rate vs false-alarm rate (B&W friendly, hatched bars) ----
fig, ax = plt.subplots(figsize=(11, 4.8))
xs = np.arange(len(summary))
ax.bar(xs - 0.2, summary['Detection rate, all faults (%)'], width=0.4, color='white',
       edgecolor='black', hatch='//', label='Faults detected (%) - higher is better')
ax.bar(xs + 0.2, summary['False alarms on real days (%)'], width=0.4, color='0.75',
       edgecolor='black', hatch='..', label='Real days flagged (%) - lower is better')
for x, (d, f) in zip(xs, summary[['Detection rate, all faults (%)',
                                  'False alarms on real days (%)']].values):
    ax.text(x - 0.2, d, f"{d:.0f}", ha='center', va='bottom', fontsize=9)
    ax.text(x + 0.2, f, f"{f:.0f}", ha='center', va='bottom', fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([n.replace(' (proposed)', '\n(proposed)') for n in summary.index], fontsize=9)
ax.set_ylabel("%")
ax.set_ylim(0, 110)
ax.set_title("Anomaly detection: faults caught vs real days flagged", fontweight='bold')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('anomaly_method_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

summary.to_csv('anomaly_detection_results.csv')
print("\nCell 18 complete. Saved: anomaly_fault_examples.png, "
      "anomaly_method_comparison.png, anomaly_detection_results.csv")


# ============================================================
# CELL 19: Digital Twin interface
# Builds a self-contained HTML Digital Twin of the 4 assets from the
# results of Cells 15-18 and saves it as 'digital_twin.html'.
# The twin replays the held-out TEST period hour by hour and shows, for
# the selected asset:
#   * a plant-floor view with an industrial symbol for each asset (press,
#     AHU fan, compressor, conveyor) coloured by state: running / idle / anomaly
#   * machine state (running / idle) and how long it has been in it
#   * measured demand and the next-hour forecast (Enhanced XGBoost),
#     next to the persistence baseline
#   * day health from the context-aware kNN anomaly detector (Cell 18
#     method), with the top-5% alert threshold
#   * today's load curve against the typical weekday / weekend curve
#   * LF, PBR, VI of the day, forecast accuracy, and an alert log
# Open the downloaded file in any web browser (Chrome / Edge).
# Requires: Cells 1-17 (uses final_df, scale, ON_THRESHOLD, hourly,
# daily_indices_ctx, SELECTED_ASSETS, POWER_COL).
# ============================================================

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROLE = {'p_c': 'Press drive (production)', 'ahu_a': 'Air handling unit (HVAC)',
        'comp_b': 'Air compressor', 'mh_a': 'Material handling'}
K_NN, ALERT_Q = 5, 0.95

twin = {'assets': {},
        'note': ('Replay of the chronological test period (last 20% of the timeline, never used for '
                 'training). Forecast = Enhanced XGBoost (hourly + 15-min + ON/OFF + cross-asset '
                 'features). Day health = mean distance to the 5 most similar real days of the same '
                 'day type; alert when above the 95th percentile of all real days of that asset. '
                 'Data: open industrial asset-level energy dataset, 15-min aggregates, Demand_kW.')}

prof_all = hourly.pivot_table(index=['AssetId', 'date'], columns='hour', values=POWER_COL).dropna()
ctx_idx = daily_indices_ctx.copy()
ctx_idx['date'] = pd.to_datetime(ctx_idx['date'])

for a in SELECTED_ASSETS:
    amax = float(scale[a])
    sub = final_df[final_df['AssetId'] == a].set_index('ts').sort_index()
    # continuous hourly axis over the test period (gaps stay empty)
    rng_t = pd.date_range(sub.index.min(), sub.index.max(), freq='h')
    sub = sub.reindex(rng_t)
    r2 = lambda s: [None if pd.isna(v) else round(float(v), 3) for v in s]

    # context-aware kNN day score for every complete real day of this asset
    P = prof_all.loc[a]
    dates = pd.to_datetime(P.index)
    L = P.values / amax
    wk = np.asarray(dates.dayofweek >= 5).astype(int)
    score = np.zeros(len(L))
    typical = {}
    for t in [0, 1]:
        m = wk == t
        if m.sum() > K_NN + 1:
            d, _ = NearestNeighbors(n_neighbors=K_NN + 1).fit(L[m]).kneighbors(L[m])
            score[m] = d[:, 1:].mean(axis=1)          # skip the day itself
            typical[str(t)] = np.round(np.median(P.values[m], axis=0), 3).tolist()
        elif m.any():
            d, _ = NearestNeighbors(n_neighbors=K_NN).fit(L).kneighbors(L[m])
            score[m] = d.mean(axis=1)
    for t in ['0', '1']:
        typical.setdefault(t, np.round(np.median(P.values, axis=0), 3).tolist())
    thr = float(np.quantile(score, ALERT_Q))

    idx_a = ctx_idx[ctx_idx['AssetId'] == a].set_index('date')
    first_day = rng_t[0].normalize() - pd.Timedelta(days=2)
    days = {}
    for d, s, w in zip(dates, score, wk):
        if d < first_day:
            continue
        rec = {'s': round(float(s), 4), 'w': int(w)}
        if d in idx_a.index:
            row = idx_a.loc[d]
            rec['LF'] = None if pd.isna(row['LF']) else round(float(row['LF']), 4)
            rec['PBR'] = None if pd.isna(row['PBR']) else round(float(row['PBR']), 3)
            rec['VI'] = None if pd.isna(row['VI']) else round(float(row['VI']), 3)
        days[d.strftime('%Y-%m-%d')] = rec

    t_str = [t.strftime('%Y-%m-%d %H') for t in rng_t]
    start = next((k for k, t in enumerate(t_str) if t >= '2025-10-21 10'), 48)
    start = min(max(start, 48), len(t_str) - 2)
    ok = sub['Actual'].notna()
    twin['assets'][a] = {
        'role': ROLE.get(a, a), 'max': round(amax, 3), 'onKw': round(ON_THRESHOLD * amax, 3),
        't': t_str, 'a': r2(sub['Actual']), 'f': r2(sub['Enhanced XGBoost']), 'p': r2(sub['Persistence']),
        'rmse_enh': float(np.sqrt(((sub.loc[ok, 'Actual'] - sub.loc[ok, 'Enhanced XGBoost']) ** 2).mean())),
        'rmse_pers': float(np.sqrt(((sub.loc[ok, 'Actual'] - sub.loc[ok, 'Persistence']) ** 2).mean())),
        'thr': round(thr, 4), 'days': days, 'typical': typical, 'start': start}
    n_alert = sum(1 for dd, r in days.items() if r['s'] > thr and dd >= t_str[0][:10])
    print(f"{a}: {len(t_str)} test hours, {n_alert} anomalous test days, alert threshold {thr:.4f}")

TWIN_HEAD = r"""<title>Industrial Energy Digital Twin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&display=swap">
<style>
/* Layout: control-room console. Header + asset tabs + time scrubber, then a 4-tile
   status row, then trend + day-profile panels, then indices / accuracy / alerts. */
:root {
  color-scheme: dark;
  --bg: #0d1520;          /* console background, blue-biased */
  --panel: #142031;       /* instrument panel */
  --panel-2: #1a293d;     /* raised element / track */
  --line: #25374f;        /* hairlines, grid */
  --fg: #dfe8f3;          /* primary text */
  --muted: #8195ad;       /* secondary text */
  --actual: #eef3f9;      /* measured load trace */
  --forecast: #4cc3e6;    /* model forecast trace (cyan instrument) */
  --persist: #7d8ea4;     /* persistence baseline */
  --ok: #3ecf8e;          /* state: normal / ON */
  --warn: #f2b33d;        /* state: warning */
  --crit: #f0625a;        /* state: anomaly */
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --f-body: "IBM Plex Sans", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --f-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
body { background: var(--bg); color: var(--fg); font-family: var(--f-body); font-size: 14px; }
.wrap { max-width: 1280px; margin: 0 auto; padding-inline: 20px; padding-block: 18px 28px; display: grid; gap: 14px; }
header.top { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px; }
.brand h1 { font-family: var(--f-display); font-weight: 600; font-size: 1.55rem; letter-spacing: .01em; margin: 0; text-wrap: balance; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: .82rem; }
.live { font-family: var(--f-data); font-size: .8rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.live .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 25%, transparent); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; }
.tab { font: 500 .85rem var(--f-body); color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; cursor: pointer; display: grid; text-align: left; gap: 1px; }
.tab small { font-size: .7rem; color: var(--muted); font-weight: 400; }
.tab[aria-selected="true"] { color: var(--fg); border-color: var(--forecast); background: var(--panel-2); }
.tab:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--forecast); outline-offset: 2px; }
.scrub { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
.scrub button { font: 600 .8rem var(--f-body); background: var(--forecast); color: #06202b; border: 0; border-radius: 5px; padding: 7px 12px; cursor: pointer; min-width: 64px; }
.scrub button.ghost { background: var(--panel-2); color: var(--fg); border: 1px solid var(--line); min-width: 0; }
.scrub input[type=range] { flex: 1 1 260px; accent-color: var(--forecast); min-width: 0; }
.scrub .when { font-family: var(--f-data); font-size: .92rem; min-width: 190px; font-variant-numeric: tabular-nums; }
.grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.grid-main { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(0, 1fr); gap: 12px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-width: 0; display: grid; gap: 8px; align-content: start; }
.label { font: 600 .7rem var(--f-body); text-transform: uppercase; letter-spacing: .09em; color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
.big { font-family: var(--f-data); font-size: 1.9rem; font-weight: 500; font-variant-numeric: tabular-nums; line-height: 1.1; }
.big small { font-size: .9rem; color: var(--muted); margin-left: 4px; }
.sub { color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: 6px; font: 600 .78rem var(--f-body); letter-spacing: .04em; padding: 4px 10px; border-radius: 999px; width: fit-content; }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.ok { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
.pill.off { color: var(--muted); background: var(--panel-2); }
.pill.warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 14%, transparent); }
.pill.crit { color: var(--crit); background: color-mix(in srgb, var(--crit) 16%, transparent); }
.meter { height: 8px; background: var(--panel-2); border-radius: 4px; position: relative; overflow: hidden; }
.meter > i { position: absolute; inset: 0 auto 0 0; background: var(--forecast); border-radius: 4px; }
.meter.state > i { background: var(--ok); }
.meter .mark { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--warn); }
.card.alert-crit { border-color: color-mix(in srgb, var(--crit) 60%, var(--line)); }
.chart { width: 100%; height: auto; display: block; }
.chart text { fill: var(--muted); font: 10px var(--f-data); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .75rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
dl.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; font-variant-numeric: tabular-nums; }
dl.kv dt { color: var(--muted); font-size: .8rem; }
dl.kv dd { margin: 0; font-family: var(--f-data); text-align: right; }
.log { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; max-height: 190px; overflow-y: auto; }
.log li { display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; font-size: .8rem; padding: 6px 8px; background: var(--panel-2); border-radius: 5px; cursor: pointer; border: 1px solid transparent; }
.log li:hover { border-color: var(--line); }
.log li b { font-family: var(--f-data); font-weight: 500; }
.log .sev { width: 8px; height: 8px; border-radius: 50%; background: var(--crit); }
.foot { color: var(--muted); font-size: .75rem; line-height: 1.5; max-width: 110ch; }
@media (max-width: 980px) { .grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid-main, .grid3 { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 520px) { .grid4 { grid-template-columns: minmax(0, 1fr); } .wrap { padding-inline: 16px; } .big { font-size: 1.6rem; } }
@media (prefers-reduced-motion: no-preference) { .meter > i { transition: width .25s ease; } }

/* plant floor: machine symbols tinted by state */
.floor { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.mach { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; cursor: pointer;
  display: grid; grid-template-columns: 58px minmax(0, 1fr); gap: 4px 12px; align-items: center; text-align: left; color: var(--fg); font: inherit; }
.mach[aria-selected="true"] { border-color: var(--forecast); background: var(--panel-2); box-shadow: inset 0 0 0 1px var(--forecast); }
.mach .ico { grid-row: span 3; width: 58px; height: 58px; color: var(--muted); }
.mach.s-run .ico { color: var(--ok); } .mach.s-anom .ico { color: var(--crit); }
.mach.s-anom { border-color: color-mix(in srgb, var(--crit) 55%, var(--line)); }
.mach b { font: 600 .95rem var(--f-display); letter-spacing: .02em; }
.mach small { color: var(--muted); font-size: .72rem; }
.mach .st { font: 500 .78rem var(--f-data); font-variant-numeric: tabular-nums; }
.mach.s-run .st { color: var(--ok); } .mach.s-anom .st { color: var(--crit); }
.bigico { width: 84px; height: 84px; color: var(--muted); }
.bigico.s-run { color: var(--ok); } .bigico.s-anom { color: var(--crit); }
.statebox { display: flex; align-items: center; gap: 14px; }
.ico svg, .bigico svg { width: 100%; height: 100%; display: block; overflow: visible; }
@media (prefers-reduced-motion: no-preference) {
  .s-run .spin { animation: spin 1.1s linear infinite; transform-origin: 32px 32px; transform-box: view-box; }
  .s-run .ram { animation: ram 1.2s ease-in-out infinite; }
  .s-run .item { animation: belt 2.4s linear infinite; }
  .s-run .needle { animation: needle 1.6s ease-in-out infinite alternate; transform-origin: 20px 18px; transform-box: view-box; }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes ram { 50% { transform: translateY(12px); } }
@keyframes belt { from { transform: translateX(-14px); } to { transform: translateX(14px); } }
@keyframes needle { from { transform: rotate(-35deg); } to { transform: rotate(35deg); } }
@media (max-width: 980px) { .floor { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .floor { grid-template-columns: minmax(0, 1fr); } }
</style>"""

TWIN_BODY = r"""
<div class="wrap">
  <header class="top">
    <div class="brand">
      <h1>Industrial Energy Digital Twin</h1>
      <p>Asset-level load profiling, next-hour forecasting and anomaly alerts &middot; replaying the held-out test period</p>
    </div>
    <div class="live"><span class="dot" id="liveDot"></span><span id="liveTxt">Replay paused</span></div>
  </header>

  <nav class="floor" id="tabs" role="tablist" aria-label="Plant floor: select an asset"></nav>

  <section class="scrub" aria-label="Time control">
    <button id="play" type="button">Play</button>
    <button id="stepB" class="ghost" type="button" aria-label="Back one hour">&minus;1 h</button>
    <button id="stepF" class="ghost" type="button" aria-label="Forward one hour">+1 h</button>
    <input id="slider" type="range" min="0" max="1" value="0" aria-label="Replay time">
    <div class="when" id="when">&ndash;</div>
  </section>

  <section class="grid4">
    <div class="card" id="cState">
      <div class="label"><span>Machine state</span><span id="stateName"></span></div>
      <div class="statebox"><div class="bigico" id="bigIco"></div>
        <div style="display:grid;gap:6px;min-width:0"><div id="statePill" class="pill ok">ON</div>
        <div class="sub" id="stateSub">&ndash;</div></div></div>
    </div>
    <div class="card">
      <div class="label"><span>Measured demand</span><span id="maxTxt"></span></div>
      <div class="big" id="nowKw">&ndash;<small>kW</small></div>
      <div class="meter"><i id="nowBar"></i></div>
    </div>
    <div class="card">
      <div class="label"><span>Next-hour forecast</span><span>Enhanced XGBoost</span></div>
      <div class="big" id="fcKw">&ndash;<small>kW</small></div>
      <div class="sub" id="fcSub">&ndash;</div>
    </div>
    <div class="card" id="cHealth">
      <div class="label"><span>Day health</span><span>context kNN</span></div>
      <div id="healthPill" class="pill ok">Normal</div>
      <div class="meter"><i id="healthBar"></i><span class="mark" id="healthMark"></span></div>
      <div class="sub" id="healthSub">&ndash;</div>
    </div>
  </section>

  <section class="grid-main">
    <div class="card">
      <div class="label"><span>Last 48 hours and next-hour forecast</span><span id="trendNote"></span></div>
      <svg class="chart" id="trend" viewBox="0 0 720 250" role="img" aria-label="Actual and forecast demand"></svg>
      <div class="legend">
        <span><i style="border-color:var(--actual)"></i>Measured</span>
        <span><i style="border-color:var(--forecast)"></i>Forecast (Enhanced XGBoost)</span>
        <span><i style="border-color:var(--persist);border-top-style:dashed"></i>Persistence baseline</span>
      </div>
    </div>
    <div class="card">
      <div class="label"><span>Today vs typical day</span><span id="dayType"></span></div>
      <svg class="chart" id="dayChart" viewBox="0 0 420 250" role="img" aria-label="Today's profile against the typical profile"></svg>
      <div class="legend">
        <span><i style="border-color:var(--actual)"></i>Today so far</span>
        <span><i style="border-color:var(--warn);border-top-style:dashed"></i>Typical (median)</span>
      </div>
    </div>
  </section>

  <section class="grid3">
    <div class="card">
      <div class="label"><span>Load-profile indices</span><span>today</span></div>
      <dl class="kv">
        <dt>Load factor (LF)</dt><dd id="iLF">&ndash;</dd>
        <dt>Peak-to-base ratio (PBR)</dt><dd id="iPBR">&ndash;</dd>
        <dt>Variability index (VI, kW&sup2;)</dt><dd id="iVI">&ndash;</dd>
        <dt>Energy so far</dt><dd id="iE">&ndash;</dd>
      </dl>
    </div>
    <div class="card">
      <div class="label"><span>Forecast accuracy</span><span>test period</span></div>
      <dl class="kv">
        <dt>RMSE, Enhanced XGBoost</dt><dd id="aEnh">&ndash;</dd>
        <dt>RMSE, persistence</dt><dd id="aPers">&ndash;</dd>
        <dt>Skill vs persistence</dt><dd id="aSkill">&ndash;</dd>
        <dt>Rolling 24 h MAE</dt><dd id="aRoll">&ndash;</dd>
      </dl>
    </div>
    <div class="card">
      <div class="label"><span>Anomaly alerts</span><span id="alertCount"></span></div>
      <ul class="log" id="log"></ul>
    </div>
  </section>

  <p class="foot" id="foot"></p>
</div>

<script>
const TWIN = /*__DATA__*/null;
(function () {
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (x, d = 2) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);

  const ICON = {
    press: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="9" y="5" width="46" height="8" rx="1.5"/><line x1="14" y1="13" x2="14" y2="54"/><line x1="50" y1="13" x2="50" y2="54"/>
      <g class="ram"><line x1="32" y1="13" x2="32" y2="19"/><rect x="21" y="19" width="22" height="11" rx="1" fill="currentColor" fill-opacity=".25"/></g>
      <rect x="21" y="45" width="22" height="9" rx="1"/><rect x="6" y="54" width="52" height="6" rx="1.5" fill="currentColor" fill-opacity=".25"/></svg>`,
    fan: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="5" y="5" width="54" height="54" rx="6"/><circle cx="32" cy="32" r="22" stroke-opacity=".45"/>
      <g class="spin" fill="currentColor" fill-opacity=".3">
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z"/>
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z" transform="rotate(120 32 32)"/>
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z" transform="rotate(240 32 32)"/>
        <circle cx="32" cy="32" r="3.5" fill-opacity="1"/></g></svg>`,
    compressor: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="4" y="30" width="42" height="20" rx="10" fill="currentColor" fill-opacity=".2"/>
      <line x1="12" y1="50" x2="10" y2="58"/><line x1="38" y1="50" x2="40" y2="58"/>
      <rect x="44" y="20" width="16" height="16" rx="2"/><line x1="46" y1="25" x2="58" y2="25"/><line x1="46" y1="30" x2="58" y2="30"/>
      <line x1="20" y1="26" x2="20" y2="30"/><circle cx="20" cy="18" r="8"/>
      <line class="needle" x1="20" y1="18" x2="20" y2="12" stroke-linecap="round"/></svg>`,
    conveyor: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="36" width="58" height="12" rx="6"/>
      <circle cx="9" cy="42" r="2.6"/><circle cx="24" cy="42" r="2.6"/><circle cx="40" cy="42" r="2.6"/><circle cx="55" cy="42" r="2.6"/>
      <line x1="12" y1="48" x2="10" y2="60"/><line x1="52" y1="48" x2="54" y2="60"/>
      <rect class="item" x="23" y="20" width="18" height="16" rx="1.5" fill="currentColor" fill-opacity=".25"/></svg>`
  };
  const ICON_OF = { p_c: "press", ahu_a: "fan", comp_b: "compressor", mh_a: "conveyor" };
  const iconFor = (a) => ICON[ICON_OF[a] || "press"];
  const TIDX = {}; Object.keys(TWIN.assets).forEach((a) => { const m = {}; TWIN.assets[a].t.forEach((x, k) => m[x] = k); TIDX[a] = m; });
  const assets = Object.keys(TWIN.assets);
  let cur = assets.includes("p_c") ? "p_c" : assets[0];
  let i = 0, timer = null;

  // tabs
  assets.forEach((a) => {
    const b = document.createElement("button");
    b.className = "mach"; b.type = "button"; b.setAttribute("role", "tab"); b.id = "tab-" + a;
    b.innerHTML = `<span class="ico">${iconFor(a)}</span><b>${a}</b><small>${TWIN.assets[a].role}</small><span class="st" id="st-${a}">&ndash;</span>`;
    b.onclick = () => { const t = TWIN.assets[cur].t[i]; cur = a; const k = TIDX[a][t]; if (k != null) i = k; setup(); };
    $("tabs").appendChild(b);
  });

  function setup() {
    const A = TWIN.assets[cur];
    assets.forEach((a) => $("tab-" + a).setAttribute("aria-selected", a === cur ? "true" : "false"));
    $("slider").max = A.t.length - 2;
    i = Math.min(Math.max(i, 48), A.t.length - 2);
    if (A.start != null && i === 48) i = A.start;
    $("slider").value = i;
    $("maxTxt").textContent = `max ${fmt(A.max, 1)} kW`;
    $("aEnh").textContent = fmt(A.rmse_enh, 3) + " kW";
    $("aPers").textContent = fmt(A.rmse_pers, 3) + " kW";
    $("aSkill").textContent = fmt(100 * (1 - A.rmse_enh / A.rmse_pers), 1) + " %";
    // alert log: anomalous days inside the test period
    const days = Object.entries(A.days).filter(([d, r]) => r.s > A.thr && d >= A.t[0].slice(0, 10))
      .sort((x, y) => y[1].s - x[1].s);
    $("alertCount").textContent = `${days.length} days above threshold`;
    const log = $("log"); log.innerHTML = "";
    days.slice(0, 40).forEach(([d, r]) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="sev"></span><span>${d} (${r.w ? "weekend" : "weekday"})</span><b>${fmt(r.s / A.thr, 2)}&times;</b>`;
      li.title = "Jump to this day";
      li.onclick = () => { const k = A.t.findIndex((t) => t.startsWith(d)); if (k >= 0) { i = Math.min(k + 18, A.t.length - 2); render(); } };
      log.appendChild(li);
    });
    if (!days.length) log.innerHTML = `<li><span></span><span>No anomalous days in the test period</span><b></b></li>`;
    $("foot").textContent = TWIN.note;
    render();
  }

  function hoursSame(A, k) { // hours the machine has stayed in its current state
    const on = (x) => x > A.onKw;
    let n = 0; const s = on(A.a[k]);
    for (let j = k; j >= 0 && on(A.a[j]) === s && n < 999; j--) n++;
    return n;
  }

  function render() {
    const A = TWIN.assets[cur];
    const t = A.t[i], day = t.slice(0, 10), hr = +t.slice(11, 13);
    $("when").textContent = `${t}:00 UTC`;
    $("slider").value = i;
    // state
    const on = A.a[i] > A.onKw, hs = hoursSame(A, i);
    $("statePill").className = "pill " + (on ? "ok" : "off");
    $("statePill").textContent = on ? "RUNNING" : "IDLE";
    $("stateName").textContent = A.role;
    const anomToday = A.days[day] && A.days[day].s > A.thr;
    $("bigIco").innerHTML = iconFor(cur);
    $("bigIco").className = "bigico " + (anomToday ? "s-anom" : on ? "s-run" : "");
    assets.forEach((a) => {
      const B = TWIN.assets[a], k = TIDX[a][t], el = $("tab-" + a);
      if (k == null || B.a[k] == null) { el.className = "mach"; $("st-" + a).textContent = "no data"; return; }
      const r = B.days[t.slice(0, 10)], an = r && r.s > B.thr, run = B.a[k] > B.onKw;
      el.className = "mach " + (an ? "s-anom" : run ? "s-run" : "");
      $("st-" + a).textContent = `${an ? "ANOMALY" : run ? "RUNNING" : "IDLE"} · ${fmt(B.a[k], 1)} kW`;
    });
    $("stateSub").textContent = `${on ? "Running" : "Idle"} for ${hs >= 999 ? "999+" : hs} h · ON above ${fmt(A.onKw, 1)} kW`;
    // demand
    $("nowKw").innerHTML = `${fmt(A.a[i])}<small>kW</small>`;
    $("nowBar").style.width = Math.min(100, 100 * A.a[i] / A.max) + "%";
    // forecast for next hour (index i+1)
    const nt = A.t[i + 1];
    $("fcKw").innerHTML = `${fmt(A.f[i + 1])}<small>kW</small>`;
    const err = A.a[i] - A.f[i];
    $("fcSub").textContent = `for ${nt.slice(11, 13)}:00 · persistence ${fmt(A.p[i + 1])} kW · last-hour error ${err >= 0 ? "+" : ""}${fmt(err)} kW`;
    // day health
    const r = A.days[day];
    if (r) {
      const ratio = r.s / A.thr, bad = r.s > A.thr, near = ratio > 0.8;
      $("healthPill").className = "pill " + (bad ? "crit" : near ? "warn" : "ok");
      $("healthPill").textContent = bad ? "ANOMALY" : near ? "WATCH" : "NORMAL";
      $("healthBar").style.width = Math.min(100, 100 * ratio / 2) + "%";
      $("healthBar").style.background = css(bad ? "--crit" : near ? "--warn" : "--ok");
      $("healthMark").style.left = "50%";
      $("healthSub").textContent = `score ${fmt(r.s, 3)} · alert threshold ${fmt(A.thr, 3)} (top 5 %)`;
      $("cHealth").classList.toggle("alert-crit", bad);
      $("iLF").textContent = fmt(r.LF, 3); $("iPBR").textContent = r.PBR == null ? "n/a (base < 0.5 kW)" : fmt(r.PBR, 2);
      $("iVI").textContent = fmt(r.VI, 2);
    } else {
      $("healthPill").className = "pill off"; $("healthPill").textContent = "INCOMPLETE DAY";
      $("healthBar").style.width = "0%"; $("healthSub").textContent = "Day has missing hours; not scored";
      $("cHealth").classList.remove("alert-crit");
      ["iLF", "iPBR", "iVI"].forEach((k) => $(k).textContent = "–");
    }
    // energy so far today and rolling MAE
    let e = 0, k = i; while (k >= 0 && A.t[k].startsWith(day)) { e += A.a[k]; k--; }
    $("iE").textContent = fmt(e, 1) + " kWh";
    let s = 0, n = 0; for (let j = Math.max(0, i - 23); j <= i; j++) { s += Math.abs(A.a[j] - A.f[j]); n++; }
    $("aRoll").textContent = fmt(s / n, 3) + " kW";
    drawTrend(A); drawDay(A, day);
    $("trendNote").textContent = A.a[i + 1] != null ? `actual next hour ${fmt(A.a[i + 1])} kW` : "";
  }

  function axis(svg, x0, x1, y0, y1, ymax, ticks) {
    let g = "";
    for (let k = 0; k <= ticks; k++) {
      const v = ymax * k / ticks, y = y1 - (y1 - y0) * k / ticks;
      g += `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" stroke="${css("--line")}" stroke-width="1"/>`;
      g += `<text x="${x0 - 6}" y="${y + 3}" text-anchor="end">${String(+v.toFixed(2))}</text>`;
    }
    return g;
  }
  const niceMax = (m) => { const p = Math.pow(10, Math.floor(Math.log10(m || 1))); for (const s of [1, 2, 2.5, 5, 10]) if (s * p >= m) return s * p; return 10 * p; };

  function path(xs, ys) {
    let d = "", pen = false;
    xs.forEach((x, k) => { const y = ys[k]; if (y == null || isNaN(y)) { pen = false; return; } d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; });
    return d;
  }

  function drawTrend(A) {
    const W = 720, H = 250, x0 = 44, x1 = W - 14, y0 = 12, y1 = H - 28;
    const a0 = Math.max(0, i - 47), idx = []; for (let k = a0; k <= i + 1; k++) idx.push(k);
    const vals = idx.flatMap((k) => [A.a[k], A.f[k], A.p[k]]).filter((v) => v != null);
    const ymax = niceMax(Math.max(...vals, A.onKw) * 1.05);
    const X = (k) => x0 + (x1 - x0) * (k - a0) / 48, Y = (v) => y1 - (y1 - y0) * v / ymax;
    let g = axis(null, x0, x1, y0, y1, ymax, 4);
    // time labels every 12 h
    idx.forEach((k) => { if (+A.t[k].slice(11, 13) % 12 === 0) g += `<text x="${X(k)}" y="${H - 10}" text-anchor="middle">${A.t[k].slice(5, 10)} ${A.t[k].slice(11, 13)}h</text>`; });
    // ON threshold
    g += `<line x1="${x0}" x2="${x1}" y1="${Y(A.onKw)}" y2="${Y(A.onKw)}" stroke="${css("--ok")}" stroke-opacity=".35" stroke-dasharray="2 4"/>`;
    // now marker
    g += `<line x1="${X(i)}" x2="${X(i)}" y1="${y0}" y2="${y1}" stroke="${css("--muted")}" stroke-dasharray="3 3"/>`;
    g += `<text x="${X(i) - 4}" y="${y0 + 10}" text-anchor="end">now</text>`;
    const past = idx.filter((k) => k <= i);
    g += `<path d="${path(idx.map(X), idx.map((k) => A.p[k] == null ? null : Y(A.p[k])))}" fill="none" stroke="${css("--persist")}" stroke-width="1.3" stroke-dasharray="5 4"/>`;
    g += `<path d="${path(idx.map(X), idx.map((k) => A.f[k] == null ? null : Y(A.f[k])))}" fill="none" stroke="${css("--forecast")}" stroke-width="1.8"/>`;
    g += `<path d="${path(past.map(X), past.map((k) => A.a[k] == null ? null : Y(A.a[k])))}" fill="none" stroke="${css("--actual")}" stroke-width="2.2"/>`;
    // next-hour forecast point
    g += `<circle cx="${X(i + 1)}" cy="${Y(A.f[i + 1])}" r="5" fill="${css("--forecast")}" stroke="${css("--bg")}" stroke-width="2"/>`;
    $("trend").innerHTML = g;
  }

  function drawDay(A, day) {
    const W = 420, H = 250, x0 = 40, x1 = W - 10, y0 = 12, y1 = H - 28;
    const wk = A.days[day] ? A.days[day].w : ((new Date(day + "T00:00:00Z").getUTCDay() % 6 === 0) ? 1 : 0);
    const typ = A.typical[String(wk)];
    $("dayType").textContent = wk ? "weekend reference" : "weekday reference";
    const today = new Array(24).fill(null);
    for (let k = i; k >= 0 && A.t[k].startsWith(day); k--) today[+A.t[k].slice(11, 13)] = A.a[k];
    const ymax = niceMax(Math.max(...typ, ...today.filter((v) => v != null), 0.1) * 1.08);
    const X = (h) => x0 + (x1 - x0) * h / 23, Y = (v) => y1 - (y1 - y0) * v / ymax;
    let g = axis(null, x0, x1, y0, y1, ymax, 4);
    [0, 6, 12, 18, 23].forEach((h) => g += `<text x="${X(h)}" y="${H - 10}" text-anchor="middle">${String(h).padStart(2, "0")}h</text>`);
    const hrs = [...Array(24).keys()];
    g += `<path d="${path(hrs.map(X), typ.map(Y))}" fill="none" stroke="${css("--warn")}" stroke-width="1.6" stroke-dasharray="6 4"/>`;
    const area = path(hrs.map(X), today.map((v) => v == null ? null : Y(v)));
    g += `<path d="${area}" fill="none" stroke="${css("--actual")}" stroke-width="2.2"/>`;
    $("dayChart").innerHTML = g;
  }

  function play() {
    if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; $("liveTxt").textContent = "Replay paused"; return; }
    $("play").textContent = "Pause"; $("liveTxt").textContent = "Replaying test data";
    timer = setInterval(() => { const A = TWIN.assets[cur]; i = i >= A.t.length - 2 ? 48 : i + 1; render(); }, 350);
  }
  $("play").onclick = play;
  $("stepF").onclick = () => { i = Math.min(i + 1, TWIN.assets[cur].t.length - 2); render(); };
  $("stepB").onclick = () => { i = Math.max(i - 1, 0); render(); };
  $("slider").oninput = (e) => { i = +e.target.value; render(); };
  setup();
})();
</script>
"""

html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + TWIN_HEAD + '\n</head>\n<body>\n'
        + TWIN_BODY.replace('/*__DATA__*/null', json.dumps(twin, separators=(',', ':')))
        + '\n</body>\n</html>\n')
with open('digital_twin.html', 'w', encoding='utf-8') as fh:
    fh.write(html)
print(f"\nSaved digital_twin.html ({len(html)/1e6:.1f} MB).")

# Download it from Colab and open in a browser:
try:
    from google.colab import files
    files.download('digital_twin.html')
except Exception:
    print("Not running in Colab -- open digital_twin.html from the working folder.")
print("Cell 19 complete.")


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
