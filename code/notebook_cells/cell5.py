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