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
