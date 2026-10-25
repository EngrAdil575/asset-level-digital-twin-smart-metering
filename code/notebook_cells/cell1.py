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