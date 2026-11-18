# Code

`asset_level_pipeline.py` is the complete analysis as a single Google Colab cell. It covers eight assets: next-hour forecasting, day-level anomaly detection with seven injected fault types plus an unseen-fault test (mixtures of the injected faults), the digital twin, and all tables and 600-dpi figures.

## Run it

1. In Colab: Runtime → Change runtime type → T4 GPU (the GPU only speeds up the neural baselines).
2. Put the dataset ZIP in Google Drive as `MyDrive/industrial_energy_dataset.zip`, or change `DATA_ZIP` at the top. The original dataset ZIP (data_csv / data_parquet) or a ZIP holding `all_assets_rel.parquet` both work.
3. Paste the file into one cell and run it. At the end `results.zip` (figures, tables, LaTeX parts, twin, and the code) is produced. If Colab disconnects, run the cell again — finished steps are cached in the output folder.

## Flags (top of the file)

- `QUICK` — `True` runs one seed for a fast check; `False` runs the five seeds used in the paper (42, 7, 123, 2024, 31), about 1–2 hours.
- `RUN_NN` — toggles the neural baselines (LSTM, CNN-LSTM, TCN) and the LSTM+XGBoost hybrid and weighted ensemble.
- `ASSETS` / `ROLE` — the eight assets and their roles.
- `MODEL_NAME` — `SA-XGBoost`, the switching-aware XGBoost forecaster (the proposed model).

The `PIPELINE_TEST` environment switch is a developer-only mechanical test that runs with stand-in libraries; leave it unset for real runs.
