# Code

`asset_level_pipeline.py` is the complete analysis as a single Google Colab cell. It covers eight assets: next-hour forecasting, day-level anomaly detection with seven injected fault types plus an unseen-fault test (mixtures of the injected faults), the metering view (daily energy, 15-minute peak demand, excess energy of alert days), the digital twin, and all tables and 600-dpi figures.

## Run it

1. In Colab: paste the whole file into one cell and Runtime → Run all (a GPU is optional and only speeds up the neural baselines).
2. The cell downloads `eight_assets_15min.parquet` automatically from this repository's `data/` folder (`DATA_URL` at the top). Without internet, upload that file when asked, or set `DATA_FILE` to its path.
3. At the end `results.zip` (figures, tables, LaTeX parts, twin, and the code) is produced. If Colab disconnects, run the cell again — finished steps are cached in the output folder.

## Flags (top of the file)

- `QUICK` — `False` (the default) runs the five seeds used in the paper (42, 7, 123, 2024, 31), about 1–2 hours; `True` runs one seed for a fast check.
- `RUN_NN` — toggles the neural baselines (LSTM, CNN-LSTM, TCN) and the LSTM+XGBoost hybrid and weighted ensemble.
- `ASSETS` / `ROLE` — the eight assets and their roles.
- `MODEL_NAME` — `SA-XGBoost`, the switching-aware XGBoost forecaster (the proposed model).

The `PIPELINE_TEST` environment switch is a developer-only mechanical test that runs with stand-in libraries; leave it unset for real runs.
