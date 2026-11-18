# AI-Driven Predictive Analytics and Anomaly Detection for Asset-Level Industrial Smart Metering Using a Digital Twin Framework

Dataset, code, output files, and digital twin for the conference paper:

> **AI-Driven Predictive Analytics and Anomaly Detection for Asset-Level Industrial Smart Metering Using a Digital Twin Framework**
> Bilal Ahmed, Muhammad Amad Khan, Saqib Nawaz, Adil Zohaib, Sohail Khalid, Haris Sheh Zad.

This repository is the reproducibility record for the conference paper. It holds the dataset used in the study (in reduced form), the full analysis pipeline, the figures and tables, and the offline digital twin.

## What the paper does

The paper presents an asset-level digital-twin framework that integrates load profiling, short-term demand forecasting, and day-level anomaly detection for individual industrial machines. It is evaluated on an open year-long dataset of 15-minute readings from a manufacturing facility, covering eight assets: a press drive (`p_c`), an air-handling unit (`ahu_a`), an air compressor (`comp_b`), a material-handling system (`mh_a`), a main press (`p_h`), a utility circuit (`u_b`), an extraction fan (`ex_a`), and a process mixer (`mix_c`). A switching-aware XGBoost forecaster (SA-XGBoost) that augments standard lag features with sub-hourly readings, ON/OFF state descriptors, and cross-asset signals is compared over five seeds against classical baselines and deep-learning models (LSTM, CNN-LSTM, TCN, and hybrid/ensemble variants). Day-level anomaly detection is evaluated against seven injected fault types and an unseen-fault test built from mixtures of those faults, at a fixed false-alarm budget, and an offline digital twin visualizes machine state, forecasts, and day-level health.

## Repository layout

```
asset-level-digital-twin-smart-metering/
├── data/                              Dataset (reduced form, CC BY 4.0 — see data/DATASET_README.txt)
│   ├── four_assets_15min.parquet      Reliable 15-min windows of the first four assets
│   ├── four_assets_15min.csv          Same data as CSV
│   ├── four_assets_hourly.csv         Hourly mean demand per asset
│   └── DATASET_README.txt             Source, assets, columns, and cross-checks
├── code/
│   ├── asset_level_pipeline.py  Complete pipeline in one Google Colab cell (eight assets)
│   ├── CODE_README.md                  How to run it and what the flags do
│   └── requirements.txt
├── outputs/                           Figures, tables, and LaTeX parts from one run of the pipeline
│   ├── figures/                       Figs 1–7 (PNG, with vector PDFs for Figs 1–6)
│   ├── tables/                        Result CSVs, numbers.json, and claims_check.txt
│   ├── latex_parts/                   LaTeX snippets for the numbers and tables
│   └── OUTPUTS_README.txt             File-by-file description
├── digital_twin/
│   └── Asset_Level_Digital_Twin.html   Self-contained offline digital twin (open in a browser)
├── CITATION.cff
└── LICENSE                            MIT (code). Data keep their original CC BY 4.0 license.
```

## Dataset

The dataset is redistributed in reduced form from:

> C. Flynn, T. Murphy, J. Walsh, and D. Riordan, "An open industrial energy dataset with asset-level measurements and high-coverage 15-minute aggregates from a manufacturing facility," *Data*, vol. 11, no. 5, p. 101, 2026. doi:[10.3390/data11050101](https://doi.org/10.3390/data11050101). License: CC BY 4.0.

The reduced copy in `data/` holds the first four assets as reliable three-phase-total 15-minute windows (31 December 2024 to 31 December 2025, UTC), under the same CC BY 4.0 license. The full pipeline uses all eight assets and reads the original dataset ZIP (or an `all_assets_rel.parquet`); obtain it from the source above. See `data/DATASET_README.txt`.

## Reproducing the results

Open `code/asset_level_pipeline.py` as a single cell in Google Colab (Runtime → T4 GPU), place the dataset ZIP in Google Drive as `MyDrive/industrial_energy_dataset.zip` (or edit `DATA_ZIP`), and run the cell. It prepares the data, fits the forecaster and all baselines, runs the anomaly study with the seven injected faults and the unseen-fault test, builds the digital twin, and writes the figures, tables, and LaTeX parts, downloading them as `results.zip`.

Flags at the top control scope: `QUICK = True` runs one seed for a fast check; `QUICK = False` runs the five seeds used in the paper (42, 7, 123, 2024, 31) and takes about 1–2 hours. `RUN_NN` toggles the neural baselines. See `code/CODE_README.md`.

Dependencies are in `code/requirements.txt`: numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, pyarrow, matplotlib, and tensorflow (neural baselines).

## Outputs

`outputs/figures/` holds Figures 1–7, `outputs/tables/` holds the result CSVs plus `numbers.json` and `claims_check.txt`, and `outputs/latex_parts/` holds the LaTeX snippets. `outputs/OUTPUTS_README.txt` maps each file. `claims_check.txt` lists each statement used in the paper text and whether the committed run supports it.

## Offline digital twin

`digital_twin/Asset_Level_Digital_Twin.html` is a self-contained offline interface — open it in any browser, no server or internet needed. It shows, per asset, the operating state, the measured demand, the next-hour forecast, the day-level health, and the recent profile against the typical weekday.

## Citation

If you use this code or data, please cite the conference paper (see `CITATION.cff`) and the original dataset of Flynn et al. (2026) above.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The dataset in `data/` keeps its original CC BY 4.0 license and is attributed to Flynn et al. (2026).
