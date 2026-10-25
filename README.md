# AI-Driven Predictive Analytics and Anomaly Detection for Asset-Level Industrial Smart Metering Using a Digital Twin Framework

Dataset, code, and output files for the conference paper:

> **AI-Driven Predictive Analytics and Anomaly Detection for Asset-Level Industrial Smart Metering Using a Digital Twin Framework**
> Bilal Ahmed, Muhammad Amad Khan, Saqib Nawaz, Adil Zohaib, Sohail Khalid, Haris Sheh Zad.

This repository is the reproducibility record for the conference paper. It holds the dataset used in the study (in reduced form), the full analysis code, and the figures and tables reported in the paper.

## What the paper does

The paper presents an asset-level digital-twin framework that integrates load profiling, short-term demand forecasting, and day-level anomaly detection for individual industrial machines. It is evaluated on an open year-long dataset of 15-minute readings from a manufacturing facility, covering four assets: a press drive (`p_c`), an air-handling unit (`ahu_a`), an air compressor (`comp_b`), and a material-handling system (`mh_a`). An enhanced XGBoost forecaster that augments standard lag features with sub-hourly readings, ON/OFF state descriptors, and cross-asset signals is compared over five seeds against the baseline-paper algorithms and deep-learning models. A context-aware nearest-neighbour rule detects injected faults at a fixed false-alarm budget, and an interactive interface visualizes machine state, forecasts, and day-level health.

## Repository layout

```
asset-level-digital-twin-smart-metering/
├── data/                              Dataset (reduced form, CC BY 4.0 — see data/DATASET_README.txt)
│   ├── four_assets_15min.parquet      Reliable 15-min windows of the four assets (108,740 rows)
│   ├── four_assets_15min.csv          Same data as CSV
│   ├── four_assets_hourly.csv         Hourly mean demand per asset (27,215 asset-hours)
│   └── DATASET_README.txt             Source, assets, columns, and cross-checks against the paper
├── code/                              Analysis code
│   ├── Conference_Notebook.ipynb      Google Colab notebook, Cells 1–21
│   ├── Conference_Code_All_Cells.py   All cells as a single script
│   ├── notebook_cells/                The notebook cells as separate files (cell1.py … cell21.py)
│   ├── reproduce_paper_results/       Scripts that produced the five-seed numbers (Tables IV and V)
│   ├── paper_results/                 Result files used in the paper
│   ├── CODE_README.md                 Cell-by-cell description and run instructions
│   └── requirements.txt
├── outputs/                           Figures and tables as reported
│   ├── figures/                       Figs 1–6 as 300-dpi PNG and vector PDF
│   ├── tables/                        Tables I–V as CSV, plus Conference_Tables.xlsx
│   └── OUTPUTS_README.txt             File-by-file description mapped to the paper
├── CITATION.cff
└── LICENSE                            MIT (code). Data keep their original CC BY 4.0 license.
```

## Dataset

The dataset is redistributed in reduced form from:

> C. Flynn, T. Murphy, J. Walsh, and D. Riordan, "An open industrial energy dataset with asset-level measurements and high-coverage 15-minute aggregates from a manufacturing facility," *Data*, vol. 11, no. 5, p. 101, 2026. doi:[10.3390/data11050101](https://doi.org/10.3390/data11050101). License: CC BY 4.0.

Only the four assets used in the paper are included, as reliable three-phase-total 15-minute windows for 31 December 2024 to 31 December 2025 (UTC). The data are shared under the same CC BY 4.0 license. See `data/DATASET_README.txt` for columns and cross-checks against the paper's row counts.

## Reproducing the results

**Notebook (Google Colab).** Open `code/Conference_Notebook.ipynb`, run the cells in order, and upload the dataset ZIP of Flynn et al. when Cell 1 asks for it.

**Paper tables (five seeds: 42, 7, 123, 2024, 31).** Extract the dataset, then:

```bash
export DATASET_DIR=".../data_csv"
export OUT_DIR=seeds
python code/reproduce_paper_results/run_five_seeds.py     # trains every model with five seeds, plus the anomaly study (~30 min)
python code/reproduce_paper_results/summarise_seeds.py    # mean and SD per model, Diebold–Mariano test
python code/reproduce_paper_results/make_tables.py        # Table3.csv (paper Table IV), Table4.csv (paper Table V)
```

`code/reproduce_paper_results/prep_cells1to5.py` is the non-interactive version of Cells 1–5. See `code/CODE_README.md` for the full cell list.

Dependencies are in `code/requirements.txt`: numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, tensorflow, and matplotlib.

## Outputs

`outputs/figures/` holds Figures 1–6 (300-dpi PNG and vector PDF), and `outputs/tables/` holds Tables I–V as CSV plus `Conference_Tables.xlsx`. `outputs/OUTPUTS_README.txt` maps each file to the paper.

## Citation

If you use this code or data, please cite the conference paper (see `CITATION.cff`) and the original dataset of Flynn et al. (2026) above.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The dataset in `data/` keeps its original CC BY 4.0 license and is attributed to Flynn et al. (2026).
