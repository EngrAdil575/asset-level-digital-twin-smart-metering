# Code: AI-Enabled Asset-Level Digital Twin for Industrial Load Profiling, Short-Term Load Forecasting, and Anomaly Detection

Analysis code of the conference paper. Data: open dataset of Flynn et al., "An open industrial energy dataset with
asset-level measurements and high-coverage 15-minute aggregates from a manufacturing facility," *Data* 11(5):101, 2026
(CC BY 4.0). Assets: p_c (press drive), ahu_a (air-handling unit), comp_b (air compressor), mh_a (material handling).

## Contents
| Path | What it is |
|---|---|
| `Conference_Notebook.ipynb` | Google Colab notebook, Cells 1–21 (same code as `notebook_cells/`) |
| `notebook_cells/cell1.py … cell21.py` | The notebook cells as separate files |
| `reproduce_paper_results/` | Scripts that produced the five-seed numbers of Table IV and Table V |
| `paper_results/` | The result files used in the paper (Table3.csv = Table IV, Table4.csv = Table V, summary.csv, dm.csv, anomaly_seeds.csv) |

### Notebook cells
1 Load and clean the dataset · 2 K-means (baseline) vs DBSCAN · 3 DBSCAN eps sweep · 4 Days flagged by DBSCAN with LF/PBR/VI ·
5 DBSCAN-based anomaly function · 6 Forecasting setup (split and features) · 7 Seasonal naive · 8 XGBoost · 9 LSTM ·
10 CNN-LSTM · 11 LSTM + XGBoost · 12 DBSCAN + XGBoost · 13 Weighted ensemble and accuracy tables · 14 Comparison graphs ·
15 Persistence and Enhanced XGBoost (proposed) · 16 Two-stage hybrid · 17 Final forecasting comparison ·
18 Anomaly detection with synthetic fault injection · 19 Digital twin interface · 20 Baseline-paper algorithms
(K-means profile, ARIMA, VAR, random forest, TCN) · 21 Five-seed evaluation and Diebold–Mariano test

## How to run
**Notebook (Google Colab):** open `Conference_Notebook.ipynb`, run the cells in order, and upload the dataset ZIP of
Flynn et al. when Cell 1 asks for it.

**Paper tables (five seeds 42, 7, 123, 2024, 31):** extract the dataset ZIP, then
```
export DATASET_DIR=".../Industrial Asset-Level Electrical Energy Dataset from a Manufacturing Facility (15-Minute Aggregates)/data_csv"
export OUT_DIR=seeds
python reproduce_paper_results/run_five_seeds.py     # trains every model with five seeds, anomaly study with five seeds (~30 min)
python reproduce_paper_results/summarise_seeds.py    # mean and SD per model, Diebold-Mariano test
python reproduce_paper_results/make_tables.py        # Table3.csv (paper Table IV) and Table4.csv (paper Table V)
```
`prep_cells1to5.py` is the non-interactive version of Cells 1–5 (loads the four assets directly instead of the upload step).

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, tensorflow, matplotlib.

## Reproducibility note
Re-running `reproduce_paper_results/` gives the same numbers as `paper_results/` for every deterministic and tree-based
model (K-means profile, ARIMA, VAR, random forest, XGBoost, TCN, DBSCAN + XGBoost, two-stage hybrid, proposed enhanced
XGBoost), the same Diebold–Mariano results, and the same Table V. The LSTM-based models (LSTM, CNN-LSTM, LSTM + XGBoost,
weighted ensemble) can differ in the third decimal between runs because TensorFlow training on CPU/GPU is not bit-for-bit
deterministic.
