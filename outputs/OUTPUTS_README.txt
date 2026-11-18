Output files produced by code/asset_level_pipeline.py (eight assets).

figures/
  fig1_framework.png / .pdf     Framework / methodology diagram
  fig2_faults.png / .pdf        The injected fault types
  fig3_profiles.png / .pdf      Mean weekday load profiles per asset
  fig4_forecast.png / .pdf      Next-hour forecast example
  fig5_switching.png / .pdf     Steady vs switching-hour error
  fig6_detection.png / .pdf     Anomaly-detection results
  fig7_twin.png                 Digital-twin interface view

tables/
  assets.csv                    Assets and profile indices
  clustering.csv                Clustering quality
  per_asset_rmse.csv            Per-asset next-hour RMSE (persistence, XGBoost, proposed)
  forecast_summary.csv          Forecast accuracy across all compared models
  dm_tests.csv                  Diebold-Mariano test results
  anomaly_summary.csv           Anomaly detection per fault type and method
  anomaly_all_rows.csv          Full per-day anomaly scores
  numbers.json                  All scalar numbers used in the text
  claims_check.txt              Each statement in the paper text and whether this run supports it

latex_parts/
  numbers.tex, tab_*.tex        LaTeX snippets for the numbers and tables

Note: these files come from a single run of the pipeline. Re-run with QUICK=False
over the five seeds to regenerate the five-seed paper numbers; claims_check.txt
reports which statements the current run supports.
