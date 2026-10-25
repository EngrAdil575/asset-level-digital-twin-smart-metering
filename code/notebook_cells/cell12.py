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
