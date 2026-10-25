# ============================================================
# CELL 8: Model 2 -- XGBoost (single ML model)
# Features: last 24 hours, same hour last week, 24-h rolling mean/std,
# hour/day-of-week (sin/cos), weekend flag, asset one-hot.
# Requires: Cell 6.
# ============================================================

import xgboost as xgb

def fit_xgb(feats):
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    m.fit(data.loc[tr, feats], y_tr, eval_set=[(data.loc[va, feats], y_va)], verbose=False)
    return m

xgb_model = fit_xgb(base_feats)
preds_va['XGBoost'] = xgb_model.predict(data.loc[va, base_feats])
preds_te['XGBoost'] = xgb_model.predict(data.loc[te, base_feats])
print("XGBoost trained.")
