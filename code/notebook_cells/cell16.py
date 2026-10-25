# ============================================================
# CELL 16: Two-Stage Hybrid (ON/OFF classifier + load regressors)
# Stage 1: an XGBoost CLASSIFIER predicts the probability that the
#          machine will be ON in the next hour.
# Stage 2: two XGBoost REGRESSORS predict the load IF the machine is on,
#          and IF it is off.
# Final forecast = P(on) x load_if_on + (1 - P(on)) x load_if_off
# This targets the ON/OFF switching hours, where most error comes from.
# Uses the enhanced features from Cell 15.
# Requires: Cell 15.
# ============================================================

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

on_tr = (y_tr > ON_THRESHOLD).astype(int)
on_va = (y_va > ON_THRESHOLD).astype(int)
Xtr_e, Xva_e, Xte_e = data.loc[tr, enh_feats], data.loc[va, enh_feats], data.loc[te, enh_feats]

# ---- Stage 1: ON/OFF classifier ----
clf = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        early_stopping_rounds=100, eval_metric='logloss',
                        random_state=SEED, n_jobs=-1)
clf.fit(Xtr_e, on_tr, eval_set=[(Xva_e, on_va)], verbose=False)
p_on_va, p_on_te = clf.predict_proba(Xva_e)[:, 1], clf.predict_proba(Xte_e)[:, 1]
print(f"Stage 1 (ON/OFF classifier) validation AUC: {roc_auc_score(on_va, p_on_va):.4f}")

# ---- Stage 2: separate regressors for ON hours and OFF hours ----
def fit_reg(mask_tr, mask_va):
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    m.fit(Xtr_e[mask_tr], y_tr[mask_tr], eval_set=[(Xva_e[mask_va], y_va[mask_va])], verbose=False)
    return m

reg_on = fit_reg(on_tr == 1, on_va == 1)
reg_off = fit_reg(on_tr == 0, on_va == 0)

def two_stage(X, p_on):
    return p_on * reg_on.predict(X) + (1 - p_on) * reg_off.predict(X)

preds_va['Two-Stage Hybrid'] = two_stage(Xva_e, p_on_va)
preds_te['Two-Stage Hybrid'] = two_stage(Xte_e, p_on_te)
print("Two-Stage Hybrid trained.")
print("\nCell 16 complete.")
