# ============================================================
# CELL 7: Model 1 -- Seasonal Naive (BASELINE)
# Prediction = load at the same hour last week
# (falls back to the same hour yesterday if last week is missing).
# Requires: Cell 6.
# ============================================================

naive = data['lag_168'].fillna(data['lag_24'])
preds_va['Seasonal Naive'] = naive[va].values
preds_te['Seasonal Naive'] = naive[te].values
print("Seasonal Naive baseline ready.")
