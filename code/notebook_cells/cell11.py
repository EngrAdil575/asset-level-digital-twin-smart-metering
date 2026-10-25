# ============================================================
# CELL 11: Model 5 -- LSTM + XGBoost (HYBRID, residual correction)
# LSTM captures the temporal pattern; a small XGBoost learns what the
# LSTM gets wrong (its residual) from the tabular features, and the two
# are added: final = LSTM prediction + predicted LSTM error.
# Requires: Cells 6, 8 and 9.
# ============================================================

res_tr = y_tr - lstm_model.predict(Xtr_nn, verbose=0).ravel()
res_va = y_va - preds_va['LSTM']
res_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8,
                             early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
res_model.fit(data.loc[tr, base_feats], res_tr,
              eval_set=[(data.loc[va, base_feats], res_va)], verbose=False)
preds_va['LSTM + XGBoost'] = preds_va['LSTM'] + res_model.predict(data.loc[va, base_feats])
preds_te['LSTM + XGBoost'] = preds_te['LSTM'] + res_model.predict(data.loc[te, base_feats])
print("LSTM + XGBoost trained.")
