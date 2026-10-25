# ============================================================
# CELL 10: Model 4 -- CNN-LSTM (HYBRID)
# Two 1-D convolution layers extract short local patterns (ramps, on/off
# steps) from the last 24 hours, then an LSTM learns the time dependency.
# Same inputs and training settings as the LSTM in Cell 9.
# Requires: Cells 6 and 9 (uses build_nn / fit_nn defined in Cell 9).
# ============================================================

cnn_lstm_model = fit_nn('cnn_lstm')
preds_va['CNN-LSTM'] = cnn_lstm_model.predict(Xva_nn, verbose=0).ravel()
preds_te['CNN-LSTM'] = cnn_lstm_model.predict(Xte_nn, verbose=0).ravel()
print("CNN-LSTM trained.")
