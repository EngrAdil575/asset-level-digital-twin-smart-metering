# ============================================================
# CELL 9: Model 3 -- LSTM (single deep-learning model)
# Input 1: last 24 hours of load (oldest -> newest)
# Input 2: calendar + asset features of the hour being predicted
# Requires: Cell 6.
# ============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

seq_cols = lag_cols[::-1]                # lag_24 ... lag_1
aux_cols = cal_cols + asset_cols
def nn_inputs(mask):
    return [data.loc[mask, seq_cols].values[..., None].astype('float32'),
            data.loc[mask, aux_cols].values.astype('float32')]
Xtr_nn, Xva_nn, Xte_nn = nn_inputs(tr), nn_inputs(va), nn_inputs(te)

def build_nn(kind):
    seq_in = keras.Input(shape=(N_LAGS, 1))
    aux_in = keras.Input(shape=(len(aux_cols),))
    x = seq_in
    if kind == 'cnn_lstm':
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
    x = layers.LSTM(64)(x)
    x = layers.Concatenate()([x, aux_in])
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    m = keras.Model([seq_in, aux_in], out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return m

def fit_nn(kind):
    tf.random.set_seed(SEED)
    m = build_nn(kind)
    m.fit(Xtr_nn, y_tr, validation_data=(Xva_nn, y_va), epochs=EPOCHS, batch_size=128,
          verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    return m

lstm_model = fit_nn('lstm')
preds_va['LSTM'] = lstm_model.predict(Xva_nn, verbose=0).ravel()
preds_te['LSTM'] = lstm_model.predict(Xte_nn, verbose=0).ravel()
print("LSTM trained.")
