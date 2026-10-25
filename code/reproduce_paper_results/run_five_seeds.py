# Multi-seed evaluation of every learned forecasting model + anomaly detectors, and a
# Diebold-Mariano test. Saves per-seed test predictions and summary tables.
import matplotlib; matplotlib.use('Agg')
import os, json, time, warnings
warnings.filterwarnings('ignore')
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, '..', 'notebook_cells')
def cell(n): return open(os.path.join(CELLS, f'cell{n}.py')).read()
exec(open(os.path.join(HERE, 'prep_cells1to5.py')).read())
exec(cell(6))
exec(cell(7))
src9 = cell(9)
src20 = cell(20)
OUT = os.environ.get('OUT_DIR', 'seeds'); os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 7, 123, 2024, 31]
fixed_done = False
for sd in SEEDS:
    t = time.time()
    SEED = sd
    import random; random.seed(sd); np.random.seed(sd); tf.random.set_seed(sd); os.environ['PYTHONHASHSEED'] = str(sd)
    for c in [8]:
        exec(cell(c))
    exec(src9)                                   # LSTM (defines Xtr_nn, fit_nn ...)
    for c in [10, 11, 12, 13, 15, 16]:
        exec(cell(c))
    if not fixed_done:                           # K-means profile, ARIMA, VAR are deterministic
        exec(src20[:src20.index("# ---------- 4) Random Forest")])
        fixed = {k: np.asarray(preds_te[k]).copy() for k in ['K-means profile', 'ARIMA', 'VAR']}
        fixed_done = True
    else:
        preds_te.update(fixed)
    exec(src20[src20.index("# ---------- 4) Random Forest"):src20.index("# ---------- Final table")])
    np.savez(f'{OUT}/preds_seed{sd}.npz', **{k: np.asarray(v, dtype=float) for k, v in preds_te.items()})
    print(f'seed {sd} done in {time.time()-t:.0f}s', flush=True)

# shared test info
info = data.loc[te, ['AssetId', 'ts']].reset_index(drop=True)
info['scale'] = info['AssetId'].map(scale).values
info['y'] = y_te * info['scale'].values
info.to_csv(f'{OUT}/test_info.csv', index=False)
print('forecast part done', round(time.time() - T0), 's', flush=True)

# ---------------- anomaly detection with 5 injection seeds ----------------
src18 = cell(18)
src18 = src18[:src18.index("res = pd.DataFrame(results)")]     # computation only
an = []
for sd in SEEDS:
    src = src18.replace("rng = np.random.default_rng(42)", f"rng = np.random.default_rng({sd})") \
               .replace("IsolationForest(n_estimators=300, random_state=42)", f"IsolationForest(n_estimators=300, random_state={sd})") \
               .replace("KMeans(n_clusters=2, random_state=42, n_init=10)", f"KMeans(n_clusters=2, random_state={sd}, n_init=10)")
    exec(src)
    r = pd.DataFrame(results); r['seed'] = sd; an.append(r)
    print('anomaly seed', sd, flush=True)
pd.concat(an).to_csv(f'{OUT}/anomaly_seeds.csv', index=False)
print('ALL DONE', round(time.time() - T0), 's', flush=True)
