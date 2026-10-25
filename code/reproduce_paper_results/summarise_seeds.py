import numpy as np, pandas as pd, glob, json
from scipy import stats
import os
S = os.environ.get('OUT_DIR', 'seeds')
info = pd.read_csv(f'{S}/test_info.csv')
y = info['y'].values; sc = info['scale'].values
files = sorted(glob.glob(f'{S}/preds_seed*.npz'))
MAPE_MIN = 0.5
def metr(pr, y, sw):
    mk = y >= MAPE_MIN
    return dict(RMSE=np.sqrt(np.mean((y-pr)**2)), MAE=np.mean(np.abs(y-pr)),
                MAPE=100*np.mean(np.abs((y[mk]-pr[mk])/y[mk])), R2=1-np.sum((y-pr)**2)/np.sum((y-y.mean())**2),
                SW=np.sqrt(np.mean((y[sw]-pr[sw])**2)), ST=np.sqrt(np.mean((y[~sw]-pr[~sw])**2)))
rows = []; P = {}
for f in files:
    sd = int(f.split('preds_seed')[1].split('.')[0]); d = np.load(f)
    pers = d['Persistence'] * sc; sw = np.abs(y - pers) > 0.2 * sc
    rp = np.sqrt(np.mean((y - pers) ** 2))
    for k in d.files:
        pr = np.clip(np.nan_to_num(d[k], nan=0.0), 0, None) * sc
        m = metr(pr, y, sw); m['Skill'] = 100 * (1 - m['RMSE'] / rp); m['model'] = k; m['seed'] = sd
        rows.append(m); P[(k, sd)] = pr
R = pd.DataFrame(rows)
agg = R.groupby('model').agg(['mean', 'std'])
agg.to_csv(f'{S}/summary.csv')
print(agg[[('RMSE','mean'),('RMSE','std'),('MAE','mean'),('R2','mean'),('SW','mean'),('Skill','mean')]].sort_values(('RMSE','mean')).round(4).to_string())

def dm_test(e1, e2, h=1):
    d = e1**2 - e2**2
    n = len(d); dbar = d.mean()
    gamma0 = np.mean((d - dbar)**2)
    var = gamma0 / n
    stat = dbar / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p
seeds = sorted(R['seed'].unique())
b2 = ['ARIMA', 'VAR', 'Random Forest', 'XGBoost', 'LSTM', 'TCN']
best = agg.loc[b2, ('RMSE', 'mean')].idxmin()
dm = []
for sd in seeds:
    e_prop = y - P[('Enhanced XGBoost', sd)]; e_b = y - P[(best, sd)]
    st, p = dm_test(e_b, e_prop); dm.append(dict(seed=sd, best=best, DM=st, p=p))
dm = pd.DataFrame(dm); dm.to_csv(f'{S}/dm.csv', index=False)
print('\nbest baseline-2:', best); print(dm.to_string())
json.dump({'best': best, 'seeds': [int(s) for s in seeds]}, open(f'{S}/meta.json', 'w'))
