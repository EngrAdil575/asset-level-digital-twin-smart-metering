# ============================================================
# CELL 18: Anomaly detection -- DBSCAN vs better alternatives,
# evaluated with SYNTHETIC ANOMALY INJECTION.
# Problem: DBSCAN flags ~78% of days, far too many for a practical
# alert system, and the dataset has no labelled anomalies to test with.
# Standard solution: copy real days, inject known faults into the copies,
# and measure how many injected faults each method catches (recall)
# while flagging few normal days (false-alarm rate).
# Injected fault types (realistic for industrial assets):
#   1. Left ON overnight     -- the day's OFF hours run at 80% of the day's peak
#   2. Missed shift          -- the day's running hours drop to its base level
#   3. Load spike            -- 2 consecutive hours at 2.5x the normal load
#   4. Schedule shift        -- the whole day's pattern moves 4 hours later
#   5. Base-load increase    -- +30% of the asset's max added to every hour
# Methods compared (all per asset):
#   * DBSCAN (fixed eps, as in Cells 3-5)  -- flags whatever DBSCAN calls noise
#   * DBSCAN density score  -- same DBSCAN rule (distance to the 5th nearest
#                              day), but only the top 5% most unusual days flagged
#   * K-means distance      -- distance to nearest K-means centroid (baseline paper's method)
#   * Isolation Forest
#   * Context-aware kNN (PROPOSED) -- compares each day only with REAL days of
#     the same type (weekday vs weekend), using the actual load level (not
#     just the shape), and scores it by the average distance to its 5
#     most similar days.
# All scoring methods use the SAME 5% alert budget, so the comparison is fair.
# Requires: 'hourly', 'SELECTED_ASSETS', 'POWER_COL', 'chosen_eps',
# 'min_samples' (Cells 2-3).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from IPython.display import display

rng = np.random.default_rng(42)
ALERT_BUDGET = 0.05          # scoring methods flag the top 5% most unusual days
N_INJECT_PER_TYPE = 40       # injected copies per fault type per asset

# ---- Raw 24-hour profiles (kW) per asset-day; only complete days ----
prof = hourly.pivot_table(index=['AssetId', 'date'], columns='hour', values=POWER_COL)
prof = prof.dropna()
print(f"Complete asset-days used: {len(prof)}")

def inject(p, kind, asset_max):
    """Return a modified copy of a 24-h profile p (kW) with one fault."""
    q = p.copy()
    lo, hi = p.min(), p.max()
    off_hours = p <= lo + 0.2 * (hi - lo)      # hours the machine is (nearly) off on THIS day
    on_hours = p >= lo + 0.5 * (hi - lo)       # hours the machine is running on THIS day
    if kind == 'Left ON overnight':            # left running during its off-hours
        q[off_hours] = 0.8 * hi
    elif kind == 'Missed shift':               # running hours drop to the off/base level
        q[on_hours] = lo
    elif kind == 'Load spike':
        s = rng.integers(6, 20)
        q[s:s + 2] = 2.5 * np.maximum(p[s:s + 2], 0.1 * asset_max)
    elif kind == 'Schedule shift':
        q = np.roll(p, 4)
    elif kind == 'Base-load increase':
        q = p + 0.3 * asset_max
    return q

FAULTS = ['Left ON overnight', 'Missed shift', 'Load spike', 'Schedule shift', 'Base-load increase']

def day_features(P, asset_max):
    """Features for each day: normalized shape (as in Cell 2) + level/variation."""
    P = np.asarray(P, dtype=float)
    mx = P.max(axis=1, keepdims=True)
    shape = np.divide(P, mx, out=np.zeros_like(P), where=mx > 0)
    level = P / asset_max
    extra = np.c_[level.mean(axis=1), level.max(axis=1), level.min(axis=1), level.std(axis=1)]
    return shape, np.c_[shape, extra]

results, examples = [], {}
for a in SELECTED_ASSETS:
    P = prof.loc[a].values
    asset_max = P.max()
    wk = np.asarray(pd.to_datetime(prof.loc[a].index).dayofweek >= 5).astype(int)   # 1 = weekend
    # build injected copies (only keep injections that really change the day)
    inj_P, inj_kind, inj_wk, inj_src = [], [], [], []
    for kind in FAULTS:
        tries = 0
        while sum(k == kind for k in inj_kind) < N_INJECT_PER_TYPE and tries < 2000:
            tries += 1
            i = rng.integers(len(P))
            p = P[i]
            q = inject(p, kind, asset_max)
            if np.abs(q - p).mean() > 0.05 * asset_max:
                inj_P.append(q); inj_kind.append(kind); inj_wk.append(wk[i]); inj_src.append(i)
                examples.setdefault((a, kind), (p, q))
    inj_P, inj_kind = np.array(inj_P), np.array(inj_kind)

    all_P = np.vstack([P, inj_P])
    is_inj = np.r_[np.zeros(len(P), bool), np.ones(len(inj_P), bool)]
    kinds = np.r_[np.array(['Normal'] * len(P)), inj_kind]
    day_type = np.r_[wk, inj_wk]
    src_day = np.r_[np.arange(len(P)), inj_src]      # real day each row comes from (itself for real days)
    shape, feats = day_features(all_P, asset_max)

    flags = {}
    # DBSCAN, same eps / min_samples as Cell 3, on the normalized shape
    flags['DBSCAN'] = DBSCAN(eps=chosen_eps, min_samples=min_samples).fit(shape).labels_ == -1

    def top_budget(score):                     # threshold from the REAL days only
        thr = np.quantile(score[~is_inj], 1 - ALERT_BUDGET)
        return score > thr

    def knn_excl(Xall, ref_idx, k):
        # distance to the k nearest REAL days (rows ref_idx), always excluding the day itself
        # (real days) or the real day the fault was injected into (injected days)
        nn_ = NearestNeighbors(n_neighbors=min(k + 1, len(ref_idx))).fit(Xall[ref_idx])
        d_, j_ = nn_.kneighbors(Xall)
        out = np.zeros((len(Xall), k))
        for r in range(len(Xall)):
            keep = ref_idx[j_[r]] != src_day[r]
            out[r] = d_[r][keep][:k]
        return out

    # DBSCAN density score: distance to the 5th nearest REAL day (DBSCAN's own rule)
    real_idx = np.where(~is_inj)[0]
    dens = knn_excl(shape, real_idx, min_samples)[:, -1]
    flags['DBSCAN density score'] = top_budget(dens)

    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(feats[~is_inj])
    flags['K-means distance'] = top_budget(km.transform(feats).min(axis=1))

    iso = IsolationForest(n_estimators=300, random_state=42).fit(feats[~is_inj])
    flags['Isolation Forest'] = top_budget(-iso.score_samples(feats))

    # PROPOSED: context-aware kNN on the load level, within the same day type
    level = all_P / asset_max
    score = np.zeros(len(all_P))
    K = 5
    for t in [0, 1]:
        ref_idx = np.where(~is_inj & (day_type == t))[0]
        if len(ref_idx) <= K + 1:               # too few days of this type -> use all real days
            ref_idx = real_idx
        m_t = np.where(day_type == t)[0]
        d_all = knn_excl(level, ref_idx, K)    # excludes the day itself / its source day
        score[m_t] = d_all[m_t].mean(axis=1)
    flags['Context-aware kNN (proposed)'] = top_budget(score)

    for m, f in flags.items():
        row = {'Asset': a, 'Method': m,
               'False alarms on real days (%)': 100 * f[~is_inj].mean(),
               'Detection rate, all faults (%)': 100 * f[is_inj].mean()}
        for kind in FAULTS:
            row[kind] = 100 * f[kinds == kind].mean() if (kinds == kind).any() else np.nan
        results.append(row)

res = pd.DataFrame(results)
summary = res.groupby('Method', sort=False).mean(numeric_only=True).round(1)
summary = summary.sort_values('Detection rate, all faults (%)', ascending=False)

print("=" * 70)
print("TABLE: Anomaly detection with injected faults (average over 4 assets)")
print(f"Scoring methods use a {int(ALERT_BUDGET*100)}% alert budget; DBSCAN uses eps={chosen_eps}")
print("=" * 70)
display(summary)

print("\nTABLE: Detection rate (%) per asset")
display(res.pivot(index='Method', columns='Asset', values='Detection rate, all faults (%)').round(1))

best_m = summary.drop(index='DBSCAN', errors='ignore').index[0]
db_s = summary.loc['DBSCAN density score', 'Detection rate, all faults (%)']
print(f"\nBest method at a {int(ALERT_BUDGET*100)}% alert budget: {best_m} -- catches "
      f"{summary.loc[best_m, 'Detection rate, all faults (%)']:.1f}% of injected faults while flagging "
      f"{summary.loc[best_m, 'False alarms on real days (%)']:.1f}% of real days. "
      f"At the same budget, the DBSCAN density rule catches only {db_s:.1f}%. "
      f"Plain DBSCAN (fixed eps) flags {summary.loc['DBSCAN', 'False alarms on real days (%)']:.1f}% of real days.")

# ---- Figure: example of each injected fault (asset p_c or first available) ----
ex_asset = 'p_c' if any(k[0] == 'p_c' for k in examples) else SELECTED_ASSETS[0]
fig, axes = plt.subplots(1, len(FAULTS), figsize=(18, 3.6), sharey=True)
for ax, kind in zip(axes, FAULTS):
    if (ex_asset, kind) in examples:
        p, q = examples[(ex_asset, kind)]
        ax.plot(range(24), p, color='black', lw=2.2, ls='-', label='Original day')
        ax.plot(range(24), q, color='0.45', lw=1.8, ls=(0, (5, 2)), marker='o',
                markersize=4, markerfacecolor='white', label='Injected fault')
    ax.set_title(kind, fontweight='bold', fontsize=10)
    ax.set_xlabel("Hour of day")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(f"{POWER_COL} (kW)")
axes[0].legend(fontsize=8, frameon=False)
plt.suptitle(f"Injected fault types (example days, asset {ex_asset})", fontweight='bold')
plt.tight_layout()
plt.savefig('anomaly_fault_examples.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- Figure: detection rate vs false-alarm rate (B&W friendly, hatched bars) ----
fig, ax = plt.subplots(figsize=(11, 4.8))
xs = np.arange(len(summary))
ax.bar(xs - 0.2, summary['Detection rate, all faults (%)'], width=0.4, color='white',
       edgecolor='black', hatch='//', label='Faults detected (%) - higher is better')
ax.bar(xs + 0.2, summary['False alarms on real days (%)'], width=0.4, color='0.75',
       edgecolor='black', hatch='..', label='Real days flagged (%) - lower is better')
for x, (d, f) in zip(xs, summary[['Detection rate, all faults (%)',
                                  'False alarms on real days (%)']].values):
    ax.text(x - 0.2, d, f"{d:.0f}", ha='center', va='bottom', fontsize=9)
    ax.text(x + 0.2, f, f"{f:.0f}", ha='center', va='bottom', fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([n.replace(' (proposed)', '\n(proposed)') for n in summary.index], fontsize=9)
ax.set_ylabel("%")
ax.set_ylim(0, 110)
ax.set_title("Anomaly detection: faults caught vs real days flagged", fontweight='bold')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('anomaly_method_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

summary.to_csv('anomaly_detection_results.csv')
print("\nCell 18 complete. Saved: anomaly_fault_examples.png, "
      "anomaly_method_comparison.png, anomaly_detection_results.csv")
