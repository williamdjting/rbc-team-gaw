"""Metrics for run_fuel_wind.py from its saved scores (results/scores/<centre>_<year>.parquet).
Total F1 (thresholds per Fire Centre on 2022; G0 and ST32 keep Exp 32's thresholds), results by fire type,
next-door AP, new-fire AP and watch lists, and a paired bootstrap over ISO weeks with per-week counts precomputed.
usage: python metrics_fuel_wind.py REPO_ROOT"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(sys.argv[1]); X32 = ROOT / 'experiments/32_exp_two_satellite_bc'; RES = X32 / 'fuel_wind/results'
REGIONS = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
GEN = ['G0', 'G_lc', 'G_lchw', 'ST32', 'ST_full']; ND = ['N_shape', 'N_lc', 'N_lchw', 'NU_lchw_t']
TYPES = ['continuing', 'next_door', 'recent_nearby', 'new_quiet']
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
def best_thr(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys)
    return float(s[o][int(np.argmax(2 * tp / (tp + fp + ys.sum())))])
S = {(y, r): pd.read_parquet(RES / f'scores/{r}_{y}.parquet') for y in (2022, 2023) for r in REGIONS}
thr = {'G0': {r: v['A_full_base'] for r, v in json.load(open(X32 / 'exp30/results/metrics.json'))['thresholds_2022'].items()},
       'ST32': json.load(open(X32 / 'exp31/results/stack_metrics.json'))['thresholds_2022']}
for n in GEN:
    thr.setdefault(n, {r: best_thr(S[(2022, r)].y.to_numpy(), S[(2022, r)][n].to_numpy()) for r in REGIONS})
T = pd.concat([S[(2023, r)] for r in REGIONS], ignore_index=True)
for n in GEN:
    T[n + '_flag'] = np.concatenate([S[(2023, r)][n].to_numpy() >= thr[n][r] for r in REGIONS])
T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['continuing', 'next_door', 'new_quiet'], 'recent_nearby')
T['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(T.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)
y = T.y.to_numpy(); R = {'thresholds_2022': thr, 'total': {}}
for n in ['rule'] + GEN:
    fl = T.rule.to_numpy() if n == 'rule' else T[n + '_flag'].to_numpy()
    tp, fp, fn = int((fl & y).sum()), int((fl & ~y).sum()), int((~fl & y).sum())
    R['total'][n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
                     'by_type': {t: {'fires': int((y & (T.type == t)).sum()), 'caught': int((fl & y & (T.type == t)).sum()),
                                     'flagged': int((fl & (T.type == t)).sum())} for t in TYPES}}
R['f1_by_centre'] = {n: {r: float(f1(*(lambda g: [(g[n + '_flag'] & g.y).sum(), (g[n + '_flag'] & ~g.y).sum(), (~g[n + '_flag'] & g.y).sum()])(T[T.region == r])))
                         for r in REGIONS} for n in GEN}
nd = T[T.type == 'next_door'].reset_index(drop=True); ynd = nd.y.to_numpy()
R['next_door_ap'] = {n: float(average_precision_score(ynd, nd[n].fillna(0))) for n in ['G0', 'ST32', 'ST_full'] + ND}
q = T[T.quiet].reset_index(drop=True); yq = q.y.to_numpy(); R['new_fires'] = {}; top50 = {}
for n in ['G0', 'G_lc', 'G_lchw', 'ST32', 'ST_full']:
    rank = q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy()
    top50[n] = rank <= 50
    R['new_fires'][n] = {'ap': float(average_precision_score(yq, q[n])), **{f'top{N}': float(((rank <= N) & yq).sum() / yq.sum()) for N in (25, 50, 100)}}
# bootstrap
Wk = np.sort(T.week.unique()); widx = {w: i for i, w in enumerate(Wk)}; nw = len(Wk)
def per_week(mask_vals, weeks):
    return np.bincount(np.vectorize(widx.get)(weeks) if False else pd.Series(weeks).map(widx).to_numpy(), weights=mask_vals.astype(float), minlength=nw)
tw = T.week.to_numpy(); WF = {}
for n in GEN:
    fl = T[n + '_flag'].to_numpy(); WF[n] = np.stack([per_week(fl & y, tw), per_week(fl & ~y, tw), per_week(~fl & y, tw)], 1)
qwk = q.week.to_numpy(); WQ = {n: (per_week(top50[n] & yq, qwk), per_week(yq, qwk)) for n in top50}
ndw = pd.Series(nd.week.to_numpy()).map(widx).to_numpy(); ndg = [np.flatnonzero(ndw == i) for i in range(nw)]
rng = np.random.default_rng(42)
comps = [('G_lc', 'G0'), ('G_lchw', 'G_lc'), ('G_lchw', 'G0'), ('ST_full', 'ST32'), ('ST_full', 'G0')]
bs = {f'f1_{a}_minus_{b}': [] for a, b in comps}
wl_comps = [('G_lc', 'G0'), ('G_lchw', 'G_lc'), ('G_lchw', 'G0'), ('ST_full', 'ST32')]
bs.update({f'watch50_{a}_minus_{b}': [] for a, b in wl_comps})
nd_comps = [('N_lc', 'N_shape'), ('N_lchw', 'N_lc'), ('NU_lchw_t', 'N_lchw'), ('ST_full', 'ST32')]
bs.update({f'ndAP_{a}_minus_{b}': [] for a, b in nd_comps})
ndsc = {n: nd[n].fillna(0).to_numpy() for n in set(sum(nd_comps, ()))}
for _ in range(1000):
    i = rng.integers(0, nw, nw); c = np.bincount(i, minlength=nw).astype(float)
    for a, b in comps: bs[f'f1_{a}_minus_{b}'].append(f1(*(WF[a] * c[:, None]).sum(0)) - f1(*(WF[b] * c[:, None]).sum(0)))
    for a, b in wl_comps:
        bs[f'watch50_{a}_minus_{b}'].append((WQ[a][0] @ c) / (WQ[a][1] @ c) - (WQ[b][0] @ c) / (WQ[b][1] @ c))
    ii = np.concatenate([ndg[j] for j in i]); yy = ynd[ii]
    for a, b in nd_comps: bs[f'ndAP_{a}_minus_{b}'].append(average_precision_score(yy, ndsc[a][ii]) - average_precision_score(yy, ndsc[b][ii]))
R['bootstrap_95'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
imp = json.load(open(RES / 'training.json'))
R['importance_top'] = {n: sorted(imp[n]['importance'].items(), key=lambda x: -x[1])[:12] for n in ('G_lchw', 'N_lchw', 'ST_full')}
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
print(json.dumps({'total': {n: {x: round(v[x], 4) for x in ('f1', 'precision', 'recall')} for n, v in R['total'].items()},
                  'by_type': {n: R['total'][n]['by_type'] for n in ['rule', 'G0', 'G_lchw', 'ST32', 'ST_full']},
                  'next_door_ap': {k: round(v, 4) for k, v in R['next_door_ap'].items()},
                  'new_fires': {k: {x: round(v[x], 4) for x in v} for k, v in R['new_fires'].items()},
                  'bootstrap_95': {k: [round(a, 4) for a in v] for k, v in R['bootstrap_95'].items()},
                  'f1_by_centre': {n: {r: round(v, 4) for r, v in d.items()} for n, d in R['f1_by_centre'].items()},
                  'importance_top': R['importance_top']}, indent=1))
