"""Threshold strategies for the jump models (scores from run_jumps.py). All choices use 2022 only; 2023 is the test.
  A per-centre threshold (as before)
  B one pooled BC-wide threshold
  C rule volume: each centre-day flags as many cells as the rule (no tuning)
  D routed: cells burning yesterday keep the stacked (no-jump) flag; all other cells use stacked_j with a per-centre
    threshold chosen on 2022 to maximise total F1 given those fixed flags
  E routed as D with one pooled threshold
Also: F1 by Fire Centre, results by fire type, watch list for N = 10..200, and precision on the list.
usage: python tune.py REPO_ROOT"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1]); X32 = ROOT / 'experiments/32_exp_two_satellite_bc'; RES = X32 / 'jumps/results'
REGIONS = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
S = {(y, r): pd.read_parquet(RES / f'scores/{r}_{y}.parquet') for y in (2022, 2023) for r in REGIONS}
old = json.load(open(RES / 'metrics.json'))['thresholds_2022']
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
def best(y, s, rest=np.zeros(3)):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys) + rest[0]; fp = np.cumsum(~ys) + rest[1]; fn = ys.sum() - np.cumsum(ys) + rest[2]
    v = 2 * tp / (2 * tp + fp + fn); k = int(np.argmax(v)); return float(s[o][k])
def counts(fl, y): return np.array([(fl & y).sum(), (fl & ~y).sum(), (~fl & y).sum()], float)
flags = {}
for n in ('main', 'stacked', 'main_j', 'stacked_j'):
    flags[n + ' / per-centre'] = {r: S[(2023, r)][n].to_numpy() >= old[n][r] for r in REGIONS}
for n in ('main_j', 'stacked_j'):
    a = pd.concat([S[(2022, r)] for r in REGIONS]); t = best(a.y.to_numpy(), a[n].to_numpy())
    flags[n + ' / pooled'] = {r: S[(2023, r)][n].to_numpy() >= t for r in REGIONS}
    fl = {}
    for r in REGIONS:  # rule volume: per day flag as many as the rule flags that day
        d = S[(2023, r)]; k = d.groupby('day').rule.transform('sum').to_numpy()
        rank = d.groupby('day')[n].rank(ascending=False, method='first').to_numpy(); fl[r] = rank <= k
    flags[n + ' / rule volume'] = fl
# routed
cont_old = {(y, r): S[(y, r)].stacked.to_numpy() >= old['stacked'][r] for y in (2022, 2023) for r in REGIONS}
thr_r = {}
for r in REGIONS:
    d = S[(2022, r)]; c = d.local1d.to_numpy(); y = d.y.to_numpy()
    rest = counts(cont_old[(2022, r)][c], y[c]); thr_r[r] = best(y[~c], d.stacked_j.to_numpy()[~c], rest)
flags['routed / per-centre'] = {r: np.where(S[(2023, r)].local1d, cont_old[(2023, r)], S[(2023, r)].stacked_j.to_numpy() >= thr_r[r]) for r in REGIONS}
a = pd.concat([S[(2022, r)] for r in REGIONS]); c = a.local1d.to_numpy(); y = a.y.to_numpy()
co = np.concatenate([cont_old[(2022, r)] for r in REGIONS]); tp = best(y[~c], a.stacked_j.to_numpy()[~c], counts(co[c], y[c]))
flags['routed / pooled'] = {r: np.where(S[(2023, r)].local1d, cont_old[(2023, r)], S[(2023, r)].stacked_j.to_numpy() >= tp) for r in REGIONS}

T = pd.concat([S[(2023, r)] for r in REGIONS], ignore_index=True); yt = T.y.to_numpy()
T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['continuing', 'next_door', 'new_quiet'], 'recent_nearby')
T['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(T.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)
Wk = np.sort(T.week.unique()); wi = pd.Series(T.week.to_numpy()).map({w: i for i, w in enumerate(Wk)}).to_numpy()
def weekly(fl): return np.stack([np.bincount(wi, weights=(fl & yt).astype(float), minlength=len(Wk)), np.bincount(wi, weights=(fl & ~yt).astype(float), minlength=len(Wk)),
                                 np.bincount(wi, weights=(~fl & yt).astype(float), minlength=len(Wk))], 1)
ref = np.concatenate([flags['stacked / per-centre'][r] for r in REGIONS]); RW = weekly(ref); rule = T.rule.to_numpy(); RL = weekly(rule)
rng = np.random.default_rng(42); C = [np.bincount(rng.integers(0, len(Wk), len(Wk)), minlength=len(Wk)).astype(float) for _ in range(2000)]
rows = []
for name, fd in [('rule', None)] + list(flags.items()):
    fl = rule if fd is None else np.concatenate([fd[r] for r in REGIONS]); W = weekly(fl)
    tp, fp, fn = counts(fl, yt); bt = {t: (int((fl & yt & (T.type == t)).sum()), int((fl & (T.type == t)).sum())) for t in ('continuing', 'next_door', 'recent_nearby', 'new_quiet')}
    d_ref = [f1(*(W * c[:, None]).sum(0)) - f1(*(RW * c[:, None]).sum(0)) for c in C]; d_rule = [f1(*(W * c[:, None]).sum(0)) - f1(*(RL * c[:, None]).sum(0)) for c in C]
    pc = {r: round(f1(*counts((rule if fd is None else fd[r])[0:0] if False else ((T.rule.to_numpy() if fd is None else np.concatenate([fd[x] for x in REGIONS]))[(T.region == r).to_numpy()]), yt[(T.region == r).to_numpy()])), 3) for r in REGIONS}
    rows.append({'strategy': name, 'f1': round(f1(tp, fp, fn), 4), 'precision': round(tp / (tp + fp), 3), 'recall': round(tp / (tp + fn), 3),
                 'vs_stacked_95': [round(np.percentile(d_ref, 2.5), 4), round(np.percentile(d_ref, 97.5), 4)],
                 'vs_rule_95': [round(np.percentile(d_rule, 2.5), 4), round(np.percentile(d_rule, 97.5), 4)],
                 **{f'{t}_caught/flagged': f'{a}/{b}' for t, (a, b) in bt.items()}, **{f'f1_{r}': v for r, v in pc.items()}})
R = pd.DataFrame(rows); pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30)
print(R[['strategy', 'f1', 'precision', 'recall', 'vs_stacked_95', 'vs_rule_95']].to_string(index=False))
print(); print(R[['strategy'] + [c for c in R if 'caught' in c]].to_string(index=False))
print(); print(R[['strategy'] + [c for c in R if c.startswith('f1_')]].to_string(index=False))
# watch list curve
q = T[T.quiet].reset_index(drop=True); yq = q.y.to_numpy(); wl = []
for n in ('stacked', 'stacked_j', 'main_j'):
    rk = q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy()
    for N in (10, 25, 50, 100, 150, 200):
        on = rk <= N; wl.append({'model': n, 'N_per_centre_per_day': N, 'caught': int((on & yq).sum()), 'share': round((on & yq).sum() / yq.sum(), 3),
                                 'precision': round((on & yq).sum() / on.sum(), 4), 'lift_vs_random': round(((on & yq).sum() / on.sum()) / yq.mean(), 1)})
W = pd.DataFrame(wl); print(); print(W.pivot(index='N_per_centre_per_day', columns='model', values=['share', 'precision', 'lift_vs_random']).to_string())
R.to_csv(RES / 'tuning_strategies.csv', index=False); W.to_csv(RES / 'watch_curve.csv', index=False)
json.dump({'routed_thresholds': thr_r, 'routed_pooled_threshold': tp}, open(RES / 'tuning_thresholds.json', 'w'), indent=1)
