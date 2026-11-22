"""Exp 31 stacked model: the next-door spread model's score (S3) becomes an input to the full next-day model.
- Training rows 2018-2021 (Exp 25 samples): next-door rows get an out-of-fold S3 score (leave-one-year-out, same
  iterations as S3); other rows get NaN.
- 2022/2023 rows: S3 model trained on 2018-2021 (results/S3_wind.cbm).
- Stacked CatBoost: Exp 25's 21 features + nd_score, sample weights as Exp 30, iterations by early stopping on 2021
  then refit on 2018-2021. Thresholds per Fire Centre on full 2022; full 2023 grid reported.
- Compared with the rule and Exp 30 model A (same features without nd_score), by fire type, with week bootstrap.
usage: python stack.py REPO_ROOT"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(sys.argv[1]); HERE = ROOT / 'experiments/31_exp_spread_features_bc'
sys.argv = [sys.argv[0], str(ROOT), 'none']
spec = importlib.util.spec_from_file_location('e31', HERE / 'run_experiment.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
RES, SRC, REGIONS, BASE, SETS, PARAMS = E.RES, E.SRC, E.REGIONS, E.BASE, E.SETS, E.PARAMS
S3 = SETS['S3_wind']; S3_IT = json.load(open(RES / 'training.json'))['S3_wind']['iterations']
EXP30 = E.EXP30_RES
say = E.say

# ---- training table with out-of-fold next-door scores
full, nd = [], []
for r in REGIONS:
    for y in range(2018, 2022):
        d = pd.read_parquet(SRC / f'training_samples/{r}/snpp_lightning_{y}.parquet'); d['region'] = r; d['year'] = y
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]'); full.append(d)
        j = E.joined('train', y, r); j['year'] = y; nd.append(j)
full = pd.concat(full, ignore_index=True); nd = pd.concat(nd, ignore_index=True)
nd['nd_score'] = np.nan
for y in range(2018, 2022):
    tr, te = nd[nd.year != y], nd[nd.year == y]
    m = CatBoostClassifier(iterations=S3_IT, **PARAMS); m.fit(Pool(tr[S3], tr.target_snpp_filled.astype(int)))
    nd.loc[te.index, 'nd_score'] = m.predict_proba(te[S3])[:, 1]
n0 = len(full)
full = full.merge(nd[['region', 'h3_cell', 'target_date', 'nd_score']].drop_duplicates(['region', 'h3_cell', 'target_date']),
                  on=['region', 'h3_cell', 'target_date'], how='left')
assert len(full) == n0
say('stack training rows', len(full), 'with nd_score', int(full.nd_score.notna().sum()))
F = BASE + ['nd_score']
mk = lambda x: Pool(x[F], x.target_snpp_filled.astype(int), weight=x.sample_weight)
m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
m.fit(mk(full[full.year <= 2020]), eval_set=mk(full[full.year == 2021])); it = max(100, m.get_best_iteration() + 1)
stack = CatBoostClassifier(iterations=it, **PARAMS); stack.fit(mk(full)); stack.save_model(str(RES / 'stacked.cbm'))
imp = dict(zip(F, stack.get_feature_importance().round(2).tolist()))
say('stacked iterations', it, 'nd_score importance', imp['nd_score'])

# ---- score 2022 / 2023
s3 = CatBoostClassifier().load_model(str(RES / 'S3_wind.cbm'))
ev = {}
for y in (2022, 2023):
    for r in REGIONS:
        d = pd.read_parquet(SRC / f'evaluation/{y}/{r}_snpp_lightning_full_grid.parquet', columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled', 'snpp_persistence'])
        j = E.joined('eval', y, r); d['nd_score'] = np.nan; d.loc[d.index[j.row.to_numpy()], 'nd_score'] = s3.predict_proba(j[S3])[:, 1]
        sc = np.empty(len(d), np.float32)
        for a in range(0, len(d), 2_000_000):
            sc[a:a + 2_000_000] = stack.predict_proba(d.iloc[a:a + 2_000_000][F])[:, 1]
        A = pd.read_parquet(EXP30 / f'scores/{r}_{y}.parquet', columns=['A_full_base', 'quiet'])
        ev[(y, r)] = pd.DataFrame({'y': d.target_snpp_filled.to_numpy(bool), 'rule': d.snpp_persistence.to_numpy().astype(bool),
                                   'stk': sc, 'A': A.A_full_base.to_numpy(), 'quiet': A.quiet.to_numpy(),
                                   'local1d': d.local_count_1d.to_numpy() > 0, 'nb1d': d.neighbour_count_1d.to_numpy() > 0,
                                   'week': pd.to_datetime(d.target_date).dt.isocalendar().week.to_numpy(), 'region': r})
        say('scored', y, r)
        (RES / 'stack_scores').mkdir(exist_ok=True); ev[(y, r)].to_parquet(RES / f'stack_scores/{r}_{y}.parquet', index=False)

thrA = json.load(open(EXP30 / 'metrics.json'))['thresholds_2022']
thr = {r: E.best_f1_thr(ev[(2022, r)].y.to_numpy(), ev[(2022, r)].stk.to_numpy()) for r in REGIONS}
T = pd.concat([ev[(2023, r)].assign(stack_flag=lambda x, r=r: x['stk'] >= thr[r], A_flag=lambda x, r=r: x.A >= thrA[r]['A_full_base']) for r in REGIONS], ignore_index=True)
T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['1 continuing (burning yesterday)', '2 next-door spread (neighbour burning yesterday)', '4 new, quiet area'],
                      '3 recent fire nearby (within 7 days, not yesterday next door)')
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
R = {'thresholds_2022': thr, 'iterations': it, 'nd_score_importance': imp['nd_score']}
tot = {}
for n in ('rule', 'A_flag', 'stack_flag'):
    fl = T[n].to_numpy(); y = T.y.to_numpy(); tp, fp, fn = int((fl & y).sum()), int((fl & ~y).sum()), int((~fl & y).sum())
    tot[n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'tp': tp, 'fp': fp, 'fn': fn}
R['total_2023'] = tot
rows = []
for t, g in T.groupby('type'):
    row = {'type': t, 'cell_days': len(g), 'fires': int(g.y.sum()), 'burn_rate': float(g.y.mean())}
    for n in ('rule', 'A_flag', 'stack_flag'):
        fl = g[n].to_numpy(); y = g.y.to_numpy()
        row[f'{n}_flagged'] = int(fl.sum()); row[f'{n}_caught'] = int((fl & y).sum()); row[f'{n}_right_share'] = float((fl & y).sum() / max(fl.sum(), 1))
    rows.append(row)
B = pd.DataFrame(rows); B['share_of_fires'] = B.fires / B.fires.sum()
R['by_type_2023'] = B.to_dict('records')
W = np.sort(T.week.unique()); rng = np.random.default_rng(42)
def wk(col):
    return pd.DataFrame({'tp': T[col] & T.y, 'fp': T[col] & ~T.y, 'fn': ~T[col] & T.y, 'w': T.week}).groupby('w').sum().reindex(W).to_numpy()
cs, ca, cr = wk('stack_flag'), wk('A_flag'), wk('rule')
d1, d2 = [], []
for _ in range(2000):
    i = rng.integers(0, len(W), len(W)); s, a, r_ = cs[i].sum(0), ca[i].sum(0), cr[i].sum(0)
    d1.append(f1(*s) - f1(*a)); d2.append(f1(*s) - f1(*r_))
R['bootstrap_95'] = {'stack_minus_A': [float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))],
                     'stack_minus_rule': [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]}
# per centre
pc = []
for r, g in T.groupby('region'):
    for n in ('rule', 'A_flag', 'stack_flag'):
        fl = g[n].to_numpy(); y = g.y.to_numpy(); pc.append({'region': r, 'model': n, 'f1': f1((fl & y).sum(), (fl & ~y).sum(), (~fl & y).sum())})
R['f1_by_centre'] = pd.DataFrame(pc).pivot(index='region', columns='model', values='f1').round(4).reset_index().to_dict('records')
json.dump(R, open(RES / 'stack_metrics.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
say(json.dumps(tot, indent=1)); say(B.round(3).to_string()); say(R['bootstrap_95']); say(pd.DataFrame(R['f1_by_centre']).to_string())
