"""Rule, main model and stacked model on label v1 or label v2 (identical pipeline; only the target differs).
  main    CatBoost on Exp 25's 21 inputs (fire history from both satellites + lightning), sample weights as Exp 30
  stacked main inputs + nd_score, the next-door spread model's score (fire history + fire-shape features only; no wind,
          rain or terrain), out-of-fold (leave one year out) on training years
Train 2018-2021, thresholds per Fire Centre on 2022 (max exact F1 on that label), test the full 2023 grid.
For label v2 the models are also scored against label v1 to separate "easier label" from "better model".
usage: python run_label.py REPO_ROOT v1|v2"""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(sys.argv[1]); LABEL = sys.argv[2]; EXP = ROOT / 'experiments'
HERE = EXP / '32_exp_two_satellite_bc/label_v2'; RES = HERE / f'results_{LABEL}'; (RES / 'scores').mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(ROOT), 'none']
spec = importlib.util.spec_from_file_location('e31', EXP / '31_exp_spread_features_bc/run_experiment.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
E.SRC = ROOT / ('data/pair_v2_lightning_h3r7' if LABEL == 'v2' else 'data/pair_lightning_h3r7'); SRC = E.SRC
REGIONS, BASE, SHAPE, PARAMS = E.REGIONS, E.BASE, E.SHAPE, E.PARAMS
ND_F = BASE + SHAPE; ST_F = BASE + ['nd_score']
log = open(HERE / f'run_{LABEL}.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
def fit(name, d, feats, weight):
    yr = pd.to_datetime(d.target_date).dt.year
    mk = lambda x: Pool(x[feats], x.target_snpp_filled.astype(int), weight=x.sample_weight if weight else None)
    m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
    m.fit(mk(d[yr <= 2020]), eval_set=mk(d[yr == 2021])); it = max(100, m.get_best_iteration() + 1)
    m = CatBoostClassifier(iterations=it, **PARAMS); m.fit(mk(d)); m.save_model(str(RES / f'{name}.cbm')); say(LABEL, name, 'iterations', it)
    return m, it
t0 = time.time()
nd = pd.concat([E.joined('train', y, r).assign(year=y) for r in REGIONS for y in range(2018, 2022)], ignore_index=True)
say(LABEL, 'next-door rows', len(nd), 'burn rate', round(nd.target_snpp_filled.mean(), 4))
NDm, nd_it = fit('nextdoor', nd, ND_F, weight=False)
nd['nd_score'] = np.nan
for y in range(2018, 2022):
    m = CatBoostClassifier(iterations=nd_it, **PARAMS); tr = nd[nd.year != y]
    m.fit(Pool(tr[ND_F], tr.target_snpp_filled.astype(int))); nd.loc[nd.year == y, 'nd_score'] = m.predict_proba(nd.loc[nd.year == y, ND_F])[:, 1]
full = []
for r in REGIONS:
    for y in range(2018, 2022):
        d = pd.read_parquet(SRC / f'training_samples/{r}/snpp_lightning_{y}.parquet'); d['region'] = r
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]'); full.append(d)
full = pd.concat(full, ignore_index=True); n0 = len(full)
full = full.merge(nd[['region', 'h3_cell', 'target_date', 'nd_score']].drop_duplicates(['region', 'h3_cell', 'target_date']), on=['region', 'h3_cell', 'target_date'], how='left')
assert len(full) == n0
say(LABEL, 'training rows', len(full), 'positives', int(full.target_snpp_filled.sum()))
Gm, _ = fit('main', full, BASE, weight=True); Sm, _ = fit('stacked', full, ST_F, weight=True)
del full, nd
ev = {}
for y in (2022, 2023):
    for r in REGIONS:
        cols = BASE + ['h3_cell', 'target_date', 'target_snpp_filled', 'snpp_persistence'] + (['target_v1'] if LABEL == 'v2' else [])
        d = pd.read_parquet(SRC / f'evaluation/{y}/{r}_snpp_lightning_full_grid.parquet', columns=cols)
        j = E.joined('eval', y, r); d['nd_score'] = np.nan; d.loc[d.index[j.row.to_numpy()], 'nd_score'] = NDm.predict_proba(j[ND_F])[:, 1]
        out = pd.DataFrame({'y': d.target_snpp_filled.to_numpy(bool), 'y_v1': (d.target_v1 if LABEL == 'v2' else d.target_snpp_filled).to_numpy(bool),
                            'rule': d.snpp_persistence.to_numpy().astype(bool), 'quiet': d.quiet_stratum.to_numpy() == 1,
                            'local1d': d.local_count_1d.to_numpy() > 0, 'nb1d': d.neighbour_count_1d.to_numpy() > 0,
                            'day': (pd.to_datetime(d.target_date) - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(), 'region': r})
        for n, m, f in (('main', Gm, BASE), ('stacked', Sm, ST_F)):
            s = np.empty(len(d), np.float32)
            for a in range(0, len(d), 2_000_000): s[a:a + 2_000_000] = m.predict_proba(d.iloc[a:a + 2_000_000][f])[:, 1]
            out[n] = s
        out.to_parquet(RES / f'scores/{r}_{y}.parquet', index=False); ev[(y, r)] = out; say(LABEL, 'scored', y, r)
# metrics
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
def best_thr(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys)
    return float(s[o][int(np.argmax(2 * tp / (tp + fp + ys.sum())))])
thr = {n: {r: best_thr(ev[(2022, r)].y.to_numpy(), ev[(2022, r)][n].to_numpy()) for r in REGIONS} for n in ('main', 'stacked')}
T = pd.concat([ev[(2023, r)] for r in REGIONS], ignore_index=True)
for n in ('main', 'stacked'): T[n + '_flag'] = np.concatenate([ev[(2023, r)][n].to_numpy() >= thr[n][r] for r in REGIONS])
T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['continuing', 'next_door', 'new_quiet'], 'recent_nearby')
T['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(T.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)
R = {'label': LABEL, 'thresholds_2022': thr}
for target in (['y'] + (['y_v1'] if LABEL == 'v2' else [])):
    y = T[target].to_numpy(); res = {}
    for n in ('rule', 'main', 'stacked'):
        fl = T.rule.to_numpy() if n == 'rule' else T[n + '_flag'].to_numpy()
        tp, fp, fn = int((fl & y).sum()), int((fl & ~y).sum()), int((~fl & y).sum())
        res[n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'tp': tp, 'fp': fp, 'fn': fn,
                  'by_type': {t: {'fires': int((y & (T.type == t)).sum()), 'cells': int((T.type == t).sum()), 'caught': int((fl & y & (T.type == t)).sum()),
                                  'flagged': int((fl & (T.type == t)).sum())} for t in ('continuing', 'next_door', 'recent_nearby', 'new_quiet')}}
    Wk = np.sort(T.week.unique()); wi = pd.Series(T.week.to_numpy()).map({w: i for i, w in enumerate(Wk)}).to_numpy()
    WF = {n: np.stack([np.bincount(wi, weights=(fl & y).astype(float), minlength=len(Wk)), np.bincount(wi, weights=(fl & ~y).astype(float), minlength=len(Wk)),
                       np.bincount(wi, weights=(~fl & y).astype(float), minlength=len(Wk))], 1)
          for n, fl in (('rule', T.rule.to_numpy()), ('main', T.main_flag.to_numpy()), ('stacked', T.stacked_flag.to_numpy()))}
    rng = np.random.default_rng(42); bs = {'main_minus_rule': [], 'stacked_minus_rule': [], 'stacked_minus_main': []}
    for _ in range(2000):
        c = np.bincount(rng.integers(0, len(Wk), len(Wk)), minlength=len(Wk)).astype(float); v = {n: f1(*(WF[n] * c[:, None]).sum(0)) for n in WF}
        bs['main_minus_rule'].append(v['main'] - v['rule']); bs['stacked_minus_rule'].append(v['stacked'] - v['rule']); bs['stacked_minus_main'].append(v['stacked'] - v['main'])
    res['bootstrap_95'] = {k: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] for k, a in bs.items()}
    R['scored_against_' + ('label_' + LABEL if target == 'y' else 'label_v1')] = res
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
for k, res in R.items():
    if not k.startswith('scored'): continue
    say(LABEL, k); [say(' ', n, {x: round(res[n][x], 4) for x in ('f1', 'precision', 'recall')}, res[n]['by_type']) for n in ('rule', 'main', 'stacked')]
    say('  bootstrap', {a: [round(x, 4) for x in b] for a, b in res['bootstrap_95'].items()})
say(LABEL, 'done', round(time.time() - t0), 's')
