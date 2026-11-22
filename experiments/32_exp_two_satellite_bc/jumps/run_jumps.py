"""Jump features: can the model see a big fire 3-24 km away and expect it to jump? Label v1, two-satellite table.
New inputs (all from S-NPP or NOAA-20 detections up to day t-1):
  jp_dist1      H3 rings to the nearest cell burning on t-1 (cap 11 = none within 10 rings, ~24 km)
  jp_dist3      same for any detection on t-3..t-1
  jp_n3 / jp_n5 / jp_n8   burning cells within 3 / 5 / 8 rings on t-1 (~7 / 12 / 19 km)
  jp_g5 / jp_g8 change in those counts since t-2
  jp_frp5       FRP within 5 rings on t-1
Models (same settings as label_v2/run_label.py, label v1):
  main_j    = 21 base inputs + jump inputs
  stacked_j = main_j inputs + next-door spread score (fire history + fire shape, out-of-fold)
Compared with label_v2/results_v1 (main 0.625, stacked 0.627). Watch list on quiet cells, split into genuine new
ignitions, jumps of older BCWS fires, and other burns (same rules as new_fire_split.py).
usage: python run_jumps.py REPO_ROOT"""
import sys, json, glob, importlib.util, io, zipfile, time
from pathlib import Path
import numpy as np, pandas as pd, h3, scipy.sparse as sp
from scipy.spatial import cKDTree
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'; X32 = EXP / '32_exp_two_satellite_bc'
HERE = X32 / 'jumps'; RES = HERE / 'results'; (RES / 'scores').mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(ROOT), 'none']
spec = importlib.util.spec_from_file_location('e31', EXP / '31_exp_spread_features_bc/run_experiment.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
SRC, REGIONS, BASE, SHAPE, PARAMS = E.SRC, E.REGIONS, E.BASE, E.SHAPE, E.PARAMS
JUMP = ['jp_dist1', 'jp_dist3', 'jp_n3', 'jp_n5', 'jp_n8', 'jp_g5', 'jp_g8', 'jp_frp5']
ND_F = BASE + SHAPE; MAIN_F = BASE + JUMP; ST_F = BASE + JUMP + ['nd_score']
SEASON = 153; t0 = time.time()
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell', 'frp'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date', 'frp'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
det = pd.concat([sn, n2[['acq_date', 'h3_cell', 'frp']]]); det['date'] = pd.to_datetime(det.acq_date)
det = det[det.date.dt.month.between(5, 9) & det.date.dt.year.between(2018, 2023)]
det['year'] = det.date.dt.year; det['day'] = (det.date - pd.to_datetime(det.year.astype(str) + '-05-01')).dt.days

def disk(cells, k):
    r, c = [], []
    for i, x in enumerate(cells):
        j = cells.get_indexer(list(h3.grid_ring(x, 1) if k == 1 else h3.grid_disk(x, k))); j = j[j >= 0]; r += [i] * len(j); c += list(j)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(cells),) * 2)
GRID = {}
def jump_grids(slug, year):
    cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
    if slug not in GRID: GRID[slug] = (disk(cells, 1), disk(cells, 3), disk(cells, 5), disk(cells, 8))
    A1, A3, A5, A8 = GRID[slug]; n = len(cells)
    d = det[det.year == year]; ci = cells.get_indexer(d.h3_cell); k = ci >= 0
    b = np.zeros((n, SEASON), bool); b[ci[k], d.day.to_numpy()[k]] = True
    frp = np.zeros((n, SEASON), np.float32); np.add.at(frp, (ci[k], d.day.to_numpy()[k]), d.frp.to_numpy(np.float32)[k])
    def dist(src):
        out = np.full(n, 11, np.float32); out[src] = 0; reached = src.copy(); front = src.astype(np.float32)
        for s in range(1, 11):
            nxt = (A1 @ front) > 0; new = nxt & ~reached
            if not new.any(): break
            out[new] = s; reached |= new; front = new.astype(np.float32)
        return out
    f = {c: np.full((n, SEASON), np.nan, np.float32) for c in JUMP}
    for t in range(1, SEASON):
        h = t - 1; b1 = b[:, h]; b3 = b[:, max(0, h - 2):h + 1].any(1)
        f['jp_dist1'][:, t] = dist(b1); f['jp_dist3'][:, t] = dist(b3)
        x1 = b1.astype(np.float32); x2 = b[:, h - 1].astype(np.float32) if h >= 1 else np.zeros(n, np.float32)
        n3, n5, n8 = A3 @ x1, A5 @ x1, A8 @ x1
        f['jp_n3'][:, t], f['jp_n5'][:, t], f['jp_n8'][:, t] = n3, n5, n8
        f['jp_g5'][:, t], f['jp_g8'][:, t] = n5 - A5 @ x2, n8 - A8 @ x2
        f['jp_frp5'][:, t] = A5 @ (frp[:, h] * x1)
    return cells, f
def gather(df, cells, f):
    ci = cells.get_indexer(df.h3_cell.to_numpy()); t = (pd.to_datetime(df.target_date) - pd.to_datetime(pd.to_datetime(df.target_date).dt.year.astype(str) + '-05-01')).dt.days.to_numpy()
    ok = ci >= 0; out = {}
    for c, v in f.items():
        a = np.full(len(df), np.nan, np.float32); a[ok] = v[ci[ok], t[ok]]; out[c] = a
    return out
def fit(name, d, feats, weight):
    yr = pd.to_datetime(d.target_date).dt.year
    mk = lambda x: Pool(x[feats], x.target_snpp_filled.astype(int), weight=x.sample_weight if weight else None)
    m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
    m.fit(mk(d[yr <= 2020]), eval_set=mk(d[yr == 2021])); it = max(100, m.get_best_iteration() + 1)
    m = CatBoostClassifier(iterations=it, **PARAMS); m.fit(mk(d)); m.save_model(str(RES / f'{name}.cbm')); say(name, 'iterations', it)
    return m, it

# ---- training
nd = pd.concat([E.joined('train', y, r).assign(year=y) for r in REGIONS for y in range(2018, 2022)], ignore_index=True)
NDm, it = fit('nextdoor', nd, ND_F, weight=False); nd['nd_score'] = np.nan
for y in range(2018, 2022):
    m = CatBoostClassifier(iterations=it, **PARAMS); tr = nd[nd.year != y]
    m.fit(Pool(tr[ND_F], tr.target_snpp_filled.astype(int))); nd.loc[nd.year == y, 'nd_score'] = m.predict_proba(nd.loc[nd.year == y, ND_F])[:, 1]
full = []
for r in REGIONS:
    for y in range(2018, 2022):
        cells, f = jump_grids(r, y)
        d = pd.read_parquet(SRC / f'training_samples/{r}/snpp_lightning_{y}.parquet'); d['region'] = r
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]')
        for c, v in gather(d, cells, f).items(): d[c] = v
        full.append(d)
    say('jump features train', r)
full = pd.concat(full, ignore_index=True); n0 = len(full)
full = full.merge(nd[['region', 'h3_cell', 'target_date', 'nd_score']].drop_duplicates(['region', 'h3_cell', 'target_date']), on=['region', 'h3_cell', 'target_date'], how='left')
assert len(full) == n0
Gm, _ = fit('main_j', full, MAIN_F, True); Sm, _ = fit('stacked_j', full, ST_F, True)
imp = {'main_j': dict(zip(MAIN_F, Gm.get_feature_importance().round(2).tolist())), 'stacked_j': dict(zip(ST_F, Sm.get_feature_importance().round(2).tolist()))}
json.dump(imp, open(RES / 'importance.json', 'w'), indent=1); del full, nd

# ---- scoring
OLD = X32 / 'label_v2/results_v1'
ev = {}
for y in (2022, 2023):
    for r in REGIONS:
        cells, f = jump_grids(r, y)
        d = pd.read_parquet(SRC / f'evaluation/{y}/{r}_snpp_lightning_full_grid.parquet', columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled', 'snpp_persistence'])
        for c, v in gather(d, cells, f).items(): d[c] = v
        j = E.joined('eval', y, r); d['nd_score'] = np.nan; d.loc[d.index[j.row.to_numpy()], 'nd_score'] = NDm.predict_proba(j[ND_F])[:, 1]
        o = pd.read_parquet(OLD / f'scores/{r}_{y}.parquet')
        o['h3_cell'] = d.h3_cell.to_numpy(); o['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]').to_numpy()
        for n, m, F in (('main_j', Gm, MAIN_F), ('stacked_j', Sm, ST_F)):
            s = np.empty(len(d), np.float32)
            for a in range(0, len(d), 2_000_000): s[a:a + 2_000_000] = m.predict_proba(d.iloc[a:a + 2_000_000][F])[:, 1]
            o[n] = s
        o['jp_dist1'] = d.jp_dist1.to_numpy()
        o.to_parquet(RES / f'scores/{r}_{y}.parquet', index=False); ev[(y, r)] = o; say('scored', y, r)

# ---- new-fire categories for 2023 quiet fires
Y = 2023; inc = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv')
fi = inc[(inc.FIRE_YEAR == Y) & (inc.FIRE_TYPE == 'Fire')].drop_duplicates('FIRE_NUMBER').copy()
fi['ign'] = (pd.to_datetime(fi.IGNITION_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
fi['out'] = (pd.to_datetime(fi.FIRE_OUT_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
D = fi.set_index('FIRE_NUMBER')[['ign', 'out']]; cf = {}
for g in json.load(open(ROOT / 'data/bcws_perimeters/bc_fire_perimeters_2018_2023.geojson'))['features']:
    p = g['properties']
    if p['FIRE_YEAR'] != Y or p['FIRE_NUMBER'] not in D.index or g['geometry'] is None: continue
    try: cs = set(h3.geo_to_cells(g['geometry'], 7))
    except Exception: continue
    cs |= {n for c in cs for n in h3.grid_ring(c, 1)}
    for c in cs: cf.setdefault(c, []).append(p['FIRE_NUMBER'])
k = np.cos(np.radians(52)); tree = cKDTree(np.c_[fi.LATITUDE * 111, fi.LONGITUDE * 111 * k]); ig = fi.ign.to_numpy()
def category(c, t):
    lat, lon = h3.cell_to_latlng(c); js = tree.query_ball_point([lat * 111, lon * 111 * k], 5)
    if any(-3 <= ig[j] - t <= 2 for j in js) or any(-3 <= D.loc[fn, 'ign'] - t <= 2 for fn in cf.get(c, [])): return 'ignition'
    if any((D.loc[fn, 'ign'] < t - 3) and not (D.loc[fn, 'out'] < t - 1) for fn in cf.get(c, [])): return 'jump'
    return 'other'

# ---- metrics
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
def best_thr(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys)
    return float(s[o][int(np.argmax(2 * tp / (tp + fp + ys.sum())))])
old_thr = json.load(open(OLD / 'metrics.json'))['thresholds_2022']
MODELS = ['main', 'stacked', 'main_j', 'stacked_j']
thr = {n: old_thr[n] if n in old_thr else {r: best_thr(ev[(2022, r)].y.to_numpy(), ev[(2022, r)][n].to_numpy()) for r in REGIONS} for n in MODELS}
T = pd.concat([ev[(2023, r)] for r in REGIONS], ignore_index=True)
for n in MODELS: T[n + '_flag'] = np.concatenate([ev[(2023, r)][n].to_numpy() >= thr[n][r] for r in REGIONS])
T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['continuing', 'next_door', 'new_quiet'], 'recent_nearby')
T['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(T.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)
y = T.y.to_numpy(); R = {'thresholds_2022': thr, 'total': {}}
for n in ['rule'] + MODELS:
    fl = T.rule.to_numpy() if n == 'rule' else T[n + '_flag'].to_numpy()
    tp, fp, fn = int((fl & y).sum()), int((fl & ~y).sum()), int((~fl & y).sum())
    R['total'][n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
                     'by_type': {t: {'fires': int((y & (T.type == t)).sum()), 'caught': int((fl & y & (T.type == t)).sum()), 'flagged': int((fl & (T.type == t)).sum())}
                                 for t in ('continuing', 'next_door', 'recent_nearby', 'new_quiet')}}
q = T[T.quiet].reset_index(drop=True); yq = q.y.to_numpy()
q['cat'] = 'none'; q.loc[yq, 'cat'] = [category(c, t) for c, t in zip(q.h3_cell[yq], q.day[yq])]
R['new_fire_categories'] = q[yq].cat.value_counts().to_dict(); R['watch'] = {}
top = {}
for n in MODELS:
    rk = q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy(); top[n] = rk
    R['watch'][n] = {'ap': float(average_precision_score(yq, q[n]))}
    for N in (25, 50, 100):
        on = rk <= N; R['watch'][n][f'top{N}'] = float((on & yq).sum() / yq.sum())
        for c in ('ignition', 'jump', 'other'):
            m = yq & (q.cat == c).to_numpy(); R['watch'][n][f'top{N}_{c}'] = float((on & m).sum() / max(m.sum(), 1))
# bootstrap over weeks
Wk = np.sort(T.week.unique()); wi = pd.Series(T.week.to_numpy()).map({w: i for i, w in enumerate(Wk)}).to_numpy()
WF = {n: np.stack([np.bincount(wi, weights=(T[n + '_flag'].to_numpy() & y).astype(float), minlength=len(Wk)),
                   np.bincount(wi, weights=(T[n + '_flag'].to_numpy() & ~y).astype(float), minlength=len(Wk)),
                   np.bincount(wi, weights=(~T[n + '_flag'].to_numpy() & y).astype(float), minlength=len(Wk))], 1) for n in MODELS}
WF['rule'] = np.stack([np.bincount(wi, weights=(T.rule.to_numpy() & y).astype(float), minlength=len(Wk)), np.bincount(wi, weights=(T.rule.to_numpy() & ~y).astype(float), minlength=len(Wk)),
                       np.bincount(wi, weights=(~T.rule.to_numpy() & y).astype(float), minlength=len(Wk))], 1)
qi = pd.Series(q.week.to_numpy()).map({w: i for i, w in enumerate(Wk)}).to_numpy()
WQ = {n: (np.bincount(qi, weights=((top[n] <= 50) & yq).astype(float), minlength=len(Wk)), np.bincount(qi, weights=yq.astype(float), minlength=len(Wk))) for n in MODELS}
rng = np.random.default_rng(42); comps = [('main_j', 'main'), ('stacked_j', 'stacked'), ('stacked_j', 'rule'), ('main_j', 'rule')]
bs = {f'f1_{a}_minus_{b}': [] for a, b in comps}; bs.update({'watch50_main_j_minus_main': [], 'watch50_stacked_j_minus_stacked': []})
for _ in range(2000):
    c = np.bincount(rng.integers(0, len(Wk), len(Wk)), minlength=len(Wk)).astype(float)
    for a, b in comps: bs[f'f1_{a}_minus_{b}'].append(f1(*(WF[a] * c[:, None]).sum(0)) - f1(*(WF[b] * c[:, None]).sum(0)))
    w = {n: (WQ[n][0] @ c) / (WQ[n][1] @ c) for n in WQ}
    bs['watch50_main_j_minus_main'].append(w['main_j'] - w['main']); bs['watch50_stacked_j_minus_stacked'].append(w['stacked_j'] - w['stacked'])
R['bootstrap_95'] = {k2: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k2, v in bs.items()}
R['importance_top'] = {n: sorted(v.items(), key=lambda x: -x[1])[:10] for n, v in imp.items()}
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
say(json.dumps({'total': {n: {x: round(v[x], 4) for x in ('f1', 'precision', 'recall')} | {'by_type': v['by_type']} for n, v in R['total'].items()},
                'categories': R['new_fire_categories'], 'watch': {n: {x: round(v2, 4) for x, v2 in v.items()} for n, v in R['watch'].items()},
                'bootstrap': {k2: [round(a, 4) for a in v] for k2, v in R['bootstrap_95'].items()}, 'importance': R['importance_top']}, indent=1))
say('done', round(time.time() - t0), 's')
