"""Holdover lightning x dryness on top of stacked + jumps (label v1, two-satellite table, one pooled threshold on 2022).
New inputs (all known by the end of day t-1):
  lt_cell_7d / lt_cell_14d   CanCPLD flashes in the cell's 0.1-degree box over the last 7 / 14 days
  lt_3x3_7d / lt_3x3_14d     same in the surrounding 3 x 3 boxes (~30 km)
  lt_days_cell               days since a flash in the cell's own box (cap 30)
  dr_ffmc, dr_dmc, dr_dc, dr_bui, dr_isi, dr_fwi   BCWS daily fire-weather codes (noon LST of local day t-1) at the
                             nearest of the cell's 3 closest stations (within 100 km) reporting that day
  lt14_x_dc, lt7_x_dmc       interactions
Models: stacked_j (from jumps/, scores reused) and stacked_jl = stacked_j inputs + the new inputs (retrained, same
settings). Reported: total F1, by fire type, next-door AP, watch list by new-fire group, week bootstrap.
usage: python run_lightning.py REPO_ROOT"""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np, pandas as pd, h3, scipy.ndimage as ndi
from scipy.spatial import cKDTree
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'; X32 = EXP / '32_exp_two_satellite_bc'
HERE = X32 / 'lightning'; RES = HERE / 'results'; (RES / 'scores').mkdir(parents=True, exist_ok=True)
argv = sys.argv[:]
sys.argv = [argv[0], str(ROOT)]
src = open(X32 / 'jumps/run_jumps.py', encoding='utf-8').read()
cut = src.index('# ---- training')  # reuse only the definitions (detections, jump grids, helpers), not the run
J = {'__name__': 'jmp', '__file__': str(X32 / 'jumps/run_jumps.py')}; exec(compile(src[:cut], 'run_jumps_defs', 'exec'), J)
E, SRC, REGIONS, BASE, SHAPE, PARAMS, JUMP = J['E'], J['SRC'], J['REGIONS'], J['BASE'], J['SHAPE'], J['PARAMS'], J['JUMP']
jump_grids, gather = J['jump_grids'], J['gather']
LT = ['lt_cell_7d', 'lt_cell_14d', 'lt_3x3_7d', 'lt_3x3_14d', 'lt_days_cell']
DR = ['dr_ffmc', 'dr_dmc', 'dr_dc', 'dr_bui', 'dr_isi', 'dr_fwi']
IX = ['lt14_x_dc', 'lt7_x_dmc']
ND_F = BASE + SHAPE; ST_J = BASE + JUMP + ['nd_score']; ST_JL = ST_J + LT + DR + IX
SEASON = 153; t0 = time.time()
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

st = pd.read_csv(ROOT / 'data/bcws_weather/stations_2018_2023.csv').drop_duplicates('STATION_CODE', keep='last')
st = st[np.isfinite(st.LATITUDE) & np.isfinite(st.LONGITUDE)].reset_index(drop=True); st['STATION_CODE'] = st.STATION_CODE.astype(str)
SIDX = {c: i for i, c in enumerate(st.STATION_CODE)}; KM = np.cos(np.radians(52)); stree = cKDTree(np.c_[st.LATITUDE * 111, st.LONGITUDE * 111 * KM])
FW = pd.read_parquet(ROOT / 'data/bcws_weather/bcws_daily_2018_2023.parquet', columns=['station_code', 'date', 'ffmc', 'dmc', 'dc', 'bui', 'isi', 'fwi'])
FW['STATION_CODE'] = FW.station_code.astype(str); FW = FW[FW.STATION_CODE.isin(SIDX)]
for c, hi in (('ffmc', 101), ('dmc', 500), ('dc', 1500), ('bui', 500), ('isi', 100), ('fwi', 150)):
    FW.loc[(FW[c] < 0) | (FW[c] > hi), c] = np.nan
CUBE = {}
for y in range(2018, 2024):
    w = FW[FW.date.dt.year == y]; d = (w.date - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(); ok = (d >= 0) & (d < SEASON)
    c = np.full((len(st), SEASON, 6), np.nan, np.float32); c[w.STATION_CODE.map(SIDX).to_numpy()[ok], d[ok]] = w[['ffmc', 'dmc', 'dc', 'bui', 'isi', 'fwi']].to_numpy(np.float32)[ok]
    CUBE[y] = c
GEO = {}
def light_dry_grids(slug, year):
    if slug not in GEO:
        g = pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell', 'center_latitude', 'center_longitude'])
        dist, nb = stree.query(np.c_[g.center_latitude * 111, g.center_longitude * 111 * KM], k=3); nb[dist > 100] = -1
        GEO[slug] = (pd.Index(g.h3_cell), g.center_latitude.to_numpy(), g.center_longitude.to_numpy(), nb)
    cells, lat, lon, nb = GEO[slug]; n = len(cells)
    z = np.load(ROOT / f'data/cancpld_lightning/bc_cancpld_cg_daily_utc_{year}.npz'); daily, la, lo = z['daily'], z['lat'], z['lon']
    iy = np.abs(lat[:, None] - la).argmin(1); ix = np.abs(lon[:, None] - lo).argmin(1)
    cell = np.nan_to_num(daily)[:, iy, ix].T  # [n, 153] flashes per UTC day
    blk = (ndi.uniform_filter(np.nan_to_num(daily), size=(1, 3, 3), mode='constant') * 9)[:, iy, ix].T
    roll = lambda a, w: np.concatenate([np.zeros((n, 1)), np.cumsum(a, 1)], 1)[:, 1:] - np.concatenate([np.zeros((n, 1)), np.cumsum(a, 1)], 1)[:, np.maximum(0, np.arange(SEASON) + 1 - w)]
    f = {}
    h = {'lt_cell_7d': roll(cell, 7), 'lt_cell_14d': roll(cell, 14), 'lt_3x3_7d': roll(blk, 7), 'lt_3x3_14d': roll(blk, 14)}
    since = np.zeros((n, SEASON), np.float32); cnt = np.full(n, 30, np.float32)
    for t in range(SEASON):
        cnt = np.minimum(cnt + 1, 30); cnt[cell[:, t] > 0] = 0; since[:, t] = cnt
    h['lt_days_cell'] = since
    dry = np.full((n, SEASON, 6), np.nan, np.float32)
    for j in range(3):
        ok = nb[:, j] >= 0; src = np.full((n, SEASON, 6), np.nan, np.float32); src[ok] = CUBE[year][nb[ok, j]]
        m = np.isnan(dry[..., 0]) & ~np.isnan(src[..., 0]); dry[m] = src[m]
    for i, c in enumerate(DR): h[c] = dry[..., i]
    for k, v in h.items():  # value as of history day h -> target day t = h + 1
        f[k] = np.concatenate([np.full((n, 1), np.nan, np.float32), v[:, :-1].astype(np.float32)], 1)
    f['lt14_x_dc'] = f['lt_3x3_14d'] * f['dr_dc']; f['lt7_x_dmc'] = f['lt_3x3_7d'] * f['dr_dmc']
    return cells, f

# ---- training: next-door score (out-of-fold) as in run_jumps, then stacked_jl
nd = pd.concat([E.joined('train', y, r).assign(year=y) for r in REGIONS for y in range(2018, 2022)], ignore_index=True)
m0 = CatBoostClassifier().load_model(str(X32 / 'jumps/results/nextdoor.cbm')); it_nd = m0.tree_count_; nd['nd_score'] = np.nan
for y in range(2018, 2022):
    m = CatBoostClassifier(iterations=it_nd, **PARAMS); tr = nd[nd.year != y]
    m.fit(Pool(tr[ND_F], tr.target_snpp_filled.astype(int))); nd.loc[nd.year == y, 'nd_score'] = m.predict_proba(nd.loc[nd.year == y, ND_F])[:, 1]
full = []
for r in REGIONS:
    for y in range(2018, 2022):
        cj, fj = jump_grids(r, y); cl, fl = light_dry_grids(r, y)
        d = pd.read_parquet(SRC / f'training_samples/{r}/snpp_lightning_{y}.parquet'); d['region'] = r
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]')
        for c, v in gather(d, cj, fj).items(): d[c] = v
        for c, v in gather(d, cl, fl).items(): d[c] = v
        full.append(d)
    say('features train', r)
full = pd.concat(full, ignore_index=True); n0 = len(full)
full = full.merge(nd[['region', 'h3_cell', 'target_date', 'nd_score']].drop_duplicates(['region', 'h3_cell', 'target_date']), on=['region', 'h3_cell', 'target_date'], how='left')
assert len(full) == n0
say('training rows', len(full), 'missing dryness', round(float(np.isnan(full.dr_dc).mean()), 3))
yr = full.target_date.dt.year; mk = lambda x: Pool(x[ST_JL], x.target_snpp_filled.astype(int), weight=x.sample_weight)
m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
m.fit(mk(full[yr <= 2020]), eval_set=mk(full[yr == 2021])); it = max(100, m.get_best_iteration() + 1)
JL = CatBoostClassifier(iterations=it, **PARAMS); JL.fit(mk(full)); JL.save_model(str(RES / 'stacked_jl.cbm')); say('stacked_jl iterations', it)
imp = dict(zip(ST_JL, JL.get_feature_importance().round(2).tolist())); json.dump(imp, open(RES / 'importance.json', 'w'), indent=1)
del full, nd
NDm = CatBoostClassifier().load_model(str(X32 / 'jumps/results/nextdoor.cbm'))

# ---- scoring
ev = {}
for y in (2022, 2023):
    for r in REGIONS:
        cj, fj = jump_grids(r, y); cl, fl = light_dry_grids(r, y)
        d = pd.read_parquet(SRC / f'evaluation/{y}/{r}_snpp_lightning_full_grid.parquet', columns=BASE + ['h3_cell', 'target_date'])
        for c, v in gather(d, cj, fj).items(): d[c] = v
        for c, v in gather(d, cl, fl).items(): d[c] = v
        j = E.joined('eval', y, r); d['nd_score'] = np.nan; d.loc[d.index[j.row.to_numpy()], 'nd_score'] = NDm.predict_proba(j[ND_F])[:, 1]
        o = pd.read_parquet(X32 / f'jumps/results/scores/{r}_{y}.parquet')
        s = np.empty(len(d), np.float32)
        for a in range(0, len(d), 2_000_000): s[a:a + 2_000_000] = JL.predict_proba(d.iloc[a:a + 2_000_000][ST_JL])[:, 1]
        o['stacked_jl'] = s; o.to_parquet(RES / f'scores/{r}_{y}.parquet', index=False); ev[(y, r)] = o; say('scored', y, r)

# ---- metrics (pooled threshold on 2022)
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
def best(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys); return float(s[o][int(np.argmax(2 * tp / (tp + fp + ys.sum())))])
A = pd.concat([ev[(2022, r)] for r in REGIONS]); T = pd.concat([ev[(2023, r)] for r in REGIONS], ignore_index=True)
MODELS = ['stacked', 'stacked_j', 'stacked_jl']; thr = {n: best(A.y.to_numpy(), A[n].to_numpy()) for n in MODELS}
y = T.y.to_numpy(); T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['continuing', 'next_door', 'new_quiet'], 'recent_nearby')
wk = (pd.Timestamp('2023-05-01') + pd.to_timedelta(T.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int); W = np.unique(wk); wi = np.searchsorted(W, wk)
wf = lambda fl: np.stack([np.bincount(wi, (fl & y).astype(float), len(W)), np.bincount(wi, (fl & ~y).astype(float), len(W)), np.bincount(wi, (~fl & y).astype(float), len(W))], 1)
R = {'pooled_thresholds_2022': thr, 'total': {}}; WF = {'rule': wf(T.rule.to_numpy())}
for n in ['rule'] + MODELS:
    fl = T.rule.to_numpy() if n == 'rule' else T[n].to_numpy() >= thr[n]
    if n != 'rule': WF[n] = wf(fl)
    tp, fp, fn = (fl & y).sum(), (fl & ~y).sum(), (~fl & y).sum()
    R['total'][n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
                     'by_type': {t: {'fires': int((y & (T.type == t)).sum()), 'caught': int((fl & y & (T.type == t)).sum()), 'flagged': int((fl & (T.type == t)).sum())}
                                 for t in ('continuing', 'next_door', 'recent_nearby', 'new_quiet')}}
nd23 = T[T.type == 'next_door']; R['next_door_ap'] = {n: float(average_precision_score(nd23.y, nd23[n])) for n in MODELS}
# watch list by new-fire group (categories as in run_jumps)
q = T[T.quiet].reset_index(drop=True); yq = q.y.to_numpy(); q['cat'] = 'none'
exec(compile(src[src.index('# ---- new-fire categories'):src.index('# ---- metrics')], 'cats', 'exec'), J)
q.loc[yq, 'cat'] = [J['category'](c, t) for c, t in zip(q.h3_cell[yq], q.day[yq])]
R['watch'] = {}; top50 = {}
for n in MODELS:
    rk = q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy(); top50[n] = rk <= 50
    R['watch'][n] = {'ap': float(average_precision_score(yq, q[n]))}
    for N in (25, 50, 100, 200):
        on = rk <= N; R['watch'][n][f'top{N}'] = float((on & yq).sum() / yq.sum())
        for c in ('ignition', 'jump', 'other'):
            mm = yq & (q.cat == c).to_numpy(); R['watch'][n][f'top{N}_{c}'] = float((on & mm).sum() / max(mm.sum(), 1))
qwk = np.searchsorted(W, (pd.Timestamp('2023-05-01') + pd.to_timedelta(q.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int))
ign = yq & (q.cat == 'ignition').to_numpy()
WQ = {n: (np.bincount(qwk, (top50[n] & yq).astype(float), len(W)), np.bincount(qwk, yq.astype(float), len(W)),
          np.bincount(qwk, (top50[n] & ign).astype(float), len(W)), np.bincount(qwk, ign.astype(float), len(W))) for n in MODELS}
ndw = np.searchsorted(W, (pd.Timestamp('2023-05-01') + pd.to_timedelta(nd23.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)); ndg = [np.flatnonzero(ndw == i) for i in range(len(W))]
ynd = nd23.y.to_numpy(); rng = np.random.default_rng(42)
bs = {k: [] for k in ('f1_jl_minus_j', 'f1_jl_minus_stacked', 'f1_jl_minus_rule', 'watch50_jl_minus_j', 'watch50_ignition_jl_minus_j', 'ndAP_jl_minus_j')}
for _ in range(1000):
    i = rng.integers(0, len(W), len(W)); c = np.bincount(i, minlength=len(W)).astype(float); F = {n: f1(*(WF[n] * c[:, None]).sum(0)) for n in WF}
    bs['f1_jl_minus_j'].append(F['stacked_jl'] - F['stacked_j']); bs['f1_jl_minus_stacked'].append(F['stacked_jl'] - F['stacked']); bs['f1_jl_minus_rule'].append(F['stacked_jl'] - F['rule'])
    w = {n: ((WQ[n][0] @ c) / (WQ[n][1] @ c), (WQ[n][2] @ c) / max(WQ[n][3] @ c, 1)) for n in WQ}
    bs['watch50_jl_minus_j'].append(w['stacked_jl'][0] - w['stacked_j'][0]); bs['watch50_ignition_jl_minus_j'].append(w['stacked_jl'][1] - w['stacked_j'][1])
    ii = np.concatenate([ndg[j] for j in i]); bs['ndAP_jl_minus_j'].append(average_precision_score(ynd[ii], nd23.stacked_jl.to_numpy()[ii]) - average_precision_score(ynd[ii], nd23.stacked_j.to_numpy()[ii]))
R['bootstrap_95'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
R['importance_top'] = sorted(imp.items(), key=lambda x: -x[1])[:15]
R['importance_new_groups'] = {'lightning_ext': round(sum(imp[c] for c in LT), 2), 'dryness': round(sum(imp[c] for c in DR), 2), 'interactions': round(sum(imp[c] for c in IX), 2)}
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
say(json.dumps({'total': {n: {x: round(v[x], 4) for x in ('f1', 'precision', 'recall')} | {'by_type': v['by_type']} for n, v in R['total'].items()},
                'next_door_ap': {k: round(v, 4) for k, v in R['next_door_ap'].items()}, 'watch': {n: {k: round(x, 4) for k, x in v.items()} for n, v in R['watch'].items()},
                'bootstrap': {k: [round(a, 4) for a in v] for k, v in R['bootstrap_95'].items()}, 'importance_groups': R['importance_new_groups'],
                'importance_top': R['importance_top']}, indent=1, default=float))
say('done', round(time.time() - t0), 's')
