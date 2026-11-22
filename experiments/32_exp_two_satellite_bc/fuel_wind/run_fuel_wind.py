"""Exp 32 part 2: do land cover (fuel) and afternoon station wind improve overall F1, next-door spread and new fires?
Two-satellite table (data/pair_lightning_h3r7). Same splits as Exp 32: train 2018-2021, thresholds per Fire Centre
on 2022, test full 2023 grid.
Inputs added:
  LC  ESA WorldCover 2021 shares in the cell (data/static/landcover_h3r7.parquet): tree, shrub, grass, crop, built, bare,
      snow, water, wetland, burnable.
  HW  BCWS afternoon station weather (12-16 LST of day t-1; data/bcws_weather/afternoon_2018_2023.parquet) from the nearest
      of the cell's 3 closest stations (within 100 km) reporting that day: mean wind, max gust, wind-to u/v, min RH,
      max temperature, max hourly ISI, max hourly FFMC. Next-door rows also get alignment of that wind with the
      direction from the burning neighbours (x speed, x gust). HWt = the same for the target day (upper bound only).
Models:
  general  G0 (Exp 32 general model A, loaded scores), G_lc = base + LC, G_lchw = base + LC + HW
  next-door N_shape = base + fire shape, N_lc = + LC, N_lchw = + LC + HW, NU_lchw_t = + LC + HWt (upper bound)
  stacked  ST_full = base + LC + HW + nd_score from N_lchw (out-of-fold on training years), vs Exp 32 stacked (0.629)
usage: python run_fuel_wind.py REPO_ROOT"""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np, pandas as pd, h3
from scipy.spatial import cKDTree
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'
HERE = EXP / '32_exp_two_satellite_bc/fuel_wind'; RES = HERE / 'results'; (RES / 'scores').mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(ROOT), 'none']
spec = importlib.util.spec_from_file_location('e31', EXP / '31_exp_spread_features_bc/run_experiment.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
assert E.PAIR
SRC, REGIONS, BASE, SHAPE, PARAMS = E.SRC, E.REGIONS, E.BASE, E.SHAPE, E.PARAMS
X32 = EXP / '32_exp_two_satellite_bc'
LC = ['lc_tree', 'lc_shrub', 'lc_grass', 'lc_crop', 'lc_built', 'lc_bare', 'lc_snow', 'lc_water', 'lc_wetland', 'lc_burnable']
HWB = ['hw_wind_mean', 'hw_gust_max', 'hw_u', 'hw_v', 'hw_rh_min', 'hw_temp_max', 'hw_isi_max', 'hw_ffmc_max']
ALIGN = ['hw_align', 'hw_align_x_speed', 'hw_align_x_gust']
HWt = [c + '_t' for c in HWB]; ALIGNt = [c + '_t' for c in ALIGN]
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

# ---------------- static land cover and station weather lookups
lc = pd.read_parquet(ROOT / 'data/static/landcover_h3r7.parquet').set_index('h3_cell')[LC]
st = pd.read_csv(ROOT / 'data/bcws_weather/stations_2018_2023.csv').drop_duplicates('STATION_CODE', keep='last')
st = st[np.isfinite(st.LATITUDE) & np.isfinite(st.LONGITUDE)].reset_index(drop=True)
st['STATION_CODE'] = st.STATION_CODE.astype(str)
W = pd.read_parquet(ROOT / 'data/bcws_weather/afternoon_2018_2023.parquet')
W.loc[(W.hw_gust_max > 150) | (W.hw_gust_max < 0), 'hw_gust_max'] = np.nan
W.loc[(W.hw_temp_max > 50) | (W.hw_temp_max < -30), 'hw_temp_max'] = np.nan
W['hw_rh_min'] = W.hw_rh_min.clip(upper=100); W.loc[W.hw_isi_max > 100, 'hw_isi_max'] = np.nan
W = W[W.STATION_CODE.isin(st.STATION_CODE)]
SIDX = {c: i for i, c in enumerate(st.STATION_CODE)}
k = np.cos(np.radians(52)); tree = cKDTree(np.c_[st.LATITUDE * 111, st.LONGITUDE * 111 * k])
CUBE = {}
for y in range(2018, 2024):
    w = W[W.date.dt.year == y]; d = (w.date - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(); ok = (d >= 0) & (d < 153)
    c = np.full((len(st), 153, len(HWB)), np.nan, np.float32)
    c[w.STATION_CODE.map(SIDX).to_numpy()[ok], d[ok]] = w[HWB].to_numpy(np.float32)[ok]
    CUBE[y] = c
NEAR = {}
def nearest(cells):
    key = len(cells), cells[0]
    if key not in NEAR:
        ll = np.array([h3.cell_to_latlng(c) for c in cells])
        dist, idx = tree.query(np.c_[ll[:, 0] * 111, ll[:, 1] * 111 * k], k=3)
        idx[dist > 100] = -1; NEAR[key] = idx
    return NEAR[key]

def add_hw(df, shift):
    """Station afternoon weather for local day (target day - shift), from the first of 3 nearest stations reporting."""
    out = np.full((len(df), len(HWB)), np.nan, np.float32)
    cells = df.h3_cell.to_numpy(); uc, inv = np.unique(cells, return_inverse=True); nb = nearest(uc)[inv]
    d = pd.to_datetime(df.target_date); yr = d.dt.year.to_numpy(); di = (d - pd.to_datetime(d.dt.year.astype(str) + '-05-01')).dt.days.to_numpy() - shift
    for y in np.unique(yr):
        m = (yr == y) & (di >= 0) & (di < 153)
        for j in range(3):
            need = m & np.isnan(out[:, 0]) & (nb[:, j] >= 0)
            if need.any():
                out[need] = CUBE[y][nb[need, j], di[need]]
    return out

def enrich(df, nextdoor=False):
    df = df.reset_index(drop=True)
    df[LC] = lc.reindex(df.h3_cell.to_numpy()).to_numpy(np.float32)
    for shift, names, al in ((1, HWB, ALIGN), (0, HWt, ALIGNt)):
        if shift == 0 and not nextdoor: continue
        a = add_hw(df, shift); df[names] = a
        if nextdoor:
            u, v = a[:, HWB.index('hw_u')], a[:, HWB.index('hw_v')]; spd = np.hypot(u, v)
            with np.errstate(invalid='ignore', divide='ignore'):
                al_ = (df.sp_sx.to_numpy() * u + df.sp_sy.to_numpy() * v) / spd
            df[al[0]] = al_; df[al[1]] = al_ * a[:, HWB.index('hw_wind_mean')]; df[al[2]] = al_ * a[:, HWB.index('hw_gust_max')]
    return df

def fit(name, d, feats, weight=True):
    yr = pd.to_datetime(d.target_date).dt.year
    mk = lambda x: Pool(x[feats], x.target_snpp_filled.astype(int), weight=x.sample_weight if weight else None)
    m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
    m.fit(mk(d[yr <= 2020]), eval_set=mk(d[yr == 2021])); it = max(100, m.get_best_iteration() + 1)
    m = CatBoostClassifier(iterations=it, **PARAMS); m.fit(mk(d)); m.save_model(str(RES / f'{name}.cbm'))
    say(name, 'iterations', it); return m, it, dict(zip(feats, m.get_feature_importance().round(2).tolist()))

SETS_ND = {'N_shape': BASE + SHAPE, 'N_lc': BASE + SHAPE + LC, 'N_lchw': BASE + SHAPE + LC + HWB + ALIGN,
           'NU_lchw_t': BASE + SHAPE + LC + HWt + ALIGNt}
SETS_G = {'G_lc': BASE + LC, 'G_lchw': BASE + LC + HWB}
F_ST = BASE + LC + HWB + ['nd_score']
info = {}
t0 = time.time()
# ---------------- next-door models
nd = pd.concat([E.joined('train', y, r).assign(year=y) for r in REGIONS for y in range(2018, 2022)], ignore_index=True)
nd = enrich(nd, nextdoor=True)
say('next-door training rows', len(nd), 'HW missing', round(float(np.isnan(nd.hw_wind_mean).mean()), 3), 'LC missing', round(float(nd.lc_tree.isna().mean()), 3))
NDM = {}
for n, f in SETS_ND.items():
    NDM[n], it, imp = fit(n, nd, f, weight=False); info[n] = {'iterations': it, 'importance': imp}
# out-of-fold nd_score for stacking (N_lchw)
nd['nd_score'] = np.nan
for y in range(2018, 2022):
    m = CatBoostClassifier(iterations=info['N_lchw']['iterations'], **PARAMS)
    tr = nd[nd.year != y]; m.fit(Pool(tr[SETS_ND['N_lchw']], tr.target_snpp_filled.astype(int)))
    nd.loc[nd.year == y, 'nd_score'] = m.predict_proba(nd.loc[nd.year == y, SETS_ND['N_lchw']])[:, 1]
# ---------------- general and stacked models
full = []
for r in REGIONS:
    for y in range(2018, 2022):
        d = pd.read_parquet(SRC / f'training_samples/{r}/snpp_lightning_{y}.parquet'); d['region'] = r
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]'); full.append(d)
full = enrich(pd.concat(full, ignore_index=True)); n0 = len(full)
full = full.merge(nd[['region', 'h3_cell', 'target_date', 'nd_score']].drop_duplicates(['region', 'h3_cell', 'target_date']), on=['region', 'h3_cell', 'target_date'], how='left')
assert len(full) == n0
say('general training rows', len(full), 'HW missing', round(float(np.isnan(full.hw_wind_mean).mean()), 3), 'with nd_score', int(full.nd_score.notna().sum()))
GM = {}
for n, f in SETS_G.items():
    GM[n], it, imp = fit(n, full, f); info[n] = {'iterations': it, 'importance': imp}
GM['ST_full'], it, imp = fit('ST_full', full, F_ST); info['ST_full'] = {'iterations': it, 'importance': imp}
json.dump(info, open(RES / 'training.json', 'w'), indent=1)
del full, nd
say('training done', round(time.time() - t0), 's')

# ---------------- score 2022 / 2023
ev = {}
for y in (2022, 2023):
    for r in REGIONS:
        d = pd.read_parquet(SRC / f'evaluation/{y}/{r}_snpp_lightning_full_grid.parquet', columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled', 'snpp_persistence'])
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]')
        d = enrich(d)
        j = enrich(E.joined('eval', y, r), nextdoor=True); rows = j.row.to_numpy()
        ndsc = {n: m.predict_proba(j[SETS_ND[n]])[:, 1] for n, m in NDM.items()}
        d['nd_score'] = np.nan; d.loc[rows, 'nd_score'] = ndsc['N_lchw']
        out = pd.DataFrame({'y': d.target_snpp_filled.to_numpy(bool), 'rule': d.snpp_persistence.to_numpy().astype(bool),
                            'quiet': d.quiet_stratum.to_numpy() == 1, 'local1d': d.local_count_1d.to_numpy() > 0, 'nb1d': d.neighbour_count_1d.to_numpy() > 0,
                            'day': (d.target_date - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(), 'region': r})
        out['G0'] = pd.read_parquet(X32 / f'exp30/results/scores/{r}_{y}.parquet', columns=['A_full_base']).A_full_base.to_numpy()
        out['ST32'] = pd.read_parquet(X32 / f'exp31/results/stack_scores/{r}_{y}.parquet', columns=['stk']).stk.to_numpy()
        for n, m in GM.items():
            f = F_ST if n == 'ST_full' else SETS_G[n]; s = np.empty(len(d), np.float32)
            for a in range(0, len(d), 2_000_000):
                s[a:a + 2_000_000] = m.predict_proba(d.iloc[a:a + 2_000_000][f])[:, 1]
            out[n] = s
        for n, v in ndsc.items():
            out[n] = np.nan; out.loc[rows, n] = v
        out.to_parquet(RES / f'scores/{r}_{y}.parquet', index=False); ev[(y, r)] = out
        say('scored', y, r)

# ---------------- metrics
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
thr_old = {'G0': {r: v['A_full_base'] for r, v in json.load(open(X32 / 'exp30/results/metrics.json'))['thresholds_2022'].items()},
           'ST32': json.load(open(X32 / 'exp31/results/stack_metrics.json'))['thresholds_2022']}
GEN = ['G0', 'G_lc', 'G_lchw', 'ST32', 'ST_full']
thr = {n: thr_old.get(n) or {r: E.best_f1_thr(ev[(2022, r)].y.to_numpy(), ev[(2022, r)][n].to_numpy()) for r in REGIONS} for n in GEN}
T = pd.concat([ev[(2023, r)] for r in REGIONS], ignore_index=True)
for n in GEN:
    T[n + '_flag'] = np.concatenate([ev[(2023, r)][n].to_numpy() >= thr[n][r] for r in REGIONS])
T['type'] = np.select([T.local1d, ~T.local1d & T.nb1d, T.quiet], ['continuing', 'next_door', 'new_quiet'], 'recent_nearby')
T['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(T.day, unit='D')).dt.isocalendar().week.to_numpy()
R = {'thresholds_2022': thr}
y = T.y.to_numpy()
R['total'] = {}
for n in ['rule'] + GEN:
    fl = T.rule.to_numpy() if n == 'rule' else T[n + '_flag'].to_numpy()
    tp, fp, fn = int((fl & y).sum()), int((fl & ~y).sum()), int((~fl & y).sum())
    R['total'][n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'tp': tp, 'fp': fp, 'fn': fn,
                     'by_type': {t: {'fires': int((y & (T.type == t)).sum()), 'caught': int((fl & y & (T.type == t)).sum()),
                                     'flagged': int((fl & (T.type == t)).sum())} for t in ('continuing', 'next_door', 'recent_nearby', 'new_quiet')}}
# next-door AP
nd23 = T[T.type == 'next_door']; ynd = nd23.y.to_numpy()
R['next_door_ap'] = {n: float(average_precision_score(ynd, nd23[n].fillna(0))) for n in ['G0', 'ST32', 'ST_full'] + list(SETS_ND)}
# new fires: AP on quiet rows and top-50 watch list per centre per day
q = T[T.quiet]; yq = q.y.to_numpy(); R['new_fires'] = {}
for n in ['G0', 'G_lc', 'G_lchw', 'ST32', 'ST_full']:
    rank = q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy()
    R['new_fires'][n] = {'ap': float(average_precision_score(yq, q[n])), **{f'top{N}': float(((rank <= N) & yq).sum() / yq.sum()) for N in (25, 50, 100)}}
# bootstrap over ISO weeks
Wk = np.sort(T.week.unique()); rng = np.random.default_rng(42)
def weekly(col):
    fl = T[col].to_numpy()
    return pd.DataFrame({'tp': fl & y, 'fp': fl & ~y, 'fn': ~fl & y, 'w': T.week}).groupby('w').sum().reindex(Wk).to_numpy()
WK = {n: weekly(n + '_flag') for n in GEN}
qrank = {n: (q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy() <= 50) for n in ['G0', 'G_lc', 'G_lchw']}
qw = q.week.to_numpy(); ndw = nd23.week.to_numpy()
def wl(n, wks):
    m = np.isin(qw, wks); return (qrank[n][m] & yq[m]).sum() / max(yq[m].sum(), 1)
comps = [('G_lc', 'G0'), ('G_lchw', 'G_lc'), ('ST_full', 'ST32'), ('ST_full', 'G0')]
bs = {f'f1_{a}_minus_{b}': [] for a, b in comps}
bs.update({'watch50_G_lc_minus_G0': [], 'watch50_G_lchw_minus_G_lc': [], 'ndAP_N_lc_minus_N_shape': [], 'ndAP_N_lchw_minus_N_lc': [], 'ndAP_NU_minus_N_lchw': []})
ndg = [np.flatnonzero(ndw == w) for w in Wk]
for _ in range(1000):
    i = rng.integers(0, len(Wk), len(Wk)); wks = Wk[i]
    for a, b in comps:
        bs[f'f1_{a}_minus_{b}'].append(f1(*WK[a][i].sum(0)) - f1(*WK[b][i].sum(0)))
    cnt = pd.Series(wks).value_counts()
    # weighted watch-list share with repeated weeks
    def wlw(n):
        num = sum(((qrank[n] & yq)[qw == w].sum()) * c for w, c in cnt.items()); den = sum(yq[qw == w].sum() * c for w, c in cnt.items()); return num / max(den, 1)
    bs['watch50_G_lc_minus_G0'].append(wlw('G_lc') - wlw('G0')); bs['watch50_G_lchw_minus_G_lc'].append(wlw('G_lchw') - wlw('G_lc'))
    ii = np.concatenate([ndg[j] for j in i]); yy = ynd[ii]
    ap = {n: average_precision_score(yy, nd23[n].to_numpy()[ii]) for n in ['N_shape', 'N_lc', 'N_lchw', 'NU_lchw_t']}
    bs['ndAP_N_lc_minus_N_shape'].append(ap['N_lc'] - ap['N_shape']); bs['ndAP_N_lchw_minus_N_lc'].append(ap['N_lchw'] - ap['N_lc'])
    bs['ndAP_NU_minus_N_lchw'].append(ap['NU_lchw_t'] - ap['N_lchw'])
R['bootstrap_95'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
R['f1_by_centre'] = {n: {r: float(f1(*(lambda g: [(g[n + '_flag'] & g.y).sum(), (g[n + '_flag'] & ~g.y).sum(), (~g[n + '_flag'] & g.y).sum()])(T[T.region == r])))
                         for r in REGIONS} for n in ['rule'] + GEN if n != 'rule'}
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
say(json.dumps({'total': {n: {x: round(v[x], 4) for x in ('f1', 'precision', 'recall')} for n, v in R['total'].items()},
                'next_door_ap': {k: round(v, 4) for k, v in R['next_door_ap'].items()},
                'new_fires': {k: {x: round(v[x], 4) for x in v} for k, v in R['new_fires'].items()},
                'bootstrap_95': {k: [round(a, 4) for a in v] for k, v in R['bootstrap_95'].items()}}, indent=1))
for n in ['rule', 'G0', 'ST32', 'ST_full']:
    say(n, {t: v for t, v in R['total'][n]['by_type'].items()})
