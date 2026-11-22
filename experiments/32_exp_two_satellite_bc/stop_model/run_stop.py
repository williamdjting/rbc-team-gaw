"""Stop model: for cells burning yesterday (two-satellite detection on t-1), will the cell still burn on t?
The rule flags all of them (2023: 56,746 cell-days, 35% stop = 19,907 false alarms). A dedicated CatBoost on these
rows decides which to keep; every other row keeps the best general model's flag (G_lchw, fuel_wind). Threshold per
Fire Centre chosen on 2022 to maximise total F1. Test: full 2023 grid. Train 2018-2021 (all burning-yesterday rows).
Feature sets (cumulative):
  K0 base: Exp 25's 21 features + land cover + afternoon station weather (same inputs as G_lchw)
  K1 + fire lifecycle: consecutive burning days, days since first detection, season burn days, FRP on t-1/t-2/t-3 and
       trend, detection count trend, burning cells within 2 and 5 rings and their change, satellites seeing it on t-1,
       day/night split, mean scan (pixel size) and low-confidence share of t-1 detections
  K2 + rain: station 24 h precipitation on local day t-1, 3-day sum (nearest of 3 stations within 100 km)
  K3 + BCWS incidents (known by t-1): active incidents within 10 km (ignited <= t-1, not declared out before day t),
       distance to nearest active, its age, incidents declared out in the last 3 days within 10 km
  KU upper bound: K3 + target-day precipitation and afternoon weather (observed, not usable in practice)
usage: python run_stop.py REPO_ROOT"""
import io, sys, json, zipfile, time
from pathlib import Path
import numpy as np, pandas as pd, h3, scipy.sparse as sp
from scipy.spatial import cKDTree
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'; X32 = EXP / '32_exp_two_satellite_bc'
HERE = X32 / 'stop_model'; RES = HERE / 'results'; RES.mkdir(parents=True, exist_ok=True)
SRC = ROOT / 'data/pair_lightning_h3r7'
REGIONS = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
SEASON = 153
BASE = ["local_count_1d", "local_count_3d", "local_count_7d", "neighbour_count_1d", "neighbour_count_3d",
        "neighbour_count_7d", "frp_sum_1d", "frp_sum_7d", "frp_max_1d", "night_count_1d", "active_days_7d",
        "disk1_active_days_7d", "days_since_detection", "history_left_censored", "quiet_stratum",
        "ltg_cell_1d", "ltg_3x3_1d", "ltg_3x3_3d", "ltg_9x9_1d", "ltg_9x9_3d", "ltg_days_since_3x3"]
LC = ['lc_tree', 'lc_shrub', 'lc_grass', 'lc_crop', 'lc_built', 'lc_bare', 'lc_snow', 'lc_water', 'lc_wetland', 'lc_burnable']
HWB = ['hw_wind_mean', 'hw_gust_max', 'hw_u', 'hw_v', 'hw_rh_min', 'hw_temp_max', 'hw_isi_max', 'hw_ffmc_max']
LIFE = ['lf_run_days', 'lf_age_days', 'lf_season_days', 'lf_frp1', 'lf_frp2', 'lf_frp3', 'lf_frp_trend', 'lf_cnt_trend',
        'lf_size2', 'lf_size5', 'lf_grow2', 'lf_grow5', 'lf_nsat1', 'lf_day1', 'lf_scan1', 'lf_lowconf1']
RAIN = ['rn_precip1', 'rn_precip3']
INC = ['in_active10', 'in_dist_active', 'in_age_active', 'in_out3_10']
UPPER = ['rn_precip_t'] + [c + '_t' for c in HWB]
SETS = {'K0_base': BASE + LC + HWB, 'K1_life': BASE + LC + HWB + LIFE, 'K2_rain': BASE + LC + HWB + LIFE + RAIN,
        'K3_incidents': BASE + LC + HWB + LIFE + RAIN + INC, 'KU_target_day': BASE + LC + HWB + LIFE + RAIN + INC + UPPER}
PARAMS = dict(learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function='Logloss', random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
t0 = time.time()

# ---------------- detections (both satellites, with pixel size and confidence)
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell', 'frp', 'daynight', 'scan', 'confidence'])
sn['sat'] = 0
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date', 'frp', 'daynight', 'scan', 'confidence'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]; n2['sat'] = 1
det = pd.concat([sn, n2.drop(columns=['latitude', 'longitude'])], ignore_index=True)
det['date'] = pd.to_datetime(det.acq_date); det = det[det.date.dt.month.between(5, 9) & det.date.dt.year.between(2018, 2023)]
det['year'] = det.date.dt.year; det['day'] = (det.date - pd.to_datetime(det.year.astype(str) + '-05-01')).dt.days
det['low'] = (det.confidence.astype(str).str.lower() == 'l').astype(np.float32); det['isday'] = (det.daynight == 'D').astype(np.float32)
say('detections', len(det))

# ---------------- static lookups: land cover, stations, incidents
lc = pd.read_parquet(ROOT / 'data/static/landcover_h3r7.parquet').set_index('h3_cell')[LC]
st = pd.read_csv(ROOT / 'data/bcws_weather/stations_2018_2023.csv').drop_duplicates('STATION_CODE', keep='last')
st = st[np.isfinite(st.LATITUDE) & np.isfinite(st.LONGITUDE)].reset_index(drop=True); st['STATION_CODE'] = st.STATION_CODE.astype(str)
SIDX = {c: i for i, c in enumerate(st.STATION_CODE)}
KM = np.cos(np.radians(52)); stree = cKDTree(np.c_[st.LATITUDE * 111, st.LONGITUDE * 111 * KM])
W = pd.read_parquet(ROOT / 'data/bcws_weather/afternoon_2018_2023.parquet')
W.loc[(W.hw_gust_max > 150) | (W.hw_gust_max < 0), 'hw_gust_max'] = np.nan
W.loc[(W.hw_temp_max > 50) | (W.hw_temp_max < -30), 'hw_temp_max'] = np.nan
W['hw_rh_min'] = W.hw_rh_min.clip(upper=100); W.loc[W.hw_isi_max > 100, 'hw_isi_max'] = np.nan
P = pd.read_parquet(ROOT / 'data/bcws_weather/bcws_daily_2018_2023.parquet', columns=['station_code', 'date', 'precip_mm'])
P['STATION_CODE'] = P.station_code.astype(str); P.loc[(P.precip_mm < 0) | (P.precip_mm > 300), 'precip_mm'] = np.nan
WCUBE, PCUBE = {}, {}
for y in range(2018, 2024):
    c = np.full((len(st), SEASON, len(HWB)), np.nan, np.float32); w = W[(W.date.dt.year == y) & W.STATION_CODE.isin(SIDX)]
    d = (w.date - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(); ok = (d >= 0) & (d < SEASON)
    c[w.STATION_CODE.map(SIDX).to_numpy()[ok], d[ok]] = w[HWB].to_numpy(np.float32)[ok]; WCUBE[y] = c
    c = np.full((len(st), SEASON), np.nan, np.float32); p = P[(P.date.dt.year == y) & P.STATION_CODE.isin(SIDX)]
    d = (p.date - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(); ok = (d >= 0) & (d < SEASON)
    c[p.STATION_CODE.map(SIDX).to_numpy()[ok], d[ok]] = p.precip_mm.to_numpy(np.float32)[ok]; PCUBE[y] = c
inc = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv', usecols=['FIRE_YEAR', 'FIRE_TYPE', 'LATITUDE', 'LONGITUDE', 'IGNITION_DATE_UTC', 'FIRE_OUT_DATE_UTC'])
inc = inc[inc.FIRE_TYPE.astype(str).str.contains('Fire', case=False) & np.isfinite(inc.LATITUDE)]
inc['ign'] = pd.to_datetime(inc.IGNITION_DATE_UTC, utc=True).dt.tz_localize(None); inc['out'] = pd.to_datetime(inc.FIRE_OUT_DATE_UTC, utc=True).dt.tz_localize(None)

def adjacency(cells, k):
    r, c = [], []
    for i, x in enumerate(cells):
        j = cells.get_indexer(list(h3.grid_ring(x, 1) if k == 1 else h3.grid_disk(x, k))); j = j[j >= 0]; r += [i] * len(j); c += list(j)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(cells),) * 2)

def region_year_features(slug, year):
    """Grid features [cells x season day] for target day t (inputs from t-1 and earlier, except *_t upper bound)."""
    cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
    n = len(cells); A2, A5 = adjacency(cells, 2), adjacency(cells, 5)
    d = det[det.year == year]; ci = cells.get_indexer(d.h3_cell); k = ci >= 0; d = d[k]; ci = ci[k]; dy = d.day.to_numpy()
    G = lambda w: (lambda g: (np.add.at(g, (ci, dy), w), g)[1])(np.zeros((n, SEASON), np.float32))
    cnt = G(np.ones(len(d), np.float32)); frp = G(d.frp.to_numpy(np.float32)); scan = G(d.scan.to_numpy(np.float32))
    low = G(d.low.to_numpy()); dday = G(d.isday.to_numpy()); s0 = G((d.sat == 0).to_numpy(np.float32)); s1 = G((d.sat == 1).to_numpy(np.float32))
    b = cnt > 0
    run = np.zeros_like(cnt); age = np.full_like(cnt, np.nan); seas = np.cumsum(b, 1).astype(np.float32)
    first = np.full(n, -1)
    for t in range(SEASON):
        run[:, t] = (run[:, t - 1] + 1) * b[:, t] if t else b[:, t]
        first[(first < 0) & b[:, t]] = t; age[:, t] = np.where(first >= 0, t - first, np.nan)
    size2 = (A2 @ b.astype(np.float32)); size5 = (A5 @ b.astype(np.float32))
    shift = lambda a, s: np.concatenate([np.full((n, s), np.nan, np.float32), a[:, :-s]], 1)  # value from s days earlier
    with np.errstate(invalid='ignore', divide='ignore'):
        f = {'lf_run_days': run, 'lf_age_days': age, 'lf_season_days': seas, 'lf_frp1': frp, 'lf_frp2': shift(frp, 1), 'lf_frp3': shift(frp, 2),
             'lf_frp_trend': frp / (0.5 * (shift(frp, 1) + shift(frp, 2)) + 1), 'lf_cnt_trend': cnt - shift(cnt, 1),
             'lf_size2': size2, 'lf_size5': size5, 'lf_grow2': size2 - shift(size2, 1), 'lf_grow5': size5 - shift(size5, 1),
             'lf_nsat1': (s0 > 0).astype(np.float32) + (s1 > 0), 'lf_day1': dday / np.maximum(cnt, 1), 'lf_scan1': scan / np.maximum(cnt, 1),
             'lf_lowconf1': low / np.maximum(cnt, 1)}
    # these are "as of day h"; target day t uses h = t-1
    f = {k2: np.concatenate([np.full((n, 1), np.nan, np.float32), v[:, :-1]], 1) for k2, v in f.items()}
    # stations: nearest of 3 within 100 km reporting that day
    ll = np.array([h3.cell_to_latlng(c) for c in cells]); dist, nb = stree.query(np.c_[ll[:, 0] * 111, ll[:, 1] * 111 * KM], k=3); nb[dist > 100] = -1
    def pick(cube, day_shift):
        out = np.full((n, SEASON) + cube.shape[2:], np.nan, np.float32)
        for j in range(3):
            ok = nb[:, j] >= 0
            src = np.full((n, SEASON) + cube.shape[2:], np.nan, np.float32); src[ok] = cube[nb[ok, j]]
            if day_shift: src = np.concatenate([np.full((n, day_shift) + cube.shape[2:], np.nan, np.float32), src[:, :-day_shift]], 1)
            m = np.isnan(out[..., 0] if out.ndim == 3 else out) & ~np.isnan(src[..., 0] if src.ndim == 3 else src)
            out[m] = src[m]
        return out
    hw1, hwt = pick(WCUBE[year], 1), pick(WCUBE[year], 0); p1, pt = pick(PCUBE[year], 1), pick(PCUBE[year], 0)
    for i, c in enumerate(HWB): f[c] = hw1[..., i]; f[c + '_t'] = hwt[..., i]
    p3 = np.nansum(np.stack([p1, np.concatenate([np.full((n, 1), np.nan, np.float32), p1[:, :-1]], 1), np.concatenate([np.full((n, 2), np.nan, np.float32), p1[:, :-2]], 1)]), 0)
    p3[np.isnan(p1)] = np.nan
    f['rn_precip1'], f['rn_precip3'], f['rn_precip_t'] = p1, p3, pt
    # incidents within 10 km, known at the end of UTC day t-1 (i.e., before 00:00 UTC of day t)
    iy = inc[inc.FIRE_YEAR == year]; xy_c = np.c_[ll[:, 0] * 111, ll[:, 1] * 111 * KM]; xy_i = np.c_[iy.LATITUDE * 111, iy.LONGITUDE * 111 * KM]
    a10 = np.zeros((n, SEASON), np.float32); dact = np.full((n, SEASON), np.nan, np.float32); aact = np.full((n, SEASON), np.nan, np.float32); o3 = np.zeros((n, SEASON), np.float32)
    ign = iy.ign.to_numpy(); out_ = iy.out.to_numpy()
    for t in range(1, SEASON):
        T0 = np.datetime64(f'{year}-05-01') + np.timedelta64(t, 'D')  # 00:00 UTC of target day t
        act = (ign < T0) & ~(out_ < T0); rec = (out_ < T0) & (out_ >= T0 - np.timedelta64(3, 'D'))
        if act.any():
            tr = cKDTree(xy_i[act]); dd, jj = tr.query(xy_c, k=1); a10[:, t] = tr.query_ball_point(xy_c, 10, return_length=True)
            dact[:, t] = np.where(dd <= 50, dd, np.nan); aact[:, t] = np.where(dd <= 50, (T0 - ign[act][jj]) / np.timedelta64(1, 'D'), np.nan)
        if rec.any():
            o3[:, t] = cKDTree(xy_i[rec]).query_ball_point(xy_c, 10, return_length=True)
    f['in_active10'], f['in_dist_active'], f['in_age_active'], f['in_out3_10'] = a10, dact, aact, o3
    return cells, f

def gather(df, cells, f):
    ci = cells.get_indexer(df.h3_cell.to_numpy()); t = (pd.to_datetime(df.target_date) - pd.to_datetime(pd.to_datetime(df.target_date).dt.year.astype(str) + '-05-01')).dt.days.to_numpy()
    ok = ci >= 0; out = {}
    for k2, v in f.items():
        a = np.full(len(df), np.nan, np.float32); a[ok] = v[ci[ok], t[ok]]; out[k2] = a
    return pd.DataFrame(out, index=df.index)

# ---------------- build training and evaluation rows (burning yesterday only)
train, evrows = [], {}
for slug in REGIONS:
    for year in range(2018, 2024):
        cells, f = region_year_features(slug, year)
        if year <= 2021:
            d = pd.read_parquet(SRC / f'training_samples/{slug}/snpp_lightning_{year}.parquet'); d = d[d.local_count_1d > 0].copy()
        else:
            d = pd.read_parquet(SRC / f'evaluation/{year}/{slug}_snpp_lightning_full_grid.parquet', columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled'])
            d['row'] = np.arange(len(d)); d = d[d.local_count_1d > 0].copy()
        d = pd.concat([d, gather(d, cells, f)], axis=1); d[LC] = lc.reindex(d.h3_cell.to_numpy()).to_numpy(np.float32); d['region'] = slug
        if year <= 2021: train.append(d)
        else: evrows[(year, slug)] = d
        say('rows', slug, year, len(d), 'stop share', round(1 - d.target_snpp_filled.mean(), 3))
train = pd.concat(train, ignore_index=True); yr = pd.to_datetime(train.target_date).dt.year
say('training rows', len(train), 'missing: precip', round(float(np.isnan(train.rn_precip1).mean()), 3), 'incident dist', round(float(np.isnan(train.in_dist_active).mean()), 3))
models, info = {}, {}
for name, feats in SETS.items():
    mk = lambda x: Pool(x[feats], x.target_snpp_filled.astype(int))
    m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
    m.fit(mk(train[yr <= 2020]), eval_set=mk(train[yr == 2021])); it = max(100, m.get_best_iteration() + 1)
    m = CatBoostClassifier(iterations=it, **PARAMS); m.fit(mk(train)); m.save_model(str(RES / f'{name}.cbm')); models[name] = m
    info[name] = {'iterations': it, 'importance': dict(zip(feats, m.get_feature_importance().round(2).tolist()))}; say(name, 'iterations', it)
json.dump(info, open(RES / 'training.json', 'w'), indent=1)

# ---------------- evaluate: route burning-yesterday rows to the stop model, everything else keeps G_lchw
FW = X32 / 'fuel_wind/results'
fwm = json.load(open(FW / 'metrics.json')); thrG = fwm['thresholds_2022']['G_lchw']
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
base_counts, cont = {}, {}
for (year, slug), d in evrows.items():
    g = pd.read_parquet(FW / f'scores/{slug}_{year}.parquet', columns=['y', 'G_lchw', 'rule', 'day'])
    fl = g.G_lchw.to_numpy() >= thrG[slug]; y = g.y.to_numpy(); m = np.ones(len(g), bool); m[d.row.to_numpy()] = False
    base_counts[(year, slug)] = np.array([(fl & y & m).sum(), (fl & ~y & m).sum(), (~fl & y & m).sum()])
    d['G_flag'] = fl[d.row.to_numpy()]; d['G_score'] = g.G_lchw.to_numpy()[d.row.to_numpy()]; d['day'] = g.day.to_numpy()[d.row.to_numpy()]
    for n_, mm in models.items(): d[n_] = mm.predict_proba(d[SETS[n_]])[:, 1]
    cont[(year, slug)] = d
thr = {}
for n_ in SETS:
    thr[n_] = {}
    for slug in REGIONS:
        d = cont[(2022, slug)]; y = d.target_snpp_filled.to_numpy(bool); s = d[n_].to_numpy(); rc = base_counts[(2022, slug)]; best = (-1, 0.0)
        for c in np.quantile(s, np.linspace(0, 0.95, 300)):
            fl = s >= c; T = rc + [(fl & y).sum(), (fl & ~y).sum(), (~fl & y).sum()]; v = f1(*T)
            if v > best[0]: best = (v, float(c))
        thr[n_][slug] = best[1]
C = pd.concat([cont[(2023, s)] for s in REGIONS], ignore_index=True); yc = C.target_snpp_filled.to_numpy(bool)
rest = sum(base_counts[(2023, s)] for s in REGIONS)
R = {'continuing_rows_2023': len(C), 'continuing_fires_2023': int(yc.sum()), 'stopped_2023': int((~yc).sum()), 'thresholds': thr}
def summary(fl):
    tp, fp, fn = int((fl & yc).sum()), int((fl & ~yc).sum()), int((~fl & yc).sum()); T = rest + [tp, fp, fn]
    return {'kept_burning_caught': tp, 'false_alarms_on_stopped': fp, 'burning_dropped': fn, 'stopped_correctly_dropped': int((~fl & ~yc).sum()),
            'total_f1': f1(*T), 'total_precision': T[0] / (T[0] + T[1]), 'total_recall': T[0] / (T[0] + T[2])}
R['rule_all_flagged'] = summary(np.ones(len(C), bool)); R['G_lchw'] = summary(C.G_flag.to_numpy())
R['ap_keep_burning'] = {'G_lchw': float(average_precision_score(yc, C.G_score))}
for n_ in SETS:
    fl = np.concatenate([cont[(2023, s)][n_].to_numpy() >= thr[n_][s] for s in REGIONS]); C[n_ + '_flag'] = fl
    R[n_] = summary(fl); R['ap_keep_burning'][n_] = float(average_precision_score(yc, C[n_]))
# bootstrap over ISO weeks: total F1 vs G_lchw, and AP steps
C['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(C.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)
restw = {}
for s in REGIONS:
    g = pd.read_parquet(FW / f'scores/{s}_2023.parquet', columns=['y', 'G_lchw', 'day']); fl = g.G_lchw.to_numpy() >= thrG[s]; y = g.y.to_numpy()
    m = np.ones(len(g), bool); m[cont[(2023, s)].row.to_numpy()] = False
    wk = (pd.Timestamp('2023-05-01') + pd.to_timedelta(g.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int)
    restw[s] = pd.DataFrame({'tp': fl & y & m, 'fp': fl & ~y & m, 'fn': ~fl & y & m, 'w': wk}).groupby('w').sum()
RW = sum(restw.values()); Wk = RW.index.to_numpy(); RW = RW.to_numpy()
def cw(fl):
    return pd.DataFrame({'tp': fl & yc, 'fp': fl & ~yc, 'fn': ~fl & yc, 'w': C.week}).groupby('w').sum().reindex(Wk).fillna(0).to_numpy()
CW = {n_: cw(C[n_ + '_flag'].to_numpy()) for n_ in SETS}; CW['G_lchw'] = cw(C.G_flag.to_numpy())
grp = [np.flatnonzero(C.week.to_numpy() == w) for w in Wk]; rng = np.random.default_rng(42)
bs = {f'f1_{n_}_minus_G_lchw': [] for n_ in SETS}
steps = [('K1_life', 'K0_base'), ('K2_rain', 'K1_life'), ('K3_incidents', 'K2_rain'), ('KU_target_day', 'K3_incidents'), ('K0_base', 'G_score')]
bs.update({f'ap_{a}_minus_{b}': [] for a, b in steps})
for _ in range(1000):
    i = rng.integers(0, len(Wk), len(Wk)); c = np.bincount(i, minlength=len(Wk)).astype(float)
    base = f1(*((RW + CW['G_lchw']) * c[:, None]).sum(0))
    for n_ in SETS: bs[f'f1_{n_}_minus_G_lchw'].append(f1(*((RW + CW[n_]) * c[:, None]).sum(0)) - base)
    ii = np.concatenate([grp[j] for j in i]); yy = yc[ii]
    for a, b_ in steps:
        bs[f'ap_{a}_minus_{b_}'].append(average_precision_score(yy, C[a].to_numpy()[ii]) - average_precision_score(yy, C[b_].to_numpy()[ii]))
R['bootstrap_95'] = {k2: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k2, v in bs.items()}
R['importance_K3'] = sorted(info['K3_incidents']['importance'].items(), key=lambda x: -x[1])[:15]
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
say(json.dumps({k2: ({x: (round(v2, 4) if isinstance(v2, float) else v2) for x, v2 in v.items()} if isinstance(v, dict) and k2 != 'thresholds' else v)
                for k2, v in R.items() if k2 != 'thresholds'}, indent=1, default=float))
say('total seconds', round(time.time() - t0))
