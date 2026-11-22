"""Experiment 31: can spread features, terrain and wind improve next-door spread (and total F1)?
Next-door rows = Exp 25 rows with no S-NPP detection in the cell yesterday but at least one in a ring-1 neighbour
(local_count_1d == 0 and neighbour_count_1d > 0). Target = Exp 25's S-NPP gap-filled label.
A dedicated CatBoost per feature set is trained on next-door rows 2018-2021 (all such rows are in Exp 25's samples,
weight 1), threshold chosen on 2022, tested on 2023. For total F1, every other row keeps Exp 30 model A's flags
(fire history + lightning, same table) and the next-door threshold is chosen on 2022 to maximise total F1.
Feature sets (cumulative):
  S0 base       Exp 25's 21 features
  S1 +shape     burning / newly burning neighbours, their FRP, fire size and growth within 5 rings, growth direction,
                burned in previous 15 days, NOAA-20 burning neighbours
  S2 +terrain   cell elevation mean / std / range, elevation difference to burning neighbours (mean, max, min), uphill share
  S3 +wind      ERA5-Land day t-1 wind speed, alignment of the wind with the burning neighbours (mean cos), cos x speed
  S4 +era5      S3 + Exp 30's 22 ERA5 features
  U  upper bound: S3 with target-day (t) wind instead of t-1 ("perfect wind forecast", not a usable result)
usage: python run_experiment.py REPO_ROOT [features|train|metrics|all]"""
import io, json, sys, glob, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3, scipy.sparse as sp, pyarrow.csv as pv
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

import os
ROOT = Path(sys.argv[1]); STAGE = sys.argv[2] if len(sys.argv) > 2 else 'all'
PAIR = os.environ.get('GAW_TABLE', 'pair') == 'pair'  # two-satellite table, spread features from either satellite (Exp 32)
EXP = ROOT / 'experiments'
HERE = EXP / ('32_exp_two_satellite_bc/exp31' if PAIR else '31_exp_spread_features_bc'); RES = HERE / 'results'; RES.mkdir(parents=True, exist_ok=True)
SRC = ROOT / ('data/pair_lightning_h3r7' if PAIR else 'data/snpp_lightning_h3r7'); E5 = ROOT / 'data/era5_h3r7'
FEAT = ROOT / ('data/spread_pair_h3r7' if PAIR else 'data/spread_h3r7'); FEAT.mkdir(parents=True, exist_ok=True)
EXP30_RES = EXP / ('32_exp_two_satellite_bc/exp30/results' if PAIR else '30_exp_era5_features_bc/results')
REGIONS = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
SEASON = 153
BASE = ["local_count_1d", "local_count_3d", "local_count_7d", "neighbour_count_1d", "neighbour_count_3d",
        "neighbour_count_7d", "frp_sum_1d", "frp_sum_7d", "frp_max_1d", "night_count_1d", "active_days_7d",
        "disk1_active_days_7d", "days_since_detection", "history_left_censored", "quiet_stratum",
        "ltg_cell_1d", "ltg_3x3_1d", "ltg_3x3_3d", "ltg_9x9_1d", "ltg_9x9_3d", "ltg_days_since_3x3"]
SHAPE = ['sp_burn_nb', 'sp_new_nb', 'sp_frp_nb', 'sp_size5', 'sp_growth5', 'sp_cos_growth', 'sp_burned_15d', 'sp_n20_burn_nb']
TERRAIN = ['tr_elev_mean', 'tr_elev_std', 'tr_elev_range', 'tr_dz_mean', 'tr_dz_max', 'tr_dz_min', 'tr_uphill_share']
WIND = ['wd_speed', 'wd_align', 'wd_align_x_speed']
WIND_T = [w + '_t' for w in WIND]
ERA = json.load(open(E5 / 'era5_meta.json'))['features']
SETS = {'S0_base': BASE, 'S1_shape': BASE + SHAPE, 'S2_terrain': BASE + SHAPE + TERRAIN, 'S3_wind': BASE + SHAPE + TERRAIN + WIND,
        'S4_era5': BASE + SHAPE + TERRAIN + WIND + ERA, 'U_wind_target_day': BASE + SHAPE + TERRAIN + WIND_T}
PARAMS = dict(learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function='Logloss', random_seed=42, thread_count=-1,
              verbose=0, allow_writing_files=False)
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

def adjacency(cells, k):
    r, c = [], []
    for i, x in enumerate(cells):
        j = cells.get_indexer(list(h3.grid_ring(x, 1) if k == 1 else h3.grid_disk(x, k))); j = j[j >= 0]
        r += [i] * len(j); c += list(j)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(cells),) * 2)

def features():
    """Per region-year grids [cells x season day] of spread, terrain and wind features for target day t (inputs <= t-1,
    except the *_t wind upper bound). Saved as one parquet per region-year holding only next-door cell-days."""
    cols = ['latitude', 'longitude', 'acq_date', 'frp']
    sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=cols + ['h3_cell'], parse_dates=['acq_date'])
    sn = sn[sn.acq_date.dt.year.between(2018, 2023) & sn.acq_date.dt.month.between(5, 9)]
    with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
        n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date', 'frp'], parse_dates=['acq_date'])
    n2 = n2[n2.acq_date.dt.year.between(2018, 2023) & n2.acq_date.dt.month.between(5, 9) & n2.latitude.between(48, 60.1) & n2.longitude.between(-139.1, -114)]
    n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
    for d in (sn, n2):
        d['year'] = d.acq_date.dt.year; d['day'] = (d.acq_date - pd.to_datetime(d.year.astype(str) + '-05-01')).dt.days
    terr = pd.read_parquet(ROOT / 'data/static/terrain_h3r7.parquet').set_index('h3_cell')
    # ERA5 u10/v10 per point-day
    t = pv.read_csv(ROOT / 'dataset_ERA5_18_24.csv', convert_options=pv.ConvertOptions(include_columns=['valid_time', 'latitude', 'longitude_deg', 'u10', 'v10'])).to_pandas()
    t['valid_time'] = pd.to_datetime(t.valid_time); t['yi'] = t.valid_time.dt.year - 2018
    t['day'] = (t.valid_time - pd.to_datetime(t.valid_time.dt.year.astype(str) + '-05-01')).dt.days
    pts = t[['latitude', 'longitude_deg']].drop_duplicates().sort_values(['latitude', 'longitude_deg']).reset_index(drop=True); pts['pid'] = np.arange(len(pts))
    t = t.merge(pts, on=['latitude', 'longitude_deg']); t = t[(t.day >= 0) & (t.day < SEASON) & (t.yi <= 5)]
    UV = np.full((6, len(pts), SEASON, 2), np.nan, np.float32); UV[t.yi, t.pid, t.day] = t[['u10', 'v10']].to_numpy(np.float32); del t
    c2p = pd.read_parquet(E5 / 'cell_to_point.parquet').set_index('h3_cell')
    for slug in REGIONS:
        cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
        n = len(cells); A1, A3, A5 = adjacency(cells, 1), adjacency(cells, 3), adjacency(cells, 5)
        ll = np.array([h3.cell_to_latlng(c) for c in cells]); lat0 = np.radians(ll[:, 0].mean())
        xy = np.c_[ll[:, 1] * np.cos(lat0) * 111.0, ll[:, 0] * 111.0]
        # unit vectors from each neighbour j to cell i, stored on A1's sparsity pattern
        Ai, Aj = A1.nonzero(); vec = xy[Ai] - xy[Aj]; vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        Vx = sp.csr_matrix((vec[:, 0], (Ai, Aj)), shape=(n, n)); Vy = sp.csr_matrix((vec[:, 1], (Ai, Aj)), shape=(n, n))
        tz = terr.reindex(cells); z = tz.elev_mean_m.to_numpy(np.float32)
        dzv = np.nan_to_num(z[Ai] - z[Aj]).astype(np.float32)  # cell minus neighbour (positive = cell is uphill)
        DZ = sp.csr_matrix((dzv, (Ai, Aj)), shape=(n, n))
        DZhi = sp.csr_matrix((dzv + 1e4, (Ai, Aj)), shape=(n, n)); DZlo = sp.csr_matrix((1e4 - dzv, (Ai, Aj)), shape=(n, n))
        UP = sp.csr_matrix(((dzv > 0).astype(np.float32), (Ai, Aj)), shape=(n, n))
        pid = c2p.reindex(cells); ok_p = (pid.dist_km <= 10).to_numpy(); pidv = np.where(ok_p, pid.pid.to_numpy(), 0).astype(np.int64)
        for year in range(2018, 2024):
            def grid(d, w=None):
                d = d[d.year == year]; ci = cells.get_indexer(d.h3_cell); k = ci >= 0
                g = np.zeros((n, SEASON), np.float32); np.add.at(g, (ci[k], d.day.to_numpy()[k]), 1 if w is None else d[w].to_numpy(np.float32)[k])
                return g
            snc, frp, n2c = grid(sn), grid(sn, 'frp'), grid(n2)
            if PAIR: frp = frp + grid(n2, 'frp')
            b = ((snc + n2c) > 0) if PAIR else (snc > 0); bn2 = n2c > 0
            rows = []
            for tday in range(7, SEASON):
                b1, b2, b3 = b[:, tday - 1], b[:, tday - 2], b[:, tday - 3]
                nb = A1 @ b1.astype(np.float32)
                idx = np.flatnonzero(~b1 & (nb > 0))
                if not len(idx): continue
                new1 = b1 & ~b2 & ~b3; prior = (b2 | b3) & ~new1
                cn, cp = A3 @ new1.astype(np.float32), A3 @ prior.astype(np.float32)
                mn = (A3 @ (xy * new1[:, None])) / np.maximum(cn, 1)[:, None]; mp = (A3 @ (xy * prior[:, None])) / np.maximum(cp, 1)[:, None]
                g = (mn - mp)[idx]; dv = (xy - mn)[idx]
                with np.errstate(invalid='ignore', divide='ignore'):
                    cosg = (g * dv).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(dv, axis=1))
                cosg[(cn[idx] == 0) | (cp[idx] == 0) | (np.linalg.norm(g, axis=1) < 0.5)] = np.nan
                bf = b1.astype(np.float32)
                dz_sum = (DZ.multiply(bf[None, :])).sum(1).A1[idx]
                dz_max = DZhi[idx].multiply(bf[None, :]).max(1).toarray().ravel() - 1e4
                dz_min = 1e4 - DZlo[idx].multiply(bf[None, :]).max(1).toarray().ravel()
                up = (UP[idx] @ bf) / nb[idx]
                # wind alignment: mean over burning neighbours of cos(direction neighbour->cell, wind-to direction)
                sx, sy = (Vx @ bf)[idx] / nb[idx], (Vy @ bf)[idx] / nb[idx]
                feats = {'region': slug, 'h3_cell': cells[idx].to_numpy(), 'day': tday,
                         'sp_burn_nb': nb[idx], 'sp_new_nb': (A1 @ new1.astype(np.float32))[idx], 'sp_frp_nb': (A1 @ (frp[:, tday - 1] * bf))[idx],
                         'sp_size5': (A5 @ bf)[idx], 'sp_growth5': (A5 @ bf)[idx] - (A5 @ b2.astype(np.float32))[idx], 'sp_cos_growth': cosg,
                         'sp_burned_15d': b[idx, max(0, tday - 15):tday - 1].any(1).astype(np.float32),
                         'sp_n20_burn_nb': (A1 @ bn2[:, tday - 1].astype(np.float32))[idx] if year >= 2018 else np.nan,
                         'tr_elev_mean': z[idx], 'tr_elev_std': tz.elev_std_m.to_numpy(np.float32)[idx],
                         'tr_elev_range': (tz.elev_max_m - tz.elev_min_m).to_numpy(np.float32)[idx],
                         'tr_dz_mean': dz_sum / nb[idx], 'tr_dz_max': dz_max, 'tr_dz_min': dz_min, 'tr_uphill_share': up,
                         'sp_sx': sx, 'sp_sy': sy}  # mean unit vector from burning neighbours to the cell (east, north)
                for suffix, wd in (('', tday - 1), ('_t', tday)):
                    u, v = UV[year - 2018, pidv[idx], wd, 0], UV[year - 2018, pidv[idx], wd, 1]
                    u = np.where(ok_p[idx], u, np.nan); v = np.where(ok_p[idx], v, np.nan); spd = np.hypot(u, v)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        al = (sx * u + sy * v) / spd
                    feats['wd_speed' + suffix] = spd; feats['wd_align' + suffix] = al; feats['wd_align_x_speed' + suffix] = al * spd
                rows.append(pd.DataFrame(feats))
            out = pd.concat(rows, ignore_index=True)
            out['target_date'] = pd.Timestamp(f'{year}-05-01') + pd.to_timedelta(out.day, unit='D')
            out.drop(columns='day').to_parquet(FEAT / f'{slug}_{year}.parquet', index=False)
            say('features', slug, year, len(out))

def era5(df):
    cube = np.load(E5 / 'era5_point_features.npy', mmap_mode='r'); m = pd.read_parquet(E5 / 'cell_to_point.parquet').set_index('h3_cell')
    pid = m.pid.astype(float).where(m.dist_km <= 10).reindex(df.h3_cell.to_numpy()).to_numpy()
    d = pd.to_datetime(df.target_date); yi = (d.dt.year - 2018).to_numpy(); di = (d - pd.to_datetime(d.dt.year.astype(str) + '-05-01')).dt.days.to_numpy() - 1
    ok = ~np.isnan(pid); out = np.full((len(df), len(ERA)), np.nan, np.float32); out[ok] = cube[yi[ok], pid[ok].astype(np.int64), di[ok]]
    return pd.DataFrame(out, columns=ERA, index=df.index)

def joined(kind, year, region):
    if kind == 'train':
        f = SRC / f'training_samples/{region}/snpp_lightning_{year}.parquet'; d = pd.read_parquet(f)
    else:
        f = SRC / f'evaluation/{year}/{region}_snpp_lightning_full_grid.parquet'
        d = pd.read_parquet(f, columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled']); d['row'] = np.arange(len(d))
    d = d[(d.local_count_1d == 0) & (d.neighbour_count_1d > 0)]
    s = pd.read_parquet(FEAT / f'{region}_{year}.parquet'); s['target_date'] = pd.to_datetime(s.target_date).astype('datetime64[ns]')
    d = d.assign(target_date=pd.to_datetime(d.target_date).astype('datetime64[ns]')).merge(s.drop(columns='region'), on=['h3_cell', 'target_date'], how='left')
    d = pd.concat([d.reset_index(drop=True), era5(d.reset_index(drop=True))], axis=1); d['region'] = region
    return d

def train():
    t = pd.concat([joined('train', y, r) for r in REGIONS for y in range(2018, 2022)], ignore_index=True)
    say('next-door training rows', len(t), 'burn rate', round(t.target_snpp_filled.mean(), 4), 'spread features missing', round(float(t.sp_burn_nb.isna().mean()), 4))
    yr = t.target_date.dt.year; info = {}
    for name, feats in SETS.items():
        mk = lambda x: Pool(x[feats], x.target_snpp_filled.astype(int))
        m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
        m.fit(mk(t[yr <= 2020]), eval_set=mk(t[yr == 2021])); it = max(100, m.get_best_iteration() + 1)
        m = CatBoostClassifier(iterations=it, **PARAMS); m.fit(mk(t)); m.save_model(str(RES / f'{name}.cbm'))
        info[name] = {'iterations': it, 'importance': dict(zip(feats, m.get_feature_importance().round(3).tolist()))}
        say(name, 'iterations', it)
    json.dump(info, open(RES / 'training.json', 'w'), indent=1)

def best_f1_thr(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys)
    f1 = 2 * tp / (tp + fp + ys.sum()); k = int(np.argmax(f1)); return float(s[o][k])

def metrics():
    models = {n: CatBoostClassifier().load_model(str(RES / f'{n}.cbm')) for n in SETS}
    A = {(y, r): pd.read_parquet(EXP30_RES / f'scores/{r}_{y}.parquet', columns=['y', 'A_full_base'])
         for y in (2022, 2023) for r in REGIONS}
    a_thr = json.load(open(EXP30_RES / 'metrics.json'))['thresholds_2022']
    S = {}
    for y in (2022, 2023):
        for r in REGIONS:
            d = joined('eval', y, r)
            for n, feats in SETS.items():
                d[n] = models[n].predict_proba(d[feats])[:, 1]
            d['A'] = A[(y, r)].A_full_base.to_numpy()[d.row.to_numpy()]; d['A_flag'] = d.A >= a_thr[r]['A_full_base']
            S[(y, r)] = d
    # rest-of-grid counts from model A (non next-door rows)
    rest = {}
    for (y, r), d in S.items():
        a = A[(y, r)]; fl = a.A_full_base.to_numpy() >= a_thr[r]['A_full_base']; yy = a.y.to_numpy()
        mask = np.ones(len(a), bool); mask[d.row.to_numpy()] = False
        rest[(y, r)] = np.array([(fl & yy & mask).sum(), (fl & ~yy & mask).sum(), (~fl & yy & mask).sum()])
    R = {}
    nd23 = pd.concat([S[(2023, r)] for r in REGIONS]); y23 = nd23.target_snpp_filled.to_numpy(bool)
    rest23 = sum(rest[(2023, r)] for r in REGIONS)
    def pooled(flags, d, restc):
        y = d.target_snpp_filled.to_numpy(bool); tp, fp, fn = (flags & y).sum(), (flags & ~y).sum(), (~flags & y).sum()
        T = restc + [tp, fp, fn]; return {'nd_recall': tp / (tp + fn), 'nd_precision': tp / max(tp + fp, 1), 'nd_f1': 2 * tp / (2 * tp + fp + fn),
                                          'total_f1': 2 * T[0] / (2 * T[0] + T[1] + T[2])}
    R['next_door_rows_2023'] = int(len(nd23)); R['next_door_fires_2023'] = int(y23.sum())
    R['model_A_on_next_door'] = pooled(nd23.A_flag.to_numpy(), nd23, rest23) | {'ap': average_precision_score(y23, nd23.A)}
    for n in SETS:
        thr = {}
        for r in REGIONS:  # threshold per centre on 2022 maximising that centre's total F1 (rest fixed at model A)
            d = S[(2022, r)]; yy = d.target_snpp_filled.to_numpy(bool); s = d[n].to_numpy(); rc = rest[(2022, r)]
            cand = np.quantile(s, np.linspace(0.5, 0.9995, 400)); best = (-1, None)
            for c in cand:
                fl = s >= c; T = rc + [(fl & yy).sum(), (fl & ~yy).sum(), (~fl & yy).sum()]; f = 2 * T[0] / (2 * T[0] + T[1] + T[2])
                if f > best[0]: best = (f, float(c))
            thr[r] = best[1]
        fl = np.concatenate([S[(2023, r)][n].to_numpy() >= thr[r] for r in REGIONS])
        R[n] = pooled(fl, nd23, rest23) | {'ap': average_precision_score(y23, nd23[n])}
    # paired bootstrap over ISO weeks of next-door AP differences
    nd23[['region', 'h3_cell', 'target_date', 'target_snpp_filled', 'A'] + list(SETS)].to_parquet(RES / 'next_door_scores_2023.parquet', index=False)
    wk = nd23.target_date.dt.isocalendar().week.to_numpy(); W = np.unique(wk); rng = np.random.default_rng(42)
    groups = [np.flatnonzero(wk == w) for w in W]
    pairs = [('S1_shape', 'A'), ('S1_shape', 'S0_base'), ('S2_terrain', 'S1_shape'), ('S3_wind', 'S2_terrain'),
             ('S4_era5', 'S3_wind'), ('U_wind_target_day', 'S3_wind'), ('S3_wind', 'A')]
    bs = {f'{a}_minus_{b}': [] for a, b in pairs}
    sc = {k: nd23[k].to_numpy() for k in ['A'] + list(SETS)}
    for _ in range(1000):
        ii = np.concatenate([groups[j] for j in rng.integers(0, len(W), len(W))]); yy = y23[ii]
        ap = {k: average_precision_score(yy, v[ii]) for k, v in sc.items()}
        for a, b in pairs: bs[f'{a}_minus_{b}'].append(ap[a] - ap[b])
    R['bootstrap_ap_95'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    say(json.dumps({k: ({x: (round(float(v), 4) if np.isscalar(v) else [round(q, 4) for q in v]) for x, v in R[k].items()} if isinstance(R[k], dict) else R[k]) for k in R}, indent=1))
    tr = json.load(open(RES / 'training.json'))
    R['importance_S3'] = sorted(tr['S3_wind']['importance'].items(), key=lambda x: -x[1])[:15]
    json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
    say('top S3 importance', R['importance_S3'])

if STAGE in ('features', 'all'): features()
if STAGE in ('train', 'all'): train()
if STAGE in ('metrics', 'all'): metrics()
