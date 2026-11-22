"""How Exp 21 CatBoost (`exp16_spread`) handles fire spread in space, 2023.
On Exp 38's 18 scored dates (every 8th day, every cell), with the rebuilt next-day two-satellite label (as cb_nextday.py)
and Exp 21's per-centre 2022 thresholds:
  - distance (H3 rings) from each cell to the nearest cell burning yesterday (S-NPP or NOAA-20 on day t-1);
  - recall, precision, flag rate and median score by distance band;
  - for fires 1-3 rings from yesterday's fire: is the new cell downwind, crosswind or upwind of its nearest burning
    cell (ERA5-Land 10 m wind on day t-1, mean of 5 snapshots)?
usage: python cb_spread.py REPO_ROOT OUT_DIR"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3, pyarrow.csv as pv, pyarrow.compute as pc

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); EXP = ROOT / 'experiments'
SC = EXP / '38_exp_canadafiresat_protocol_bc/results/scores'
SEASON, FIRST, WIN, MAXD = 153, 7, 8, 10
ISSUE = np.arange(FIRST, SEASON - WIN + 1, WIN)
thr = (pd.read_csv(EXP / '21_exp_weather_lightning_spread_bc/results/metrics_by_year.csv')
       .query("year==2023 and definition=='res7_exact' and model=='catboost_spread_noday' and method=='f1_2022'")
       .set_index('region').threshold.to_dict())
season_day = lambda d: (d - pd.to_datetime(d.dt.year.astype(str) + '-05-01')).dt.days.to_numpy()

sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell'], parse_dates=['acq_date'])
sn = sn[(sn.acq_date.dt.year == 2023) & sn.acq_date.dt.month.between(5, 9)]
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date'], parse_dates=['acq_date'])
n2 = n2[(n2.acq_date.dt.year == 2023) & n2.acq_date.dt.month.between(5, 9) & n2.latitude.between(48, 60.1) & n2.longitude.between(-139.1, -114)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]

# ERA5-Land wind for 2023, history days only
hist_dates = set((pd.Timestamp('2023-05-01') + pd.to_timedelta(ISSUE - 1, unit='D')).strftime('%Y-%m-%d'))
t = pv.read_csv(ROOT / 'dataset_ERA5_18_24.csv', convert_options=pv.ConvertOptions(include_columns=['valid_time', 'latitude', 'longitude_deg', 'u10', 'v10'],
                                                                                    column_types={'valid_time': 'string'}))
t = t.filter(pc.is_in(t['valid_time'], value_set=__import__('pyarrow').array(sorted(hist_dates)))).to_pandas()
t['day'] = (pd.to_datetime(t.valid_time) - pd.Timestamp('2023-05-01')).dt.days
c2p = pd.read_parquet(ROOT / 'data/era5_h3r7/cell_to_point.parquet').set_index('h3_cell')
meta_pts = t[['latitude', 'longitude_deg']].drop_duplicates().sort_values(['latitude', 'longitude_deg']).reset_index(drop=True)
meta_pts['pid'] = np.arange(len(meta_pts))  # same ordering as build_era5_features.py
t = t.merge(meta_pts, on=['latitude', 'longitude_deg'])
wind = {(int(p), int(d)): (u, v) for p, d, u, v in zip(t.pid, t.day, t.u10, t.v10)}
print('ERA5 wind rows', len(t), flush=True)

def grid(det, idx):
    c = idx.get_indexer(det.h3_cell); k = c >= 0
    g = np.zeros((len(idx), SEASON), bool); g[c[k], season_day(det.acq_date[k])] = True
    return g
shift = lambda a, k: np.pad(a, ((0, 0), (k, 0)))[:, :-k] if k > 0 else np.pad(a, ((0, 0), (0, -k)))[:, -k:]
BANDS = [(0, 0, 'burning yesterday (0)'), (1, 1, '1 ring (~2.4 km)'), (2, 2, '2 rings (~5 km)'), (3, 3, '3 rings (~7 km)'),
         (4, 5, '4-5 rings (~10-12 km)'), (6, 10, '6-10 rings (~15-24 km)'), (11, 99, 'farther / no fire nearby')]
rows, wrows = [], []
for f in sorted(SC.glob('*_2023.parquet')):
    slug = f.name.replace('_2023.parquet', '')
    cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
    n = len(cells)
    nb = np.full((n, 6), -1, np.int64)
    for i, c in enumerate(cells):
        r = cells.get_indexer(list(h3.grid_ring(c, 1))); r = r[r >= 0]; nb[i, :len(r)] = r
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    pid = c2p.reindex(cells).pid.to_numpy(); far = c2p.reindex(cells).dist_km.to_numpy() > 10
    d = pd.read_parquet(f, columns=['region', 'cell', 'issue', 'exp16_spread'])
    region = d.region.iloc[0]
    pair = grid(sn, cells) | grid(n2, cells)
    label = pair | (shift(pair, 1) & shift(pair, -1))
    for ii, day in enumerate(ISSUE):
        dd = d[d.issue == ii]; ci = dd.cell.to_numpy()
        burning = pair[:, day - 1]
        dist = np.full(n, 99, np.int16); src = np.full(n, -1, np.int64)
        dist[burning] = 0; src[burning] = np.flatnonzero(burning); front = np.flatnonzero(burning)
        for k in range(1, MAXD + 1):
            if not len(front): break
            cand = nb[front].ravel(); par = np.repeat(src[front], 6)
            ok = (cand >= 0); cand, par = cand[ok], par[ok]
            ok = dist[cand] == 99; cand, par = cand[ok], par[ok]
            cand, first = np.unique(cand, return_index=True)
            dist[cand] = k; src[cand] = par[first]; front = cand
        y = label[ci, day]; s = dd.exp16_spread.to_numpy(); fl = s >= thr[region]; dc = dist[ci]
        for lo, hi, name in BANDS:
            m = (dc >= lo) & (dc <= hi)
            rows.append({'region': region, 'band': name, 'cells': int(m.sum()), 'fires': int((y & m).sum()), 'caught': int((y & fl & m).sum()),
                         'flagged': int((fl & m).sum()), 'score_fire_sum': float(s[y & m].sum())})
        # wind alignment for fire cells 1-3 rings away
        sel = np.flatnonzero(y & (dc >= 1) & (dc <= 3))
        for j in sel:
            c, sidx = ci[j], src[ci[j]]
            if far[c] or np.isnan(pid[c]): continue
            u, v = wind.get((int(pid[c]), int(day - 1)), (np.nan, np.nan))
            sp = np.hypot(u, v)
            if not sp > 1: continue
            lat0 = np.radians(ll[sidx, 0])
            vec = np.array([(ll[c, 1] - ll[sidx, 1]) * np.cos(lat0), ll[c, 0] - ll[sidx, 0]])
            cosang = vec @ np.array([u, v]) / (np.linalg.norm(vec) * sp)
            wrows.append({'region': region, 'dist': int(dc[j]), 'cos': float(cosang), 'caught': bool(fl[j]), 'score': float(s[j])})
    print('done', region, flush=True)

R = pd.DataFrame(rows).groupby('band', sort=False).sum(numeric_only=True)
R['share_of_fires'] = R.fires / R.fires.sum(); R['recall'] = R.caught / R.fires
R['precision'] = R.caught / R.flagged.replace(0, np.nan); R['flag_rate'] = R.flagged / R.cells; R['fire_rate'] = R.fires / R.cells
R = R.drop(columns='score_fire_sum')
W = pd.DataFrame(wrows)
W['direction'] = pd.cut(W.cos, [-1.01, -0.5, 0.5, 1.01], labels=['upwind (against wind)', 'crosswind', 'downwind'])
WD = W.groupby('direction', observed=True).agg(fires=('caught', 'size'), caught=('caught', 'mean'), median_score=('score', 'median'))
WD['share'] = WD.fires / WD.fires.sum()
WD2 = W.groupby(['dist', 'direction'], observed=True).agg(fires=('caught', 'size'), recall=('caught', 'mean')).reset_index()
pd.set_option('display.width', 200)
print(R.round(4).to_string()); print(WD.round(3).to_string()); print(WD2.round(3).to_string())
json.dump({'bands': R.reset_index().to_dict('records'), 'wind': WD.reset_index().astype({'direction': str}).to_dict('records'),
           'wind_by_dist': WD2.astype({'direction': str}).to_dict('records')}, open(OUT / 'cb_spread.json', 'w'), indent=1)
