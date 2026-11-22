"""Quick checks for Exp 31: which simple features separate next-door cells that burn tomorrow from those that don't?
Next-door cell on target day t = not burning on t-1, with at least one ring-1 neighbour burning on t-1
(burning = S-NPP or NOAA-20 detection, FIRMS). Label = next-day two-satellite gap-filled label on t.
All target days May 8 - Sep 30, 2022 and 2023, six Fire Centres. Prints burn rate by feature bin; writes ring1_checks.json.
usage: python ring1_checks.py REPO_ROOT OUT_DIR"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3, scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); EXP = ROOT / 'experiments'
SEASON, FIRST, YEARS = 153, 7, (2022, 2023)
season_day = lambda d: (d - pd.to_datetime(d.dt.year.astype(str) + '-05-01')).dt.days.to_numpy()
cols = ['latitude', 'longitude', 'acq_date', 'frp']
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=cols + ['h3_cell'], parse_dates=['acq_date'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=cols, parse_dates=['acq_date'])
n2 = n2[n2.latitude.between(48, 60.1) & n2.longitude.between(-139.1, -114)]
det = pd.concat([sn, n2], ignore_index=True)
det = det[det.acq_date.dt.year.isin(YEARS) & det.acq_date.dt.month.between(5, 9)].copy()
miss = det.h3_cell.isna()
det.loc[miss, 'h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(det.latitude[miss], det.longitude[miss])]
det['year'] = det.acq_date.dt.year; det['day'] = season_day(det.acq_date)
print('detections', len(det), flush=True)

def adjacency(cells, k):
    rows, colsx = [], []
    for i, c in enumerate(cells):
        r = cells.get_indexer(list(h3.grid_disk(c, k) if k > 1 else h3.grid_ring(c, 1))); r = r[r >= 0]
        rows += [i] * len(r); colsx += list(r)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, colsx)), shape=(len(cells),) * 2)

out = []
spill = []
for slug in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']:
    cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
    n = len(cells); A1, A3, A5 = adjacency(cells, 1), adjacency(cells, 3), adjacency(cells, 5)
    ll = np.array([h3.cell_to_latlng(c) for c in cells]); lat0 = np.radians(ll[:, 0].mean())
    xy = np.c_[ll[:, 1] * np.cos(lat0) * 111.0, ll[:, 0] * 111.0]
    for year in YEARS:
        d = det[det.year == year]; ci = cells.get_indexer(d.h3_cell); k = ci >= 0; d = d[k]; ci = ci[k]
        pair = np.zeros((n, SEASON), bool); pair[ci, d.day.to_numpy()] = True
        frp = np.zeros((n, SEASON), np.float32); np.add.at(frp, (ci, d.day.to_numpy()), d.frp.to_numpy(np.float32))
        lab = pair.copy(); lab[:, 1:-1] |= pair[:, :-2] & pair[:, 2:]
        pts = {dd: g[['longitude', 'latitude']].to_numpy() for dd, g in d.groupby('day')}
        for t in range(FIRST, SEASON):
            b1, b2, b3 = pair[:, t - 1], pair[:, t - 2], pair[:, t - 3]
            nb1 = A1 @ b1.astype(np.float32)
            ring1 = (~b1) & (nb1 > 0)
            if not ring1.any(): continue
            idx = np.flatnonzero(ring1)
            new1 = b1 & ~b2 & ~b3                     # cells newly burning yesterday (active front)
            prior = (b2 | b3) & ~new1
            burned_before = pair[:, max(0, t - 15):t - 1].any(1)
            size5 = A5 @ b1.astype(np.float32); size5_prev = A5 @ b2.astype(np.float32)
            # growth direction: centroid of new cells minus centroid of prior cells within 3 rings
            cn = A3 @ new1.astype(np.float32); cp = A3 @ prior.astype(np.float32)
            mx_n = (A3 @ (xy * new1[:, None])) / np.maximum(cn, 1)[:, None]
            mx_p = (A3 @ (xy * prior[:, None])) / np.maximum(cp, 1)[:, None]
            g = mx_n - mx_p; dvec = xy - mx_n
            with np.errstate(invalid='ignore', divide='ignore'):
                cosg = (g * dvec).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(dvec, axis=1))
            cosg[(cn == 0) | (cp == 0) | (np.linalg.norm(g, axis=1) < 0.5)] = np.nan
            frp_nb = A1 @ (frp[:, t - 1] * b1)
            y = lab[idx, t]
            out.append(pd.DataFrame({'region': slug, 'year': year, 'y': y, 'n_burn_nb': nb1[idx], 'n_new_nb': (A1 @ new1.astype(np.float32))[idx],
                                     'burned_before_15d': burned_before[idx], 'size5': size5[idx], 'growth5': size5[idx] - size5_prev[idx],
                                     'cos_growth': cosg[idx], 'frp_nb': frp_nb[idx]}))
            # spill-over: for next-door cells that burn on t, distance from their day-t detections to yesterday's detections
            if y.any() and t in pts and (t - 1) in pts:
                tree = cKDTree(pts[t - 1] * [np.cos(lat0) * 111, 111])
                dt = d[(d.day == t)]; dci = cells.get_indexer(dt.h3_cell)
                for c in idx[y]:
                    p = dt[dci == c][['longitude', 'latitude']].to_numpy()
                    if len(p):
                        spill.append(float(tree.query(p * [np.cos(lat0) * 111, 111])[0].max()))
    print('done', slug, flush=True)

D = pd.concat(out, ignore_index=True)
base = D.y.mean()
res = {'cell_days': len(D), 'base_burn_rate': float(base)}
def show(name, bins):
    t = D.groupby(bins, observed=True).y.agg(['size', 'mean']).rename(columns={'size': 'cells', 'mean': 'burn_rate'})
    t['lift'] = t.burn_rate / base
    print(f'\n{name}'); print(t.round(3).to_string()); res[name] = t.reset_index().astype({t.index.name or 'index': str}).to_dict('records')
show('burning neighbours (of 6)', pd.cut(D.n_burn_nb, [0, 1, 2, 3, 6], labels=['1', '2', '3', '4-6']).rename('burning_neighbours'))
show('newly burning neighbours (active front)', pd.cut(D.n_new_nb, [-1, 0, 1, 6], labels=['0', '1', '2+']).rename('new_neighbours'))
show('cell burned in previous 15 days', D.burned_before_15d.rename('burned_before'))
show('fire size: burning cells within 5 rings', pd.cut(D.size5, [0, 2, 5, 15, 50, 1e9], labels=['1-2', '3-5', '6-15', '16-50', '50+']).rename('size5'))
show('growth in 5 rings since day before', pd.cut(D.growth5, [-1e9, -1, 0, 5, 1e9], labels=['shrinking', 'flat', '+1-5', '+6 or more']).rename('growth5'))
show('direction of recent growth', pd.cut(D.cos_growth, [-1.01, -0.5, 0.5, 1.01], labels=['behind the front', 'side', 'ahead of the front']).rename('growth_direction'))
show('FRP of burning neighbours (MW)', pd.qcut(D.frp_nb, 4, duplicates='drop').rename('frp_nb'))
S = np.array(spill)
res['spill_km'] = {'n': len(S), 'le_0_5': float((S <= 0.5).mean()), 'le_1': float((S <= 1).mean()), 'gt_2': float((S > 2).mean())}
print('\nspill-over: farthest day-t detection from any day t-1 detection, next-door burned cells', res['spill_km'])
json.dump(res, open(OUT / 'ring1_checks.json', 'w'), indent=1, default=str)
