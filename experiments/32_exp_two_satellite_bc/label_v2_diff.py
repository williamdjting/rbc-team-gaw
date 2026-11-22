"""Compare label v1 (current: S-NPP or NOAA-20, 1-day gaps filled) with label v2 (also fill 2-4 day gaps inside a
BCWS wildfire's final perimeter, 1-ring buffer, or within 10 km of its ignition point, when that fire ignited before the
gap and was not declared out before it ended). Counts per year (May 8 - Sep 30 target days), and one example cell.
usage: python label_v2_diff.py REPO_ROOT"""
import io, sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3
from scipy.spatial import cKDTree
ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
d = pd.concat([sn, n2[['acq_date', 'h3_cell']]]); d['date'] = pd.to_datetime(d.acq_date)
d = d[d.date.dt.month.between(5, 9) & d.date.dt.year.between(2018, 2023)]
grid = set(pd.concat([pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{s}/data/h3_grid.parquet', columns=['h3_cell'])
           for s in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']]).h3_cell)
d = d[d.h3_cell.isin(grid)]; d['year'] = d.date.dt.year; d['day'] = (d.date - pd.to_datetime(d.year.astype(str) + '-05-01')).dt.days
u = d[['year', 'h3_cell', 'day']].drop_duplicates().sort_values(['year', 'h3_cell', 'day'])
incA = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv'); incA = incA[incA.FIRE_TYPE == 'Fire'].copy()
per = json.load(open(ROOT / 'data/bcws_perimeters/bc_fire_perimeters_2018_2023.geojson'))['features']
k = np.cos(np.radians(52)); rows = []; example = None
for Y in range(2018, 2024):
    inc = incA[incA.FIRE_YEAR == Y].copy()
    inc['ign'] = (pd.to_datetime(inc.IGNITION_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
    inc['out'] = (pd.to_datetime(inc.FIRE_OUT_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
    dates = inc.drop_duplicates('FIRE_NUMBER').set_index('FIRE_NUMBER')[['ign', 'out']]
    cf = {}
    for f in per:
        p = f['properties']
        if p['FIRE_YEAR'] != Y or p['FIRE_NUMBER'] not in dates.index or f['geometry'] is None: continue
        try: cs = set(h3.geo_to_cells(f['geometry'], 7))
        except Exception: continue
        cs |= {n for c in cs for n in h3.grid_ring(c, 1)}
        for c in cs: cf.setdefault(c, []).append(p['FIRE_NUMBER'])
    tree = cKDTree(np.c_[inc.LATITUDE * 111, inc.LONGITUDE * 111 * k]); iv, ov = inc.ign.to_numpy(), inc.out.to_numpy()
    act = lambda i, o, a, b: (i <= a + 1) and not (o < b - 1)
    uy = u[u.year == Y].copy(); uy['prev'] = uy.groupby('h3_cell').day.shift(); uy['gap'] = uy.day - uy.prev - 1
    g = uy[(uy.gap >= 2) & (uy.gap <= 4)].copy(); g['prev'] = g.prev.astype(int)
    ev = []
    for c, a, b in zip(g.h3_cell, g.prev, g.day.astype(int)):
        e = any(act(*dates.loc[fn].values, a, b) for fn in cf.get(c, []))
        if not e:
            lat, lon = h3.cell_to_latlng(c); e = any(act(iv[j], ov[j], a, b) for j in tree.query_ball_point([lat * 111, lon * 111 * k], 10))
        ev.append(e)
    g['filled'] = ev
    one = uy[uy.gap == 1]
    det_days = uy[uy.day.between(6, 151)]  # detections on history days of target days May 8 - Sep 30 are not the label; count label days directly
    v1 = len(uy[uy.day.between(7, 152)]) + int(one.day.between(8, 153).sum())  # detected target days + 1-day fills (approx. at season edge)
    added = int(sum(gp for gp, f, dd in zip(g.gap, g.filled, g.day) if f))
    rows.append({'year': Y, 'label_v1_positives': v1, 'gaps_2_4': len(g), 'gaps_filled': int(g.filled.sum()),
                 'share_filled': round(g.filled.mean(), 3) if len(g) else np.nan, 'cell_days_added': added,
                 'label_v2_positives': v1 + added, 'increase': round(added / v1, 3), 'stops_removed': int(g.filled.sum())})
    if Y == 2023 and example is None:
        ex = g[g.filled & (g.gap == 3)].iloc[0]; c = ex.h3_cell
        days = uy[uy.h3_cell == c].day.to_numpy(); lo, hi = int(ex.prev) - 3, int(ex.day) + 3
        example = (c, lo, hi, set(days), set(range(int(ex.prev) + 1, int(ex.day))), [fn for fn in cf.get(c, [])])
T = pd.DataFrame(rows); print(T.to_string(index=False))
print('\nall years: gaps', int(T.gaps_2_4.sum()), 'filled', int(T.gaps_filled.sum()), 'cell-days added', int(T.cell_days_added.sum()))
c, lo, hi, det, gap, fires = example
print(f'\nExample cell {c} (BCWS fire {fires[0] if fires else "-"}), days {lo}-{hi} after May 1, 2023:')
print('day      ' + ' '.join(f'{(pd.Timestamp("2023-05-01") + pd.Timedelta(days=t)).strftime("%m-%d")}' for t in range(lo, hi + 1)))
print('detected ' + ' '.join('  X  ' if t in det else '  .  ' for t in range(lo, hi + 1)))
print('label v1 ' + ' '.join('  1  ' if t in det else '  0  ' for t in range(lo, hi + 1)))
print('label v2 ' + ' '.join('  1  ' if (t in det or t in gap) else '  0  ' for t in range(lo, hi + 1)))
