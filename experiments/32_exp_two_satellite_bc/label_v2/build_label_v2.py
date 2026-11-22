"""Label v2: label v1 (S-NPP or NOAA-20, 1-day gaps filled) plus 2-4 day gaps filled when the cell is inside a BCWS
wildfire's final perimeter (1-ring buffer) or within 10 km of its ignition point, and that fire ignited before the gap
and was not declared out before it ended. Labels may use hindsight; inputs are unchanged (raw detections to t-1).
Writes data/pair_v2_lightning_h3r7/ with the same files as data/pair_lightning_h3r7/; `target_snpp_filled` = label v2,
`target_v1` = label v1. Also writes added_cell_days.parquet. usage: python build_label_v2.py REPO_ROOT"""
import io, sys, json, zipfile, glob
from pathlib import Path
import numpy as np, pandas as pd, h3
from scipy.spatial import cKDTree
ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'; SRC = ROOT / 'data/pair_lightning_h3r7'; OUT = ROOT / 'data/pair_v2_lightning_h3r7'
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
d = pd.concat([sn, n2[['acq_date', 'h3_cell']]]); d['date'] = pd.to_datetime(d.acq_date)
d = d[d.date.dt.month.between(5, 9) & d.date.dt.year.between(2018, 2023)]
d['year'] = d.date.dt.year; d['day'] = (d.date - pd.to_datetime(d.year.astype(str) + '-05-01')).dt.days
u = d[['year', 'h3_cell', 'day']].drop_duplicates().sort_values(['year', 'h3_cell', 'day'])
incA = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv'); incA = incA[incA.FIRE_TYPE == 'Fire'].copy()
per = json.load(open(ROOT / 'data/bcws_perimeters/bc_fire_perimeters_2018_2023.geojson'))['features']
k = np.cos(np.radians(52)); added = []
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
    g = uy[(uy.gap >= 2) & (uy.gap <= 4)]
    for c, a, b in zip(g.h3_cell, g.prev.astype(int), g.day.astype(int)):
        e = any(act(*dates.loc[fn].values, a, b) for fn in cf.get(c, []))
        if not e:
            lat, lon = h3.cell_to_latlng(c); e = any(act(iv[j], ov[j], a, b) for j in tree.query_ball_point([lat * 111, lon * 111 * k], 10))
        if e:
            added += [(c, pd.Timestamp(f'{Y}-05-01') + pd.Timedelta(days=t)) for t in range(a + 1, b)]
    print(Y, 'filled cell-days so far', len(added), flush=True)
A = pd.DataFrame(added, columns=['h3_cell', 'target_date']).drop_duplicates(); A['add'] = True
OUT.mkdir(parents=True, exist_ok=True); A.to_parquet(OUT / 'added_cell_days.parquet', index=False)
for f in sorted(glob.glob(str(SRC / '**/*.parquet'), recursive=True)):
    t = pd.read_parquet(f); t['target_date'] = pd.to_datetime(t.target_date).astype('datetime64[ns]'); n0 = len(t)
    t = t.merge(A, on=['h3_cell', 'target_date'], how='left'); assert len(t) == n0
    t['target_v1'] = t.target_snpp_filled.astype(bool); t['target_snpp_filled'] = t.target_v1 | t['add'].fillna(False).astype(bool)
    t = t.drop(columns='add'); o = OUT / Path(f).relative_to(SRC); o.parent.mkdir(parents=True, exist_ok=True); t.to_parquet(o, index=False)
    print(Path(f).relative_to(SRC), 'v1', int(t.target_v1.sum()), 'v2', int(t.target_snpp_filled.sum()), flush=True)
