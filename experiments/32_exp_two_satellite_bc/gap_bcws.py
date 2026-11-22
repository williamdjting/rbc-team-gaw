"""Can BCWS fires cover 2-4 day detection gaps (2023)? A gap in cell c counts as covered if a BCWS wildfire
(a) has c inside its final perimeter (H3 cells whose centre is inside, plus a 1-ring buffer), or (b) for comparison,
has its ignition point within 10 km of c; and the fire was active over the gap: ignited before the gap's first day,
not declared out before the gap's last day. Perimeters are final burned areas (labels may use hindsight).
usage: python gap_bcws.py REPO_ROOT"""
import io, sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3
from scipy.spatial import cKDTree
ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'; Y = 2023
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5) & n2.acq_date.str.startswith(str(Y))]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
d = pd.concat([sn[sn.acq_date.str.startswith(str(Y))], n2[['acq_date', 'h3_cell']]]); d['date'] = pd.to_datetime(d.acq_date)
d = d[d.date.dt.month.between(5, 9)]
grid = set(pd.concat([pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{s}/data/h3_grid.parquet', columns=['h3_cell'])
           for s in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']]).h3_cell)
d = d[d.h3_cell.isin(grid)]; d['day'] = (d.date - pd.Timestamp(f'{Y}-05-01')).dt.days
u = d[['h3_cell', 'day']].drop_duplicates().sort_values(['h3_cell', 'day']); seen = set(zip(u.h3_cell, u.day))
u['prev'] = u.groupby('h3_cell').day.shift(); u['gap'] = u.day - u.prev - 1
g = u[(u.gap >= 2) & (u.gap <= 4)].copy(); g['prev'] = g.prev.astype(int)
inc = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv')
inc = inc[(inc.FIRE_YEAR == Y) & (inc.FIRE_TYPE == 'Fire')].copy()
inc['ign'] = (pd.to_datetime(inc.IGNITION_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
inc['out'] = (pd.to_datetime(inc.FIRE_OUT_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
dates = inc.set_index('FIRE_NUMBER')[['ign', 'out']]
per = json.load(open(ROOT / 'data/bcws_perimeters/bc_fire_perimeters_2018_2023.geojson'))['features']
cell_fires = {}
nper = 0
for f in per:
    p = f['properties']
    if p['FIRE_YEAR'] != Y or p['FIRE_NUMBER'] not in dates.index or f['geometry'] is None: continue
    try: cs = set(h3.geo_to_cells(f['geometry'], 7))
    except Exception: continue
    nper += 1
    cs |= {n for c in cs for n in h3.grid_ring(c, 1)}
    for c in cs: cell_fires.setdefault(c, []).append(p['FIRE_NUMBER'])
k = np.cos(np.radians(52)); tree = cKDTree(np.c_[inc.LATITUDE * 111, inc.LONGITUDE * 111 * k])
ignv, outv = inc.ign.to_numpy(), inc.out.to_numpy()
def active(fire_ign, fire_out, a, b):  # gap days a+1 .. b-1
    return (fire_ign <= a + 1) and not (fire_out < b - 1)
per_ev, pt_ev, nb_ev = [], [], []
for c, a, b in zip(g.h3_cell, g.prev, g.day.astype(int)):
    per_ev.append(any(active(*dates.loc[fn].values, a, b) for fn in cell_fires.get(c, [])))
    lat, lon = h3.cell_to_latlng(c)
    pt_ev.append(any(active(ignv[j], outv[j], a, b) for j in tree.query_ball_point([lat * 111, lon * 111 * k], 10)))
    nb_ev.append(any((n, t) in seen for n in h3.grid_ring(c, 1) for t in range(a + 1, b)))
g['perimeter'] = per_ev; g['point10'] = pt_ev; g['neighbour'] = nb_ev
g['in_any_perimeter'] = [c in cell_fires for c in g.h3_cell]
t = g.groupby('gap')[['in_any_perimeter', 'perimeter', 'point10', 'neighbour']].mean()
t['gaps'] = g.groupby('gap').size()
t['perimeter_or_point'] = g.assign(x=g.perimeter | g.point10).groupby('gap').x.mean()
t['bcws_or_neighbour'] = g.assign(x=g.perimeter | g.point10 | g.neighbour).groupby('gap').x.mean()
print('perimeters used', nper); print(t.round(3).to_string())
tot = lambda m: (int(m.sum()), int((m * g.gap).sum()))
print('gaps / cell-days filled: perimeter', tot(g.perimeter), '| perimeter or point', tot(g.perimeter | g.point10),
      '| neighbour only (no BCWS)', tot(g.neighbour & ~(g.perimeter | g.point10)), '| none', tot(~(g.perimeter | g.point10 | g.neighbour)))
