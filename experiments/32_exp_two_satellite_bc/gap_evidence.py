"""For 2-4 day detection gaps in a cell (2023), is there evidence the fire was alive during the gap?
(a) a ring-1 neighbour detected on any gap day; (b) a BCWS incident within 10 km ignited before the gap and not
declared out before the gap ended; (c) either. usage: python gap_evidence.py REPO_ROOT"""
import io, sys, zipfile
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
cells = set(pd.concat([pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{s}/data/h3_grid.parquet', columns=['h3_cell'])
            for s in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']]).h3_cell)
d = d[d.h3_cell.isin(cells)]; d['day'] = (d.date - pd.Timestamp(f'{Y}-05-01')).dt.days
u = d[['h3_cell', 'day']].drop_duplicates().sort_values(['h3_cell', 'day'])
seen = set(zip(u.h3_cell, u.day))
u['prev'] = u.groupby('h3_cell').day.shift(); u['gap'] = u.day - u.prev - 1
g = u[(u.gap >= 2) & (u.gap <= 4)].copy()
inc = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv', usecols=['FIRE_YEAR', 'FIRE_TYPE', 'LATITUDE', 'LONGITUDE', 'IGNITION_DATE_UTC', 'FIRE_OUT_DATE_UTC'])
inc = inc[(inc.FIRE_YEAR == Y) & inc.FIRE_TYPE.astype(str).str.contains('Fire', case=False)]
ign = (pd.to_datetime(inc.IGNITION_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days.to_numpy()
out = (pd.to_datetime(inc.FIRE_OUT_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days.to_numpy()
k = np.cos(np.radians(52)); tree = cKDTree(np.c_[inc.LATITUDE * 111, inc.LONGITUDE * 111 * k])
nb_ev, inc_ev = [], []
for c, a, b in zip(g.h3_cell, g.prev.astype(int), g.day.astype(int)):
    gap_days = range(a + 1, b)
    nb_ev.append(any((n, t) in seen for n in h3.grid_ring(c, 1) for t in gap_days))
    lat, lon = h3.cell_to_latlng(c); near = tree.query_ball_point([lat * 111, lon * 111 * k], 10)
    inc_ev.append(any(ign[j] <= a + 1 and not (out[j] < b) for j in near))
g['nb'] = nb_ev; g['inc'] = inc_ev; g['any'] = g.nb | g.inc
t = g.groupby('gap').agg(gaps=('nb', 'size'), neighbour_detected=('nb', 'mean'), bcws_active=('inc', 'mean'), either=('any', 'mean'))
t['gaps_filled_by_either'] = (g.groupby('gap')['any'].sum()).astype(int); t['cell_days_added'] = t.gaps_filled_by_either * t.index
print(t.round(3).to_string()); print('total gaps', len(g), 'filled by either', int(g['any'].sum()), 'cell-days added', int((g['any'] * g.gap).sum()))
