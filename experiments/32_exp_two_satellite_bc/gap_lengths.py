"""How long are the detection gaps inside a cell? For every cell and season (May 1 - Sep 30), find runs of days with
no S-NPP or NOAA-20 detection between two detected days, by length. A 1-day gap is already filled in the label.
usage: python gap_lengths.py REPO_ROOT"""
import io, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3
ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
d = pd.concat([sn, n2[['acq_date', 'h3_cell']]]); d['date'] = pd.to_datetime(d.acq_date)
d = d[d.date.dt.month.between(5, 9) & d.date.dt.year.between(2018, 2023)]
cells = set(pd.concat([pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{s}/data/h3_grid.parquet', columns=['h3_cell'])
            for s in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']]).h3_cell)
d = d[d.h3_cell.isin(cells)]; d['year'] = d.date.dt.year; d['day'] = (d.date - pd.to_datetime(d.year.astype(str) + '-05-01')).dt.days
u = d[['year', 'h3_cell', 'day']].drop_duplicates().sort_values(['year', 'h3_cell', 'day'])
u['gap'] = u.groupby(['year', 'h3_cell']).day.diff() - 1
g = u.dropna(subset=['gap']); g = g[g.gap >= 1]
bins = [0, 1, 2, 3, 4, 7, 14, 200]; labels = ['1', '2', '3', '4', '5-7', '8-14', '15+']
g['len'] = pd.cut(g.gap, bins, labels=labels)
t = g.groupby(['len'], observed=False).size().rename('gaps').to_frame()
t['share'] = t.gaps / t.gaps.sum(); t['missing_cell_days'] = g.groupby('len', observed=False).gap.sum()
print('All years 2018-2023'); print(t.to_string())
y = g[g.year == 2023].groupby('len', observed=False).agg(gaps=('gap', 'size'), missing_cell_days=('gap', 'sum')); y['share'] = y.gaps / y.gaps.sum()
print('\n2023'); print(y.to_string())
print('\nGaps per year (2-4 days):'); print(g[g.gap.between(2, 4)].groupby('year').size().to_string())
print('\nDetected cell-days 2023:', int((u.year == 2023).sum()), '| cells with any detection 2023:', u[u.year == 2023].h3_cell.nunique())
