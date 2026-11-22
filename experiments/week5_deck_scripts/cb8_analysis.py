"""Exp 21 CatBoost (Exp 38's `exp16_spread` scores) on the 8-day target `burn_8d`, every cell, 2023.
Thresholds: best F1 on 2022 per Fire Centre. Outputs cb8.json (F1 by centre x month, map cells) and cb8_map.png.
usage: python cb8_analysis.py REPO_ROOT OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, h3
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from sklearn.metrics import precision_recall_curve

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
SC = ROOT / 'experiments/38_exp_canadafiresat_protocol_bc/results/scores'
GRIDS = ROOT / 'experiments/02_hotspot_persistence_bc/regions'
ISSUE = pd.Timestamp('2023-05-08') + pd.to_timedelta(np.arange(18) * 8, unit='D')
heat, mapinfo = [], None
for f in sorted(SC.glob('*_2023.parquet')):
    slug = f.name.replace('_2023.parquet', '')
    a = pd.read_parquet(SC / f'{slug}_2022.parquet', columns=['burn_8d', 'exp16_spread'])
    p, r, t = precision_recall_curve(a.burn_8d, a.exp16_spread)
    with np.errstate(divide='ignore', invalid='ignore'):
        thr = t[np.nan_to_num(2 * p * r / (p + r))[:-1].argmax()]
    d = pd.read_parquet(f, columns=['region', 'cell', 'issue', 'month', 'burn_8d', 'exp16_spread'])
    d['flag'] = d.exp16_spread >= thr; y = d.burn_8d
    region = d.region.iloc[0]
    for mo, g in d.groupby('month'):
        tp = int((g.flag & g.burn_8d).sum()); fp = int((g.flag & ~g.burn_8d).sum()); fn = int((~g.flag & g.burn_8d).sum())
        heat.append({'region': region, 'month': int(mo), 'f1': 2 * tp / max(1, 2 * tp + fp + fn), 'fires': tp + fn})
    if region == 'Kamloops':
        aug = [i for i, s in enumerate(ISSUE) if s.month == 8]
        issue = int(d[d.issue.isin(aug)].groupby('issue').burn_8d.sum().idxmax())
        x = d[d.issue == issue]
        cells = pd.read_parquet(GRIDS / slug / 'data/h3_grid.parquet', columns=['h3_cell']).h3_cell.to_numpy()
        pick = lambda m: cells[x.cell[m].to_numpy()].tolist()
        mapinfo = {'start': str(ISSUE[issue].date()), 'tp': pick(x.flag & x.burn_8d), 'fn': pick(~x.flag & x.burn_8d),
                   'fp': pick(x.flag & ~x.burn_8d), 'all': cells.tolist()}
    print(region, 'threshold', round(float(thr), 4), flush=True)
poly = lambda cs: [[(lon, lat) for lat, lon in h3.cell_to_boundary(c)] for c in cs]
fig, ax = plt.subplots(figsize=(9, 8.2)); ax.set_aspect(1 / np.cos(np.deg2rad(51))); ax.axis('off')
ax.add_collection(PolyCollection(poly(mapinfo['all']), facecolor='#f1eeea', edgecolor='#e2ddd7', linewidth=0.1))
for k, c in [('fp', '#f2a33a'), ('fn', '#d23b3b'), ('tp', '#2a78d6')]:
    ax.add_collection(PolyCollection(poly(mapinfo[k]), facecolor=c, edgecolor=c, linewidth=0.4))
ax.autoscale_view(); fig.tight_layout(); fig.savefig(OUT / 'cb8_map.png', dpi=110, facecolor='white')
json.dump({'heat': heat, 'map': {'start': mapinfo['start'], **{k: len(mapinfo[k]) for k in ('tp', 'fn', 'fp')}}}, open(OUT / 'cb8.json', 'w'))
print('map', mapinfo['start'], {k: len(mapinfo[k]) for k in ('tp', 'fn', 'fp')})
