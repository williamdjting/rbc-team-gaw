"""Error analysis of Exp 25 (S-NPP + lightning MLP, next-day target) on 2023, every cell-day."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
EVAL = ROOT / 'data/snpp_lightning_h3r7/evaluation/2023'
met = pd.read_csv(ROOT / 'experiments/25_exp_snpp_lightning_2024/results/metrics_2023.csv')
thr = met[met.model == 'mlp_snpp_lightning'].set_index('region').threshold.to_dict()
cols = ['h3_cell', 'target_date', 'target_snpp_filled', 'mlp_score', 'snpp_persistence', 'local_count_1d',
        'neighbour_count_1d', 'local_count_7d', 'neighbour_count_7d', 'active_days_7d']
TYPES = ['Burning yesterday', 'Neighbour burning yesterday', 'Fire nearby this week', 'New (quiet week)']
caught = {m: {k: [0, 0] for k in TYPES} for m in ['model', 'rule']}   # [caught, missed]
fp = {m: {k: 0 for k in TYPES} for m in ['model', 'rule']}
reappear = {'model': [0, 0], 'rule': [0, 0]}  # [false alarms on burning-yesterday cells that burn again within 3 days, all such]
keep = {d: [0, 0] for d in range(1, 8)}       # burning yesterday: [burn today, total] by active days in last 7
heat = []
mapday = None
for f in sorted(EVAL.glob('*_snpp_lightning_full_grid.parquet')):
    region = {'prince_george': 'Prince George'}.get(f.name.split('_snpp')[0], f.name.split('_snpp')[0].title())
    d = pd.read_parquet(f, columns=cols).sort_values(['h3_cell', 'target_date']).reset_index(drop=True)
    y = d.target_snpp_filled.to_numpy().astype(bool)
    flags = {'model': d.mlp_score.to_numpy() >= thr[region], 'rule': d.snpp_persistence.to_numpy().astype(bool)}
    t = np.select([d.local_count_1d > 0, d.neighbour_count_1d > 0, (d.local_count_7d > 0) | (d.neighbour_count_7d > 0)],
                  TYPES[:3], TYPES[3])
    same = d.h3_cell.to_numpy()
    later = np.zeros(len(d), bool)
    for k in (1, 2, 3):
        nxt = np.roll(y, -k); ok = np.roll(same, -k) == same; ok[-k:] = False
        later |= nxt & ok
    for m, fl in flags.items():
        for k in TYPES:
            s = t == k
            caught[m][k][0] += int((fl & y & s).sum()); caught[m][k][1] += int((~fl & y & s).sum())
            fp[m][k] += int((fl & ~y & s).sum())
        s = fl & ~y & (t == TYPES[0])
        reappear[m][0] += int((s & later).sum()); reappear[m][1] += int(s.sum())
    b = t == TYPES[0]
    for a in range(1, 8):
        s = b & (d.active_days_7d.to_numpy() == a)
        keep[a][0] += int((s & y).sum()); keep[a][1] += int(s.sum())
    month = d.target_date.dt.month.to_numpy()
    for mo in range(5, 10):
        s = month == mo
        for m, fl in flags.items():
            tp = int((fl & y & s).sum()); fpn = int((fl & ~y & s).sum()); fn = int((~fl & y & s).sum())
            heat.append({'region': region, 'month': mo, 'model': m, 'f1': 2 * tp / max(1, 2 * tp + fpn + fn), 'fires': tp + fn})
    if region == 'Kamloops':
        dd = d.assign(y=y, fl=flags['model'])
        aug = dd[dd.target_date.dt.month == 8]
        day = aug.groupby('target_date').y.sum().idxmax()
        x = dd[dd.target_date == day]
        mapday = {'date': str(day.date()), 'tp': x.h3_cell[x.y & x.fl].tolist(), 'fp': x.h3_cell[~x.y & x.fl].tolist(),
                  'fn': x.h3_cell[x.y & ~x.fl].tolist(), 'all': x.h3_cell.tolist()}
    print(region, 'done', flush=True)
out = {'types': TYPES, 'caught': caught, 'fp': fp, 'reappear': reappear,
       'keep': {str(k): v for k, v in keep.items()}, 'heat': heat, 'mapday': mapday}
json.dump(out, open(OUT, 'w'))
for m in ['model', 'rule']:
    print(m, 'caught/missed', caught[m], 'fp', fp[m], 'reappear', reappear[m])
print('keep', keep)
print('map', mapday['date'], len(mapday['tp']), len(mapday['fp']), len(mapday['fn']))
