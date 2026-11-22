"""How confident is Exp 21 CatBoost when it is right vs wrong? Next-day label, 2023, every cell on the 18 dates Exp 38
scored (same label rebuild as cb_nextday.py). Outputs cbconf.json: score quantiles per outcome (TP, FP, FN, TN) and
calibration bins. usage: python cb_confidence.py REPO_ROOT OUT_DIR"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); EXP = ROOT / 'experiments'
SC = EXP / '38_exp_canadafiresat_protocol_bc/results/scores'
SEASON, FIRST, WIN = 153, 7, 8
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

def grid(det, idx):
    c = idx.get_indexer(det.h3_cell); k = c >= 0
    g = np.zeros((len(idx), SEASON), bool); g[c[k], season_day(det.acq_date[k])] = True
    return g
shift = lambda a, k: np.pad(a, ((0, 0), (k, 0)))[:, :-k] if k > 0 else np.pad(a, ((0, 0), (0, -k)))[:, -k:]
scores = {k: [] for k in ('tp', 'fp', 'fn', 'tn')}; ys, ss = [], []
for f in sorted(SC.glob('*_2023.parquet')):
    slug = f.name.replace('_2023.parquet', '')
    cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
    d = pd.read_parquet(f, columns=['region', 'cell', 'issue', 'exp16_spread'])
    pair = grid(sn, cells) | grid(n2, cells); label = pair | (shift(pair, 1) & shift(pair, -1))
    y = label[d.cell.to_numpy(), ISSUE[d.issue.to_numpy()]]; s = d.exp16_spread.to_numpy(); fl = s >= thr[d.region.iloc[0]]
    for k, m in (('tp', fl & y), ('fp', fl & ~y), ('fn', ~fl & y), ('tn', ~fl & ~y)):
        scores[k].append(s[m])
    ys.append(y); ss.append(s)
    print(slug, flush=True)
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
out = {'box': {k: {'n': int(sum(len(a) for a in v)), 'q': np.quantile(np.concatenate(v), qs).tolist(), 'mean': float(np.concatenate(v).mean())}
               for k, v in scores.items()}, 'thr': [min(thr.values()), max(thr.values())]}
y = np.concatenate(ys); s = np.concatenate(ss)
edges = [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
b = np.digitize(s, edges[1:-1])
out['cal'] = [{'lo': edges[i], 'hi': min(edges[i + 1], 1), 'pred': float(s[b == i].mean()), 'obs': float(y[b == i].mean()), 'n': int((b == i).sum())}
              for i in range(len(edges) - 1) if (b == i).sum() > 0]
json.dump(out, open(OUT / 'cbconf.json', 'w'))
for k, v in out['box'].items():
    print(k, v['n'], 'median', round(v['q'][2], 4), 'IQR', [round(x, 4) for x in v['q'][1:4:2]], 'mean', round(v['mean'], 4))
print('cal', [(round(c['pred'], 3), round(c['obs'], 3), c['n']) for c in out['cal']])
