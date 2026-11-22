"""Next-day two-satellite label on the cell-dates Exp 38 scored, for Exp 21 CatBoost (`exp16_spread`).
NOAA-20 comes from the FIRMS archive (Exp 21 used CWFIS's copy), so the label is checked first against Exp 38's
`burn_8d` (any S-NPP or NOAA-20 detection on t..t+7), then: label(t) = filled(pair)[t], thresholds = Exp 21's
per-centre 2022 thresholds. Outputs cbn.json (check, F1 by centre x month, map counts) and cbn_map.png.
usage: python cb_nextday.py REPO_ROOT OUT_DIR"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, h3
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); EXP = ROOT / 'experiments'
SC = EXP / '38_exp_canadafiresat_protocol_bc/results/scores'
SEASON, FIRST, WIN = 153, 7, 8
ISSUE = np.arange(FIRST, SEASON - WIN + 1, WIN)  # season-day index of each forecast date (May 8, 16, ...)
thr = (pd.read_csv(EXP / '21_exp_weather_lightning_spread_bc/results/metrics_by_year.csv')
       .query("year==2023 and definition=='res7_exact' and model=='catboost_spread_noday' and method=='f1_2022'")
       .set_index('region').threshold.to_dict())

def season_day(d):
    return (d - pd.to_datetime(d.dt.year.astype(str) + '-05-01')).dt.days.to_numpy()

# detections, years 2022-2023, May-Sep
sn = pd.read_csv(EXP / 'firms_summer_2018_2024/data/firms_h3_detections.csv', usecols=['acq_date', 'h3_cell'], parse_dates=['acq_date'])
sn = sn[sn.acq_date.dt.year.isin([2022, 2023]) & sn.acq_date.dt.month.between(5, 9)]
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date'], parse_dates=['acq_date'])
n2 = n2[n2.acq_date.dt.year.isin([2022, 2023]) & n2.acq_date.dt.month.between(5, 9)
        & n2.latitude.between(48, 60.1) & n2.longitude.between(-139.1, -114)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
print('detections: S-NPP', len(sn), 'NOAA-20', len(n2), flush=True)

def grid(det, idx, year):
    d = det[det.acq_date.dt.year == year]
    c = idx.get_indexer(d.h3_cell); k = c >= 0
    g = np.zeros((len(idx), SEASON), bool); g[c[k], season_day(d.acq_date[k])] = True
    return g

shift = lambda a, k: np.roll(a, k, axis=1) if k == 0 else (np.pad(a, ((0, 0), (k, 0)))[:, :-k] if k > 0 else np.pad(a, ((0, 0), (0, -k)))[:, -k:])
check, heat, mapinfo, total = [], [], None, {'tp': 0, 'fp': 0, 'fn': 0}
for f in sorted(SC.glob('*_2023.parquet')):
    slug = f.name.replace('_2023.parquet', '')
    cells = pd.Index(pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet', columns=['h3_cell']).h3_cell)
    d = pd.read_parquet(f, columns=['region', 'cell', 'issue', 'month', 'burn_8d', 'exp16_spread'])
    region = d.region.iloc[0]
    pair = grid(sn, cells, 2023) | grid(n2, cells, 2023)
    label = pair | (shift(pair, 1) & shift(pair, -1))
    cs = np.concatenate([np.zeros((len(cells), 1), np.int32), pair.cumsum(axis=1, dtype=np.int32)], axis=1)
    ours8 = ((cs[:, ISSUE + WIN] - cs[:, ISSUE]) > 0)[d.cell.to_numpy(), d.issue.to_numpy()]
    theirs8 = d.burn_8d.to_numpy()
    check.append({'region': region, 'theirs': int(theirs8.sum()), 'ours': int(ours8.sum()), 'both': int((ours8 & theirs8).sum())})
    y = label[d.cell.to_numpy(), ISSUE[d.issue.to_numpy()]]
    fl = d.exp16_spread.to_numpy() >= thr[region]
    for k, v in (('tp', fl & y), ('fp', fl & ~y), ('fn', ~fl & y)):
        total[k] += int(v.sum())
    for mo in range(5, 10):
        s = d.month.to_numpy() == mo
        tp, fp, fn = int((fl & y & s).sum()), int((fl & ~y & s).sum()), int((~fl & y & s).sum())
        heat.append({'region': region, 'month': mo, 'f1': 2 * tp / max(1, 2 * tp + fp + fn), 'fires': tp + fn})
    if region == 'Kamloops':
        aug = [i for i, t in enumerate(ISSUE) if (pd.Timestamp('2023-05-01') + pd.Timedelta(days=int(t))).month == 8]
        dd = pd.DataFrame({'issue': d.issue, 'y': y, 'fl': fl, 'cell': d.cell})
        issue = int(dd[dd.issue.isin(aug)].groupby('issue').y.sum().idxmax())
        x = dd[dd.issue == issue]; arr = cells.to_numpy()
        mapinfo = {'date': str((pd.Timestamp('2023-05-01') + pd.Timedelta(days=int(ISSUE[issue]))).date()),
                   'tp': arr[x.cell[x.y & x.fl]].tolist(), 'fn': arr[x.cell[x.y & ~x.fl]].tolist(), 'fp': arr[x.cell[~x.y & x.fl]].tolist(), 'all': arr.tolist()}
    print(region, check[-1], flush=True)
c = pd.DataFrame(check).sum(numeric_only=True)
agree = {'theirs': int(c.theirs), 'ours': int(c.ours), 'both': int(c.both), 'jaccard': round(c.both / (c.theirs + c.ours - c.both), 4)}
f1 = 2 * total['tp'] / (2 * total['tp'] + total['fp'] + total['fn'])
print('8-day label check vs Exp 38:', agree, '| next-day F1 on 18 dates', round(f1, 3), total)
poly = lambda cs: [[(lon, lat) for lat, lon in h3.cell_to_boundary(q)] for q in cs]
fig, ax = plt.subplots(figsize=(9, 8.2)); ax.set_aspect(1 / np.cos(np.deg2rad(51))); ax.axis('off')
ax.add_collection(PolyCollection(poly(mapinfo['all']), facecolor='#f1eeea', edgecolor='#e2ddd7', linewidth=0.1))
for k, col in [('fp', '#f2a33a'), ('fn', '#d23b3b'), ('tp', '#2a78d6')]:
    ax.add_collection(PolyCollection(poly(mapinfo[k]), facecolor=col, edgecolor=col, linewidth=0.4))
ax.autoscale_view(); fig.tight_layout(); fig.savefig(OUT / 'cbn_map.png', dpi=110, facecolor='white')
json.dump({'check': agree, 'f1_18dates': f1, 'total': total, 'heat': heat,
           'map': {'date': mapinfo['date'], **{k: len(mapinfo[k]) for k in ('tp', 'fn', 'fp')}}}, open(OUT / 'cbn.json', 'w'))
print('map', mapinfo['date'], {k: len(mapinfo[k]) for k in ('tp', 'fn', 'fp')})
