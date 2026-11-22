"""Split new fires in quiet areas (label v1; no detection in the cell or its neighbours for 7 days) into BCWS
wildfires and other burns. A new-fire cell-day on day t counts as a BCWS wildfire if
  (a) the cell is inside the final perimeter (+1 ring) of a BCWS fire (FIRE_TYPE 'Fire') that ignited by t+2 and was
      not declared out before t-1, or
  (b) a BCWS fire's ignition point is within R km of the cell centre and it ignited between t-3 and t+2.
BCWS 'Nuisance Fire' records (reported non-wildfire burning) are counted separately. R = 5 (and 10 for sensitivity).
usage: python new_fire_split.py REPO_ROOT"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd, h3
from scipy.spatial import cKDTree
ROOT = Path(sys.argv[1]); SRC = ROOT / 'data/pair_lightning_h3r7'
incA = pd.read_csv(ROOT / 'data/bcws_incidents/bc_historical_incidents_2018_2023.csv')
per = json.load(open(ROOT / 'data/bcws_perimeters/bc_fire_perimeters_2018_2023.geojson'))['features']
k = np.cos(np.radians(52)); rows = []
for Y in range(2018, 2024):
    f = glob.glob(str(SRC / (f'training_samples/*/snpp_lightning_{Y}.parquet' if Y <= 2021 else f'evaluation/{Y}/*.parquet')))
    d = pd.concat([pd.read_parquet(x, columns=['h3_cell', 'target_date', 'quiet_stratum', 'target_snpp_filled']) for x in f])
    d = d[(d.quiet_stratum == 1) & d.target_snpp_filled].drop_duplicates(['h3_cell', 'target_date']).copy()
    d['t'] = (pd.to_datetime(d.target_date) - pd.Timestamp(f'{Y}-05-01')).dt.days.to_numpy()
    inc = incA[incA.FIRE_YEAR == Y].copy()
    inc['ign'] = (pd.to_datetime(inc.IGNITION_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
    inc['out'] = (pd.to_datetime(inc.FIRE_OUT_DATE_UTC, utc=True).dt.tz_localize(None) - pd.Timestamp(f'{Y}-05-01')).dt.days
    fire = inc[inc.FIRE_TYPE == 'Fire']; nuis = inc[inc.FIRE_TYPE == 'Nuisance Fire']
    dates = fire.drop_duplicates('FIRE_NUMBER').set_index('FIRE_NUMBER')[['ign', 'out']]
    cf = {}
    for g in per:
        p = g['properties']
        if p['FIRE_YEAR'] != Y or p['FIRE_NUMBER'] not in dates.index or g['geometry'] is None: continue
        try: cs = set(h3.geo_to_cells(g['geometry'], 7))
        except Exception: continue
        cs |= {n for c in cs for n in h3.grid_ring(c, 1)}
        for c in cs: cf.setdefault(c, []).append(p['FIRE_NUMBER'])
    ll = np.array([h3.cell_to_latlng(c) for c in d.h3_cell]); xy = np.c_[ll[:, 0] * 111, ll[:, 1] * 111 * k]
    def near(df, R, lo, hi):
        if not len(df): return np.zeros(len(d), bool)
        tr = cKDTree(np.c_[df.LATITUDE * 111, df.LONGITUDE * 111 * k]); ig = df.ign.to_numpy()
        return np.array([any(lo <= ig[j] - t <= hi for j in js) for js, t in zip(tr.query_ball_point(xy, R), d.t)])
    in_per = np.array([any((dates.loc[fn, 'ign'] <= t + 2) and not (dates.loc[fn, 'out'] < t - 1) for fn in cf.get(c, [])) for c, t in zip(d.h3_cell, d.t)])
    p5, p10 = near(fire, 5, -3, 2), near(fire, 10, -3, 2)
    n5 = near(nuis, 5, -3, 2)
    w5 = in_per | p5; w10 = in_per | p10
    rows.append({'year': Y, 'new_fire_cell_days': len(d), 'wildfire_5km': int(w5.sum()), 'other_5km': int((~w5).sum()),
                 'share_wildfire_5km': round(w5.mean(), 3), 'wildfire_10km': int(w10.sum()), 'share_wildfire_10km': round(w10.mean(), 3),
                 'by_perimeter': int(in_per.sum()), 'by_point_only_5km': int((p5 & ~in_per).sum()),
                 'other_but_bcws_nuisance_5km': int((~w5 & n5).sum())})
    if Y == 2023: d.assign(wildfire=w5, nuisance=n5).to_parquet(ROOT / 'data/pair_lightning_h3r7/new_fire_split_2023.parquet', index=False)
T = pd.DataFrame(rows); print(T.to_string(index=False))
s = T[['new_fire_cell_days', 'wildfire_5km', 'other_5km', 'wildfire_10km']].sum(); print('\nall years:', s.to_dict(), 'share wildfire 5km', round(s.wildfire_5km / s.new_fire_cell_days, 3))
