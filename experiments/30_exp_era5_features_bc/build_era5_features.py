"""Build daily ERA5-Land features per 0.1-degree land point and map every H3 res-7 cell to its
nearest point. Output (git-ignored) in data/era5_h3r7/:
  era5_point_features.npy   float32 [n_points, n_days, n_features], day 0 = 2018-05-01 per season
  era5_meta.json            feature names, point coordinates, season day index
  cell_to_point.parquet     h3_cell -> point index, distance (km)
Source: dataset_ERA5_18_24.csv (repo root). Its daily value is the mean of 5 snapshots (00/05/10/15/20 UTC),
so accumulated fields (tp, pev, ssrd) are a monotone proxy, not a true daily total (see docs/week-5-tasks.md 7).
usage: python build_era5_features.py REPO_ROOT"""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.csv as pv, h3
from scipy.spatial import cKDTree

ROOT = Path(sys.argv[1]); OUT = ROOT / 'data/era5_h3r7'; OUT.mkdir(parents=True, exist_ok=True)
COLS = ['valid_time', 'latitude', 'longitude_deg', 'swvl1', 'swvl2', 'swvl3', 't2m', 'd2m', 'u10', 'v10',
        'tp', 'pev', 'ssrd', 'skt', 'sde']
tab = pv.read_csv(ROOT / 'dataset_ERA5_18_24.csv', convert_options=pv.ConvertOptions(include_columns=COLS))
df = tab.to_pandas(); del tab
df['valid_time'] = pd.to_datetime(df.valid_time)
print('rows', len(df), df.valid_time.min(), df.valid_time.max())

pts = df[['latitude', 'longitude_deg']].drop_duplicates().sort_values(['latitude', 'longitude_deg']).reset_index(drop=True)
pts['pid'] = np.arange(len(pts))
df = df.merge(pts, on=['latitude', 'longitude_deg'])
years = list(range(2018, 2025)); season = pd.date_range('2018-05-01', '2018-09-30')  # 153 days
ND = len(season)
df['year'] = df.valid_time.dt.year
df['doy'] = (df.valid_time - pd.to_datetime(df.year.astype(str) + '-05-01')).dt.days
df = df[(df.doy >= 0) & (df.doy < ND)]

raw = ['swvl1', 'swvl2', 'swvl3', 't2m', 'd2m', 'u10', 'v10', 'tp', 'pev', 'ssrd', 'skt', 'sde']
P = len(pts)
cube = np.full((len(years), P, ND, len(raw)), np.nan, np.float32)
yi = df.year.to_numpy() - 2018
cube[yi, df.pid.to_numpy(), df.doy.to_numpy()] = df[raw].to_numpy(np.float32)
print('missing share', np.isnan(cube).mean())
del df
R = {k: cube[..., i] for i, k in enumerate(raw)}

def roll(a, w, fn):  # trailing window over days incl. today, partial windows allowed
    out = np.full_like(a, np.nan)
    for d in range(a.shape[-1]):
        out[..., d] = fn(a[..., max(0, d - w + 1):d + 1], axis=-1)
    return out

t = R['t2m'] - 273.15; td = R['d2m'] - 273.15
es = 6.112 * np.exp(17.67 * t / (t + 243.5)); ea = 6.112 * np.exp(17.67 * td / (td + 243.5))
rh = np.clip(100 * ea / es, 0, 100); vpd = np.clip(es - ea, 0, None)
wind = np.hypot(R['u10'], R['v10'])
tp = np.clip(R['tp'], 0, None) * 1000  # mm-equivalent proxy
dry = (tp < 1.0).astype(np.float32)
# soil-moisture anomaly vs the point's 2018-2021 mean for the same day-of-season (+-7 days), training years only
clim = np.nanmean(R['swvl1'][:4], axis=0)
clim = roll(clim[None], 15, np.nanmean)[0]
clim = np.concatenate([clim[:, 7:], np.repeat(clim[:, -1:], 7, axis=1)], axis=1)  # centre the window
F = {
    'era5_t2m': t, 'era5_rh': rh, 'era5_vpd': vpd, 'era5_wind': wind, 'era5_skt': R['skt'] - 273.15,
    'era5_swvl1': R['swvl1'], 'era5_swvl2': R['swvl2'], 'era5_swvl3': R['swvl3'],
    'era5_swvl1_anom': R['swvl1'] - clim[None],
    'era5_swvl1_chg7': R['swvl1'] - np.concatenate([np.repeat(R['swvl1'][..., :1], 7, -1), R['swvl1'][..., :-7]], -1),
    'era5_tp': tp, 'era5_tp_3d': roll(tp, 3, np.nansum), 'era5_tp_7d': roll(tp, 7, np.nansum),
    'era5_tp_14d': roll(tp, 14, np.nansum), 'era5_dry_days_7d': roll(dry, 7, np.nansum),
    'era5_t2m_max7': roll(t, 7, np.nanmax), 'era5_t2m_mean7': roll(t, 7, np.nanmean),
    'era5_rh_mean7': roll(rh, 7, np.nanmean), 'era5_vpd_mean7': roll(vpd, 7, np.nanmean),
    'era5_pev': R['pev'] * 1000, 'era5_ssrd': R['ssrd'] / 1e6, 'era5_snow_depth': R['sde'],
}
names = list(F)
feat = np.stack([F[k] for k in names], -1).astype(np.float32)  # [year, point, day, feat]
np.save(OUT / 'era5_point_features.npy', feat)

cells = set()
for f in glob.glob(str(ROOT / 'data/snpp_lightning_h3r7/evaluation/2023/*.parquet')):
    cells |= set(pd.read_parquet(f, columns=['h3_cell']).h3_cell.unique())
for f in glob.glob(str(ROOT / 'data/snpp_lightning_h3r7/training_samples/*/*.parquet')):
    cells |= set(pd.read_parquet(f, columns=['h3_cell']).h3_cell.unique())
cells = sorted(cells)
ll = np.array([h3.cell_to_latlng(c) for c in cells])
k = np.cos(np.radians(52))
tree = cKDTree(np.c_[pts.latitude, pts.longitude_deg * k])
dist, idx = tree.query(np.c_[ll[:, 0], ll[:, 1] * k])
pd.DataFrame({'h3_cell': cells, 'pid': idx.astype(np.int32), 'dist_km': dist * 111.0}).to_parquet(OUT / 'cell_to_point.parquet', index=False)
json.dump({'features': names, 'years': years, 'season_start': '05-01', 'n_days': ND,
           'n_points': P, 'mapping': 'nearest ERA5-Land land point to the H3 cell centre (lon scaled by cos 52 deg)'},
          open(OUT / 'era5_meta.json', 'w'), indent=1)
print('points', P, 'cells', len(cells), 'max dist km', round(dist.max() * 111, 1), 'median', round(np.median(dist) * 111, 1))
print('features', len(names), names)
