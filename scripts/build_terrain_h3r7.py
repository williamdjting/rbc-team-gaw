"""Terrain per H3 res-7 cell from the Copernicus GLO-90 DEM (public, AWS Open Data, free licence with attribution:
"produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018,
provided under COPERNICUS by the European Union and ESA").
Downloads the 1-degree tiles covering the six Fire Centre grids into data/raw_archives/copernicus_dem_90m/ (git-ignored),
samples each cell's 49 res-9 child centres, writes data/static/terrain_h3r7.parquet:
  h3_cell, elev_mean_m, elev_std_m, elev_centre_m, elev_min_m, elev_max_m
usage: python build_terrain_h3r7.py REPO_ROOT"""
import sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, h3, rasterio

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'
TILES = ROOT / 'data/raw_archives/copernicus_dem_90m'; TILES.mkdir(parents=True, exist_ok=True)
cells = pd.concat([pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{s}/data/h3_grid.parquet', columns=['h3_cell'])
                   for s in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']]).h3_cell.unique()
print('cells', len(cells), flush=True)
kids = [h3.cell_to_children(c, 9) for c in cells]
owner = np.repeat(np.arange(len(cells)), [len(k) for k in kids])
pts = np.array([h3.cell_to_latlng(k) for ks in kids for k in ks])
centre = np.array([h3.cell_to_latlng(c) for c in cells])

def tile_name(lat, lon):
    ns = f'N{int(np.floor(lat)):02d}' if lat >= 0 else f'S{int(-np.floor(lat)):02d}'
    ew = f'W{int(-np.floor(lon)):03d}' if lon < 0 else f'E{int(np.floor(lon)):03d}'
    return f'Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM'

def sample(latlon):
    z = np.full(len(latlon), np.nan, np.float32)
    keys = pd.Series([tile_name(a, b) for a, b in zip(np.floor(latlon[:, 0]), np.floor(latlon[:, 1]))])
    for name, idx in keys.groupby(keys).groups.items():
        f = TILES / f'{name}.tif'
        if not f.exists():
            try:
                urllib.request.urlretrieve(f'https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif', f)
            except Exception as e:
                print('missing tile', name, e); continue
        with rasterio.open(f) as r:
            a = r.read(1); ii = np.asarray(idx)
            row, col = rasterio.transform.rowcol(r.transform, latlon[ii, 1], latlon[ii, 0])
            row = np.clip(np.asarray(row), 0, a.shape[0] - 1); col = np.clip(np.asarray(col), 0, a.shape[1] - 1)
            z[ii] = a[row, col]
    return z

zc = sample(centre); zk = sample(pts)
df = pd.DataFrame({'owner': owner, 'z': zk}).groupby('owner').z.agg(['mean', 'std', 'min', 'max'])
out = pd.DataFrame({'h3_cell': cells, 'elev_mean_m': df['mean'].to_numpy(), 'elev_std_m': df['std'].to_numpy(),
                    'elev_centre_m': zc, 'elev_min_m': df['min'].to_numpy(), 'elev_max_m': df['max'].to_numpy()})
(ROOT / 'data/static').mkdir(exist_ok=True)
out.to_parquet(ROOT / 'data/static/terrain_h3r7.parquet', index=False)
print(out.describe().round(1).to_string()); print('tiles', len(list(TILES.glob('*.tif'))), 'missing elev', int(out.elev_mean_m.isna().sum()))
