"""Land-cover shares per H3 res-7 cell from ESA WorldCover 2021 v200 (10 m; public on AWS, CC BY 4.0:
"ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium").
Reads each 3-degree tile at its 4x overview (~40 m) over HTTP, samples each cell's 49 res-9 child centres, and writes
data/static/landcover_h3r7.parquet: h3_cell, lc_tree, lc_shrub, lc_grass, lc_crop, lc_built, lc_bare, lc_snow,
lc_water, lc_wetland, lc_moss, lc_burnable (tree + shrub + grass + wetland), lc_n (points sampled).
WorldCover 2021 is one snapshot (after 2018 and 2021 fires), so burn scars from 2018-2021 can appear as shrub/grass.
usage: python build_landcover_h3r7.py REPO_ROOT"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, h3, rasterio
from rasterio.enums import Resampling

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'
CLASSES = {10: 'tree', 20: 'shrub', 30: 'grass', 40: 'crop', 50: 'built', 60: 'bare', 70: 'snow', 80: 'water', 90: 'wetland', 100: 'moss'}
cells = pd.concat([pd.read_parquet(EXP / f'02_hotspot_persistence_bc/regions/{s}/data/h3_grid.parquet', columns=['h3_cell'])
                   for s in ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']]).h3_cell.unique()
kids = [h3.cell_to_children(c, 9) for c in cells]
owner = np.repeat(np.arange(len(cells)), [len(k) for k in kids])
pts = np.array([h3.cell_to_latlng(k) for ks in kids for k in ks])
lab = np.zeros(len(pts), np.uint8)
tl = (np.floor(pts[:, 0] / 3) * 3).astype(int); tn = (np.floor(pts[:, 1] / 3) * 3).astype(int)
for (la, lo), idx in pd.DataFrame({'la': tl, 'lo': tn}).groupby(['la', 'lo']).groups.items():
    name = f"ESA_WorldCover_10m_2021_v200_{'N' if la >= 0 else 'S'}{abs(la):02d}{'W' if lo < 0 else 'E'}{abs(lo):03d}_Map.tif"
    url = f'/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}'
    try:
        with rasterio.open(url) as r:
            f = 4; a = r.read(1, out_shape=(r.height // f, r.width // f), resampling=Resampling.nearest)
            tr = r.transform * r.transform.scale(r.width / a.shape[1], r.height / a.shape[0])
            ii = np.asarray(idx); row, col = rasterio.transform.rowcol(tr, pts[ii, 1], pts[ii, 0])
            row = np.clip(np.asarray(row), 0, a.shape[0] - 1); col = np.clip(np.asarray(col), 0, a.shape[1] - 1)
            lab[ii] = a[row, col]
        print('tile', name, len(ii), flush=True)
    except Exception as e:
        print('missing', name, e, flush=True)
df = pd.DataFrame({'owner': owner, 'lab': lab})
share = pd.crosstab(df.owner, df.lab, normalize='index')
out = pd.DataFrame({'h3_cell': cells})
for k, v in CLASSES.items():
    out[f'lc_{v}'] = share[k].reindex(np.arange(len(cells))).fillna(0).to_numpy(np.float32) if k in share else np.float32(0)
out['lc_n'] = df[df.lab > 0].groupby('owner').size().reindex(np.arange(len(cells))).fillna(0).to_numpy()
out.loc[out.lc_n == 0, [c for c in out if c.startswith('lc_') and c != 'lc_n']] = np.nan
out['lc_burnable'] = out.lc_tree + out.lc_shrub + out.lc_grass + out.lc_wetland
(ROOT / 'data/static').mkdir(exist_ok=True); out.to_parquet(ROOT / 'data/static/landcover_h3r7.parquet', index=False)
print(out.describe().round(3).T.to_string())
