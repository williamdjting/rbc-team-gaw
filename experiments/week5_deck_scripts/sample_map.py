"""Illustration: every-cell evaluation vs a 1 fire : 2 non-fire sample, Kamloops, one 8-day window, 2023."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
EXP = ROOT / 'experiments'
grid = pd.read_parquet(EXP / '02_hotspot_persistence_bc/regions/kamloops/data/h3_grid.parquet', columns=['h3_cell'])
s = pd.read_parquet(EXP / '38_exp_canadafiresat_protocol_bc/results/scores/kamloops_2023.parquet',
                    columns=['cell', 'issue', 'burn_8d', 'prior_burned'])
assert s.cell.max() < len(grid)
issue_dates = pd.Timestamp('2023-05-08') + pd.to_timedelta(np.arange(18) * 8, unit='D')
aug = [i for i, d in enumerate(issue_dates) if d.month == 8]
by = s[s.issue.isin(aug)].groupby('issue').burn_8d.sum()
issue = int(by.idxmax())
d = s[s.issue == issue]
start = issue_dates[issue]
fire = d.loc[d.burn_8d, 'cell'].to_numpy()
pool = d.loc[~d.burn_8d & ~d.prior_burned, 'cell'].to_numpy()
neg = np.random.default_rng(0).choice(pool, size=2 * len(fire), replace=False)
print('window', start.date(), 'fires', len(fire), 'sampled non-fire', len(neg), 'cells', len(d))

def polys(cells):
    out = []
    for c in grid.h3_cell.to_numpy()[cells]:
        b = h3.cell_to_boundary(c)
        out.append([(lon, lat) for lat, lon in b])
    return out

allp = polys(np.arange(len(grid)))
fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), sharex=True, sharey=True)
red, blue, grey = '#e8642c', '#2a78d6', '#d9d4ce'
for ax, title in zip(axes, ['Every cell (ours)', 'Sampled 1 fire : 2 non-fire (theirs)']):
    ax.set_aspect(1 / np.cos(np.deg2rad(51)))
    ax.axis('off')
    ax.set_title(title, fontsize=15, pad=10)
# left: every cell scored -> all hexes light grey, fires red
axes[0].add_collection(PolyCollection(allp, facecolor='#f1eeea', edgecolor=grey, linewidth=0.15))
axes[0].add_collection(PolyCollection(polys(fire), facecolor=red, edgecolor=red, linewidth=0.3))
# right: only sampled cells scored -> outline of region faint, fires red, sampled non-fire blue
axes[1].add_collection(PolyCollection(allp, facecolor='white', edgecolor='#ece8e3', linewidth=0.15))
axes[1].add_collection(PolyCollection(polys(neg), facecolor=blue, edgecolor=blue, linewidth=0.3))
axes[1].add_collection(PolyCollection(polys(fire), facecolor=red, edgecolor=red, linewidth=0.3))
for ax in axes:
    ax.autoscale_view()
axes[0].text(0.02, 0.02, f'{len(fire):,} fire cells\n{len(d) - len(fire):,} non-fire cells scored', transform=axes[0].transAxes, fontsize=12)
axes[1].text(0.02, 0.02, f'{len(fire):,} fire cells\n{len(neg):,} non-fire cells scored', transform=axes[1].transAxes, fontsize=12)
fig.suptitle(f'Kamloops Fire Centre · fire in the 8 days from {start:%b %d}, 2023', fontsize=13, color='#4d4740', y=0.99)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(OUT, dpi=110, facecolor='white')
print('saved', OUT)
