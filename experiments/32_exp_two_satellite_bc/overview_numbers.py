"""Numbers for experiments/model_overview.md (two-satellite table, Exp 32).
1. Fire cell-days by type for every year 2018-2023 (training samples hold every positive; 2022-2023 from full grids).
2. Watch list for 2023: each Fire Centre's N highest-scored quiet cells each day (N = 10, 25, 50, 100), by the stacked
   model, the general model and a random list; share of quiet-area new fires on the list the day before, and precision.
Writes overview_numbers.json. usage: python overview_numbers.py REPO_ROOT"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1]); SRC = ROOT / 'data/pair_lightning_h3r7'; HERE = ROOT / 'experiments/32_exp_two_satellite_bc'
REGIONS = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
COLS = ['local_count_1d', 'neighbour_count_1d', 'quiet_stratum', 'target_snpp_filled']
def kind(d):
    return np.select([d.local_count_1d > 0, d.neighbour_count_1d > 0, d.quiet_stratum == 1],
                     ['continuing', 'next_door_spread', 'new_quiet'], 'recent_nearby')
rows = []
for y in range(2018, 2024):
    files = glob.glob(str(SRC / f'training_samples/*/snpp_lightning_{y}.parquet')) if y <= 2021 else glob.glob(str(SRC / f'evaluation/{y}/*.parquet'))
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files]); d = d[d.target_snpp_filled]
    c = pd.Series(kind(d)).value_counts()
    rows.append({'year': y, 'fires': int(len(d)), **{k: int(c.get(k, 0)) for k in ['continuing', 'next_door_spread', 'recent_nearby', 'new_quiet']}})
dist = pd.DataFrame(rows); print(dist.to_string())

ND = 153 - 7
wl = {}
parts = []
for r in REGIONS:
    s = pd.read_parquet(HERE / f'exp31/results/stack_scores/{r}_2023.parquet')
    s['day'] = np.arange(len(s)) % ND; s['region'] = r; parts.append(s[s.quiet])
q = pd.concat(parts, ignore_index=True)
rng = np.random.default_rng(42); q['random'] = rng.random(len(q))
for model in ('stk', 'A', 'random'):
    rank = q.groupby(['region', 'day'])[model].rank(ascending=False, method='first')
    for N in (10, 25, 50, 100):
        on = (rank <= N).to_numpy(); y = q.y.to_numpy()
        wl[f'{model}_top{N}'] = {'listed': int(on.sum()), 'caught': int((on & y).sum()), 'catch_share': float((on & y).sum() / y.sum()),
                                 'precision': float((on & y).sum() / on.sum())}
wl['quiet_fires'] = int(q.y.sum()); wl['quiet_cells_per_centre_day'] = float(q.groupby(['region', 'day']).size().mean())
for k, v in wl.items(): print(k, v)
json.dump({'dist': rows, 'watch_list_2023': wl}, open(HERE / 'overview_numbers.json', 'w'), indent=1)
