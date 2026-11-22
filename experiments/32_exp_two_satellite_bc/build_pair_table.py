"""Two-satellite version of Experiment 25's table: S-NPP OR NOAA-20 (both from FIRMS), all years 2018-2023.
Same 21 features, same sampling (all active rows and positives, seeded 1% quiet negatives at weight 100), same grids,
but every fire-history feature, the quiet flag, the target and the persistence rule use detections from either
satellite. This also fills the 2018-2020 NOAA-20 gap that Exp 21 had (task G1): FIRMS NOAA-20 covers 2018 onward.
Column names are kept from Exp 25 so the Exp 30/31 code runs unchanged: `target_snpp_filled` here means the
two-satellite gap-filled target, `snpp_persistence` the "either satellite yesterday" rule.
Writes data/pair_lightning_h3r7/{training_samples/<centre>/snpp_lightning_YYYY.parquet (Exp 25 file names kept), evaluation/YYYY/<centre>_..._full_grid.parquet}.
2024 is not built.
usage: python build_pair_table.py REPO_ROOT"""
import io, sys, zipfile, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, h3

ROOT = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('e25', ROOT / 'experiments/25_exp_snpp_lightning_2024/run_experiment.py')
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
OUT = ROOT / 'data/pair_lightning_h3r7'

sn = pd.read_csv(E.FIRMS, usecols=['acq_date', 'h3_cell', 'frp', 'daynight'])
with zipfile.ZipFile(ROOT / 'data/raw_archives/firms_noaa20_detections_2018_2026_order_835290.zip') as z:
    n2 = pd.read_csv(io.BytesIO(z.read('fire_archive_J1V-C2_835290.csv')), usecols=['latitude', 'longitude', 'acq_date', 'frp', 'daynight'])
n2 = n2[n2.latitude.between(47.5, 60.5) & n2.longitude.between(-140, -113.5)]
n2['h3_cell'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(n2.latitude, n2.longitude)]
det = pd.concat([sn.assign(sat='snpp'), n2[['acq_date', 'h3_cell', 'frp', 'daynight']].assign(sat='n20')], ignore_index=True)
det['date'] = pd.to_datetime(det.acq_date); det = det[det.date.dt.month.between(5, 9) & det.date.dt.year.between(2018, 2023)].copy()
det['year'] = det.date.dt.year; det['day'] = (det.date - pd.to_datetime(det.year.astype(str) + '-05-01')).dt.days
print(det.groupby(['year', 'sat']).size().unstack(), flush=True)

for ri, region in enumerate(E.REGIONS):
    slug = region.lower().replace(' ', '_')
    for year in range(2018, 2024):
        cells, _, features, y, baseline = E.build(region, year, det)
        if year <= 2021:  # training sample, exactly Exp 25's rule
            quiet = features[E.FEATURES.index('quiet_stratum')].astype(bool)
            draw = np.random.default_rng([E.SEED, year, ri]).random(y.shape) < E.RATE
            keep = ~quiet | y | draw; ci, ti = np.where(keep)
            s = pd.DataFrame(np.column_stack([a[keep] for a in features]), columns=E.FEATURES)
            s['h3_cell'] = cells.to_numpy()[ci]
            s['target_date'] = pd.Timestamp(f'{year}-05-01') + pd.to_timedelta(ti + E.START, unit='D')
            s['target_snpp_filled'] = y[keep]; s['sample_weight'] = np.where(quiet[keep] & ~y[keep], 1 / E.RATE, 1)
            f = OUT / 'training_samples' / slug; f.mkdir(parents=True, exist_ok=True)
            s.to_parquet(f / f'snpp_lightning_{year}.parquet', index=False)
        else:
            t = pd.DataFrame({n: a.reshape(-1) for n, a in zip(E.FEATURES, features)})
            t['h3_cell'] = np.repeat(cells.to_numpy(), E.DAYS - E.START)
            t['target_date'] = np.tile(pd.date_range(f'{year}-05-08', f'{year}-09-30').to_numpy(), len(cells))
            t['target_snpp_filled'] = y.ravel(); t['snpp_persistence'] = baseline.ravel()
            f = OUT / 'evaluation' / str(year); f.mkdir(parents=True, exist_ok=True)
            t.to_parquet(f / f'{slug}_snpp_lightning_full_grid.parquet', index=False)
        print(region, year, 'positives', int(y.sum()), flush=True)
