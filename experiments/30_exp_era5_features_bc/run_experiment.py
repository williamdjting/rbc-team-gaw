"""Experiment 30: does ERA5-Land (incl. soil moisture) help next-day fire, overall and on new fires?
Base = Experiment 25's table (15 S-NPP history + 6 lightning features, S-NPP gap-filled target).
Four CatBoost models, identical settings:
  A full_base   all rows, base features          B full_era5   all rows, base + ERA5
  C quiet_base  quiet rows only, base features   D quiet_era5  quiet rows only, base + ERA5
Quiet = no S-NPP detection in the cell or its ring-1 neighbours over the previous 7 days (Exp 25 quiet_stratum),
i.e. a new fire that a "fire yesterday" rule can never catch.
Train 2018-2021 (Exp 25 samples, 1% quiet negatives with weight 100); iterations chosen by early stopping on 2021
with 2018-2020 fitting, then refit on 2018-2021. Thresholds per Fire Centre on the full 2022 grid. Report 2023.
2024 is Exp 25's frozen holdout and is not touched.
usage: python run_experiment.py REPO_ROOT [train|score|metrics|all]"""
import json, sys, glob, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

import os
ROOT = Path(sys.argv[1]); STAGE = sys.argv[2] if len(sys.argv) > 2 else 'all'
PAIR = os.environ.get('GAW_TABLE', 'pair') == 'pair'  # two-satellite table from Exp 32 (outputs go to Exp 32)
HERE = ROOT / ('experiments/32_exp_two_satellite_bc/exp30' if PAIR else 'experiments/30_exp_era5_features_bc')
RES = HERE / 'results'; (RES / 'scores').mkdir(parents=True, exist_ok=True)
SRC = ROOT / ('data/pair_lightning_h3r7' if PAIR else 'data/snpp_lightning_h3r7'); E5 = ROOT / 'data/era5_h3r7'
BASE = ["local_count_1d", "local_count_3d", "local_count_7d", "neighbour_count_1d", "neighbour_count_3d",
        "neighbour_count_7d", "frp_sum_1d", "frp_sum_7d", "frp_max_1d", "night_count_1d", "active_days_7d",
        "disk1_active_days_7d", "days_since_detection", "history_left_censored", "quiet_stratum",
        "ltg_cell_1d", "ltg_3x3_1d", "ltg_3x3_3d", "ltg_9x9_1d", "ltg_9x9_3d", "ltg_days_since_3x3"]
META = json.load(open(E5 / 'meta.json' if (E5 / 'meta.json').exists() else E5 / 'era5_meta.json'))
ERA = META['features']
CUBE = np.load(E5 / 'era5_point_features.npy', mmap_mode='r')  # [year-2018, point, season day, feature]
_m = pd.read_parquet(E5 / 'cell_to_point.parquet')
C2P = _m.set_index('h3_cell').pid.astype(float).where(_m.set_index('h3_cell').dist_km <= 10)  # >10 km: outside ERA5 file, leave missing
REGIONS = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
MODELS = {'A_full_base': (False, BASE), 'B_full_era5': (False, BASE + ERA),
          'C_quiet_base': (True, BASE), 'D_quiet_era5': (True, BASE + ERA)}
PARAMS = dict(learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function='Logloss', random_seed=42,
              thread_count=-1, verbose=200, allow_writing_files=False)
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

def add_era5(df):
    """ERA5 for target day t is taken from day t-1 (features stop at the previous UTC day)."""
    d = pd.to_datetime(df.target_date)
    yi = (d.dt.year - 2018).to_numpy()
    di = (d - pd.to_datetime(d.dt.year.astype(str) + '-05-01')).dt.days.to_numpy() - 1
    pid = C2P.reindex(df.h3_cell.to_numpy()).to_numpy()
    ok = ~np.isnan(pid) & (di >= 0)
    out = np.full((len(df), len(ERA)), np.nan, np.float32)
    out[ok] = CUBE[yi[ok], pid[ok].astype(np.int64), di[ok]]
    return pd.DataFrame(out, columns=ERA, index=df.index)

def train():
    t = pd.concat([pd.read_parquet(f) for f in glob.glob(str(SRC / 'training_samples/*/*.parquet'))], ignore_index=True)
    t = pd.concat([t, add_era5(t)], axis=1)
    say('training rows', len(t), 'positives', int(t.target_snpp_filled.sum()), 'ERA5 missing share', round(float(t[ERA].isna().mean().mean()), 4))
    yr = pd.to_datetime(t.target_date).dt.year
    info = {}
    for name, (quiet_only, feats) in MODELS.items():
        d = t[t.quiet_stratum == 1] if quiet_only else t
        dy = yr[d.index]
        fit, val = d[dy <= 2020], d[dy == 2021]
        mk = lambda x: Pool(x[feats], x.target_snpp_filled.astype(int), weight=x.sample_weight)
        t0 = time.time()
        m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, **PARAMS)
        m.fit(mk(fit), eval_set=mk(val))
        n_it = max(100, m.get_best_iteration() + 1)
        m = CatBoostClassifier(iterations=n_it, **PARAMS); m.fit(mk(d))
        m.save_model(str(RES / f'{name}.cbm'))
        imp = dict(zip(feats, m.get_feature_importance().round(3).tolist()))
        info[name] = {'rows': len(d), 'positives': int(d.target_snpp_filled.sum()), 'iterations': n_it,
                      'seconds': round(time.time() - t0), 'importance': imp}
        say(name, 'iterations', n_it, 'rows', len(d), 'sec', info[name]['seconds'])
    json.dump(info, open(RES / 'training.json', 'w'), indent=1)

def score():
    models = {n: CatBoostClassifier().load_model(str(RES / f'{n}.cbm')) for n in MODELS}
    for year in (2022, 2023):
        for region in REGIONS:
            f = SRC / f'evaluation/{year}/{region}_snpp_lightning_full_grid.parquet'
            df = pd.read_parquet(f, columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled', 'snpp_persistence'])
            df = pd.concat([df, add_era5(df)], axis=1)
            q = df.quiet_stratum.to_numpy() == 1
            out = pd.DataFrame({'cell': pd.factorize(df.h3_cell)[0].astype(np.int32),
                                'day': (pd.to_datetime(df.target_date) - pd.Timestamp(f'{year}-05-01')).dt.days.astype(np.int16),
                                'y': df.target_snpp_filled.to_numpy(bool), 'rule': df.snpp_persistence.to_numpy().astype(bool),
                                'quiet': q, 'local1d': df.local_count_1d.to_numpy() > 0})
            for n, (quiet_only, feats) in MODELS.items():
                s = np.full(len(df), np.nan, np.float32)
                rows = q if quiet_only else np.ones(len(df), bool)
                for a in range(0, rows.sum(), 2_000_000):
                    idx = np.flatnonzero(rows)[a:a + 2_000_000]
                    s[idx] = models[n].predict_proba(df.iloc[idx][feats])[:, 1]
                out[n] = s
            out.to_parquet(RES / f'scores/{region}_{year}.parquet', index=False)
            say('scored', year, region, len(df), 'ERA5 missing share', round(float(df[ERA].isna().mean().mean()), 4))

def best_threshold(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys); P = ys.sum()
    f1 = 2 * tp / (tp + fp + P); k = int(np.argmax(f1)); return float(s[o][k])

def metrics():
    S = {y: {r: pd.read_parquet(RES / f'scores/{r}_{y}.parquet') for r in REGIONS} for y in (2022, 2023)}
    thr = {}
    for r in REGIONS:
        d = S[2022][r]; thr[r] = {}
        for n, (quiet_only, _) in MODELS.items():
            dd = d[d.quiet] if quiet_only else d
            thr[r][n] = best_threshold(dd.y.to_numpy(), dd[n].to_numpy())
    ev = []
    for r in REGIONS:
        d = S[2023][r].copy(); d['region'] = r
        d['week'] = (pd.Timestamp('2023-05-01') + pd.to_timedelta(d.day, unit='D')).dt.isocalendar().week.to_numpy()
        for n in MODELS:
            d[n + '_flag'] = d[n].to_numpy() >= thr[r][n]
        d['stratum'] = np.where(d.local1d, 'continuing', np.where(d.quiet, 'new_quiet', 'spread_nearby'))
        ev.append(d)
    ev = pd.concat(ev, ignore_index=True)
    # watch list: top-N quiet cells per centre per day, by each model
    for n in MODELS:
        qd = ev[ev.quiet]
        rank = qd.groupby(['region', 'day'])[n].rank(ascending=False, method='first')
        for N in (25, 50, 100):
            ev.loc[qd.index, f'{n}_top{N}'] = (rank <= N).to_numpy()
    ev = ev.fillna({c: False for c in ev.columns if '_top' in c})
    R = {'thresholds_2022': thr}
    def f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn)
    tot = {}
    for n in ['rule'] + [m for m in MODELS if 'full' in m]:
        fl = ev.rule.to_numpy() if n == 'rule' else ev[n + '_flag'].to_numpy(); y = ev.y.to_numpy()
        tp, fp, fn = int((fl & y).sum()), int((fl & ~y).sum()), int((~fl & y).sum())
        rec = {s: float(fl[y & (ev.stratum == s).to_numpy()].mean()) for s in ('continuing', 'spread_nearby', 'new_quiet')}
        tot[n] = {'f1': f1(tp, fp, fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'tp': tp, 'fp': fp, 'fn': fn,
                  'recall_by_stratum': rec}
    R['total_2023'] = tot
    q = ev[ev.quiet]; yq = q.y.to_numpy()
    new = {'quiet_rows': len(q), 'quiet_positives': int(yq.sum()),
           'positives_by_stratum': ev[ev.y].stratum.value_counts().to_dict()}
    for n in MODELS:
        new[n] = {'ap': float(average_precision_score(yq, q[n].to_numpy())),
                  'ap_by_centre': {r: float(average_precision_score(g.y, g[n])) for r, g in q.groupby('region')},
                  **{f'watch_top{N}_catch': float(q[f'{n}_top{N}'][q.y].mean()) for N in (25, 50, 100)}}
        if 'quiet' in n:
            fl = q[n + '_flag'].to_numpy(); tp, fp, fn = (fl & yq).sum(), (fl & ~yq).sum(), (~fl & yq).sum()
            new[n].update({'f1_quiet': float(f1(tp, fp, fn)), 'precision_quiet': float(tp / max(tp + fp, 1)), 'recall_quiet': float(tp / (tp + fn))})
    days_per_centre = q.groupby(['region', 'day']).size()
    new['random_top50_catch'] = float((50 / days_per_centre).clip(upper=1).mean())
    R['new_fires_2023'] = new
    # paired bootstrap over ISO weeks
    g = ev.groupby('week'); W = np.array(sorted(ev.week.unique())); rng = np.random.default_rng(42)
    def wk(fl, mask=None):
        m = np.ones(len(ev), bool) if mask is None else mask
        y = ev.y.to_numpy()
        return pd.DataFrame({'tp': fl & y & m, 'fp': fl & ~y & m, 'fn': ~fl & y & m, 'w': ev.week}).groupby('w').sum().reindex(W).to_numpy()
    A, B = wk(ev.A_full_base_flag.to_numpy()), wk(ev.B_full_era5_flag.to_numpy())
    qm = ev.quiet.to_numpy()
    C50 = wk(ev.C_quiet_base_top50.to_numpy().astype(bool), qm); D50 = wk(ev.D_quiet_era5_top50.to_numpy().astype(bool), qm)
    Aq = wk(ev.A_full_base_flag.to_numpy(), qm); Bq = wk(ev.B_full_era5_flag.to_numpy(), qm)
    bs = {'f1_B_minus_A': [], 'watch50_D_minus_C': [], 'newrecall_B_minus_A': []}
    for _ in range(2000):
        i = rng.integers(0, len(W), len(W))
        a, b = A[i].sum(0), B[i].sum(0); bs['f1_B_minus_A'].append(f1(*b) - f1(*a))
        c, d = C50[i].sum(0), D50[i].sum(0); bs['watch50_D_minus_C'].append(d[0] / (d[0] + d[2]) - c[0] / (c[0] + c[2]))
        a, b = Aq[i].sum(0), Bq[i].sum(0); bs['newrecall_B_minus_A'].append(b[0] / (b[0] + b[2]) - a[0] / (a[0] + a[2]))
    R['bootstrap_95'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    tr = json.load(open(RES / 'training.json'))
    for n in ('B_full_era5', 'D_quiet_era5'):
        imp = tr[n]['importance']; R.setdefault('importance_share', {})[n] = {
            'fire_history': round(sum(v for k, v in imp.items() if not k.startswith(('ltg', 'era5'))), 1),
            'lightning': round(sum(v for k, v in imp.items() if k.startswith('ltg')), 1),
            'era5': round(sum(v for k, v in imp.items() if k.startswith('era5')), 1),
            'top_era5': sorted([(k, v) for k, v in imp.items() if k.startswith('era5')], key=lambda x: -x[1])[:6]}
    json.dump(R, open(RES / 'metrics.json', 'w'), indent=1)
    say(json.dumps({'total': {k: {x: round(v[x], 4) for x in ('f1', 'precision', 'recall')} | {'new_recall': round(v['recall_by_stratum']['new_quiet'], 4)} for k, v in tot.items()},
                    'new': {k: (round(v['ap'], 4), round(v['watch_top50_catch'], 4)) for k, v in new.items() if isinstance(v, dict) and 'ap' in v},
                    'bootstrap': R['bootstrap_95'], 'importance': R['importance_share']}, indent=1))

if STAGE in ('train', 'all'): train()
if STAGE in ('score', 'all'): score()
if STAGE in ('metrics', 'all'): metrics()
