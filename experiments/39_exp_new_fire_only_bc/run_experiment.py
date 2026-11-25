"""Experiment 39: a model that only predicts new fires (mentor suggestion, Nov 25, 2026).
Rows: quiet cells only (no S-NPP or NOAA-20 detection in the cell or its ring-1 neighbours for 7 days), so the
"fire yesterday" rule cannot be learned. Label v1 (two-satellite, 1-day gaps filled). Inputs: Exp 25's 21 inputs
(fire history + lightning) + Exp 32's 8 jump features (fires up to ~24 km). Train 2018-2021 (quiet rows of
data/pair_lightning_h3r7 training samples: every quiet fire + 1% of quiet non-fires), early stopping on 2021,
refit; test on every quiet cell-day of 2023 (33.8M).
Models:
  Q_pop    CatBoost, population weights (sampled non-fires x100): the plain quiet-only baseline
  Q_wce    CatBoost, weighted cross-entropy: fire class up-weighted so both classes carry equal total weight
  Q_1to2   CatBoost on 2 random non-fires per fire (CanadaFireSat-style sampling), unweighted
  Q_focal  MLP 256/256/128 (SiLU, dropout 0.1), focal loss (gamma 2, alpha 0.75), on the 1% sample, unweighted
Reference: Exp 32 stacked + jumps (general model, all rows), scores reused.
Evaluation (2023): average precision on quiet cells; watch list (top N per Fire Centre per day); the model's top-k
most confident predictions BC-wide per day and over the season, and what % are real new fires; by new-fire group
(genuine ignition / jump of an older BCWS fire / other burn); routed total F1 (quiet cells decided by the new model,
all other cells by the best stacked model, thresholds on 2022); week bootstrap vs the reference.
usage: python run_experiment.py REPO_ROOT"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(sys.argv[1]); EXP = ROOT / 'experiments'; X32 = EXP / '32_exp_two_satellite_bc'
HERE = EXP / '39_exp_new_fire_only_bc'; RES = HERE / 'results'; (RES / 'scores').mkdir(parents=True, exist_ok=True)
src = open(X32 / 'jumps/run_jumps.py', encoding='utf-8').read()
J = {'__name__': 'jmp'}; sys.argv = [sys.argv[0], str(ROOT)]
exec(compile(src[:src.index('# ---- training')], 'run_jumps_defs', 'exec'), J)  # detections, jump grids, helpers
SRC, REGIONS, BASE, JUMP, PARAMS, jump_grids, gather = J['SRC'], J['REGIONS'], J['BASE'], J['JUMP'], J['PARAMS'], J['jump_grids'], J['gather']
F = BASE + JUMP; SEED = 42; t0 = time.time()
torch.manual_seed(SEED); np.random.seed(SEED)
log = open(HERE / 'run.log', 'a', encoding='utf-8')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

# ---------------- training rows (quiet only)
tr = []
for r in REGIONS:
    for y in range(2018, 2022):
        cells, f = jump_grids(r, y)
        d = pd.read_parquet(SRC / f'training_samples/{r}/snpp_lightning_{y}.parquet'); d = d[d.quiet_stratum == 1].copy()
        d['target_date'] = pd.to_datetime(d.target_date).astype('datetime64[ns]')
        for c, v in gather(d, cells, f).items(): d[c] = v
        d['year'] = y; d['region'] = r; tr.append(d)
    say('train features', r)
tr = pd.concat(tr, ignore_index=True); tr['y'] = tr.target_snpp_filled.astype(bool)
say('quiet training rows', len(tr), 'fires', int(tr.y.sum()), 'by year', tr[tr.y].year.value_counts().sort_index().to_dict())
fitm, valm = (tr.year <= 2020).to_numpy(), (tr.year == 2021).to_numpy()

def cat_fit(name, d_fit, d_val, w_fit, w_val, d_all, w_all):
    m = CatBoostClassifier(iterations=3000, od_type='Iter', od_wait=150, use_best_model=True, eval_metric='PRAUC', **PARAMS)
    m.fit(Pool(d_fit[F], d_fit.y.astype(int), weight=w_fit), eval_set=Pool(d_val[F], d_val.y.astype(int), weight=w_val))
    it = max(100, m.get_best_iteration() + 1)
    m = CatBoostClassifier(iterations=it, **PARAMS); m.fit(Pool(d_all[F], d_all.y.astype(int), weight=w_all))
    m.save_model(str(RES / f'{name}.cbm')); say(name, 'iterations', it); return m
def wce(d):
    w = d.sample_weight.to_numpy().astype(float); pos = d.y.to_numpy()
    w[pos] *= w[~pos].sum() / w[pos].sum(); return w
models = {}
models['Q_pop'] = cat_fit('Q_pop', tr[fitm], tr[valm], tr.sample_weight[fitm], tr.sample_weight[valm], tr, tr.sample_weight)
models['Q_wce'] = cat_fit('Q_wce', tr[fitm], tr[valm], wce(tr[fitm]), wce(tr[valm]), tr, wce(tr))
rng = np.random.default_rng(SEED)
def one_to_two(d):
    pos = d[d.y]; neg = d[~d.y]; k = min(len(neg), 2 * len(pos))
    return pd.concat([pos, neg.iloc[rng.choice(len(neg), k, replace=False)]], ignore_index=True)
s_fit, s_val, s_all = one_to_two(tr[fitm]), one_to_two(tr[valm]), one_to_two(tr)
say('1:2 samples', len(s_fit), len(s_val), len(s_all))
models['Q_1to2'] = cat_fit('Q_1to2', s_fit, s_val, None, None, s_all, None)

# ---------------- MLP with focal loss
class Prep:
    def fit(self, X):
        X = X.copy(); self.log = (np.nanmin(X, 0) >= 0) & (np.nanquantile(X, .999, 0) > 10)
        X[:, self.log] = np.log1p(X[:, self.log]); self.mu = np.nanmean(X, 0); self.sd = np.nanstd(X, 0) + 1e-6; return self
    def __call__(self, X):
        X = X.astype(np.float32).copy(); miss = np.isnan(X).astype(np.float32); X[:, self.log] = np.log1p(np.clip(X[:, self.log], 0, None))
        X = (X - self.mu) / self.sd; X[np.isnan(X)] = 0; return np.concatenate([X, miss], 1).astype(np.float32)
class Net(nn.Module):
    def __init__(self, k):
        super().__init__(); self.f = nn.Sequential(nn.Linear(k, 256), nn.SiLU(), nn.Dropout(.1), nn.Linear(256, 256), nn.SiLU(), nn.Dropout(.1), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x): return self.f(x).squeeze(1)
def focal(logit, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logit); ce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction='none')
    pt = p * y + (1 - p) * (1 - y); a = alpha * y + (1 - alpha) * (1 - y); return (a * (1 - pt) ** gamma * ce).mean()
def train_mlp(Xf, yf, Xv, yv, epochs=None):
    net = Net(Xf.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xf), torch.from_numpy(yf.astype(np.float32)); best, best_ep, wait, state = -1, 0, 0, None
    for ep in range(epochs or 20):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            b = perm[i:i + 4096]; opt.zero_grad(); loss = focal(net(Xt[b]), yt[b]); loss.backward(); opt.step()
        if epochs: continue
        net.eval()
        with torch.no_grad(): ap = average_precision_score(yv, net(torch.from_numpy(Xv)).numpy())
        say('  focal epoch', ep + 1, 'val AP (2021 sample)', round(ap, 4))
        if ap > best: best, best_ep, wait, state = ap, ep + 1, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 3: break
    if not epochs: net.load_state_dict(state)
    return net, best_ep
prep = Prep().fit(tr.loc[fitm, F].to_numpy(np.float64))
_, ep = train_mlp(prep(tr.loc[fitm, F].to_numpy()), tr.y.to_numpy()[fitm], prep(tr.loc[valm, F].to_numpy()), tr.y.to_numpy()[valm])
prep = Prep().fit(tr[F].to_numpy(np.float64)); torch.manual_seed(SEED)
mlp, _ = train_mlp(prep(tr[F].to_numpy()), tr.y.to_numpy(), None, None, epochs=ep); mlp.eval()
torch.save(mlp.state_dict(), RES / 'Q_focal.pt'); say('Q_focal epochs', ep)
del tr, s_fit, s_val, s_all

# ---------------- scoring quiet cells 2022/2023
MODELS = ['Q_pop', 'Q_wce', 'Q_1to2', 'Q_focal']
ev = {}
for y in (2022, 2023):
    for r in REGIONS:
        cells, f = jump_grids(r, y)
        d = pd.read_parquet(SRC / f'evaluation/{y}/{r}_snpp_lightning_full_grid.parquet', columns=BASE + ['h3_cell', 'target_date', 'target_snpp_filled'])
        d['row'] = np.arange(len(d)); d = d[d.quiet_stratum == 1].reset_index(drop=True)
        for c, v in gather(d, cells, f).items(): d[c] = v
        o = pd.DataFrame({'row': d.row.to_numpy(np.int32), 'y': d.target_snpp_filled.to_numpy(bool), 'day': (pd.to_datetime(d.target_date) - pd.Timestamp(f'{y}-05-01')).dt.days.to_numpy(np.int16)})
        for n in MODELS:
            s = np.empty(len(d), np.float32)
            for a in range(0, len(d), 2_000_000):
                chunk = d.iloc[a:a + 2_000_000]
                if n == 'Q_focal':
                    with torch.no_grad(): s[a:a + 2_000_000] = torch.sigmoid(mlp(torch.from_numpy(prep(chunk[F].to_numpy())))).numpy()
                else: s[a:a + 2_000_000] = models[n].predict_proba(chunk[F])[:, 1]
            o[n] = s
        ref = pd.read_parquet(X32 / f'jumps/results/scores/{r}_{y}.parquet', columns=['stacked', 'stacked_j'])
        o['ref_stacked_j'] = ref.stacked_j.to_numpy()[o.row.to_numpy()]
        o['h3_cell'] = np.where(o.y, d.h3_cell.to_numpy(), '')
        o.to_parquet(RES / f'scores/{r}_{y}.parquet', index=False); ev[(y, r)] = o; say('scored', y, r, len(o))

# ---------------- metrics
exec(compile(src[src.index('# ---- new-fire categories'):src.index('# ---- metrics')], 'cats', 'exec'), J)
ALL = MODELS + ['ref_stacked_j']
Q = pd.concat([ev[(2023, r)].assign(region=r) for r in REGIONS], ignore_index=True); yq = Q.y.to_numpy()
Q['cat'] = 'none'; Q.loc[yq, 'cat'] = [J['category'](c, t) for c, t in zip(Q.h3_cell[yq], Q.day[yq])]
R = {'quiet_rows_2023': len(Q), 'new_fires_2023': int(yq.sum()), 'groups': Q[yq].cat.value_counts().to_dict(),
     'base_rate': float(yq.mean()), 'ap': {}, 'watch_per_centre': {}, 'topk_bc_per_day': {}, 'topk_season': {}}
for n in ALL:
    R['ap'][n] = float(average_precision_score(yq, Q[n]))
    rk = Q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy()
    R['watch_per_centre'][n] = {}
    for N in (10, 25, 50, 100, 200):
        on = rk <= N; R['watch_per_centre'][n][N] = {'caught': int((on & yq).sum()), 'share': float((on & yq).sum() / yq.sum()), 'precision': float((on & yq).sum() / on.sum()),
                                                       **{f'share_{c}': float((on & yq & (Q.cat == c).to_numpy()).sum() / max((yq & (Q.cat == c).to_numpy()).sum(), 1)) for c in ('ignition', 'jump', 'other')}}
    rb = Q.groupby('day')[n].rank(ascending=False, method='first').to_numpy(); R['topk_bc_per_day'][n] = {}
    for k in (10, 50, 100, 500, 1000):
        on = rb <= k; R['topk_bc_per_day'][n][k] = {'precision': float((on & yq).sum() / on.sum()), 'share': float((on & yq).sum() / yq.sum())}
    order = np.argsort(-Q[n].to_numpy(), kind='stable'); R['topk_season'][n] = {}
    for k in (100, 1000, 5000, 10000, 50000):
        R['topk_season'][n][k] = {'precision': float(yq[order[:k]].mean()), 'fires': int(yq[order[:k]].sum())}
# routed total F1: quiet cells by the new model, all others by the best stacked model (pooled threshold 0.311 on 2022)
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
thr_st = json.load(open(X32 / 'jumps/results/tuning_thresholds.json')).get('stacked_pooled', 0.31114125)
def rest_counts(y):
    tp = fp = fn = 0
    for r in REGIONS:
        a = pd.read_parquet(X32 / f'jumps/results/scores/{r}_{y}.parquet', columns=['y', 'quiet', 'stacked'])
        fl = a.stacked.to_numpy() >= thr_st; yy = a.y.to_numpy(); m = ~a.quiet.to_numpy()
        tp += (fl & yy & m).sum(); fp += (fl & ~yy & m).sum(); fn += (~fl & yy & m).sum()
    return np.array([tp, fp, fn], float)
rest22, rest23 = rest_counts(2022), rest_counts(2023)
Q22 = pd.concat([ev[(2022, r)] for r in REGIONS], ignore_index=True); y22 = Q22.y.to_numpy()
base23 = rest23 + [0, 0, yq.sum()]
R['total_f1'] = {'best_stacked_alone': f1(*base23)}
for n in MODELS:
    s = Q22[n].to_numpy(); o = np.argsort(-s, kind='stable'); ys = y22[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys); fn = ys.sum() - tp
    v = 2 * (tp + rest22[0]) / (2 * (tp + rest22[0]) + fp + rest22[1] + fn + rest22[2]); k = int(np.argmax(v))
    best_prefix = v[k] > f1(*(rest22 + [0, 0, y22.sum()])); t = float(s[o][k]) if best_prefix else np.inf
    fl = Q[n].to_numpy() >= t; c = rest23 + [(fl & yq).sum(), (fl & ~yq).sum(), (~fl & yq).sum()]
    R['total_f1'][n] = {'threshold_2022': t, 'quiet_flagged_2023': int(fl.sum()), 'quiet_caught_2023': int((fl & yq).sum()), 'total_f1': f1(*c)}
# bootstrap over ISO weeks vs reference: watch-list share at 50, BC-wide top-100/day precision
wk = (pd.Timestamp('2023-05-01') + pd.to_timedelta(Q.day, unit='D')).dt.isocalendar().week.to_numpy().astype(int); W = np.unique(wk); wi = np.searchsorted(W, wk)
cnt = {}
for n in ALL:
    rk = Q.groupby(['region', 'day'])[n].rank(ascending=False, method='first').to_numpy() <= 50
    rb = Q.groupby('day')[n].rank(ascending=False, method='first').to_numpy() <= 100
    cnt[n] = (np.bincount(wi, (rk & yq).astype(float), len(W)), np.bincount(wi, (rb & yq).astype(float), len(W)), np.bincount(wi, rb.astype(float), len(W)))
fires_w = np.bincount(wi, yq.astype(float), len(W)); rng = np.random.default_rng(SEED); bs = {f'{n}_minus_ref': {'watch50': [], 'bc_top100_precision': []} for n in MODELS}
for _ in range(2000):
    c = np.bincount(rng.integers(0, len(W), len(W)), minlength=len(W)).astype(float); ref = cnt['ref_stacked_j']
    for n in MODELS:
        bs[f'{n}_minus_ref']['watch50'].append((cnt[n][0] @ c - ref[0] @ c) / (fires_w @ c))
        bs[f'{n}_minus_ref']['bc_top100_precision'].append((cnt[n][1] @ c) / (cnt[n][2] @ c) - (ref[1] @ c) / (ref[2] @ c))
R['bootstrap_95'] = {k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in d.items()} for k, d in bs.items()}
json.dump(R, open(RES / 'metrics.json', 'w'), indent=1, default=float)
rows = []
for n in ALL:
    w = R['watch_per_centre'][n]; b = R['topk_bc_per_day'][n]; s = R['topk_season'][n]
    rows.append({'model': n, 'ap': round(R['ap'][n], 4), **{f'watch{N}_share': round(w[N]['share'], 4) for N in (25, 50, 100, 200)},
                 'watch50_precision': round(w[50]['precision'], 4), **{f'watch50_{c}': round(w[50][f'share_{c}'], 4) for c in ('ignition', 'jump', 'other')},
                 **{f'bc_top{k}_day_precision': round(b[k]['precision'], 4) for k in (10, 100, 1000)}, **{f'season_top{k}_precision': round(s[k]['precision'], 4) for k in (1000, 10000)}})
pd.DataFrame(rows).to_csv(RES / 'summary.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
say(pd.DataFrame(rows).to_string(index=False)); say(json.dumps({'total_f1': R['total_f1'], 'bootstrap': R['bootstrap_95'], 'groups': R['groups'], 'base_rate': R['base_rate']}, indent=1, default=float))
say('done', round(time.time() - t0), 's')
