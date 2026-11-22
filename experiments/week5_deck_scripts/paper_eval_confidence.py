"""Confidence of Exp 21 CatBoost (`exp16_spread`) under the paper's evaluation: 8-day target `burn_8d`, Exp 38's
1 fire : 2 non-fire Test sample for seed 0, redrawn exactly with Exp 38's own sampling code and seed, threshold tuned
on the 2022 sample. Checks the redraw reproduces Exp 38's published F1, then writes paper_conf.json.
usage: python paper_eval_confidence.py REPO_ROOT OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); EXP = ROOT / 'experiments'
R = EXP / '38_exp_canadafiresat_protocol_bc/results'
SEED, NEG, LABEL, MODEL = 42, 2, 'burn_8d', 'exp16_spread'  # SEED from Exp 03, as imported by Exp 38

def fwi_rank_bins(fwi, rng, bins=10):  # copied from Exp 38
    order = np.lexsort((rng.random(len(fwi)), np.nan_to_num(fwi, nan=-np.inf)))
    rank = np.empty(len(fwi), np.int64); rank[order] = np.arange(len(fwi))
    return rank * bins // max(len(fwi), 1)

def draw_test(data, label, rng, excluded):  # copied from Exp 38
    positives, negatives, negative_cells = [], [], {}
    for region, d in data.groupby('region', sort=True):
        pos = d[label].to_numpy()
        if not pos.any():
            continue
        days = d.issue.to_numpy(); first, last = days[pos].min(), days[pos].max()
        pool = ~pos & (days >= first) & (days <= last) & ~d.prior_burned.to_numpy()
        if region in excluded:
            pool &= ~np.isin(d.cell.to_numpy(), excluded[region])
        pool_index = d.index.to_numpy()[pool]
        bins = fwi_rank_bins(d.wx_fwi.to_numpy()[pool], rng)
        wanted = NEG * int(pos.sum()); per_bin = np.full(10, wanted // 10) + (np.arange(10) < wanted % 10)
        chosen = np.concatenate([rng.choice(pool_index[bins == b], min(per_bin[b], (bins == b).sum()), replace=False) for b in range(10)])
        positives.append(d.index.to_numpy()[pos]); negatives.append(chosen)
        negative_cells[region] = np.unique(data.loc[chosen, 'cell'].to_numpy())
    return np.concatenate(positives), np.concatenate(negatives), negative_cells

regions = ['cariboo', 'coastal', 'kamloops', 'northwest', 'prince_george', 'southeast']
cols = ['region', 'year', 'cell', 'issue', 'prior_burned', 'wx_fwi', LABEL, MODEL]
data = pd.concat([pd.read_parquet(R / f'scores/{r}_{y}.parquet', columns=cols) for r in regions for y in (2022, 2023)], ignore_index=True)
sel, ev = data[data.year == 2022], data[data.year == 2023]
rng = np.random.default_rng([SEED, 24, 0, 0])  # seed 0, label index 0 (burn_8d)
p22, n22, negcells = draw_test(sel, LABEL, rng, {})
p23, n23, _ = draw_test(ev, LABEL, rng, negcells)
tune = data.loc[np.concatenate([p22, n22])]
p, r, t = precision_recall_curve(tune[LABEL], tune[MODEL])
with np.errstate(divide='ignore', invalid='ignore'):
    thr = float(t[np.nan_to_num(2 * p * r / (p + r))[:-1].argmax()])
test = data.loc[np.concatenate([p23, n23])]
y = test[LABEL].to_numpy(); s = test[MODEL].to_numpy(); fl = s >= thr
tp, fp, fn, tn = (fl & y).sum(), (fl & ~y).sum(), (~fl & y).sum(), (~fl & ~y).sum()
f1 = 2 * tp / (2 * tp + fp + fn)
ref = pd.read_csv(R / 'protocol_metrics_by_seed.csv').query("label==@LABEL and model==@MODEL and set=='test' and seed==0").iloc[0]
print('redraw: positives', len(p23), 'negatives', len(n23), 'threshold', thr, 'F1', round(f1, 4), '| Exp 38 published F1', round(ref.f1, 4), 'threshold', ref.threshold)
assert abs(f1 - ref.f1) < 1e-6, 'redraw does not reproduce Exp 38'
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
box = {k: {'n': int(m.sum()), 'q': np.quantile(s[m], qs).tolist()} for k, m in (('tp', fl & y), ('fp', fl & ~y), ('fn', ~fl & y), ('tn', ~fl & ~y))}
edges = [0, 1e-4, 1e-3, 1e-2, .05, .1, .2, .3, .5, .7, 1.0001]
b = np.digitize(s, edges[1:-1])
cal = [{'lo': edges[i], 'hi': min(edges[i + 1], 1), 'pred': float(s[b == i].mean()), 'obs': float(y[b == i].mean()), 'n': int((b == i).sum())}
       for i in range(len(edges) - 1) if (b == i).sum() > 0]
json.dump({'threshold': thr, 'f1': f1, 'counts': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}, 'box': box, 'cal': cal},
          open(OUT / 'paper_conf.json', 'w'))
for k, v in box.items():
    print(k, v['n'], 'median', v['q'][2], 'IQR', v['q'][1], v['q'][3])
print('cal', [(c['lo'], round(c['pred'], 4), round(c['obs'], 3), c['n']) for c in cal])
