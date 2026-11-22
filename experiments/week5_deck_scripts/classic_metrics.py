"""Classic evaluation curves for Exp 25 (S-NPP + lightning MLP), 2023, every cell-day, all six Fire Centres.
Scores are binned (log-spaced) so 34M rows fit in memory; curves come from cumulative bin counts."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
EVAL = ROOT / 'data/snpp_lightning_h3r7/evaluation/2023'
met = pd.read_csv(ROOT / 'experiments/25_exp_snpp_lightning_2024/results/metrics_2023.csv')
mlp = met[met.model == 'mlp_snpp_lightning']; rule = met[met.model == 'snpp_persistence']
edges = np.concatenate([[0], np.logspace(-6, 0, 601)[1:-1], [1.0000001]])
pos = np.zeros(len(edges) - 1); neg = np.zeros(len(edges) - 1)
for f in sorted(EVAL.glob('*_full_grid.parquet')):
    d = pd.read_parquet(f, columns=['target_snpp_filled', 'mlp_score'])
    y = d.target_snpp_filled.to_numpy().astype(bool); s = d.mlp_score.to_numpy()
    pos += np.histogram(s[y], edges)[0]; neg += np.histogram(s[~y], edges)[0]
    print(f.name, flush=True)
# threshold sweep: flag score >= edges[i]
tp = pos[::-1].cumsum()[::-1]; fp = neg[::-1].cumsum()[::-1]
P, N = pos.sum(), neg.sum()
prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan); rec = tp / P; fpr = fp / N
f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
ap = float(np.nansum(-np.diff(np.append(rec, 0)) * np.nan_to_num(prec)))
auc = float(np.trapezoid(np.append(rec, 0)[::-1], np.append(fpr, 0)[::-1]))
thr = edges[:-1]
keep = np.unique(np.linspace(0, len(thr) - 1, 300).astype(int))
# calibration on 12 bins of predicted probability
cal_edges = [0, .001, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1.0000001]
cal = []
for a, b in zip(cal_edges[:-1], cal_edges[1:]):
    m = (thr >= a) & (thr < b); n = pos[m].sum() + neg[m].sum()
    if n > 0:
        centre = float(((pos[m] + neg[m]) * np.sqrt(np.maximum(thr[m], 1e-7) * np.minimum(edges[1:][m], 1))).sum() / n)
        cal.append({'lo': a, 'hi': min(b, 1), 'pred': centre, 'obs': float(pos[m].sum() / n), 'n': int(n)})
# histogram on coarse log bins for plotting
hb = np.logspace(-6, 0, 25)
hpos = [int(pos[(thr >= a) & (thr < b)].sum()) for a, b in zip(hb[:-1], hb[1:])]
hneg = [int(neg[(thr >= a) & (thr < b)].sum()) for a, b in zip(hb[:-1], hb[1:])]
cm = {k: int(mlp[k].sum()) for k in ['tp', 'fp', 'fn', 'tn']}
rc = {k: int(rule[k].sum()) for k in ['tp', 'fp', 'fn', 'tn']}
out = {'P': int(P), 'N': int(N), 'ap': ap, 'auc': auc, 'base': float(P / (P + N)),
       'curve': {'thr': thr[keep].tolist(), 'prec': np.nan_to_num(prec[keep]).tolist(), 'rec': rec[keep].tolist(),
                 'fpr': fpr[keep].tolist(), 'f1': f1[keep].tolist()},
       'best': {'thr': float(thr[np.argmax(f1)]), 'f1': float(f1.max())},
       'cm': cm, 'rule': rc, 'thresholds': mlp.set_index('region').threshold.round(3).to_dict(),
       'cal': cal, 'hist': {'edges': hb.tolist(), 'pos': hpos, 'neg': hneg}}
json.dump(out, open(OUT, 'w'))
f = lambda c: 2 * c['tp'] / (2 * c['tp'] + c['fp'] + c['fn'])
print('AP', round(ap, 3), 'AUC', round(auc, 4), 'best pooled F1', round(f1.max(), 3), 'at', thr[np.argmax(f1)])
print('model', cm, 'F1', round(f(cm), 3), '| rule', rc, 'F1', round(f(rc), 3))
print('cal', [(c['lo'], round(c['pred'], 3), round(c['obs'], 3), c['n']) for c in cal])
