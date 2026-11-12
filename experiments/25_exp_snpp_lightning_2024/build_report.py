"""Summarise saved full-grid metrics without fitting or changing any model."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def read(name):
    with (RESULTS / name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    frozen = json.loads((RESULTS / 'frozen_before_2024.json').read_text())
    completion = json.loads((RESULTS / 'test_completion.json').read_text())
    fit = json.loads((RESULTS / 'fit.json').read_text())
    table = ['| Year | Model | F1 | Precision | Recall | Median centre AP |',
             '|---|---|---:|---:|---:|---:|']
    for year in (2023, 2024):
        for row in read(f'summary_{year}.csv'):
            numbers = ' | '.join(f'{float(row[k]):.4f}' for k in ('f1','precision','recall','median_region_ap'))
            table.append(f'| {year} | {row["model"]} | {numbers} |')
    regional = ['| Fire Centre | MLP F1 | Persistence F1 | MLP precision | MLP recall | MLP AP |',
                '|---|---:|---:|---:|---:|---:|']
    rows = read('metrics_2024.csv')
    for region in sorted({r['region'] for r in rows}):
        mlp = next(r for r in rows if r['region'] == region and r['model'] == 'mlp_snpp_lightning')
        rule = next(r for r in rows if r['region'] == region and r['model'] == 'snpp_persistence')
        values = [mlp['f1'],rule['f1'],mlp['precision'],mlp['recall'],mlp['average_precision']]
        regional.append('| '+region+' | '+' | '.join(f'{float(v):.4f}' for v in values)+' |')
    boot = json.loads((RESULTS/'bootstrap_2024.json').read_text())
    sample = read('training_sample_counts.csv')
    n = sum(int(r['rows']) for r in rows if r['model']=='mlp_snpp_lightning')
    positive = sum(int(r['tp'])+int(r['fn']) for r in rows if r['model']=='mlp_snpp_lightning')
    text = f'''# Experiment 25: S-NPP + lightning results

The 2024 evaluation is complete. This is a separately trained S-NPP/lightning
MLP, not the earlier two-satellite model. No 2024 threshold tuning was performed.

## Data and method

- Inputs: S-NPP FIRMS history and official CanCPLD cloud-to-ground lightning,
  downloaded for 2018-2024. No NOAA-20, weather or incident inputs.
- 21 features, plus missing flags; model architecture 256/256/128, seed 42.
- Target: next-day S-NPP detection, with one-day label gaps filled. This measures
  hotspot activity, not verified ignitions or confirmed fire absence.
- All six existing BC H3-resolution-7 grids, May 8-September 30.
- Training 2018-2021: {sum(int(r['kept']) for r in sample):,} sampled rows;
  {fit['fit_rows']:,} fitting rows, {fit['validation_rows']:,} held-out training-season
  rows for early stopping. Best epoch: {fit['best_epoch']}.
- Thresholds selected once on full-grid 2022; 2023 used as a development check.
- Final 2024 test: {n:,} cell-days, {positive:,} positive labels.
- Frozen UTC: {frozen['utc_frozen']}; completed UTC: {completion['completed_utc']}.

## Full-grid, pooled exact-cell results

{chr(10).join(table)}

F1 is not accuracy. AP is the median of six per-centre average precision values,
not pooled provincial AP. All evaluation negatives are retained.

2024 MLP minus persistence F1: **{boot['f1_difference']:+.4f}**;
paired whole-week bootstrap 95% interval **[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]**
from 2,000 resamples. This interval reflects within-year variation, not uncertainty
across future fire seasons or training seeds.

## 2024 by Fire Centre

{chr(10).join(regional)}

## Interpretation and limits

The MLP has higher precision but lower recall than persistence on 2024. Its pooled
F1 is slightly lower, and the paired confidence interval includes zero. This run
does not establish an overall improvement over the simple persistence rule.

Compare with the S-NPP persistence baseline in this table. The earlier 0.626 MLP
result used a different label, feature set and evaluation year, so it is not a
like-for-like comparator. No S-NPP-only model ablation was run here, so these
results do not isolate the added value of lightning. One seed and one fixed
architecture were evaluated. Missing satellite detections remain uncertain
negatives; lightning coverage gaps remain missing features.

No hyperparameters or thresholds were adjusted after seeing 2024. The 2024
holdout has now been used for this experiment and should not be reused as an
untouched test during further model selection.

## Files

- `results/metrics_2024.csv`: per-centre metrics and frozen thresholds.
- `results/summary_2024.csv`: pooled performance.
- `results/bootstrap_2024.json`: paired uncertainty interval.
- `results/frozen_before_2024.json`: model/scaler/source hashes and protocol.
- `results/test_completion.json`: completed test record.
- `data/snpp_lightning_h3r7/evaluation/2024/`: full-grid joined features,
  target, MLP score and persistence prediction, partitioned by Fire Centre.

Lightning source: [official CanCPLD directory](https://crd-data-donnees-rdc.ec.gc.ca/CDAS/products/CanCPLD/CLDN/cg/).
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    print('Wrote', HERE/'RESULTS.md')


if __name__=='__main__':
    main()
