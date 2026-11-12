"""Describe the matched lightning ablation from saved metrics only."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def main():
    complete=json.loads((OUT/'completion.json').read_text())
    assert complete['status']=='complete' and complete['source_hashes_unchanged']
    metrics=pd.concat([pd.read_csv(OUT/f'summary_{y}.csv') for y in (2023,2024)])
    boot=pd.concat([pd.read_csv(OUT/f'bootstrap_{y}.csv') for y in (2023,2024)])
    table=['| Year | Model | F1 | Precision | Recall | Median centre AP |',
           '|---|---|---:|---:|---:|---:|']
    changes=['| Year | F1 gain from lightning | Relative F1 change | Paired 95% interval (F1 points) | AP change |',
             '|---|---:|---:|---|---:|']
    for year in (2023,2024):
        subset=metrics.loc[metrics.year.eq(year)].set_index('model')
        for name,row in subset.iterrows():
            table.append(f'| {year} | {name.replace("_"," ")} | '+
                         ' | '.join(f'{row[k]:.4f}' for k in ('f1','precision','recall','median_region_ap'))+' |')
        a,b=subset.loc['with_lightning'],subset.loc['without_lightning']
        ci=boot.loc[boot.year.eq(year)&boot.scope.eq('BC-wide')].iloc[0]
        changes.append(f'| {year} | {(a.f1-b.f1)*100:+.3f} points | {(a.f1/b.f1-1)*100:+.2f}% | '
                       f'[{ci.ci_low*100:+.3f}, {ci.ci_high*100:+.3f}] | {a.median_region_ap-b.median_region_ap:+.4f} |')
    region=pd.read_csv(OUT/'metrics_2024.csv')
    regional=['| Fire Centre | F1 without lightning | F1 with lightning | Difference |',
              '|---|---:|---:|---:|']
    for centre,group in region.groupby('region'):
        group=group.set_index('model');a,b=group.loc['with_lightning'],group.loc['without_lightning']
        regional.append(f'| {centre} | {b.f1:.4f} | {a.f1:.4f} | {a.f1-b.f1:+.4f} |')
    rows=metrics.loc[metrics.year.eq(2024)].set_index('model')
    a,b=rows.loc['with_lightning'],rows.loc['without_lightning']
    ci=boot.loc[boot.year.eq(2024)&boot.scope.eq('BC-wide')].iloc[0]
    if ci.ci_low>0:
        finding='Lightning improves 2024 F1 in this matched run; the paired within-year interval is above zero.'
    elif ci.ci_high<0:
        finding='Lightning reduces 2024 F1 in this matched run; the paired within-year interval is below zero.'
    else:
        finding='The 2024 F1 difference is not clearly distinguishable from zero in this matched run.'
    fit=json.loads((OUT/'fit.json').read_text())
    text=f'''# Experiment 26: how much does lightning contribute?

**{finding}**

Adding lightning changes 2024 pooled F1 from **{b.f1:.4f} to {a.f1:.4f}**:
**{(a.f1-b.f1)*100:+.3f} F1 percentage points**, or {(a.f1/b.f1-1)*100:+.2f}% relative.
The paired 95% interval is [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}] in F1 units.

## Matched design

Both models use the same S-NPP-only, one-day-gap-filled activity label, existing
H3-resolution-7 grids, and full-grid evaluation rows. Training data, row order,
non-lightning scaling, architecture, random initialization (seed 42), optimizer,
batch order and early-stopping policy are held fixed. The ablation is retrained
from scratch with all six lightning features and their missing flags set to zero
after scaling throughout training and inference. It is not inference-only masking.

The original with-lightning checkpoint and predictions from Experiment 25 are
unchanged. The retained scaler statistics were independently checked against
the identical training sample. Perturbing all lightning fields leaves the
ablated network input exactly unchanged.

Training: 2018-2021 ({fit['fit_rows']:,} fitting and {fit['validation_rows']:,}
held-out training-season rows). The no-lightning model selects epoch
{fit['best_epoch']}; the original with-lightning model selected epoch 6.
Each model selects its own thresholds on 2022 using the same per-centre F1
criterion. No 2023/2024 threshold tuning is performed.

## Full-grid comparison

{chr(10).join(table)}

{chr(10).join(changes)}

F1 points above mean 100 times the difference in F1, not classification accuracy.
AP is the median per-centre average precision, not pooled AP. AP does not depend
on the selected classification threshold.

At the frozen operating points, adding lightning changes 2024 true positives by
**{int(a.tp-b.tp):+,}**, false positives by **{int(a.fp-b.fp):+,}**, and missed positive
cell-days by **{int(a.fn-b.fn):+,}**. These are cell-days, not distinct fires.

## Regional F1, 2024

{chr(10).join(regional)}

## Interpretation limits

This ablation was requested after the original 2024 results had been inspected.
It is a follow-up analysis on an already-used test year, not a fresh independent
holdout. No settings were tuned from the new 2023/2024 results.

Intervals pair both models on the same 2,000 whole-ISO-week resamples. They measure
within-year variability, not uncertainty from training seeds or future seasons.
One matched seed is insufficient to establish a universal contribution. The
measured difference includes each model's training-based epoch choice and
2022-selected operating point. These results do not estimate a causal effect of
lightning on wildfire ignition.

This comparison addresses lightning's contribution to this MLP. Whether the
complete model beats persistence is a separate comparison; see
[Experiment 25](../25_exp_snpp_lightning_2024/RESULTS.md).

## Artifacts

`results/metrics_2023.csv`, `metrics_2024.csv`, `summary_*.csv`, and
`bootstrap_*.csv` contain the comparisons. `frozen_before_followup_evaluation.json`
records hashes and thresholds; `completion.json` confirms source artifacts were
unchanged and the original Experiment 25 metrics were reproduced. Ablated score
arrays preserve the row order of their corresponding Experiment 25 full-grid
Parquet files, whose hashes are frozen.
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    print(text[:500])


if __name__=='__main__':main()
