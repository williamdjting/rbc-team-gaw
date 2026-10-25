"""Concise standalone overview of Experiments 01-04 using saved metrics."""
import base64
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    comparison = pd.read_csv(HERE / '04_exp_gnn_bc/results/comparison_2023.csv')
    comparison = comparison.loc[comparison.model.ne('best_rule_hindsight')]
    columns = ['precision', 'recall', 'f1', 'average_precision']
    means = comparison.groupby(['model', 'feature_set'])[columns].mean()
    pilot = pd.read_csv(HERE / '01_hotspot_persistence_kamloops/results/metrics_by_year.csv')
    pilot = pilot.loc[pilot.year.eq(2023) & pilot.rule.eq('local_1d')].iloc[0]
    configurations = [
        ('01 · Persistence (single-region pilot)', None),
        ('02 · Persistence', ('selected_rule', 'local_1d')),
        ('03 · Logistic regression', ('logistic_regression', 'cell_only')),
        ('03 · Logistic + neighbours', ('logistic_regression', 'cell_plus_neighbours')),
        ('03 · Gradient boosting', ('gradient_boosting', 'cell_only')),
        ('03 · Boosting + neighbours', ('gradient_boosting', 'cell_plus_neighbours')),
        ('04 · MLP control', ('mlp', 'cell_only_inputs')),
        ('04 · GNN', ('gnn', 'cell_only_inputs')),
    ]
    records = []
    for name, key in configurations:
        values = pilot if key is None else means.loc[key]
        if key is not None:
            selected = comparison.loc[comparison.model.eq(key[0]) & comparison.feature_set.eq(key[1])]
            assert len(selected) == 6 and selected.region.nunique() == 6
        records.append({'configuration': name, **{metric: float(values[metric]) for metric in columns}})
    aggregate = pd.DataFrame(records).set_index('configuration')
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True, constrained_layout=True)
    y = np.arange(len(aggregate))
    colours = ['#a2a9aa', '#ca641f', '#789da0', '#789da0', '#789da0', '#789da0', '#536a82', '#237e85']
    for ax, metric, title in zip(axes, columns, ['Precision', 'Recall', 'F1', 'Average precision']):
        ax.barh(y, aggregate[metric], color=colours, height=.58)
        for index, value in enumerate(aggregate[metric]):
            ax.text(value+.012, index, f'{value:.1%}' if metric in ('precision', 'recall') else f'{value:.3f}',
                    va='center', fontsize=9)
        ax.axhline(.5, color='#9ca9aa', linestyle='--', linewidth=.8)
        ax.set(title=title, xlim=(0, .74), xticks=[0, .2, .4, .6])
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle=':', alpha=.35)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(yticks=y, yticklabels=aggregate.index)
    axes[0].invert_yaxis()
    fig.suptitle('2023 development evaluation · higher is better\nPilot shown separately; remaining rows are equal-region means', fontsize=13)
    figures = HERE / 'figures'
    figures.mkdir(exist_ok=True)
    chart = figures / 'experiments_overview_metrics.png'
    fig.savefig(chart, dpi=160)
    plt.close(fig)
    encoded = base64.b64encode(chart.read_bytes()).decode('ascii')
    balance = pd.read_csv(HERE / '02_hotspot_persistence_bc/results/class_balance.csv')
    balance['period'] = balance.year.map(lambda year: 'Training · 2018–2021' if year < 2022 else
                                       'Validation · 2022' if year == 2022 else 'Development evaluation · 2023')
    split = balance.groupby('period', sort=False)[['rows', 'positives', 'negatives']].sum()
    balance_rows = ''.join(f'<tr><th>{name}</th><td>{int(row.rows):,}</td><td>{int(row.positives):,} ({row.positives/row.rows:.3%})</td><td>{int(row.negatives):,} ({row.negatives/row.rows:.3%})</td></tr>'
                           for name, row in split.iterrows())
    metric_rows = ''.join(f'<tr><th>{name}</th><td>{row.precision:.1%}</td><td>{row.recall:.1%}</td><td>{row.f1:.3f}</td><td>{row.average_precision:.3f}</td></tr>'
                         for name, row in aggregate.iterrows())
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wildfire forecasting · Experiments 01–04</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f5;color:#20343a;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 24px}}
header{{border-bottom:2px solid #20343a;padding-bottom:20px}}h1{{font-size:36px;line-height:1.15;margin:10px 0 16px}}h2{{font-size:23px;margin:0 0 12px}}
.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#b4521d;font-weight:750}}section{{padding:24px 0;border-bottom:1px solid #dce4e0}}
p{{margin:8px 0}}li{{margin:5px 0}}ul{{padding-left:22px}}.muted,figcaption{{font-size:13px;color:#566b70}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce4e0;text-align:right}}th:first-child{{text-align:left}}thead{{background:#e9efeb}}
.methods td{{text-align:left}}.methods th{{width:175px}}figure{{margin:14px 0;padding:12px;background:white;border-radius:12px;border:1px solid #dce4e0}}img{{width:100%;height:auto}}.conclusion{{background:#e9efeb;border-radius:12px;padding:20px}}
footer{{padding:18px 0;font-size:13px;color:#566b70}}@media(max-width:700px){{main{{padding:22px 14px}}h1{{font-size:28px}}}}
@media print{{body{{background:white}}main{{padding:0}}figure{{break-inside:avoid}}}}
</style></head><body><main><header><div class="eyebrow">Experiments 01–04 · Consolidated overview</div>
<h1>Can learned models improve on hotspot persistence?</h1>
<p><strong>Objective:</strong> predict whether a hotspot detection will be recorded in an H3 cell on the next day, and test the added value of neighbour features and graph message passing.</p></header>

<section><h2>Dataset and class balance</h2>
<ul><li><strong>Data:</strong> FIRMS VIIRS S-NPP observations, fixed resolution-7 H3 grids; May–September 2018–2023. Target dates start May 8 to allow seven days of history.</li>
<li><strong>Scope:</strong> Experiment 01 is a single-region pilot; Experiments 02–04 use six BC Fire Centres and 234,554 region-specific grid nodes.</li>
<li><strong>Target:</strong> 1 = at least one detection recorded on the target day; 0 = no detection recorded. Coverage remains unknown; these are not ignition or verified fire-absence labels.</li></ul>
<div class="scroll"><table><thead><tr><th>Full dataset period · Experiments 02–04</th><th>Cell-days</th><th>Positive</th><th>Negative proxy</th></tr></thead><tbody>{balance_rows}</tbody></table></div>
<ul><li><strong>01–02:</strong> all rows retained; no class weights or sampling.</li>
<li><strong>03–04:</strong> retain all positives and active rows, sample 1% of quiet negatives for training, then correct quiet-row positive odds by ×0.01. GNN/MLP sample loss contributions each step while retaining the full graph.</li>
<li><strong>Quiet:</strong> no detections in the previous seven days in the cell; for 03 neighbour-feature models, also none in its neighbours. Quiet is not the target class.</li>
<li>03 sampled training positives: <strong>3.081%</strong> for cell-only models and <strong>2.653%</strong> with neighbours. Validation/evaluation retain all rows. No separately fitted probability calibration.</li></ul>
<p class="muted">Counts sum regional cell-days; border cells may occur in multiple grids. 2024 remains excluded for final testing.</p></section>

<section><h2>Experiments and methodology</h2>
<div class="scroll"><table class="methods"><thead><tr><th>Experiment</th><th>Method</th><th>One-line purpose</th></tr></thead><tbody>
<tr><th>01 · Persistence pilot</th><td>Fixed count-based rules using 1-, 3- and 7-day local/neighbour histories in one region.</td><td>Quantify whether recent observed activity recurs the next day.</td></tr>
<tr><th>02 · BC persistence</th><td>Repeat the rules independently across all six regions.</td><td>Establish a wider regional benchmark.</td></tr>
<tr><th>03 · Learned baselines</th><td>Pooled logistic regression and gradient boosting, each with cell-only and neighbour-enhanced inputs.</td><td>Test whether learning feature combinations improves the fixed-rule benchmark.</td></tr>
<tr><th>04 · GNN + MLP</th><td>Two GraphSAGE-style mean-aggregation layers; MLP comparison without message passing.</td><td>Test the contribution of learned graph-based neighbour information.</td></tr>
</tbody></table></div>
<p><strong>Split:</strong> train 2018–2021 → select on 2022 → evaluate on 2023. Persistence selects rules; learned models select per-region thresholds; neural models also select the epoch on 2022.</p>
<p class="muted">Features use only earlier dates. No weather inputs yet. 2023 is development evaluation, not an untouched final test.</p></section>

<section><h2>Aggregate results</h2>
<p>2023 results. Experiments 02–04 use equal-weight averages across the six regions; the pilot is listed separately and is not directly comparable in scope.</p>
<figure><img src="data:image/png;base64,{encoded}" alt="Aggregate precision, recall, F1 and average precision for the eight configurations in Experiments 01 through 04"><figcaption>Dashed line separates the single-region pilot from the six-region comparisons. Average F1 is the mean of regional F1 values.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>Configuration</th><th>Precision</th><th>Recall</th><th>F1</th><th>Average precision</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<p class="muted">Precision/recall/F1 use the 2022-selected rules or thresholds. Average precision evaluates ranking across thresholds; persistence uses historical counts as its ranking score.</p></section>

<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>Persistence retains the best six-region F1: 0.543.</strong> Learned models have not beaten it at their selected thresholds.</li>
<li><strong>Neighbour features improve both tabular models’ F1 and average precision.</strong></li>
<li><strong>The GNN has the best average precision: 0.501</strong>, versus 0.457 for the MLP and 0.378 for persistence. Its F1 of 0.530 still trails persistence.</li>
<li>Next: validate robustness and threshold/calibration behaviour before the final 2024 evaluation.</li></ul></div></section>
<footer>Source: saved experiment metrics; no models rerun. This standalone HTML embeds its chart and works offline.</footer>
</main></body></html>'''
    output = HERE / 'EXPERIMENTS_OVERVIEW.html'
    output.write_text(page, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
