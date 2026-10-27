"""Shared class-balance section for the four experiment reports."""
from html import escape
from pathlib import Path
import re

import pandas as pd

START = '<!-- CLASS_BALANCE_START -->'
END = '<!-- CLASS_BALANCE_END -->'


def before_after_tables(experiment, balance):
    sample_path = experiment / 'results/training_sample.csv'
    if not sample_path.exists():
        for fallback in ('03_exp_learned_baselines_bc', '03a_exp_learned_cems_bc'):
            candidate = experiment.parent / fallback / 'results/training_sample.csv'
            if candidate.exists():
                sample_path = candidate
                break
    sample = pd.read_csv(sample_path)
    if experiment.name.startswith('01_'):
        sample = sample.loc[sample.region.eq('Kamloops')]
    totals = sample.sum(numeric_only=True)
    assert int(totals.rows) == int(balance.loc[balance.year.lt(2022), 'rows'].sum())
    mode = ('actual' if experiment.name.startswith(('03_', '03a_')) else
            'expected' if experiment.name.startswith(('04_', '04a_')) else 'unchanged')
    sets = [('cell_only', 'Cell-only history')]
    if mode != 'expected':
        sets.append(('cell_plus_neighbours', 'Cell + neighbour history'))
    sections, records = [], []
    for key, title in sets:
        quiet_positive = int(totals[f'{key}_quiet_positive_rows'])
        active_positive = int(totals.positives) - quiet_positive
        active_negative = int(totals[f'{key}_active_rows']) - active_positive
        quiet_negative = int(totals.rows) - int(totals[f'{key}_active_rows']) - quiet_positive
        before = [active_positive, active_negative, quiet_positive, quiet_negative]
        retained = (int(totals[f'{key}_sampled_quiet_negatives']) if mode == 'actual'
                    else quiet_negative * .01 if mode == 'expected' else quiet_negative)
        after = before[:3] + [retained]
        assert sum(before) == int(totals.rows)
        if mode == 'actual':
            assert sum(after) == int(totals[f'{key}_kept_rows'])
        labels = ['Active positive', 'Active negative', 'Quiet positive', 'Quiet negative']
        rows = []
        for label, original, kept in zip(labels + ['Total'], before + [sum(before)], after + [sum(after)]):
            kept_text = f'≈ {kept:,.0f}' if mode == 'expected' and label in ('Quiet negative', 'Total') else f'{kept:,.0f}'
            rows.append(f'<tr><th scope="row">{label}</th><td>{original:,}</td><td>{kept_text}</td></tr>')
            records.append({'history_definition': key, 'category': label, 'before': original,
                            'after': kept, 'after_type': mode, 'years': '2018-2021'})
        heading = 'Expected loss rows per epoch' if mode == 'expected' else 'After sampling' if mode == 'actual' else 'Used by rules (no sampling)'
        sections.append(f'<h3>{title}</h3><div style="overflow-x:auto"><table class="balance-table">'
                        f'<thead><tr><th>Row type</th><th>Before</th><th>{heading}</th></tr></thead>'
                        '<tbody>' + ''.join(rows) + '</tbody></table></div>'
                        f'<p class="muted">Positive share: {int(totals.positives)/sum(before):.3%} before; '
                        f'{int(totals.positives)/sum(after):.3%} after'
                        + (' (based on expected retained count).' if mode == 'expected' else '.') + '</p>')
    pd.DataFrame(records).to_csv(experiment / 'results/class_balance_before_after.csv', index=False)
    note = ('Actual retained training counts. Only quiet negatives are reduced.' if mode == 'actual' else
            'Expected counts over one complete training epoch, not measured draws. GNN and MLP retain all graph nodes; only loss participation is sampled. Each step draws fresh quiet negatives with probability 1%; exact per-epoch counts were not saved.' if mode == 'expected' else
            'Descriptive counts for the 2018–2021 period. These fixed-rule experiments do not fit a model or sample rows, so before and after are identical.')
    return ('<h3>Active/quiet × positive/negative: before and after</h3>'
            '<p>2018–2021; May 8–September 30 target dates. These are cell–date examples, not distinct cells.</p>'
            f'<p>{note}</p>' + ''.join(sections)
            + '<p class="muted">For cell + neighbour history, active means a detection in the cell or its immediate neighbours during the preceding seven days. The target always concerns the cell itself. Validation and evaluation are not sampled.</p>')


def add_class_balance(page, experiment):
    experiment = Path(experiment)
    root = experiment.parent
    if experiment.name.startswith('01_'):
        source = pd.read_csv(experiment / 'results/feature_checks.csv')
        balance = source[['year', 'rows', 'positives']].copy()
        balance['region'] = 'Kamloops'
    else:
        source = pd.read_csv(root / '02_hotspot_persistence_bc/results/all_region_checks.csv')
        balance = source[['region', 'year', 'feature_rows', 'positive_targets']].rename(
            columns={'feature_rows': 'rows', 'positive_targets': 'positives'})
    balance['negatives'] = balance.rows - balance.positives
    balance['positive_percent'] = 100 * balance.positives / balance.rows
    balance['negative_percent'] = 100 * balance.negatives / balance.rows
    balance['split'] = balance.year.map(lambda year: '2018–2021 training period' if year < 2022
                                       else '2022 selection' if year == 2022 else '2023 development evaluation')
    assert balance.year.between(2018, 2023).all()
    assert (balance.positives + balance.negatives).eq(balance.rows).all()
    if experiment.name.startswith(('03_', '03a_', '04_', '04a_')):
        measured = pd.read_csv(experiment / 'results/metrics_by_year.csv')
        for row in measured.itertuples():
            expected = balance.loc[balance.region.eq(row.region) & balance.year.eq(row.year)].iloc[0]
            assert expected.rows == row.rows and expected.positives == row.positive_cell_days
    balance.to_csv(experiment / 'results/class_balance.csv', index=False)

    def table(frame, label):
        body = []
        for key, row in frame.iterrows():
            positive, negative, total = int(row.positives), int(row.negatives), int(row.rows)
            body.append(f'<tr><th scope="row">{escape(str(key))}</th><td>{total:,}</td>'
                        f'<td>{positive:,} ({positive/total:.3%})</td>'
                        f'<td>{negative:,} ({negative/total:.3%})</td></tr>')
        return ('<div style="overflow-x:auto"><table class="balance-table"><thead><tr>'
                f'<th>{label}</th><th>Cell-days</th><th>Positive: detection recorded</th>'
                '<th>Negative proxy: no detection recorded</th></tr></thead><tbody>'
                + ''.join(body) + '</tbody></table></div>')

    split = balance.groupby('split', sort=True)[['rows', 'positives', 'negatives']].sum()
    evaluation = balance.loc[balance.year.eq(2023)].set_index('region')
    training_note = '<p class="muted">The persistence experiment fits no model and does not sample negatives; the training-period row is descriptive.</p>'
    if experiment.name.startswith(('03_', '03a_')):
        sample = pd.read_csv(experiment / 'results/training_sample.csv').sum(numeric_only=True)
        assert int(sample.positives) == int(balance.loc[balance.year.lt(2022), 'positives'].sum())
        sampled = pd.DataFrame([
            {'configuration': label, 'rows': int(sample[f'{key}_kept_rows']),
             'positives': int(sample.positives),
             'negatives': int(sample[f'{key}_kept_rows']-sample.positives)}
            for key, label in [('cell_only', 'Cell only'), ('cell_plus_neighbours', 'Cell + neighbours')]
        ]).set_index('configuration')
        sampled.to_csv(experiment / 'results/training_class_balance.csv')
        training_note = ('<h3>Actual sampled training data</h3>' + table(sampled, 'Feature set')
                         + '<p class="muted">All positives and recent-activity rows are retained; other quiet negatives are sampled at 1%. Selection and evaluation retain all rows.</p>')
    elif experiment.name.startswith(('04_', '04a_')):
        training_note = ('<h3>GNN / MLP training loss sampling</h3><ul>'
                         '<li>Each snapshot contains every node. The loss keeps all positives and active nodes, plus a fresh 1% draw of quiet negatives at every training step.</li>'
                         '<li>The sampled loss class balance varies by step and epoch; exact sampled counts were not saved, so the table above reports the full underlying data.</li>'
                         '<li>2022 selection and 2023 evaluation include all cell-days.</li></ul>')
    definitions = '''<h3>Quiet is not the negative class</h3>
<p><strong>Quiet/active describes the previous seven days; positive/negative describes the target day.</strong> To predict August 15, quiet means no recorded detections during August 8–14 in the relevant area.</p>
<div style="overflow-x:auto"><table class="balance-table"><thead><tr><th>Example</th><th>Previous seven days</th><th>Target day</th><th>Training retention in 03–04</th></tr></thead><tbody>
<tr><th>Quiet positive</th><td>No detections</td><td>Detection recorded</td><td>Keep all</td></tr>
<tr><th>Quiet negative</th><td>No detections</td><td>No detection recorded</td><td>Randomly retain about 1%</td></tr>
<tr><th>Active positive</th><td>Some detections</td><td>Detection recorded</td><td>Keep all</td></tr>
<tr><th>Active negative</th><td>Some detections</td><td>No detection recorded</td><td>Keep all</td></tr>
</tbody></table></div>
<ul><li><strong>03 cell-only and 04 GNN/MLP:</strong> quiet is based on the cell itself.</li>
<li><strong>03 neighbour-feature models:</strong> quiet requires zero detections in both the cell and its immediate in-region neighbours.</li>
<li>Experiments 01–02 keep all rows; the sampling definitions above explain the later learned-model experiments.</li></ul>'''
    technique = '''<h3>Class-imbalance technique: step by step</h3>
<p><strong>Selective random undersampling, followed by a sampling correction to scores.</strong> Applied in Experiments 03–04; not in persistence Experiments 01–02.</p>
<ol><li><strong>Select training rows:</strong> keep every positive and every active row. Give each quiet negative a 1% chance of being retained. This is not sampling 1% of all negatives or making a 50/50 dataset.</li>
<li><strong>Train without class weights:</strong> Experiment 03 uses the saved sampled rows. Experiment 04 keeps the full graph, but uses a fresh quiet-negative sample in the training loss each step.</li>
<li><strong>Correct quiet-row scores:</strong> sampling makes positives appear more common among quiet examples. At prediction time, multiply predicted odds by 0.01 for every quiet row; active-row scores are unchanged. This uses past-history quiet status, never the unknown target label.</li>
<li><strong>Evaluate on all rows:</strong> apply the correction before selecting thresholds on 2022 and evaluating on 2023. Evaluation data is not undersampled.</li></ol>
<h3>Why correct the scores? An illustrative example</h3>
<ul><li>Suppose the original quiet training group has <strong>100 positives and 10,000 negatives</strong>: its positive rate is <strong>100 / 10,100 ≈ 0.99%</strong>.</li>
<li>Keeping all 100 positives and approximately 1% of negatives leaves <strong>100 positives and about 100 negatives</strong>: roughly <strong>50%</strong> positive in the sample.</li>
<li>The population did not change; sampling changed what the model saw. The correction accounts for this difference.</li></ul>
<p>For an illustrative quiet-row prediction of <strong>50%</strong>:</p>
<pre style="overflow-x:auto;padding:16px;background:#e8efeb;border-radius:8px">Original odds      = 0.50 / (1 - 0.50) = 1
Corrected odds     = 1 × 0.01 = 0.01
Corrected probability = 0.01 / (1 + 0.01) ≈ 0.99%

Code equivalent: corrected log-odds = model log-odds + log(0.01)</pre>
<p class="muted">0.01 comes from the 1% negative-retention probability. It is not a buffer, clipping threshold, or training weight. This corrects the sampling-induced odds shift; it does not guarantee calibrated probabilities from a fitted model. No separate probability-calibration procedure has been fitted.</p>'''
    section = (START + '<section id="class-balance"><style>'
               '.balance-table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}'
               '.balance-table th,.balance-table td{padding:10px;border-bottom:1px solid #d9e2df;text-align:right}'
               '.balance-table th:first-child{text-align:left}'
               '</style><h2>Class balance</h2>'
               '<p>One example is one H3 cell on one target date. Counts below use <strong>May 8–September 30</strong> after the seven-day warmup.</p>'
               + table(split, 'Period') + '<h3>2023 balance by region</h3>'
               + table(evaluation, 'Fire Centre') + training_note + definitions
               + before_after_tables(experiment, balance) + technique
               + '<ul class="muted"><li>Positive means at least one recorded hotspot detection. A zero target means no recorded detection, not verified absence of fire; observation coverage remains unknown.</li>'
               '<li>BC totals count region–cell–days. Border cells may appear in multiple regional grids; these are not deduplicated province-wide cell-days.</li>'
               '<li>Heavy imbalance makes accuracy alone misleading; interpret precision, recall, F1 and average precision alongside the positive prevalence.</li></ul></section>' + END)
    if START in page:
        return re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: section, page, flags=re.S)
    assert '</header>' in page
    return page.replace('</header>', '</header>\n' + section, 1)
