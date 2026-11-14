"""Build the Week 4 presentation in the Week 3 deck's visual style."""
import csv
import html
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def read(pattern):
    path = next(HERE.glob(pattern))
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def card(title, content, style=''):
    return f'<div class="card {style}"><h3>{title}</h3><p>{content}</p></div>'


def metric(value, label):
    return f'<div class="stat"><span class="num">{value}</span><span class="lab">{label}</span></div>'


def main():
    old = (HERE/'Week3_GAW_detailed.html').read_text(encoding='utf-8')
    css = re.search(r'<style>(.*?)</style>', old, re.S)[1]
    dual = read('22_*/results/summary_2023.csv')
    drivers = read('21_*/results/summary_2023.csv')
    full = {str(y): read(f'25_*/results/summary_{y}.csv') for y in (2023, 2024)}
    ablation = {str(y): read(f'26_*/results/summary_{y}.csv') for y in (2023, 2024)}
    regional = read('25_*/results/metrics_2024.csv')
    ci = {str(y): next(r for r in read(f'26_*/results/bootstrap_{y}.csv') if r['scope']=='BC-wide') for y in (2023, 2024)}
    def dvalue(rows, model, definition='res7_exact'):
        return float(next(r for r in rows if r['model']==model and r['definition']==definition and r['method']=='f1_2022')['bc_wide_f1'])
    def format_rows(rows):
        return [[r['model'].replace('mlp_snpp_lightning','MLP + lightning').replace('snpp_persistence','S-NPP persistence'),
                 *[f'{float(r[k]):.4f}' for k in ('f1','precision','recall','median_region_ap')]] for r in rows]
    slides=[]
    def slide(section, kicker, title, content, note=''):
        slides.append(f'<section class="slide" id="slide-{len(slides)+1}" data-section="{html.escape(section)}" aria-labelledby="heading-{len(slides)+1}"><div class="inner"><span class="kicker">{kicker}</span><h2 id="heading-{len(slides)+1}">{title}</h2>{content}'+(f'<p class="note">{note}</p>' if note else '')+'</div></section>')

    slide('Week 4','RBC team GAW · Week 4 · updated November 12, 2026',
          'From richer inputs to a 2024 test',
          '<p class="lede">Continuing the Week 3 experiments: weather and neural models, a separate S-NPP test, and a matched check of what lightning adds.</p>'
          '<div class="grid3">'+metric('21 → 26','six experiment records covered')+metric('34.24M','cell-days in the 2024 test')+metric('+0.385','F1 points from lightning in 2024; interval includes zero')+'</div>'
          +card('Main finding','More inputs improve some rankings and reduce false alarms, but the tested MLP still does not clearly beat persistence on exact-cell F1.','accent')
          +'<p class="note">← → to navigate · select a slide below · print for a page-by-page version.</p>')

    slide('Week 3 hand-off','Where we left off · local Experiments 19–20',
          'Week 3 ended with a stronger label and a stubborn rule',
          '<div class="grid2">'+card('What we had learned','Combining S-NPP and NOAA-20 made hotspot labels more consistent. The two-satellite persistence rule reached <strong>0.624 exact-cell F1</strong> and <strong>0.810 with a one-cell margin</strong> on 2023.')
          +card('What Week 3 proposed next','Add weather and lightning; test whether richer inputs or a different learner can beat the rule; then evaluate a fixed model on 2024.','accent')+'</div>'
          +table(['Week 4 record','Question'],[['21 · drivers','Do weather, lightning and wind-related spread help?'],['22 · neural models','Does an MLP or GRU use those inputs better?'],['23–24 · status','Is regularisation complete, and can we run the frozen two-satellite test?'],['25–26 · S-NPP track','What happens in 2024, and what does lightning contribute?']]),
          'Week 3 mixed contributor numbering with local folder numbering. This deck consistently uses local folders: contributor 16/17/18 → local 21/22/23. Earlier multisensor experiments are local 19/20.')

    slide('Task and label','The forecast · same grid, two distinct experiment tracks',
          'Tomorrow’s hotspot activity, one H3 cell at a time',
          '<div class="grid3">'+card('Input','Past satellite observations and available environmental history, ending on UTC day <strong>t−1</strong>.')+card('Output','A score and thresholded alert for each of <strong>234,554 H3 resolution-7 cell entries</strong> across six Fire Centres.')+card('Target','A detection on day t, or a one-day missing observation bracketed by detections. This is hotspot activity, not verified ignition.')+'</div>'
          +'<div class="formula">y(c,t) = D(c,t) OR [D(c,t−1) AND D(c,t+1)]</div>'
          +table(['Track','What D means','Evaluation'],[['21–22','S-NPP OR NOAA-20 detection','2023; exact-cell and spatial-margin scores'],['25–26','S-NPP detection only','2023 and 2024; exact-cell scores only']]),
          'Gap filling is for retrospective labels only, never input features. A zero label is not confirmed fire absence. H3 resolution 7 is approximately 5 km² per cell; lightning is coarser, on a 0.1° grid.')

    slide('Data','Which sources actually enter each model',
          'The S-NPP test is a separate model configuration',
          table(['Source / feature group','Experiments 21–22','Experiments 25–26'],[
              ['FIRMS S-NPP VIIRS','Hotspot counts, FRP, nearby history','15 S-NPP/history columns, including quiet indicator'],
              ['CWFIS NOAA-20 VIIRS','Additional label and history information','Not used'],
              ['BCWS weather and incidents','Station observations and incident context','Not used'],
              ['CanCPLD cloud-to-ground lightning','Recent flash counts and nearby history','6 lightning columns; removed in the ablation'],
              ['Wind-related spread / day of year','Spread inputs; final configuration removes day of year','Neither used']])
          +card('New data acquired this week','Downloaded official CanCPLD files for <strong>2018–2024</strong>: seven annual files, about <strong>357 MB</strong>. Three-hour UTC flash counts are summed into daily grids; missing coverage remains missing.','accent'),
          'The earlier shared 2.87-million-row joined file covers only 2018–2023 and samples quiet rows. Experiment 25 rebuilt its own joins and evaluates all cell-days, including quiet negatives.')

    driver_models=[('catboost_stopgap','Fire-history CatBoost'),('catboost_drivers','+ weather + lightning'),('catboost_spread','+ upwind spread'),('catboost_spread_noday','Same inputs, no day of year')]
    slide('21 · drivers','Experiment 21 · two-satellite label · 2023',
          'Weather helps the spatial-margin result, not exact-cell F1',
          table(['Forecast','Exact-cell F1','One-cell-margin F1'],[['Two-satellite persistence','0.624','0.810']]+[[label,f'{dvalue(drivers,m):.3f}',f'{dvalue(drivers,m,"res7_margin1"):.3f}'] for m,label in driver_models])
          +'<div class="grid2">'+card('Margin improvement','The no-calendar model reaches <strong>0.824</strong>, versus 0.810 for the rule. Difference ≈ +0.015; 95% interval <strong>[+0.005, +0.025]</strong>.','good')
          +card('Operating point matters','At the rule’s daily alert count, margin F1 is <strong>0.807</strong>. The headline gain depends on the selected threshold and number of alerts, not just ranking.')+'</div>',
          'All numbers are BC-wide pooled F1. Thresholds were selected on 2022 separately by model, centre and scoring definition. Exact-cell and margin F1 are different tasks; F1 is not accuracy.')

    slide('22 · learners','Experiment 22 · same two-satellite features, different learner',
          'The MLP closes the exact-cell gap to persistence',
          '<div class="controls"><label for="dual-mode">Scoring definition</label><select id="dual-mode"><option value="res7_exact">Exact cell</option><option value="res7_margin1">One-cell margin</option></select><span class="tag">2023 · six centres</span></div><div id="dual-chart" class="bars" aria-live="polite"></div><p class="axis-label">F1 · zero-based bars, common scale 0–1</p>'
          +'<div class="grid2">'+card('MLP','Three hidden layers: 256, 256, 128. Exact-cell F1 <strong>0.626</strong> vs CatBoost 0.607. Against the rule: <strong>+0.002</strong>, 95% interval <strong>[−0.003, +0.007]</strong>.')
          +card('MLP + GRU','A 14-day sequence adds complexity but no clear gain: exact-cell F1 <strong>0.623</strong>, margin F1 <strong>0.816</strong>. The simpler MLP remains the stronger exact-cell candidate.')+'</div>',
          'Margin F1: CatBoost 0.824, MLP 0.820, GRU 0.816, persistence 0.810. The MLP improves on CatBoost but does not clearly beat persistence on the exact cell.')

    slide('23–24 · status','Experiment status · distinguish plans from completed evidence',
          'Two items were not complete province-wide tests',
          '<div class="grid2">'+card('23 · regularised networks','The supplied HTML shows <strong>Kamloops only and two training epochs</strong>. Generic “BC-wide” captions do not make it a provincial result. The longer run log stops during evaluation; final tables are missing.','bad')
          +card('24 · frozen two-satellite MLP','We preserved the saved checkpoint, scaler and 2022 thresholds. Matching NOAA-20 and other 2024 inputs were unavailable, so <strong>this model was not tested on 2024</strong>.','accent')+'</div>'
          +card('Decision','Proceed with the sources available: a new S-NPP + lightning model. Keep the two-satellite model unchanged. This changes both the feature set and the label, so the new result cannot be compared directly with 0.626 from Experiment 22.'),
          'Experiment 23’s preliminary exact-cell F1: original Kamloops MLP 0.645, regularised MLP 0.599, regularised GRU 0.610. These do not establish the outcome of a completed six-centre run.')

    slide('25 · design','Experiment 25 · S-NPP + lightning only',
          'A clean new training run using available data',
          '<div class="flow"><div class="step"><span class="mono">S-NPP</span><b>Fire history</b><span>Cell / neighbour counts, 1/3/7-day windows, FRP, night detections and time since detection.</span></div><div class="step"><span class="mono">Lightning</span><b>Recent flashes</b><span>Containing 0.1° cell, 3×3 and 9×9 blocks, three-day counts and time since nearby flashes.</span></div><div class="step"><span class="mono">Join</span><b>H3 cell + day</b><span>Only history through t−1. Missing flags retained. Same six regional grids.</span></div><div class="step"><span class="mono">MLP</span><b>Next-day score</b><span>21 features plus missing flags; 256/256/128 hidden layers; one fixed seed.</span></div></div>'
          +'<div class="grid2">'+card('Training sample','Keep all positive and recently active rows, plus 1% of quiet negatives. Apply the matching quiet-row odds correction at prediction.')
          +card('Evaluation grid','Keep <strong>every cell-day</strong>. Do not evaluate on a sampled set of quiet negatives. Save the joined features, labels and predictions for inspection.','accent')+'</div>',
          'The persistence baseline uses only yesterday’s S-NPP detection. It has no lightning. That comparison tests whether the complete MLP beats the simple rule, not lightning’s contribution alone.')

    slide('25 · frozen protocol','Chronology before the first 2024 result',
          'Train, select, freeze, then test',
          '<div class="flow"><div class="step"><span class="mono">2018–2021</span><b>Fit the model</b><span>1,510,565 fitting rows; 251,393 held-out training-season rows for epoch selection.</span></div><div class="step"><span class="mono">2022</span><b>Select thresholds</b><span>Maximise full-grid F1 separately in each Fire Centre.</span></div><div class="step"><span class="mono">2023</span><b>Development check</b><span>Score unchanged. Do not tune from these results.</span></div><div class="step"><span class="mono">Freeze → 2024</span><b>Final test</b><span>Hash the model, scaler and sources; lock the thresholds before 2024 feature generation/scoring.</span></div></div>'
          +'<div class="grid3">'+metric('6','best training epoch')+metric('34,244,884','2024 cell-days evaluated')+metric('20,082','positive 2024 cell-day labels')+'</div>'
          +card('Verification','Independent recount of saved predictions matched the reported confusion counts. The full-grid row counts, source hashes and unchanged 2022 thresholds all passed.','good'),
          'Target dates: May 8–September 30, with history starting May 1. Since this evaluation has happened, 2024 is no longer untouched for subsequent project decisions.')

    slide('25 · 2024 result','Experiment 25 · pooled exact-cell metrics',
          'Fewer false alarms, but persistence still has higher F1',
          table(['2024 forecast','F1','Precision','Recall','Median centre AP'],format_rows(full['2024']))
          +'<div id="test-chart" class="bars"></div><p class="axis-label">F1 · zero-based bars, common scale 0–1</p>'
          +'<div class="grid2">'+card('What changes','The MLP produces <strong>1,886 fewer false positives</strong> than persistence, but also <strong>959 fewer true positives</strong>. Higher precision comes at lower recall.')
          +card('Uncertainty','MLP minus persistence: <strong>−0.0084 F1</strong>, paired 95% interval <strong>[−0.0184, +0.0023]</strong>. No demonstrated overall F1 improvement.','accent')+'</div>',
          'This test uses the S-NPP-only label. Do not interpret the difference from Experiment 22’s 0.626 as a measured decline of the same model on the same target. AP is median per-centre AP, not pooled AP.')

    region_rows=[]
    for region in ['Cariboo','Coastal','Kamloops','Northwest','Prince George','Southeast']:
        m=next(r for r in regional if r['region']==region and r['model']=='mlp_snpp_lightning')
        r=next(r for r in regional if r['region']==region and r['model']=='snpp_persistence')
        region_rows.append([region,f'{float(r["f1"]):.4f}',f'{float(m["f1"]):.4f}',f'{float(m["f1"])-float(r["f1"]):+.4f}'])
    slide('25 · regional view','2024 · exact-cell F1 by Fire Centre',
          'Regional gains do not guarantee a pooled gain',
          table(['Fire Centre','Persistence','MLP + lightning','MLP − persistence'],region_rows)
          +card('Why the provincial score matters','The MLP leads in Cariboo, Kamloops and Northwest, and trails in Coastal, Prince George and Southeast. Pooled F1 combines confusion counts; it is <strong>not the average of the six F1 values</strong>.'),
          'Prince George accounts for 15,299 of 20,082 positive 2024 cell-days in this grid. Its performance has a large effect on the pooled result. “Positive cell-days” does not mean distinct fires.')

    slide('The fair comparison','Two questions, two comparisons',
          'Does the system work better—or does lightning help?',
          '<div class="grid2">'+card('Experiment 25 · system comparison','<strong>S-NPP + lightning MLP vs S-NPP persistence.</strong><br>Fair for asking whether the complete system beats the simple baseline. Inputs do not need to match for that question.','accent')
          +card('Experiment 26 · feature ablation','<strong>The same MLP with vs without lightning.</strong><br>Needed to isolate the contribution of the lightning feature group while holding the rest of the training procedure fixed.','good')+'</div>'
          +'<p class="lede">Failure to beat persistence does not, by itself, show that lightning is useless.</p>')

    slide('26 · matched ablation','Experiment 26 · follow-up after the 2024 result',
          'Retrain without lightning; keep the comparison matched',
          table(['Held fixed','What changed'],[['Same rows, labels, feature timing and data splits','Six lightning values and their missing flags set to zero'],['Same network shape, seed and initial parameter tensors','Mask applied during training and inference—not only at test time'],['Same S-NPP scaling, optimizer, batches and stopping policy','No-lightning model retrained from scratch'],['Same 2022 threshold-selection criterion','Each model selects its own 2022 thresholds']])
          +'<div class="grid2">'+card('Checks passed','Same fitting/validation rows and retained scaler statistics. Changing lightning values leaves the ablated input unchanged. Both models selected epoch 6.')
          +card('Status of the evidence','This was requested <strong>after seeing 2024</strong>. It is a one-seed follow-up analysis, not a fresh untouched test. No settings were tuned from its 2023/2024 outcomes.','accent')+'</div>')

    slide('26 · contribution','Matched ablation · use the year selector',
          'Lightning improves ranking; the F1 effect is small',
          '<div class="controls"><label for="ablation-year">Evaluation year</label><select id="ablation-year"><option>2024</option><option>2023</option></select><span class="tag">With vs without lightning</span></div><div id="ablation-table" aria-live="polite"></div><div id="ablation-chart" class="bars"></div><p id="ablation-delta" class="lede" aria-live="polite"></p>'
          +card('Read the metrics separately','F1 measures the selected yes/no alerts. Average precision measures ranking without one selected threshold. Median regional AP rises in both years, while the F1 change reverses sign.'),
          '2023: F1 0.5833 → 0.5813; median AP 0.5790 → 0.5910. 2024: F1 0.5547 → 0.5586; median AP 0.4674 → 0.4765. These are observed differences from one matched seed.')

    slide('26 · uncertainty','How much? · paired whole-week bootstrap',
          'Both F1 intervals include zero',
          '<div id="interval-chart" class="dotplot"></div>'
          +table(['Year','F1 change from lightning','95% interval, F1 points','Median AP change'],[[y,f'{float(ci[y]["f1_difference"])*100:+.3f} points',f'[{float(ci[y]["ci_low"])*100:+.3f}, {float(ci[y]["ci_high"])*100:+.3f}]',f'{float(ablation[y][1]["median_region_ap"])-float(ablation[y][0]["median_region_ap"]):+.4f}'] for y in ('2023','2024')])
          +card('Conclusion','The 2024 gain is <strong>+0.385 F1 percentage points</strong> (+0.69% relative), but it is not clearly distinguishable from zero. We do not yet have a consistent, clearly established F1 benefit.','accent'),
          '2,000 paired resamples of whole ISO weeks. Intervals describe within-year variation, not training-seed or future-season uncertainty. One F1 point = 0.01 F1, not one point of accuracy.')

    slide('26 · error trade-off','Lightning on vs off · 2024 matched models',
          '511 fewer false alarms, 100 more missed positive cell-days',
          '<div class="grid3">'+metric('−511','false-positive cell-days')+metric('−100','true-positive cell-days')+metric('+0.0091','median regional average precision')+'</div>'
          +table(['2024 outcome','Without lightning','With lightning'],[['True positives','10,178','10,078'],['False positives','6,435','5,924'],['Missed positives','9,904','10,004'],['Precision','0.6127','0.6298'],['Recall','0.5068','0.5018']])
          +card('Operational meaning','The lightning model is more selective at its 2022-chosen thresholds. That may suit a lower-false-alarm workflow, but it does not establish stronger detection of new fire starts.'),
          'This table compares two MLPs. The earlier 1,886-fewer-false-alarms result compared the lightning MLP with persistence. These are different comparisons, not conflicting counts.')

    slide('What we learned','Week 4 conclusion',
          'The baseline remains hard to beat',
          '<div class="grid2">'+card('Supported by this week’s results','<strong>21–22:</strong> richer inputs and an MLP narrow the gap on the two-satellite target.<br><strong>25:</strong> the S-NPP/lightning MLP does not clearly beat persistence in 2024.<br><strong>26:</strong> lightning improves observed median AP, but its F1 contribution is small and uncertain.','accent')
          +card('What we cannot claim','No proof that lightning has no value. No completed BC-wide regularisation result. No 2024 result for the frozen two-satellite MLP. No evidence that these hotspot models reliably predict verified ignitions.')+'</div>'
          +'<div class="flow"><div class="step"><span class="mono">Next · development only</span><b>Check robustness</b><span>Repeat matched seeds and temporal folds within development years.</span></div><div class="step"><span class="mono">Next · operating objective</span><b>Choose alert costs</b><span>Set a recall or alert-budget objective before a new evaluation.</span></div><div class="step"><span class="mono">Next · data</span><b>Restore richer inputs</b><span>Obtain matching NOAA-20/weather and a fresh evaluation season before renewed final-test claims.</span></div></div>',
          '2024 has now been inspected. Further model development must not present repeated 2024 comparisons as untouched holdout evidence.')

    links=[('21','21_exp_weather_lightning_spread_bc/RESULTS.html','Weather / lightning / spread · completed 2023'),('22','22_exp_neural_models_bc/RESULTS.html','MLP / GRU · completed 2023'),('23','23_exp_regularised_neural_bc/RESULTS.html','Regularisation · preliminary Kamloops report'),('24','24_exp_frozen_mlp_2024/README.md','Frozen two-satellite model · blocked'),('25','25_exp_snpp_lightning_2024/RESULTS.md','S-NPP + lightning · completed 2024'),('26','26_exp_lightning_ablation/RESULTS.md','Matched ablation · follow-up completed')]
    slide('Sources and index','Appendix · local experiment numbering',
          'Trace every claim to its experiment',
          table(['Experiment','Report / status'],[[n,f'<a href="{path}">{label}</a>'] for n,path,label in links])
          +'<div class="grid2">'+card('Metric sources','21–22: saved 2023 CSV summaries and HTML reports.<br>25–26: full-grid metric CSVs, paired bootstrap outputs, frozen protocols and completion checks. Values are embedded in this deck; no live data connection is needed.')
          +card('Data and continuity','<a href="Week3_GAW_detailed.html">Week 3 presentation</a><br><a href="../docs/data-guide.md">Local data guide</a><br><a href="https://crd-data-donnees-rdc.ec.gc.ca/CDAS/products/CanCPLD/CLDN/cg/">Official CanCPLD lightning source</a>')+'</div>',
          'Built with build_week4_report.py. The presentation itself works offline. Relative source links require the repository; the official source link requires internet.')

    additions='''
.slide[hidden] { display:none; }
a { color:var(--accent); text-underline-offset:3px; }
.controls { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
select, .utility { font:inherit; font-size:14px; padding:7px 10px; border:1px solid var(--rule); border-radius:6px; background:var(--surface); color:var(--fg); max-width:100%; }
.formula { font:500 clamp(14px,2.1vw,23px)/1.5 "IBM Plex Mono",monospace; padding:18px; background:var(--surface-2); border-radius:8px; overflow-x:auto; }
.axis-label { color:var(--muted); font-size:12px; }
.bar-nav select { width:230px; }
.interval-line { stroke:var(--series-model); stroke-width:4; }
.zero-line { stroke:var(--accent); stroke-width:2; stroke-dasharray:5 5; }
button,select,a { touch-action:manipulation; }
:focus-visible { outline:2px solid var(--accent); outline-offset:3px; }
.topline { display:flex; justify-content:space-between; align-items:center; gap:10px; }
@media(max-width:650px) { .bar-nav { gap:6px; flex-wrap:wrap; } .bar-nav select { flex:1; width:110px; } .bar-nav button { padding:6px 10px; } .progress { display:none; } .bar-nav .print-button { display:none; } .grid2 { grid-template-columns:minmax(0,1fr); } .grid3 { grid-template-columns:minmax(0,1fr); } .slide { padding-top:24px; } }
@media print { html,body,.deck,.stage { height:auto!important; overflow:visible!important; display:block; } .slide,.slide[hidden] { display:block!important; position:relative!important; inset:auto!important; overflow:visible!important; break-after:page; padding:24px 0; } .slide:last-child { break-after:auto; } .inner { min-height:0; gap:14px; } .bar-nav,.controls { display:none!important; } h2 { font-size:27px; } .card { break-inside:avoid; } table { font-size:12px; } th,td { padding:6px 9px; } .table-wrap { overflow:visible; } .dotplot svg { min-width:0; } body { font-size:14px; background:white; color:#1c1916; } :root { --bg:white; --surface:white; --surface-2:#efebe6; --fg:#1c1916; --fg-2:#4d4740; --muted:#7a7168; --rule:#ddd6ce; --accent:#b8430f; --accent-soft:#f6e3d8; } }
'''
    data={'dual':{d:[{'label':label,'value':dvalue(dual,m,d),'rule':m.startswith('rule')} for m,label in [('rule_pair_yesterday','Two-satellite persistence'),('catboost_spread_noday','21 · CatBoost'),('mlp','22 · MLP'),('gru','22 · MLP + GRU')]] for d in ('res7_exact','res7_margin1')},'full':full,'ablation':ablation,'ci':ci}
    script=r'''
const DATA = __DATA__;
const $ = id => document.getElementById(id);
function bars(host, rows) {
  host.replaceChildren();
  rows.forEach(r => {
    const row=document.createElement('div');row.className='bar-row';
    const name=document.createElement('span');name.className='name';name.textContent=r.label;name.title=r.label;
    const track=document.createElement('div');track.className='bar-track';
    const bar=document.createElement('div');bar.className='bar'+(r.rule?' rule':'');bar.style.width=(Number(r.value)*100)+'%';track.append(bar);
    const value=document.createElement('span');value.className='val';value.textContent=Number(r.value).toFixed(4);
    row.append(name,track,value);host.append(row);
  });
}
function dual(){bars($('dual-chart'),DATA.dual[$('dual-mode').value]);}
$('dual-mode').addEventListener('change',dual);dual();
bars($('test-chart'),DATA.full['2024'].map(r=>({label:r.model==='snpp_persistence'?'S-NPP persistence':'MLP + lightning',value:r.f1,rule:r.model==='snpp_persistence'})));
function ablation(){
  const year=$('ablation-year').value,rows=DATA.ablation[year];
  const table=document.createElement('table');
  table.innerHTML='<thead><tr><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>Median centre AP</th></tr></thead>';
  const tbody=document.createElement('tbody');
  rows.forEach(r=>{const tr=document.createElement('tr');[r.model==='with_lightning'?'With lightning':'Without lightning',...['f1','precision','recall','median_region_ap'].map(k=>Number(r[k]).toFixed(4))].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});tbody.append(tr);});table.append(tbody);
  $('ablation-table').className='table-wrap';$('ablation-table').replaceChildren(table);
  bars($('ablation-chart'),rows.map(r=>({label:r.model==='with_lightning'?'With lightning':'Without lightning',value:r.f1,rule:r.model==='without_lightning'})));
  const delta=Number(DATA.ci[year].f1_difference)*100;
  $('ablation-delta').textContent=year+': '+(delta>=0?'+':'')+delta.toFixed(3)+' F1 points from lightning. The 95% interval includes zero.';
}
$('ablation-year').addEventListener('change',ablation);ablation();
{
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox','0 0 900 210');svg.setAttribute('role','img');svg.setAttribute('aria-label','Lightning F1 effect: 2023 minus 0.201 points, interval minus 0.891 to plus 0.486; 2024 plus 0.385, interval minus 0.216 to plus 0.994. Both cross zero.');
  const x=v=>170+(v+1.1)/2.2*650;
  function el(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e;}
  [-1,-.5,0,.5,1].forEach(t=>{el('line',{x1:x(t),x2:x(t),y1:25,y2:155,class:t===0?'zero-line':'grid'});el('text',{x:x(t),y:182,'text-anchor':'middle'},(t>0?'+':'')+t.toFixed(1));});
  ['2023','2024'].forEach((year,i)=>{const c=DATA.ci[year],y=60+i*65;el('text',{x:10,y:y+4},year+' · F1 change');el('line',{x1:x(Number(c.ci_low)*100),x2:x(Number(c.ci_high)*100),y1:y,y2:y,class:'interval-line'});el('circle',{cx:x(Number(c.f1_difference)*100),cy:y,r:7,fill:'var(--series-model)'});});
  el('text',{x:495,y:207,'text-anchor':'middle'},'F1 percentage points · positive favours lightning');$('interval-chart').append(svg);
}
const slides=[...document.querySelectorAll('.slide')];let current=0;
slides.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+s.dataset.section;$('jump').append(o);});
function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.hidden=i!==current);slides[current].scrollTop=0;$('count').textContent=(current+1)+' / '+slides.length;$('jump').value=current;$('pbar').style.width=(current+1)/slides.length*100+'%';$('prev').disabled=current===0;$('next').disabled=current===slides.length-1;try{history.replaceState(null,'','#'+(current+1));}catch(e){}}
$('prev').onclick=()=>show(current-1);$('next').onclick=()=>show(current+1);$('jump').onchange=e=>show(Number(e.target.value));
document.addEventListener('keydown',e=>{if(e.target.closest('select,input,textarea,button,a'))return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1);}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1);}else if(e.key==='Home'){e.preventDefault();show(0);}else if(e.key==='End'){e.preventDefault();show(slides.length-1);}});
$('theme').onclick=()=>{const root=document.documentElement;const dark=root.dataset.theme?root.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;root.dataset.theme=dark?'light':'dark';$('theme').textContent=dark?'Dark':'Light';};
$('print').onclick=()=>window.print();
const initial=Number(location.hash.slice(1));show(Number.isInteger(initial)&&initial>0?initial-1:0);
'''.replace('__DATA__',json.dumps(data))
    document='<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><title>Week 4 · BC Hotspot Forecasting · GAW</title><style>'+css+additions+'</style></head><body><div class="deck"><main class="stage" id="stage">'+''.join(slides)+'</main><nav class="bar-nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><select id="jump" aria-label="Jump to slide"></select><div class="progress"><div id="pbar"></div></div><span class="count" id="count" aria-live="polite"></span><button id="theme" aria-label="Toggle colour theme">Theme</button><button id="print" class="print-button">Print</button></nav></div><noscript><style>.slide{position:relative;overflow:visible}.stage,.deck{height:auto}body{overflow:auto}.bar-nav{display:none}</style><p>This deck is readable without JavaScript; interactive charts require JavaScript.</p></noscript><script>'+script+'</script></body></html>'
    (HERE/'Week4_GAW_detailed.html').write_text(document,encoding='utf-8')
    print(f'Built {len(slides)} slides; {len(document):,} characters.')


if __name__=='__main__':main()
