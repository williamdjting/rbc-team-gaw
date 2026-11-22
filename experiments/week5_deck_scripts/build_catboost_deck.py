"""Error-analysis deck for Exp 21 CatBoost (best model for now), next-day two-satellite target, 2023, every cell.
Uses only Exp 21's saved results. usage: python build_catboost_deck.py EXPERIMENTS_DIR CENTRES_JSON"""
import json, re, sys
from pathlib import Path
import pandas as pd
EXP = Path(sys.argv[1]); R = EXP / '21_exp_weather_lightning_spread_bc/results'
CELL_DAYS = 34_244_884  # 234,554 cells x 146 days (May 8 - Sep 30, 2023)
m = pd.read_csv(R / 'metrics_by_year.csv').query("year==2023 and definition=='res7_exact' and method=='f1_2022'")
m = m.assign(tp=(m.recall * m.positives).round())
g = m.groupby('model')[['flagged', 'positives', 'tp']].sum()
def cm(model):
    r = g.loc[model]; tp, fp, fn = int(r.tp), int(r.flagged - r.tp), int(r.positives - r.tp)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': CELL_DAYS - tp - fp - fn}
n = pd.read_csv(R / 'new_vs_continuing_2023.csv').query("method=='f1_2022'").groupby('model').sum(numeric_only=True)
brk = {k: n.loc[k].astype(int).to_dict() for k in ['catboost_spread_noday', 'rule_pair_yesterday']}
import base64
SD = Path(__file__).resolve().parent
CB8 = json.load(open(SD / 'cbn.json')); IMG8 = base64.b64encode(open(SD / 'cbn_map.png', 'rb').read()).decode()
CAUSE = json.load(open(SD / 'bcws_cause_2023.json'))
CONF = json.load(open(SD / 'cbconf.json'))
data = {'conf': CONF, 'cause': CAUSE, 'cb8': CB8, 'cm': cm('catboost_spread_noday'), 'rule': cm('rule_pair_yesterday'), 'brk': brk, 'centres': json.load(open(sys.argv[2]))}
f1 = lambda c: 2 * c['tp'] / (2 * c['tp'] + c['fp'] + c['fn'])
w4 = open(EXP / 'Week4_GAW_detailed.html', encoding='utf-8').read()
head = re.sub(r'<title>.*?</title>', '<title>Week 5 · Error Analysis · GAW</title>', w4[:w4.index('</style></head>') + 8])
css = '''<style>
svg.chart { width:100%; height:auto; display:block; overflow:visible; }
svg.chart text { fill:var(--fg-2); font:13px "IBM Plex Sans",system-ui,sans-serif; }
svg.chart .tick { fill:var(--muted); font-size:12px; } svg.chart .grid { stroke:var(--rule); }
svg.chart .big { fill:var(--fg); font-size:14px; font-weight:600; }
.chartbox { overflow-x:auto; } .chartbox svg { min-width:480px; }
.cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:5px; font-size:13px; }
.cm .h { color:var(--muted); font:500 11px "IBM Plex Mono",monospace; text-transform:uppercase; letter-spacing:.05em; text-align:center; align-self:end; padding:2px; }
.cm .v { writing-mode:vertical-rl; transform:rotate(180deg); }
.cm .c { border-radius:8px; padding:16px 6px; text-align:center; font-variant-numeric:tabular-nums; }
.cm .c b { display:block; font-size:22px; color:var(--fg); } .cm .c span { color:var(--fg-2); font-size:12px; }
.cm .tp { background:color-mix(in srgb,var(--good) 32%,var(--surface)); } .cm .tn { background:color-mix(in srgb,var(--good) 12%,var(--surface)); }
.cm .fp, .cm .fn { background:color-mix(in srgb,var(--bad) 24%,var(--surface)); }
.heat { border-collapse:separate; border-spacing:4px; width:100%; }
.heat th { font:500 12px "IBM Plex Mono",monospace; color:var(--muted); background:none; text-transform:none; letter-spacing:0; padding:4px; }
.heat td { text-align:center; border-radius:6px; padding:10px 4px; font-variant-numeric:tabular-nums; font-weight:600; border:0; }
.heat td small { display:block; font-weight:400; font-size:11px; opacity:.75; }
.heat td.rowh { text-align:right; font-weight:500; background:none; color:var(--fg-2); }
.facts { width:100%; font-size:14px; } .facts td { padding:6px 12px; } .facts td:first-child { color:var(--muted); white-space:nowrap; width:1%; }
@media (max-width:560px) { .cm .c b { font-size:16px; } }
</style>'''
K = 'Exp 21 CatBoost · next-day target (S-NPP + NOAA-20) · 2023 · every cell, all six Fire Centres'
body = f'''<div class="deck"><main class="stage" id="stage">

<section class="slide" data-section="Confusion matrix"><div class="inner">
<span class="kicker">{K}</span>
<h2>Confusion matrix</h2>
<div class="grid3">
<div class="stat"><span class="num">{f1(data['cm']):.3f}</span><span class="lab">F1, CatBoost</span></div>
<div class="stat"><span class="num">{f1(data['rule']):.3f}</span><span class="lab">F1, "fire yesterday" rule</span></div>
<div class="stat"><span class="num">0.824</span><span class="lab">CatBoost F1 with one-cell margin (rule 0.810)</span></div>
</div>
<div class="grid2" id="cms"></div>
</div></section>

<section class="slide" data-section="What it catches"><div class="inner">
<span class="kicker">{K}</span>
<h2>Fires caught: continuing vs new</h2>
<div class="legend"><span><i class="sw" style="background:var(--series-model)"></i>CatBoost</span><span><i class="sw" style="background:var(--series-rule)"></i>"Fire yesterday" rule</span></div>
<div class="grid2">
<div class="card"><h3>Fire cells in 2023</h3><div class="chartbox" id="mix"></div></div>
<div class="card"><h3>Share caught</h3><div class="chartbox" id="caught"></div></div>
</div>
<p class="note">Continuing: the cell had a detection yesterday. New: it did not (includes fire spreading in from a neighbour).</p>
<div class="card"><h3>2023 BCWS wildfires by cause (May–Sep)</h3><div class="legend"><span><i class="sw" style="background:#1f6f8b"></i>Lightning</span><span><i class="sw" style="background:#e8642c"></i>Person</span><span><i class="sw" style="background:#9a928a"></i>Unknown</span></div><div class="chartbox" id="cause"></div></div>
</div></section>

<section class="slide" data-section="By Fire Centre"><div class="inner">
<span class="kicker">{K}</span>
<h2>F1 by Fire Centre</h2>
<div class="legend"><span><i class="sw" style="background:var(--series-model)"></i>CatBoost</span><span><i class="sw" style="background:var(--series-rule)"></i>"Fire yesterday" rule</span></div>
<div class="card" style="max-width:720px;width:100%;margin:0 auto"><h3>Exact cell</h3><div class="chartbox" id="exact"></div></div>
</div></section>

<section class="slide" data-section="One day on the map"><div class="inner">
<span class="kicker">Exp 21 CatBoost · next-day target · 2023 · every cell, every 8th day (18 dates) · Kamloops</span>
<h2>One day on the map</h2>
<div class="legend"><span><i class="sw" style="background:#2a78d6"></i>Caught (@@TP@@)</span><span><i class="sw" style="background:#d23b3b"></i>Missed (@@FN@@)</span><span><i class="sw" style="background:#f2a33a"></i>False alarm (@@FP@@)</span></div>
<figure style="margin:0 auto;max-width:500px;width:100%"><img src="data:image/png;base64,@@IMG8@@" alt="Kamloops, one day in August 2023: caught fire cells at fire cores, missed and false-alarm cells around the edges" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--rule)"></figure>
<div class="table-wrap" style="max-width:720px;width:100%;margin:0 auto"><table class="facts"><tbody>
<tr><td>Region</td><td>Kamloops Fire Centre: all 16,233 H3 cells (about 84,000 km²)</td></tr>
<tr><td>Day</td><td>One forecast: fire on @@START@@, using data up to the day before. Not aggregated over days.</td></tr>
<tr><td>Why this day</td><td>The August forecast date with the most fire cells in Kamloops (of the 18 scored dates)</td></tr>
<tr><td>Threshold</td><td>0.268: Exp 21's Kamloops threshold, chosen on 2022</td></tr>
<tr><td>Label</td><td>S-NPP or NOAA-20 detection in the cell that day (one-day gaps filled), rebuilt with FIRMS NOAA-20: 99.4% match to the original</td></tr>
</tbody></table></div>
</div></section>

<section class="slide" data-section="When and where"><div class="inner">
<span class="kicker">Exp 21 CatBoost · next-day target · 2023 · every cell, every 8th day (18 dates) · F1 by Fire Centre and month (fire cell-days below)</span>
<h2>When and where errors happen</h2>
<div class="table-wrap" style="padding:8px"><table class="heat" id="heat"></table></div>
<div class="table-wrap"><table class="facts"><tbody>
<tr><td>Each square</td><td>F1 for one Fire Centre, pooling every cell on that month's forecast dates; small number = fire cell-days</td></tr>
<tr><td>Dates scored</td><td>Every 8th day, 18 in all: May 8, 16, 24 · Jun 1, 9, 17, 25 · Jul 3, 11, 19, 27 · Aug 4, 12, 20, 28 · Sep 5, 13, 21</td></tr>
<tr><td>Thresholds</td><td>Exp 21's, one per Fire Centre, chosen on 2022 (0.17–0.34)</td></tr>
<tr><td>Grey squares</td><td>Fewer than 20 fire cell-days: too few to score</td></tr>
<tr><td>Overall</td><td>F1 over these 18 dates: @@F118@@ (all 146 days: 0.607)</td></tr>
</tbody></table></div>
</div></section>

<section class="slide" data-section="Confidence"><div class="inner">
<span class="kicker">Exp 21 CatBoost · next-day target · 2023 · every cell, every 8th day (18 dates)</span>
<h2>How confident is the model when it is right vs wrong?</h2>
<div class="grid2">
<div class="card"><h3>Model score by outcome</h3><div class="legend"><span><i class="sw" style="background:var(--good)"></i>Right</span><span><i class="sw" style="background:var(--bad)"></i>Wrong</span><span><i class="sw" style="background:var(--accent);opacity:.25"></i>Threshold range (0.17–0.34)</span></div><div class="chartbox" id="confbox"></div><p class="note">Box = middle 50% of scores, line = median, whiskers = 5th–95th percentile.</p></div>
<div class="card"><h3>Model said vs what actually happened</h3><div class="legend"><span><i class="sw" style="background:var(--series-old)"></i>Model said (average score)</span><span><i class="sw" style="background:var(--series-rule)"></i>Actually burned</span></div><div class="chartbox" id="confcal"></div></div>
</div>
<ul style="font-size:15px"><li>Right calls are confident: fires flagged at a median score of <strong>0.60</strong>; quiet cells at <strong>0.00</strong></li>
<li>False alarms are borderline: median <strong>0.39</strong>, just above the threshold</li>
<li>Missed fires got a low score (median <strong>0.08</strong>): mostly new fires with no sign in the inputs</li>
<li>Under-confident: cells scored 0.3–0.4 actually burned 54% of the time, so use scores for ranking, not as exact probabilities</li><li><strong>Model said</strong> = the average CatBoost score (0–1, its predicted chance of fire tomorrow) of the cell-days in that range</li><li><strong>Actually burned</strong> = share of those cell-days with an S-NPP or NOAA-20 fire detection in the cell the next day</li></ul>
</div></section>

</main><nav class="bar-nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><select id="jump" aria-label="Jump to slide"></select><div class="progress"><div id="pbar"></div></div><span class="count" id="count" aria-live="polite"></span><button id="theme" aria-label="Toggle colour theme">Theme</button><button id="print" class="print-button">Print</button></nav></div>
<script>
const D=@@DATA@@;const $=id=>document.getElementById(id);const NS='http://www.w3.org/2000/svg';
function svg(w,h,l){{const s=document.createElementNS(NS,'svg');s.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);s.setAttribute('class','chart');s.setAttribute('role','img');s.setAttribute('aria-label',l);return s;}}
function el(s,t,a,x){{const e=document.createElementNS(NS,t);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));if(x!==undefined)e.textContent=x;s.append(e);return e;}}
const fmt=v=>v.toLocaleString('en-US'),pct=v=>(v*100).toFixed(v<.1?1:0)+'%',MC='var(--series-model)',RC='var(--series-rule)';
function bars(host,rows,{{w=480,left=150,max=1,fmtv=pct,label=''}}={{}}){{const bh=20,gap=4,gp=16;let y=6;const H=rows.reduce((a,r)=>a+r.v.length*(bh+gap)+gp,6)+24,s=svg(w,H,label),x=v=>left+v/max*(w-left-60);
 [0,.25,.5,.75,1].map(q=>q*max).forEach(q=>{{el(s,'line',{{x1:x(q),x2:x(q),y1:0,y2:H-24,class:'grid'}});el(s,'text',{{x:x(q),y:H-6,'text-anchor':'middle',class:'tick'}},fmtv(q));}});
 rows.forEach(r=>{{const h=r.v.length*(bh+gap);el(s,'text',{{x:left-10,y:y+h/2+3,'text-anchor':'end'}},r.label);r.v.forEach(([v,c],i)=>{{const yy=y+i*(bh+gap);el(s,'rect',{{x:x(0),y:yy,width:Math.max(2,x(v)-x(0)),height:bh,rx:4,fill:c}});el(s,'text',{{x:x(v)+7,y:yy+bh-5,class:i?'tick':'big'}},fmtv(v));}});y+=h+gp;}});host.append(s);}}
// slide 1: BCWS fire cause split
{{const W=900,H=150,s=svg(W,H,'2023 BC Wildfire Service wildfires by cause, by count and by area burned'),L=170,C=D.cause,ks=['Lightning','Person','Unknown'],col={{Lightning:'#1f6f8b',Person:'#e8642c',Unknown:'#9a928a'}};
 [['Wildfires ('+fmt(C.total)+')',C.n],['Area burned ('+fmt(Math.round(Object.values(C.area).reduce((a,b)=>a+b,0)/1e6*100)/100)+'M ha)',C.area]].forEach(([lab,v],r)=>{{const tot=ks.reduce((a,k)=>a+(v[k]||0),0),y=14+r*62;let x=L;
  el(s,'text',{{x:L-12,y:y+25,'text-anchor':'end'}},lab);
  ks.forEach(k=>{{const w=(v[k]||0)/tot*(W-L-10);el(s,'rect',{{x,y,width:Math.max(0,w-2),height:40,rx:5,fill:col[k]}});if(w>46)el(s,'text',{{x:x+w/2,y:y+26,'text-anchor':'middle',style:'fill:#fff;font-weight:600'}},pct(v[k]/tot));x+=w;}});}});
 $('cause').append(s);}}
// confusion matrices
[['CatBoost',D.cm],['"Fire yesterday" rule',D.rule]].forEach(([t,m])=>{{const n=m.tp+m.fp+m.fn+m.tn,p=v=>(100*v/n).toFixed(v/n<.01?3:1)+'%';const d=document.createElement('div');d.className='card';
 d.innerHTML=`<h3>${{t}}</h3><div class="cm"><span></span><span class="h">Predicted fire</span><span class="h">Predicted no fire</span>
 <span class="h v">Fire</span><div class="c tp"><b>${{fmt(m.tp)}}</b><span>TP · ${{p(m.tp)}}</span></div><div class="c fn"><b>${{fmt(m.fn)}}</b><span>FN · ${{p(m.fn)}}</span></div>
 <span class="h v">No fire</span><div class="c fp"><b>${{fmt(m.fp)}}</b><span>FP · ${{p(m.fp)}}</span></div><div class="c tn"><b>${{fmt(m.tn)}}</b><span>TN · ${{p(m.tn)}}</span></div></div>
 <p class="note">Precision ${{(m.tp/(m.tp+m.fp)).toFixed(3)}} · Recall ${{(m.tp/(m.tp+m.fn)).toFixed(3)}}</p>`;$('cms').append(d);}});
const B=D.brk.catboost_spread_noday,Rb=D.brk.rule_pair_yesterday,tot=B.continuing_positives+B.new_positives;
// fire mix (donut-like stacked bar)
{{const W=480,H=170,s=svg(W,H,'Fire cells: continuing vs new'),a=B.continuing_positives/tot,x0=10,w=W-20;
 el(s,'rect',{{x:x0,y:20,width:w*a-3,height:60,rx:8,fill:'#1f6f8b'}});el(s,'rect',{{x:x0+w*a,y:20,width:w*(1-a),height:60,rx:8,fill:'#9a928a'}});
 el(s,'text',{{x:x0+w*a/2,y:57,'text-anchor':'middle',style:'fill:#fff;font-weight:600;font-size:16px'}},pct(a));el(s,'text',{{x:x0+w*a+w*(1-a)/2,y:57,'text-anchor':'middle',style:'fill:#fff;font-weight:600;font-size:16px'}},pct(1-a));
 el(s,'rect',{{x:x0,y:104,width:14,height:14,rx:3,fill:'#1f6f8b'}});el(s,'text',{{x:x0+22,y:116}},'Continuing · '+fmt(B.continuing_positives));
 el(s,'rect',{{x:x0,y:132,width:14,height:14,rx:3,fill:'#9a928a'}});el(s,'text',{{x:x0+22,y:144}},'New (incl. spread) · '+fmt(B.new_positives));$('mix').append(s);}}
bars($('caught'),[{{label:'Continuing',v:[[B.continuing_caught/B.continuing_positives,MC],[Rb.continuing_caught/Rb.continuing_positives,RC]]}},
 {{label:'New (incl. spread)',v:[[B.new_caught/B.new_positives,MC],[Rb.new_caught/Rb.new_positives,RC]]}}],{{label:'Share of fire cells caught'}});
// by centre
['exact'].forEach(k=>{{const d=k==='exact'?'res7_exact':'res7_margin1';const rows=Object.entries(D.centres).map(([r,v])=>({{label:r,v:[[v[d+'|catboost_spread_noday'],MC],[v[d+'|rule_pair_yesterday'],RC]]}}));
 bars($(k),rows,{{fmtv:v=>v.toFixed(2),label:'F1 by Fire Centre, '+k}});}});
// next-day heatmap (18 dates)
{{const H8=D.cb8.heat,regs=[...new Set(H8.map(h=>h.region))],mons={{5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep'}};
 let h='<tr><th></th>'+Object.values(mons).map(m=>`<th>${{m}}</th>`).join('')+'</tr>';
 regs.forEach(r=>{{h+=`<tr><td class="rowh">${{r}}</td>`;Object.keys(mons).forEach(m=>{{const c=H8.find(x=>x.region===r&&x.month==m);
  if(!c||c.fires<20){{h+=`<td style="background:var(--surface-2);color:var(--muted)">–<small>${{c?c.fires:0}}</small></td>`;return;}}
  const a=Math.round(15+c.f1*75);h+=`<td style="background:color-mix(in srgb,#1f6f8b ${{a}}%,var(--surface));color:${{c.f1>.45?'#fff':'var(--fg)'}}">${{c.f1.toFixed(2)}}<small>${{fmt(c.fires)}}</small></td>`;}});h+='</tr>';}});
 $('heat').innerHTML=h;}}
// slide 6: confidence by outcome + calibration
{{const C=D.conf,W=520,H=330,L=170,R=20,T=10,B=44,s=svg(W,H,'Model score distribution by outcome'),x=v=>L+v*(W-L-R);
 el(s,'rect',{{x:x(C.thr[0]),y:T,width:x(C.thr[1])-x(C.thr[0]),height:H-T-B,fill:'var(--accent)',opacity:.18}});
 [0,.25,.5,.75,1].forEach(q=>{{el(s,'line',{{x1:x(q),x2:x(q),y1:T,y2:H-B,class:'grid'}});el(s,'text',{{x:x(q),y:H-B+18,'text-anchor':'middle',class:'tick'}},q);}});
 el(s,'text',{{x:(L+W-R)/2,y:H-6,'text-anchor':'middle',class:'big'}},'Model score (predicted chance of fire tomorrow)');
 [['tp','Fire, flagged (TP)','var(--good)'],['tn','No fire, not flagged (TN)','var(--good)'],['fp','False alarm (FP)','var(--bad)'],['fn','Missed fire (FN)','var(--bad)']].forEach(([k,lab,c],i)=>{{
  const b=C.box[k],q=b.q,y=T+16+i*66;el(s,'text',{{x:L-12,y:y+12,'text-anchor':'end'}},lab);el(s,'text',{{x:L-12,y:y+28,'text-anchor':'end',class:'tick'}},fmt(b.n)+' cell-days');
  el(s,'line',{{x1:x(q[0]),x2:x(q[4]),y1:y+12,y2:y+12,stroke:c,'stroke-width':2}});
  el(s,'rect',{{x:x(q[1]),y:y,width:Math.max(3,x(q[3])-x(q[1])),height:24,rx:4,fill:c,opacity:.35,stroke:c,'stroke-width':1.5}});
  el(s,'line',{{x1:x(q[2]),x2:x(q[2]),y1:y-3,y2:y+27,stroke:c,'stroke-width':3.5}});
  el(s,'text',{{x:Math.min(x(q[4])+8,W-60),y:y+16,class:'big'}},q[2].toFixed(2));}});
 $('confbox').append(s);}}
{{const C=D.conf,bins=C.cal.filter(c=>c.lo>=0.05),W=520,H=400,L=44,R=10,T=14,B=70,s=svg(W,H,'Average model score vs share that actually burned, by score range'),
 n=bins.length,bw=(W-L-R)/n,Y=v=>T+(1-v)*(H-T-B);
 [0,.25,.5,.75,1].forEach(q=>{{el(s,'line',{{x1:L,x2:W-R,y1:Y(q),y2:Y(q),class:'grid'}});el(s,'text',{{x:L-6,y:Y(q)+4,'text-anchor':'end',class:'tick'}},(q*100)+'%');}});
 bins.forEach((c,i)=>{{const x0=L+i*bw,w=bw*.36;
  el(s,'rect',{{x:x0+bw*.12,y:Y(c.pred),width:w,height:Y(0)-Y(c.pred),rx:2,fill:'var(--series-old)'}});
  el(s,'rect',{{x:x0+bw*.52,y:Y(c.obs),width:w,height:Y(0)-Y(c.obs),rx:2,fill:'var(--series-rule)'}});
  el(s,'text',{{x:x0+bw*.52+w/2,y:Y(c.obs)-5,'text-anchor':'middle',class:'tick'}},Math.round(c.obs*100)+'%');
  el(s,'text',{{x:x0+bw/2,y:H-B+16,'text-anchor':'middle',class:'tick'}},String(+c.lo.toFixed(2))+'–'+(c.hi>=1?'1':String(+c.hi.toFixed(2))));
  el(s,'text',{{x:x0+bw/2,y:H-B+31,'text-anchor':'middle',class:'tick',style:'font-size:10.5px'}},fmt(c.n));}});
 el(s,'text',{{x:(L+W-R)/2,y:H-10,'text-anchor':'middle',class:'big'}},'Model score range (cell-days below)');
 $('confcal').append(s);}}
// nav
const slides=[...document.querySelectorAll('.slide')];let cur=0;
slides.forEach((s,i)=>{{s.id='slide-'+(i+1);const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+s.dataset.section;$('jump').append(o);}});
function show(n){{cur=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.hidden=i!==cur);slides[cur].scrollTop=0;$('count').textContent=(cur+1)+' / '+slides.length;$('jump').value=cur;$('pbar').style.width=(cur+1)/slides.length*100+'%';$('prev').disabled=cur===0;$('next').disabled=cur===slides.length-1;try{{history.replaceState(null,'','#'+(cur+1));}}catch(e){{}}}}
$('prev').onclick=()=>show(cur-1);$('next').onclick=()=>show(cur+1);$('jump').onchange=e=>show(Number(e.target.value));
document.addEventListener('keydown',e=>{{if(e.target.closest('select,input,textarea,button,a'))return;if(['ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();show(cur+1);}}else if(['ArrowLeft','PageUp'].includes(e.key)){{e.preventDefault();show(cur-1);}}}});
$('theme').onclick=()=>{{const r=document.documentElement;const d=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=d?'light':'dark';}};
$('print').onclick=()=>window.print();const ini=Number(location.hash.slice(1));show(Number.isInteger(ini)&&ini>0?ini-1:0);
</script></body></html>'''
body = (body.replace('@@DATA@@', json.dumps(data)).replace('@@TP@@', str(CB8['map']['tp'])).replace('@@FN@@', str(CB8['map']['fn']))
        .replace('@@FP@@', str(CB8['map']['fp'])).replace('@@START@@', CB8['map']['date']).replace('@@F118@@', f"{CB8['f1_18dates']:.3f}").replace('@@IMG8@@', IMG8))
open(EXP / 'Week5_error_analysis.html', 'w', encoding='utf-8', newline='\n').write(head + '\n' + css + '</head><body>' + body)
print('written', data['cm'], data['rule'])
