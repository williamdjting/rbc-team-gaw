"""Week 5 error-analysis deck: classic evaluation plots (Exp 25 MLP, 2023, every cell) + one-day map + F1 heatmap.
usage: python build_classic_deck.py SCRATCH_DIR EXPERIMENTS_DIR   (needs classic.json, err.json, err_map.png in SCRATCH_DIR)"""
import base64, json, re, sys
from pathlib import Path
S = Path(sys.argv[1]); EXP = Path(sys.argv[2])
C = json.load(open(S / 'classic.json')); E = json.load(open(S / 'err.json'))
img = base64.b64encode(open(S / 'err_map.png', 'rb').read()).decode()
w4 = open(EXP / 'Week4_GAW_detailed.html', encoding='utf-8').read()
head = re.sub(r'<title>.*?</title>', '<title>Week 5 · Error Analysis · GAW</title>', w4[:w4.index('</style></head>') + 8])
f1 = lambda c: 2 * c['tp'] / (2 * c['tp'] + c['fp'] + c['fn'])
data = {'C': C, 'heat': [h for h in E['heat'] if h['model'] == 'model'],
        'map': {k: len(E['mapday'][k]) for k in ('tp', 'fp', 'fn')}}
css = '''<style>
svg.chart { width:100%; height:auto; display:block; overflow:visible; }
svg.chart text { fill:var(--fg-2); font:13px "IBM Plex Sans",system-ui,sans-serif; }
svg.chart .tick { fill:var(--muted); font-size:12px; }
svg.chart .grid { stroke:var(--rule); }
svg.chart .axl { fill:var(--fg); font-size:13.5px; font-weight:600; }
.chartbox { overflow-x:auto; } .chartbox svg { min-width:520px; }
.cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:5px; font-size:13px; }
.cm .h { color:var(--muted); font:500 11px "IBM Plex Mono",monospace; text-transform:uppercase; letter-spacing:.05em; text-align:center; align-self:end; padding:2px; }
.cm .v { writing-mode:vertical-rl; transform:rotate(180deg); }
.cm .c { border-radius:8px; padding:16px 6px; text-align:center; font-variant-numeric:tabular-nums; }
.cm .c b { display:block; font-size:22px; color:var(--fg); }
.cm .c span { color:var(--fg-2); font-size:12px; }
.cm .tp { background:color-mix(in srgb,var(--good) 32%,var(--surface)); } .cm .tn { background:color-mix(in srgb,var(--good) 12%,var(--surface)); }
.cm .fp, .cm .fn { background:color-mix(in srgb,var(--bad) 24%,var(--surface)); }
.heat { border-collapse:separate; border-spacing:4px; width:100%; }
.heat th { font:500 12px "IBM Plex Mono",monospace; color:var(--muted); background:none; text-transform:none; letter-spacing:0; padding:4px; }
.heat td { text-align:center; border-radius:6px; padding:10px 4px; font-variant-numeric:tabular-nums; font-weight:600; border:0; }
.heat td small { display:block; font-weight:400; font-size:11px; opacity:.75; }
.heat td.rowh { text-align:right; font-weight:500; background:none; color:var(--fg-2); }
@media (max-width:560px) { .cm .c b { font-size:16px; } }
</style>'''
K = 'Exp 25 MLP (S-NPP + lightning) · next-day target · 2023 · every cell, all six Fire Centres'
body = f'''<div class="deck"><main class="stage" id="stage">

<section class="slide" data-section="Confusion matrix"><div class="inner">
<span class="kicker">{K}</span>
<h2>Confusion matrix</h2>
<div class="grid3">
<div class="stat"><span class="num">{f1(C['cm']):.2f}</span><span class="lab">F1, our model</span></div>
<div class="stat"><span class="num">{f1(C['rule']):.2f}</span><span class="lab">F1, "fire yesterday" rule</span></div>
<div class="stat"><span class="num">{C['ap']:.2f}</span><span class="lab">Average precision (random = {C['base']:.4f})</span></div>
</div>
<div class="grid2" id="cms"></div>
</div></section>

<section class="slide" data-section="Precision-recall curve"><div class="inner">
<span class="kicker">{K}</span>
<h2>Precision–recall curve</h2>
<div class="legend"><span><i class="sw" style="background:var(--series-model)"></i>Our model</span><span><i class="sw dot" style="background:var(--series-model)"></i>Model at chosen thresholds</span><span><i class="sw dot" style="background:var(--series-rule)"></i>"Fire yesterday" rule</span><span><i class="sw" style="border-top:2px dashed var(--muted);height:0;border-radius:0;width:18px"></i>Equal-F1 lines</span></div>
<div class="card"><div class="chartbox" id="pr" style="max-width:720px;margin:0 auto;width:100%"></div></div>
</div></section>

<section class="slide" data-section="Threshold"><div class="inner">
<span class="kicker">{K}</span>
<h2>Precision, recall and F1 by threshold</h2>
<div class="legend"><span><i class="sw" style="background:var(--series-old)"></i>Precision</span><span><i class="sw" style="background:var(--series-rule)"></i>Recall</span><span><i class="sw" style="background:var(--series-model)"></i>F1</span></div>
<div class="card"><div class="chartbox" id="thr"></div></div>
</div></section>

<section class="slide" data-section="Score histogram"><div class="inner">
<span class="kicker">{K}</span>
<h2>Score distribution: fire vs non-fire</h2>
<div class="legend"><span><i class="sw" style="background:var(--series-rule)"></i>Fire cell-days ({C['P']:,})</span><span><i class="sw" style="background:var(--series-model)"></i>Non-fire cell-days ({C['N']:,})</span></div>
<div class="card"><div class="chartbox" id="hist"></div></div>
</div></section>

<section class="slide" data-section="Calibration"><div class="inner">
<span class="kicker">{K}</span>
<h2>Calibration: predicted vs observed</h2>
<div class="legend"><span><i class="sw dot" style="background:var(--series-model)"></i>Our model (dot size = cell-days in bin)</span><span><i class="sw" style="border-top:2px dashed var(--muted);height:0;border-radius:0;width:18px"></i>Perfect calibration</span></div>
<div class="card"><div class="chartbox" id="cal" style="max-width:640px;margin:0 auto;width:100%"></div></div>
</div></section>

<section class="slide" data-section="ROC curve"><div class="inner">
<span class="kicker">{K}</span>
<h2>ROC curve</h2>
<div class="legend"><span><i class="sw" style="background:var(--series-model)"></i>Our model (AUC {C['auc']:.3f})</span><span><i class="sw dot" style="background:var(--series-rule)"></i>"Fire yesterday" rule</span></div>
<div class="card"><div class="chartbox" id="roc" style="max-width:640px;margin:0 auto;width:100%"></div><p class="note">AUC is high because 99.85% of cell-days never burn; precision–recall is the more honest view.</p></div>
</div></section>

<section class="slide" data-section="One day"><div class="inner">
<span class="kicker">Kamloops Fire Centre · {E['mapday']['date']} · our model</span>
<h2>One day on the map</h2>
<div class="legend"><span><i class="sw" style="background:#2a78d6"></i>Caught ({data['map']['tp']})</span><span><i class="sw" style="background:#d23b3b"></i>Missed ({data['map']['fn']})</span><span><i class="sw" style="background:#f2a33a"></i>False alarm ({data['map']['fp']})</span></div>
<figure style="margin:0 auto;max-width:760px;width:100%"><img src="data:image/png;base64,@@IMG@@" alt="Kamloops map: caught fire cells at fire cores, missed cells around fire edges" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--rule)"></figure>
</div></section>

<section class="slide" data-section="When and where"><div class="inner">
<span class="kicker">Our model · F1 by Fire Centre and month · 2023 (fire cell-days below)</span>
<h2>When and where errors happen</h2>
<div class="table-wrap" style="padding:8px"><table class="heat" id="heat"></table></div>
</div></section>

</main><nav class="bar-nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><select id="jump" aria-label="Jump to slide"></select><div class="progress"><div id="pbar"></div></div><span class="count" id="count" aria-live="polite"></span><button id="theme" aria-label="Toggle colour theme">Theme</button><button id="print" class="print-button">Print</button></nav></div>
<script>
const D=@@DATA@@,C=D.C;const $=id=>document.getElementById(id);const NS='http://www.w3.org/2000/svg';
function svg(w,h,l){{const s=document.createElementNS(NS,'svg');s.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);s.setAttribute('class','chart');s.setAttribute('role','img');s.setAttribute('aria-label',l);return s;}}
function el(s,t,a,x){{const e=document.createElementNS(NS,t);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));if(x!==undefined)e.textContent=x;s.append(e);return e;}}
const fmt=v=>v.toLocaleString('en-US'),MC='var(--series-model)',RC='var(--series-rule)',PC='var(--series-old)';
function axes(s,{{L,R,T,B,W,H,xt,yt,xl,yl,xf,yf,X,Y}}){{xt.forEach(q=>{{el(s,'line',{{x1:X(q),x2:X(q),y1:T,y2:H-B,class:'grid'}});el(s,'text',{{x:X(q),y:H-B+18,'text-anchor':'middle',class:'tick'}},xf(q));}});
 yt.forEach(q=>{{el(s,'line',{{x1:L,x2:W-R,y1:Y(q),y2:Y(q),class:'grid'}});el(s,'text',{{x:L-8,y:Y(q)+4,'text-anchor':'end',class:'tick'}},yf(q));}});
 el(s,'text',{{x:(L+W-R)/2,y:H-6,'text-anchor':'middle',class:'axl'}},xl);el(s,'text',{{x:16,y:(T+H-B)/2,'text-anchor':'middle',class:'axl',transform:`rotate(-90 16 ${{(T+H-B)/2}})`}},yl);}}
const line=(s,xs,ys,X,Y,c,w=3,dash)=>el(s,'polyline',{{points:xs.map((v,i)=>X(v)+','+Y(ys[i])).join(' '),fill:'none',stroke:c,'stroke-width':w,...(dash?{{'stroke-dasharray':dash}}:{{}})}});
const cv=C.curve,pt=(c)=>[c.tp/(c.tp+c.fn),c.tp/(c.tp+c.fp)];
// 1 confusion matrices
[['Our model (MLP)',C.cm],['"Fire yesterday" rule',C.rule]].forEach(([t,m])=>{{const n=m.tp+m.fp+m.fn+m.tn,p=v=>(100*v/n).toFixed(v/n<.01?3:1)+'%';const d=document.createElement('div');d.className='card';
 d.innerHTML=`<h3>${{t}}</h3><div class="cm"><span></span><span class="h">Predicted fire</span><span class="h">Predicted no fire</span>
 <span class="h v">Fire</span><div class="c tp"><b>${{fmt(m.tp)}}</b><span>TP · ${{p(m.tp)}}</span></div><div class="c fn"><b>${{fmt(m.fn)}}</b><span>FN · ${{p(m.fn)}}</span></div>
 <span class="h v">No fire</span><div class="c fp"><b>${{fmt(m.fp)}}</b><span>FP · ${{p(m.fp)}}</span></div><div class="c tn"><b>${{fmt(m.tn)}}</b><span>TN · ${{p(m.tn)}}</span></div></div>
 <p class="note">Precision ${{(m.tp/(m.tp+m.fp)).toFixed(2)}} · Recall ${{(m.tp/(m.tp+m.fn)).toFixed(2)}}</p>`;$('cms').append(d);}});
// 2 PR curve
{{const W=640,H=520,L=64,R=20,T=16,B=50,s=svg(W,H,'Precision-recall curve'),X=v=>L+v*(W-L-R),Y=v=>T+(1-v)*(H-T-B);
 axes(s,{{L,R,T,B,W,H,X,Y,xt:[0,.25,.5,.75,1],yt:[0,.25,.5,.75,1],xl:'Recall',yl:'Precision',xf:q=>q,yf:q=>q}});
 [.4,.6,.8].forEach(f=>{{const xs=[],ys=[];for(let r=f/(2-f)+.001;r<=1;r+=.01){{xs.push(r);ys.push(f*r/(2*r-f));}}line(s,xs,ys,X,Y,'var(--muted)',1.2,'5 4');el(s,'text',{{x:X(1)-4,y:Y(f/(2-f))-6,'text-anchor':'end',class:'tick'}},'F1 = '+f);}});
 const xs=[],ys=[];cv.rec.forEach((r,i)=>{{if(cv.prec[i]>0){{xs.push(r);ys.push(cv.prec[i]);}}}});line(s,xs,ys,X,Y,MC,3.5);
 [[C.cm,MC,'model'],[C.rule,RC,'rule']].forEach(([m,c,n])=>{{const [r,p]=pt(m);el(s,'circle',{{cx:X(r),cy:Y(p),r:8,fill:c,stroke:'var(--surface)','stroke-width':2.5}});
  el(s,'text',{{x:X(r)+(n==='rule'?12:-12),y:Y(p)+(n==='rule'?18:-22),'text-anchor':n==='rule'?'start':'end',class:'axl'}},`${{n}} (R ${{r.toFixed(2)}}, P ${{p.toFixed(2)}})`);}});
 $('pr').append(s);}}
// 3 threshold curves (log x)
{{const W=900,H=380,L=64,R=24,T=16,B=50,s=svg(W,H,'Precision, recall and F1 by threshold'),lx=v=>Math.log10(Math.max(v,1e-6)),X=v=>L+(lx(v)+6)/6*(W-L-R),Y=v=>T+(1-v)*(H-T-B);
 axes(s,{{L,R,T,B,W,H,X,Y,xt:[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1],yt:[0,.25,.5,.75,1],xl:'Threshold on model score (log scale)',yl:'Value',xf:q=>q>=.01?String(q):q.toExponential(0),yf:q=>q}});
 const idx=cv.thr.map((t,i)=>i).filter(i=>cv.thr[i]>=1e-6);[['prec',PC],['rec',RC],['f1',MC]].forEach(([k,c])=>line(s,idx.map(i=>cv.thr[i]),idx.map(i=>cv[k][i]),X,Y,c,3));
 const th=Object.values(C.thresholds),lo=Math.min(...th),hi=Math.max(...th);el(s,'rect',{{x:X(lo),y:T,width:X(hi)-X(lo),height:H-T-B,fill:'var(--accent)',opacity:.12}});
 el(s,'text',{{x:X(lo)-6,y:T+14,'text-anchor':'end',class:'tick'}},'chosen thresholds (per Fire Centre, from 2022)');
 el(s,'line',{{x1:X(C.best.thr),x2:X(C.best.thr),y1:T,y2:H-B,stroke:MC,'stroke-dasharray':'4 4'}});el(s,'text',{{x:X(C.best.thr)-6,y:Y(C.best.f1)-10,'text-anchor':'end',class:'axl'}},'best F1 '+C.best.f1.toFixed(2));
 $('thr').append(s);}}
// 4 histogram (log y)
{{const W=900,H=380,L=74,R=20,T=16,B=60,s=svg(W,H,'Score histogram for fire and non-fire cell-days'),h=C.hist,n=h.pos.length,bw=(W-L-R)/n,ly=v=>v>0?Math.log10(v):0,ymax=8,Y=v=>T+(1-ly(v)/ymax)*(H-T-B),X=i=>L+i*bw;
 [0,2,4,6,8].forEach(q=>{{el(s,'line',{{x1:L,x2:W-R,y1:Y(10**q),y2:Y(10**q),class:'grid'}});el(s,'text',{{x:L-8,y:Y(10**q)+4,'text-anchor':'end',class:'tick'}},q===0?'1':'10'+'⁰¹²³⁴⁵⁶⁷⁸'[q]);}});
 h.pos.forEach((v,i)=>{{if(h.neg[i])el(s,'rect',{{x:X(i)+bw*.08,y:Y(h.neg[i]),width:bw*.4,height:Y(1)-Y(h.neg[i]),fill:MC,rx:2}});if(v)el(s,'rect',{{x:X(i)+bw*.52,y:Y(v),width:bw*.4,height:Y(1)-Y(v),fill:RC,rx:2}});}});
 [0,4,8,12,16,20,24].forEach(i=>{{const e=h.edges[i];el(s,'text',{{x:X(i),y:H-B+18,'text-anchor':'middle',class:'tick'}},e>=.01?String(+e.toFixed(2)):e.toExponential(0));}});
 el(s,'text',{{x:(L+W-R)/2,y:H-14,'text-anchor':'middle',class:'axl'}},'Model score (log scale)');el(s,'text',{{x:16,y:(T+H-B)/2,'text-anchor':'middle',class:'axl',transform:`rotate(-90 16 ${{(T+H-B)/2}})`}},'Cell-days (log scale)');
 $('hist').append(s);}}
// 5 calibration
{{const W=600,H=520,L=64,R=20,T=16,B=50,s=svg(W,H,'Calibration curve'),X=v=>L+v*(W-L-R),Y=v=>T+(1-v)*(H-T-B);
 axes(s,{{L,R,T,B,W,H,X,Y,xt:[0,.25,.5,.75,1],yt:[0,.25,.5,.75,1],xl:'Predicted probability',yl:'Observed share that burned',xf:q=>q,yf:q=>q}});
 line(s,[0,1],[0,1],X,Y,'var(--muted)',1.5,'6 5');line(s,C.cal.map(c=>c.pred),C.cal.map(c=>c.obs),X,Y,MC,3);
 C.cal.forEach(c=>el(s,'circle',{{cx:X(c.pred),cy:Y(c.obs),r:4+Math.min(10,Math.log10(c.n)),fill:MC,opacity:.85,stroke:'var(--surface)','stroke-width':1.5}}));
 $('cal').append(s);}}
// 6 ROC (log x)
{{const W=600,H=520,L=64,R=20,T=16,B=50,s=svg(W,H,'ROC curve'),lx=v=>Math.log10(Math.max(v,1e-7)),X=v=>L+(lx(v)+7)/7*(W-L-R),Y=v=>T+(1-v)*(H-T-B);
 axes(s,{{L,R,T,B,W,H,X,Y,xt:[1e-7,1e-5,1e-3,1e-1],yt:[0,.25,.5,.75,1],xl:'False-positive rate (log scale)',yl:'True-positive rate (recall)',xf:q=>q.toExponential(0),yf:q=>q}});
 const ix=cv.fpr.map((f,i)=>i).filter(i=>cv.fpr[i]>0);line(s,ix.map(i=>cv.fpr[i]),ix.map(i=>cv.rec[i]),X,Y,MC,3.5);
 const fr=C.rule.fp/(C.rule.fp+C.rule.tn),tr=C.rule.tp/(C.rule.tp+C.rule.fn);el(s,'circle',{{cx:X(fr),cy:Y(tr),r:8,fill:RC,stroke:'var(--surface)','stroke-width':2.5}});
 el(s,'text',{{x:X(fr)+12,y:Y(tr)+18,class:'axl'}},'rule');$('roc').append(s);}}
// 8 heatmap
{{const regs=[...new Set(D.heat.map(h=>h.region))],mons={{5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep'}};
 let h='<tr><th></th>'+Object.values(mons).map(m=>`<th>${{m}}</th>`).join('')+'</tr>';
 regs.forEach(r=>{{h+=`<tr><td class="rowh">${{r}}</td>`;Object.keys(mons).forEach(m=>{{const c=D.heat.find(x=>x.region===r&&x.month==m);
  if(!c||c.fires<20){{h+=`<td style="background:var(--surface-2);color:var(--muted)">–<small>${{c?c.fires:0}}</small></td>`;return;}}
  const a=Math.round(15+c.f1*75);h+=`<td style="background:color-mix(in srgb,#1f6f8b ${{a}}%,var(--surface));color:${{c.f1>.45?'#fff':'var(--fg)'}}">${{c.f1.toFixed(2)}}<small>${{fmt(c.fires)}}</small></td>`;}});h+='</tr>';}});
 $('heat').innerHTML=h;}}
// nav
const slides=[...document.querySelectorAll('.slide')];let cur=0;
slides.forEach((s,i)=>{{s.id='slide-'+(i+1);const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+s.dataset.section;$('jump').append(o);}});
function show(n){{cur=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.hidden=i!==cur);slides[cur].scrollTop=0;$('count').textContent=(cur+1)+' / '+slides.length;$('jump').value=cur;$('pbar').style.width=(cur+1)/slides.length*100+'%';$('prev').disabled=cur===0;$('next').disabled=cur===slides.length-1;try{{history.replaceState(null,'','#'+(cur+1));}}catch(e){{}}}}
$('prev').onclick=()=>show(cur-1);$('next').onclick=()=>show(cur+1);$('jump').onchange=e=>show(Number(e.target.value));
document.addEventListener('keydown',e=>{{if(e.target.closest('select,input,textarea,button,a'))return;if(['ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();show(cur+1);}}else if(['ArrowLeft','PageUp'].includes(e.key)){{e.preventDefault();show(cur-1);}}}});
$('theme').onclick=()=>{{const r=document.documentElement;const d=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=d?'light':'dark';}};
$('print').onclick=()=>window.print();const ini=Number(location.hash.slice(1));show(Number.isInteger(ini)&&ini>0?ini-1:0);
</script></body></html>'''
body = body.replace('@@DATA@@', json.dumps(data)).replace('@@IMG@@', img)
open(EXP / 'Week5_error_analysis.html', 'w', encoding='utf-8', newline='\n').write(head + '\n' + css + '</head><body>' + body)
print('written')
