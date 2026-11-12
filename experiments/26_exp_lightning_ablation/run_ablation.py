"""Retrain matched no-lightning MLP; compare against unchanged Experiment 25."""
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent/'25_exp_snpp_lightning_2024'
OUT = HERE/'results'
DATA = ROOT/'data/snpp_lightning_h3r7'
spec = importlib.util.spec_from_file_location('snpp_lightning_reference', PREVIOUS/'run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
KEEP = [i for i, name in enumerate(base.FEATURES) if not name.startswith('ltg_')]
DROP = [i for i, name in enumerate(base.FEATURES) if name.startswith('ltg_')]
MASK = DROP + [len(base.FEATURES)+i for i in DROP]


def transform(scaler, X):
    Z = scaler(X)
    Z[:, MASK] = 0
    assert not Z[:, MASK].any()
    return Z


def load_scaler():
    scaler = base.Scaler()
    with np.load(PREVIOUS/'results/scaler.npz') as z:
        scaler.log=z['log'];scaler.mean=z['mean'];scaler.std=z['std']
    return scaler


def train(scaler):
    parts, labels, days = [], [], []
    for region in base.REGIONS:
        slug=region.lower().replace(' ','_')
        for year in range(2018,2022):
            frame=pd.read_parquet(DATA/'training_samples'/slug/f'snpp_lightning_{year}.parquet')
            parts.append(frame[base.FEATURES].to_numpy(np.float32))
            labels.append(frame.target_snpp_filled.to_numpy(np.float32))
            days.append((frame.target_date-pd.Timestamp(f'{year}-05-01')).dt.days.to_numpy()-1)
    X=np.concatenate(parts); y=np.concatenate(labels);day=np.concatenate(days)
    del parts,labels,days,frame
    valid=(day//7)%6==0
    fit_rows,valid_rows=np.flatnonzero(~valid),np.flatnonzero(valid)
    reference=json.loads((PREVIOUS/'results/fit.json').read_text())
    assert len(fit_rows)==reference['fit_rows'] and len(valid_rows)==reference['validation_rows']
    # Independently refit to confirm the retained-feature scaler is identical.
    checked=base.Scaler().fit(X[~valid])
    for name in ('log','mean','std'):
        np.testing.assert_array_equal(getattr(checked,name)[KEEP],getattr(scaler,name)[KEEP])
    # Changing every lightning value must not change the ablated network input.
    probe=X[:100].copy(); changed=probe.copy();changed[:,DROP]=12345
    np.testing.assert_array_equal(transform(scaler,probe),transform(scaler,changed))
    Z=torch.from_numpy(transform(scaler,X));target=torch.from_numpy(y)
    del X
    model=base.Net();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best,best_epoch,curves=float('inf'),0,[]
    for epoch in range(1,21):
        start_time=time.time();model.train();total=0.
        order=np.random.default_rng([base.SEED,epoch]).permutation(fit_rows)
        for start in range(0,len(order),2048):
            idx=order[start:start+2048]
            optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(Z[idx]),target[idx])
            loss.backward();optimizer.step();total+=loss.item()*len(idx)
        model.eval();validation=0.
        with torch.no_grad():
            for start in range(0,len(valid_rows),65536):
                idx=valid_rows[start:start+65536]
                validation+=nn.functional.binary_cross_entropy_with_logits(model(Z[idx]),target[idx],reduction='sum').item()
        validation/=len(valid_rows)
        curves.append(dict(epoch=epoch,train_loss=total/len(fit_rows),validation_loss=validation))
        print(f'Epoch {epoch}: validation={validation:.6f}, {time.time()-start_time:.0f}s',flush=True)
        if validation<best:
            best,best_epoch,state=validation,epoch,copy.deepcopy(model.state_dict())
        elif epoch-best_epoch>=3:break
    model.load_state_dict(state)
    torch.save(state,OUT/'mlp_without_lightning.pt')
    pd.DataFrame(curves).to_csv(OUT/'training_curves.csv',index=False)
    base.save_json(OUT/'fit.json',dict(best_epoch=best_epoch,fit_rows=len(fit_rows),validation_rows=len(valid_rows),
                                    seed=base.SEED,masked_features=[base.FEATURES[i] for i in DROP],
                                    retained_scaler_verified=True,lightning_invariance_check_passed=True))
    return model


def score(model,scaler,region,year):
    slug=region.lower().replace(' ','_')
    path=DATA/'evaluation'/str(year)/f'{slug}_snpp_lightning_full_grid.parquet'
    parquet=pq.ParquetFile(path)
    names=[base.FEATURES[i] for i in KEEP]+['target_snpp_filled','mlp_score']
    scores=np.empty(parquet.metadata.num_rows,np.float32)
    reference=np.empty_like(scores);truth=np.empty(len(scores),bool)
    offset=0;model.eval()
    with torch.no_grad():
        for batch in parquet.iter_batches(batch_size=65536,columns=names):
            X=np.zeros((len(batch),len(base.FEATURES)),np.float32)
            for j,i in enumerate(KEEP):X[:,i]=batch.column(j).to_numpy(zero_copy_only=False)
            z=model(torch.from_numpy(transform(scaler,X))).numpy()
            part=slice(offset,offset+len(batch))
            scores[part]=expit(z+np.log(base.RATE)*X[:,base.FEATURES.index('quiet_stratum')])
            truth[part]=batch.column(len(KEEP)).to_numpy(zero_copy_only=False)
            reference[part]=batch.column(len(KEEP)+1).to_numpy(zero_copy_only=False)
            offset+=len(batch)
    assert offset==len(scores) and np.isfinite(scores).all()
    np.save(OUT/f'scores_without_lightning_{slug}_{year}.npy',scores)
    return scores.reshape(-1,146),reference.reshape(-1,146),truth.reshape(-1,146)


def evaluate(model,scaler,year,thresholds,reference_thresholds,select=False):
    rows=[];daily={}
    old=pd.read_csv(PREVIOUS/f'results/metrics_{year}.csv')
    for region in base.REGIONS:
        start=time.time();scores,reference,y=score(model,scaler,region,year)
        if select:
            p,r,t=precision_recall_curve(y.ravel(),scores.ravel())
            f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-20)
            thresholds[region]=float(t[np.argmax(f)])
        for name,s,threshold in [('without_lightning',scores,thresholds[region]),
                                 ('with_lightning',reference,reference_thresholds[region])]:
            c=base.counts(y,s>=threshold);daily[region,name]=c
            row=dict(year=year,region=region,model=name,threshold=threshold,rows=int(y.size),
                     average_precision=float(average_precision_score(y.ravel(),s.ravel())),**base.metrics(c.sum(0)))
            rows.append(row)
            if name=='with_lightning':
                expected=old.loc[old.region.eq(region)&old.model.eq('mlp_snpp_lightning')].iloc[0]
                for key in ('tp','fp','fn','tn'):assert row[key]==int(expected[key])
                assert np.isclose(row['average_precision'],expected.average_precision,rtol=1e-12)
        print(f'{year} {region}: without={rows[-2]["f1"]:.4f}, with={rows[-1]["f1"]:.4f} ({time.time()-start:.0f}s)',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/f'metrics_{year}.csv',index=False)
    pooled={m:sum(daily[r,m] for r in base.REGIONS) for m in ('without_lightning','with_lightning')}
    summary=[dict(year=year,model=m,median_region_ap=float(frame.loc[frame.model.eq(m),'average_precision'].median()),
                  **base.metrics(c.sum(0))) for m,c in pooled.items()]
    pd.DataFrame(summary).to_csv(OUT/f'summary_{year}.csv',index=False)
    dates=pd.date_range(f'{year}-05-08',f'{year}-09-30');weeks=dates.isocalendar().week.to_numpy()
    unique=np.unique(weeks);member=(weeks[None,:]==unique[:,None]).astype(float)
    draws=np.random.default_rng(base.SEED).integers(0,len(unique),size=(2000,len(unique)))
    resample=np.array([np.bincount(d,minlength=len(unique)) for d in draws])
    def boot(c):
        tp,fp,fn,_=(resample@member@c).T
        return 2*tp/np.maximum(2*tp+fp+fn,1)
    comparisons=[]
    for scope in [*base.REGIONS,'BC-wide']:
        get=(lambda m:pooled[m]) if scope=='BC-wide' else (lambda m:daily[scope,m])
        diff=boot(get('with_lightning'))-boot(get('without_lightning'))
        comparisons.append(dict(scope=scope,year=year,
              f1_difference=base.metrics(get('with_lightning').sum(0))['f1']-base.metrics(get('without_lightning').sum(0))['f1'],
              ci_low=float(np.quantile(diff,.025)),ci_high=float(np.quantile(diff,.975))))
    pd.DataFrame(comparisons).to_csv(OUT/f'bootstrap_{year}.csv',index=False)
    np.savez(OUT/f'daily_counts_{year}.npz',**{r.replace(' ','_')+'__'+m:c for (r,m),c in daily.items()})
    print(json.dumps(summary,indent=2),flush=True)


def main():
    if (OUT/'frozen_before_followup_evaluation.json').exists():
        raise SystemExit('Ablation already frozen; refuse automatic retraining/reselection.')
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(base.SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    prior=json.loads((PREVIOUS/'results/frozen_before_2024.json').read_text())
    for path,expected in prior['source_hashes'].items():assert base.digest(ROOT/path)==expected,path
    paths=[Path(__file__),HERE/'README.md',PREVIOUS/'results/scaler.npz',PREVIOUS/'results/mlp_snpp_lightning.pt',
           PREVIOUS/'results/frozen_before_2024.json']
    paths+=sorted(DATA.rglob('*.parquet'))
    hashes={str(p.resolve().relative_to(ROOT)):base.digest(p) for p in paths}
    scaler=load_scaler();model=train(scaler);thresholds={}
    evaluate(model,scaler,2022,thresholds,prior['thresholds'],select=True)
    base.save_json(OUT/'frozen_before_followup_evaluation.json',dict(
        thresholds=thresholds,reference_thresholds=prior['thresholds'],source_hashes=hashes,
        model_sha256=base.digest(OUT/'mlp_without_lightning.pt'),
        previously_seen_2024=True,seed=base.SEED,utc_frozen=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    for year in (2023,2024):evaluate(model,scaler,year,thresholds,prior['thresholds'])
    for path,expected in hashes.items():assert base.digest(ROOT/path)==expected,path
    base.save_json(OUT/'completion.json',dict(status='complete',source_hashes_unchanged=True,
        prior_results_reproduced=True,previously_seen_2024=True,
        utc_completed=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':main()
