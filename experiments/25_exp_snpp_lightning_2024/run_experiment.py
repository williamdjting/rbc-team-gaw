"""Separate S-NPP + lightning MLP: development first, frozen 2024 evaluation second.

No NOAA-20, weather, incidents or calendar inputs. Full-grid evaluation on the
existing six region grids; next-day gap-filled S-NPP activity, not ignition.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import h3
import netCDF4
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.sparse as sp
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
DATA = ROOT / "data/snpp_lightning_h3r7"
LIGHTNING = ROOT / "data/cancpld_lightning"
FIRMS = ROOT / "experiments/firms_summer_2018_2024/data/firms_h3_detections.csv"
GRIDS = ROOT / "experiments/02_hotspot_persistence_bc/regions"
REGIONS = ["Cariboo", "Coastal", "Kamloops", "Northwest", "Prince George", "Southeast"]
SEED, RATE, DAYS, START = 42, .01, 153, 7
FEATURES = ["local_count_1d", "local_count_3d", "local_count_7d", "neighbour_count_1d",
            "neighbour_count_3d", "neighbour_count_7d", "frp_sum_1d", "frp_sum_7d",
            "frp_max_1d", "night_count_1d", "active_days_7d", "disk1_active_days_7d",
            "days_since_detection", "history_left_censored", "quiet_stratum",
            "ltg_cell_1d", "ltg_3x3_1d", "ltg_3x3_3d", "ltg_9x9_1d",
            "ltg_9x9_3d", "ltg_days_since_3x3"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rolling(a, width):
    cs = np.concatenate([np.zeros((*a.shape[:-1], 1)), np.cumsum(a, axis=-1, dtype=np.float64)], axis=-1)
    end = np.arange(a.shape[-1]) + 1
    return (cs[..., end] - cs[..., np.maximum(0, end - width)]).astype(np.float32)


def filled(detected):
    result = detected.copy()
    result[:, 1:-1] |= detected[:, :-2] & detected[:, 2:]
    return result


def lightning_daily(year):
    path = LIGHTNING / f"bc_cancpld_cg_daily_utc_{year}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    raw = LIGHTNING / "raw" / f"cg_flashes_3hr_0.1-deg_{year}.nc"
    with netCDF4.Dataset(raw) as d:
        lat, lon = np.asarray(d['lat'][:]), np.asarray(d['lon'][:])
        lon = np.where(lon > 180, lon - 360, lon)
        iy = np.flatnonzero((lat >= 47) & (lat <= 62))
        ix = np.flatnonzero((lon >= -141) & (lon <= -112))
        tv = d['time']
        dates = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
        selected = np.array([i for i, dt in enumerate(dates) if 5 <= dt.month <= 9])
        assert len(selected) == DAYS * 8
        assert all(dates[i].hour == (j % 8) * 3 for j, i in enumerate(selected))
        assert all(dates[i].year == year for i in selected)
        array = d['cg_flashes'][selected[0]:selected[-1]+1, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
        array = np.ma.asarray(array, dtype=np.float32).filled(np.nan)
        assert not np.any(array[np.isfinite(array)] < 0)
        daily = array.reshape(DAYS, 8, len(iy), len(ix)).sum(axis=1)
        # A masked/unobserved 3-hour interval makes the entire cell-day unknown.
        result = dict(daily=daily, lat=lat[iy], lon=lon[ix])
        np.savez_compressed(path, **result)
    print(f"Built UTC daily lightning {year}", flush=True)
    return result


def grid(region):
    path = GRIDS / region.lower().replace(' ', '_') / "data/h3_grid.parquet"
    frame = pd.read_parquet(path)
    cells = pd.Index(frame.h3_cell)
    assert cells.is_unique and frame.resolution.eq(7).all()
    pos = {c: i for i, c in enumerate(cells)}
    rows, cols = [], []
    for i, cell in enumerate(cells):
        for neighbour in h3.grid_disk(cell, 1):
            if neighbour in pos:
                rows.append(i); cols.append(pos[neighbour])
    disk = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(cells), len(cells)))
    return frame, cells, disk


def source(years):
    frame = pd.read_csv(FIRMS, usecols=['acq_date','h3_cell','frp','daynight','satellite'])
    frame = frame.loc[frame.acq_date.str[:4].astype(int).isin(years)].copy()
    assert frame.satellite.eq('SNPP').all()
    frame['date'] = pd.to_datetime(frame.acq_date)
    assert frame.date.dt.month.between(5, 9).all()
    frame['year'] = frame.date.dt.year
    frame['day'] = (frame.date - pd.to_datetime(frame.year.astype(str) + '-05-01')).dt.days
    return frame


def build(region, year, detections):
    geo, cells, disk = grid(region)
    n = len(cells)
    rows = detections.loc[detections.year.eq(year)]
    index = cells.get_indexer(rows.h3_cell)
    keep = index >= 0
    index, day = index[keep], rows.day.to_numpy()[keep]
    count = np.zeros((n, DAYS), np.float32)
    frp, maximum, night = count.copy(), count.copy(), count.copy()
    power = rows.frp.to_numpy(np.float32)[keep]
    assert np.isfinite(power).all()
    np.add.at(count, (index, day), 1)
    np.add.at(frp, (index, day), power)
    np.maximum.at(maximum, (index, day), power)
    np.add.at(night, (index, day), rows.daynight.eq('N').to_numpy(np.float32)[keep])
    neighbour = disk @ count - count
    local7, near7 = rolling(count, 7), rolling(neighbour, 7)
    age = np.zeros_like(count)
    seen = np.full(n, -1)
    for t in range(DAYS):
        seen[count[:, t] > 0] = t
        age[:, t] = np.where(seen >= 0, t - seen, DAYS)
    arrays = [count, rolling(count, 3), local7, neighbour, rolling(neighbour, 3), near7,
              frp, rolling(frp, 7), maximum, night, rolling(count > 0, 7),
              rolling((count + neighbour) > 0, 7), age, (age == DAYS).astype(np.float32),
              ((local7 == 0) & (near7 == 0)).astype(np.float32)]
    ltg = lightning_daily(year)
    lat, lon, daily = ltg['lat'], ltg['lon'], ltg['daily']
    iy = np.abs(geo.center_latitude.to_numpy()[:, None] - lat).argmin(axis=1)
    ix = np.abs(geo.center_longitude.to_numpy()[:, None] - lon).argmin(axis=1)
    assert np.max(np.abs(geo.center_latitude.to_numpy() - lat[iy])) < .051
    assert np.max(np.abs(geo.center_longitude.to_numpy() - lon[ix])) < .051
    pick = lambda a: a[:, iy, ix].T.copy()
    blocks = []
    for width in (3, 9):
        sums = ndi.uniform_filter(np.nan_to_num(daily), size=(1,width,width), mode='constant') * width**2
        valid = ndi.uniform_filter(np.isfinite(daily).astype(np.float32), size=(1,width,width), mode='constant')
        sums[valid < .99999] = np.nan
        blocks.append(pick(sums))
    small, big = blocks
    def valid_rolling(a):
        out = rolling(np.nan_to_num(a), 3)
        out[rolling(~np.isfinite(a), 3) > 0] = np.nan
        return out
    since = np.zeros_like(small)
    counter = np.full(n, 30, np.float32)
    for t in range(DAYS):
        counter = np.minimum(counter + 1, 30)
        counter[small[:, t] > 0] = 0
        counter[~np.isfinite(small[:, t])] = 30
        since[:, t] = np.where(np.isfinite(small[:, t]), counter, np.nan)
    arrays += [pick(daily), small, valid_rolling(small), big, valid_rolling(big), since]
    # Target index t uses only the history ending at t-1.
    features = [a[:, START-1:-1] for a in arrays]
    truth = filled(count > 0)[:, START:]
    baseline = count[:, START-1:-1] > 0
    assert all(a.shape == truth.shape for a in features)
    return cells, disk, features, truth, baseline


class Scaler:
    def fit(self, X):
        take = np.random.default_rng(SEED).choice(len(X), min(len(X), 200000), replace=False)
        values = X[take]
        self.log = np.nanquantile(np.abs(values), .999, axis=0) > 10
        values = self.logged(values)
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0) + 1e-6
        assert np.isfinite(self.mean).all() and np.isfinite(self.std).all()
        return self

    def logged(self, X):
        X = X.copy()
        X[:, self.log] = np.sign(X[:, self.log]) * np.log1p(np.abs(X[:, self.log]))
        return X

    def __call__(self, X):
        Z = (self.logged(X) - self.mean) / self.std
        return np.column_stack([np.clip(np.nan_to_num(Z), -8, 8), ~np.isfinite(X)]).astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURES)*2,256),nn.SiLU(),nn.Dropout(.1),
                                    nn.Linear(256,256),nn.SiLU(),nn.Dropout(.1),
                                    nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self, X):
        return self.layers(X).squeeze(1)


def train():
    data = source(range(2018, 2022))
    parts, labels, days, statistics = [], [], [], []
    for ri, region in enumerate(REGIONS):
        for year in range(2018, 2022):
            cells, _, features, y, _ = build(region, year, data)
            quiet = features[FEATURES.index('quiet_stratum')].astype(bool)
            draw = np.random.default_rng([SEED, year, ri]).random(y.shape) < RATE
            keep = ~quiet | y | draw
            ci, ti = np.where(keep)
            X = np.column_stack([a[keep] for a in features])
            parts.append(X); labels.append(y[keep]); days.append(ti + START - 1)
            sample = pd.DataFrame(X, columns=FEATURES)
            sample['h3_cell'] = cells.to_numpy()[ci]
            sample['target_date'] = pd.Timestamp(f'{year}-05-01') + pd.to_timedelta(ti + START, unit='D')
            sample['target_snpp_filled'] = y[keep]
            sample['sample_weight'] = np.where(quiet[keep] & ~y[keep], 1/RATE, 1)
            folder = DATA / 'training_samples' / region.lower().replace(' ','_')
            folder.mkdir(parents=True, exist_ok=True)
            sample.to_parquet(folder / f'snpp_lightning_{year}.parquet', index=False)
            statistics.append(dict(region=region, year=year, rows=int(y.size), kept=int(keep.sum()), positives=int(y.sum())))
            print(f'Training join {region} {year}: {len(X):,} rows', flush=True)
            del sample, features, y, quiet, draw, keep
            gc.collect()
    X, y, day = np.concatenate(parts), np.concatenate(labels), np.concatenate(days)
    del parts, labels, days, data
    valid = (day // 7) % 6 == 0
    scaler = Scaler().fit(X[~valid])
    Z = torch.from_numpy(scaler(X)); target = torch.from_numpy(y.astype(np.float32))
    del X
    fit_rows, valid_rows = np.flatnonzero(~valid), np.flatnonzero(valid)
    model = Net(); optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best, best_epoch, curves = float('inf'), 0, []
    for epoch in range(1, 21):
        started = time.time(); model.train(); total = 0.
        shuffled = np.random.default_rng([SEED,epoch]).permutation(fit_rows)
        for start in range(0, len(shuffled), 2048):
            idx = shuffled[start:start+2048]
            optimizer.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(Z[idx]), target[idx])
            loss.backward(); optimizer.step(); total += loss.item()*len(idx)
        model.eval(); validation = 0.
        with torch.no_grad():
            for start in range(0,len(valid_rows),65536):
                idx = valid_rows[start:start+65536]
                validation += nn.functional.binary_cross_entropy_with_logits(model(Z[idx]),target[idx],reduction='sum').item()
        validation /= len(valid_rows)
        curves.append(dict(epoch=epoch,train_loss=total/len(fit_rows),validation_loss=validation))
        print(f'Epoch {epoch}: validation={validation:.6f}, {time.time()-started:.0f}s',flush=True)
        if validation < best:
            best, best_epoch, state = validation, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= 3:
            break
    model.load_state_dict(state)
    torch.save(state,RESULTS/'mlp_snpp_lightning.pt')
    np.savez(RESULTS/'scaler.npz',log=scaler.log,mean=scaler.mean,std=scaler.std)
    pd.DataFrame(curves).to_csv(RESULTS/'training_curves.csv',index=False)
    pd.DataFrame(statistics).to_csv(RESULTS/'training_sample_counts.csv',index=False)
    save_json(RESULTS/'fit.json',dict(best_epoch=best_epoch,fit_rows=len(fit_rows),validation_rows=len(valid_rows),seed=SEED))
    return model, scaler


def predict(model, scaler, arrays):
    shape = arrays[0].shape; n = arrays[0].size
    scores = np.empty(n,np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0,n,65536):
            stop=min(n,start+65536)
            X=np.column_stack([a.reshape(-1)[start:stop] for a in arrays])
            z=model(torch.from_numpy(scaler(X))).numpy()
            scores[start:stop]=expit(z+np.log(RATE)*X[:,FEATURES.index('quiet_stratum')])
    assert np.isfinite(scores).all()
    return scores.reshape(shape)


def counts(y, flag):
    return np.column_stack([(flag & y).sum(0),(flag & ~y).sum(0),(~flag & y).sum(0),(~flag & ~y).sum(0)])


def metrics(total):
    tp,fp,fn,tn=map(int,total)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))


def evaluate(model,scaler,year,thresholds,select=False):
    data=source([year]); rows=[]; daily={}
    for region in REGIONS:
        started=time.time()
        cells,disk,arrays,y,baseline=build(region,year,data)
        score=predict(model,scaler,arrays)
        if select:
            p,r,t=precision_recall_curve(y.ravel(),score.ravel())
            f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-20)
            thresholds[region]=float(t[np.argmax(f)])
        for name,flag,s in [('mlp_snpp_lightning',score>=thresholds[region],score),('snpp_persistence',baseline,baseline)]:
            c=counts(y,flag);daily[region,name]=c
            row=dict(region=region,year=year,model=name,threshold=thresholds[region] if name.startswith('mlp') else .5,
                     rows=int(y.size),average_precision=float(average_precision_score(y.ravel(),s.ravel())),**metrics(c.sum(0)))
            rows.append(row)
        # Save a real full-grid joined table per region/year, including quiet negatives.
        folder=DATA/'evaluation'/str(year);folder.mkdir(parents=True,exist_ok=True)
        table=pd.DataFrame({name:a.reshape(-1) for name,a in zip(FEATURES,arrays)})
        table['h3_cell']=np.repeat(cells.to_numpy(),DAYS-START)
        table['target_date']=np.tile(pd.date_range(f'{year}-05-08',f'{year}-09-30').to_numpy(),len(cells))
        table['target_snpp_filled']=y.ravel();table['mlp_score']=score.ravel()
        table['snpp_persistence']=baseline.ravel()
        table.to_parquet(folder/f'{region.lower().replace(" ","_")}_snpp_lightning_full_grid.parquet',index=False)
        print(f'{year} {region}: MLP F1={rows[-2]["f1"]:.4f}, persistence={rows[-1]["f1"]:.4f} ({time.time()-started:.0f}s)',flush=True)
        del arrays,score,y,baseline,table
        gc.collect()
    frame=pd.DataFrame(rows);frame.to_csv(RESULTS/f'metrics_{year}.csv',index=False)
    pooled={name:sum(daily[r,name] for r in REGIONS) for name in ('mlp_snpp_lightning','snpp_persistence')}
    summary=[dict(year=year,model=name,median_region_ap=float(frame.loc[frame.model.eq(name),'average_precision'].median()),**metrics(c.sum(0))) for name,c in pooled.items()]
    pd.DataFrame(summary).to_csv(RESULTS/f'summary_{year}.csv',index=False)
    dates=pd.date_range(f'{year}-05-08',f'{year}-09-30');weeks=dates.isocalendar().week.to_numpy()
    unique=np.unique(weeks);member=(weeks[None,:]==unique[:,None]).astype(np.float64)
    draws=np.random.default_rng(SEED).integers(0,len(unique),size=(2000,len(unique)))
    resample=np.array([np.bincount(d,minlength=len(unique)) for d in draws])
    def boot(c):
        tp,fp,fn,_=(resample @ member @ c).T
        return 2*tp/np.maximum(2*tp+fp+fn,1)
    difference=boot(pooled['mlp_snpp_lightning'])-boot(pooled['snpp_persistence'])
    save_json(RESULTS/f'bootstrap_{year}.json',dict(f1_difference=summary[0]['f1']-summary[1]['f1'],
              ci_low=float(np.quantile(difference,.025)),ci_high=float(np.quantile(difference,.975)),resamples=2000))
    np.savez(RESULTS/f'daily_counts_{year}.npz',**{r.replace(' ','_')+'__'+m:c for (r,m),c in daily.items()})
    print(json.dumps(summary,indent=2),flush=True)


def self_check():
    toy=np.array([[0,1,0,1,0,0,0,2,0]],np.float32)
    for w in (1,3,7):
        expect=np.array([[toy[0,max(0,t-w+1):t+1].sum() for t in range(toy.shape[1])]])
        np.testing.assert_array_equal(rolling(toy,w),expect)
    original=rolling(toy,3)[:,:7]
    altered=toy.copy();altered[:,7:]=999
    np.testing.assert_array_equal(original,rolling(altered,3)[:,:7])
    np.testing.assert_array_equal(filled(toy>0)[0,:4],[False,True,True,True])
    y=np.array([[True,False],[False,True]]);flag=np.array([[True,True],[False,False]])
    assert metrics(counts(y,flag).sum(0))['f1']==.5
    print('Temporal window, label and confusion-count checks passed.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','develop','test']);args=parser.parse_args()
    self_check()
    if args.stage=='check':return
    torch.manual_seed(SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    frozen=RESULTS/'frozen_before_2024.json'
    if args.stage=='develop':
        if frozen.exists():raise SystemExit('Already frozen. Do not retrain after holdout.')
        model,scaler=train();thresholds={}
        evaluate(model,scaler,2022,thresholds,select=True)
        evaluate(model,scaler,2023,thresholds)
        paths=[Path(__file__),FIRMS,RESULTS/'mlp_snpp_lightning.pt',RESULTS/'scaler.npz']
        paths+=list(GRIDS.glob('*/data/h3_grid.parquet'))
        paths+=list((LIGHTNING/'raw').glob('*.nc'))
        save_json(frozen,dict(features=FEATURES,thresholds=thresholds,train_years=[2018,2019,2020,2021],
                  threshold_year=2022,development_year=2023,test_year=2024,
                  primary_metric='pooled exact-cell F1',target='S-NPP detection OR detections on both adjacent days',
                  source_hashes={str(p.resolve().relative_to(ROOT)):digest(p) for p in paths},
                  utc_frozen=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),holdout_evaluated=False))
        print('Frozen before any 2024 feature building or scoring.',flush=True)
    else:
        if (RESULTS/'summary_2024.csv').exists():raise SystemExit('2024 already evaluated; do not repeat or tune.')
        protocol=json.loads(frozen.read_text())
        for path,expected in protocol['source_hashes'].items():
            assert digest(ROOT/path)==expected, f'Frozen artifact changed: {path}'
        model=Net();model.load_state_dict(torch.load(RESULTS/'mlp_snpp_lightning.pt',weights_only=True))
        scaler=Scaler()
        with np.load(RESULTS/'scaler.npz') as z:
            scaler.log=z['log'];scaler.mean=z['mean'];scaler.std=z['std']
        evaluate(model,scaler,2024,protocol['thresholds'])
        save_json(RESULTS/'test_completion.json',dict(year=2024,protocol_sha256=digest(frozen),
                  model_sha256=digest(RESULTS/'mlp_snpp_lightning.pt'),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':main()
