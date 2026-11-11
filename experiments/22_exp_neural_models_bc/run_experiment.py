"""Deep learning models against CatBoost on Experiment 21's inputs and label.

Everything except the model follows Experiment 21: the S-NPP or NOAA-20 gap-filled label (2018-2020
train on gap-filled S-NPP), the same training draw (every active row, every positive, 1% of quiet
negatives, log(0.01) added to quiet-row log-odds), train 2018-2021, thresholds from 2022 (per Fire
Centre) or the two-satellite rule's daily volume, evaluate 2023, week-block bootstrap against
"S-NPP or NOAA-20 detection yesterday". Every model sees Experiment 21's catboost_spread_noday
inputs (no day_of_year). 2024 is never loaded.

Models:
- catboost_spread_noday: Experiment 21's model, loaded (not retrained); the reference to beat. It
  must reproduce Experiment 21.
- mlp: three hidden layers (256, 256, 128) on the same inputs. Heavy-tailed inputs get a signed
  log, all are standardized with training-sample statistics, and inputs with blanks (NOAA-20
  before 2021, weather beyond 100 km) get a missing flag.
- gru: the mlp plus a GRU (64 units) over the 14 history days ending on the history day. Each day
  has 25 channels per cell: S-NPP count in the cell, ring 1 and ring 2, FRP, night detections,
  NOAA-20 in the cell and disk 1, two-satellite disk 1 (blank before 2021, with an availability
  flag), the ten station weather values and a missing flag, lightning in the 3x3 and 9x9 blocks,
  upwind fire in rings 1 and 2, and an in-season flag (days before May 1 are zero).
Neural models train with AdamW on binary cross-entropy, holding out one 7-day block of the season in
six from the training sample to pick the epoch (best validation log loss, patience 3).
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_16", EXPERIMENTS / "21_exp_weather_lightning_spread_bc/run_experiment.py")
experiment_16 = sys.modules["experiment_16"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_16)
from experiment_16 import (Drivers, FULL_NOAA20_FROM, LIGHTNING, STATIONS, WEATHER_DAILY, breakdown, build,  # noqa: E402
                           volume_flags)
from experiment_15 import LABEL, RULES, DEFINITIONS, Labels, to_grid  # noqa: E402
from experiment_14 import CWFIS, FIRMS, shift  # noqa: E402
from experiment_13 import BOOTSTRAP, daily_counts, f1_from_counts  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve  # noqa: E402
from experiment_08 import (Context, FEATURE_SETS, QUIET, QUIET_SAMPLE_RATE, REFERENCE, SEASON_DAYS, SEED,  # noqa: E402
                           SELECTION_YEAR, EVALUATION_YEAR, TRAIN_YEARS, catboost_model, design, digest, load, predict,
                           quiet_stratum, season_index, sparse_row_max)

EXPERIMENT_16 = EXPERIMENTS / "21_exp_weather_lightning_spread_bc/results"
REFERENCE_MODEL = EXPERIMENT_16 / "models/catboost_spread_noday.cbm"
FEATURE_SET = "spread_noday"
MODELS = ("catboost_spread_noday", "mlp", "gru")
NEURAL = ("mlp", "gru")
QUIET_COLUMNS = [c for c in QUIET if c not in FEATURE_SETS[FEATURE_SET]]  # design() needs them
METHODS = ("f1_2022", "rule_volume")
CHUNK = 500_000

SEQUENCE_DAYS = 14
WEATHER_SCALE = {"wx_temp": 30, "wx_rh": 100, "wx_wind": 30, "wx_ffmc": 100, "wx_dmc": 100, "wx_dc": 800,
                 "wx_bui": 150, "wx_isi": 20, "wx_fwi": 40}
CHANNELS = ["snpp_cell", "snpp_ring1", "snpp_ring2", "frp_cell", "night_cell", "n20_cell", "n20_disk1", "pair_disk1",
            "n20_available", *WEATHER_SCALE, "wx_precip_24h", "wx_missing", "ltg_3x3_1d", "ltg_9x9_1d",
            "upwind_ring1_1d", "upwind_ring2_1d", "in_season"]
HIDDEN, GRU_HIDDEN, DROPOUT = (256, 256, 128), 64, 0.1
BATCH, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE = 2048, 1e-3, 1e-4, 20, 3
VALID_BLOCK, VALID_EVERY = 7, 6  # hold out season days whose 7-day block is a multiple of 6
LOG_ABOVE, CLIP = 10.0, 8.0
SCALER_ROWS = 200_000  # statistics from a subsample; transform in chunks of this size
PREDICT_BATCH = 65_536


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------- inputs

class Sequences:
    """Per region-year cube (cells x padded days x channels), history day h at index h + SEQUENCE_DAYS - 1."""

    def __init__(self, context, labels, drivers):
        self.context, self.labels, self.drivers = context, labels, drivers
        self._key, self._cube = None, None

    def cube(self, region, year):
        if self._key == (region, year):
            return self._cube
        self._key, self._cube = None, None
        g, a, lab = self.context.grids[region], self.context.arrays[year], self.labels.grids[region, year]
        d = self.drivers.arrays(region, year, lab["pair"] if year >= FULL_NOAA20_FROM else lab["snpp"])
        rings, disk1 = g["rings"], lab["disk1"]
        available = float(year >= FULL_NOAA20_FROM)  # NOAA-20 is blank before 2021, as in Experiment 21
        n20, pair = lab["n20"].astype(np.float32) * available, lab["pair"].astype(np.float32) * available
        channels = [lambda: np.log1p(rings[0] @ a["count"]), lambda: np.log1p(rings[1] @ a["count"]),
                    lambda: np.log1p(rings[2] @ a["count"]), lambda: np.log1p(rings[0] @ a["frp_sum"]),
                    lambda: np.log1p(rings[0] @ a["night"]), lambda: n20, lambda: np.log1p(disk1 @ n20),
                    lambda: np.log1p(disk1 @ pair), lambda: np.full(n20.shape, available),
                    *[lambda c=c, s=s: d[c] / s for c, s in WEATHER_SCALE.items()],
                    lambda: np.log1p(d["wx_precip_24h"]), lambda: np.isnan(d["wx_temp"]),
                    lambda: np.log1p(d["ltg_3x3_1d"]), lambda: np.log1p(d["ltg_9x9_1d"]),
                    lambda: signed_log(d["upwind_ring1_1d"]), lambda: signed_log(d["upwind_ring2_1d"]),
                    lambda: np.ones(n20.shape)]
        assert len(channels) == len(CHANNELS)
        cube = np.zeros((len(g["cells"]), SEQUENCE_DAYS - 1 + SEASON_DAYS, len(CHANNELS)), np.float16)
        for k, make in enumerate(channels):
            cube[:, SEQUENCE_DAYS - 1:, k] = np.nan_to_num(np.asarray(make(), np.float32))
        self._key, self._cube = (region, year), cube
        return cube


def gather(cube, cell, history):
    """(rows x SEQUENCE_DAYS x channels) float32, oldest day first."""
    return cube[cell[:, None], history[:, None] + np.arange(SEQUENCE_DAYS)].astype(np.float32)


class Scaler:
    """Signed log for heavy-tailed inputs, standardize, blanks to 0 plus a missing flag."""

    def __init__(self, X):
        fit = X[np.random.default_rng(SEED).choice(len(X), min(len(X), SCALER_ROWS), replace=False)]
        self.log = np.nanquantile(np.abs(fit), 0.999, axis=0) > LOG_ABOVE
        Z = self._log(fit)
        self.mean, self.std = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0) + 1e-6
        self.missing = np.isnan(X).any(axis=0)

    def _log(self, X):
        X = X.copy()
        X[:, self.log] = signed_log(X[:, self.log])
        return X

    def __call__(self, X):
        out = np.empty((len(X), len(self.mean) + int(self.missing.sum())), np.float32)
        for start in range(0, len(X), SCALER_ROWS):
            Z = (self._log(X[start:start + SCALER_ROWS]) - self.mean) / self.std
            out[start:start + len(Z)] = np.hstack([np.clip(np.nan_to_num(Z), -CLIP, CLIP), np.isnan(Z[:, self.missing])])
        return out

    def describe(self, columns):
        return {c: {"signed_log": bool(l), "mean": float(m), "std": float(s), "missing_flag": bool(f)}
                for c, l, m, s, f in zip(columns, self.log, self.mean, self.std, self.missing)}


def training_sample(regions, all_regions, context, labels, drivers, sequences):
    """Experiment 21's draw, plus each row's history sequence."""
    tables, seqs, stats = [], [], []
    for region in regions:
        index = all_regions.index(region)
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            positive = labels.grids[region, year]["label"][cells.get_indexer(frame.h3_cell), season_index(frame.target_date)]
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            keep = ~quiet_stratum(frame) | positive | drawn
            part = build(context, labels, drivers, region, year, frame.loc[keep].reset_index(drop=True))
            history = season_index(part.history_end_date)
            seqs.append(gather(sequences.cube(region, year), cells.get_indexer(part.h3_cell), history).astype(np.float16))
            tables.append(part[FEATURE_SETS[FEATURE_SET] + [*QUIET_COLUMNS, LABEL]].assign(region=region, year=year, history=history))
            stats.append({"region": region, "year": year, "kept_rows": int(keep.sum()), "positives": int(positive.sum())})
            print(f"  {region} {year}: kept {keep.sum():,}; positives {positive.sum():,}", flush=True)
            del frame, part
    return pd.concat(tables, ignore_index=True), np.concatenate(seqs), pd.DataFrame(stats)


# ---------------------------------------------------------------- models

class Net(nn.Module):
    def __init__(self, n_inputs, sequence):
        super().__init__()
        self.gru = nn.GRU(len(CHANNELS), GRU_HIDDEN, batch_first=True) if sequence else None
        layers, width = [], n_inputs + (GRU_HIDDEN if sequence else 0)
        for k, size in enumerate(HIDDEN):
            layers += [nn.Linear(width, size), nn.SiLU()] + ([nn.Dropout(DROPOUT)] if k < len(HIDDEN) - 1 else [])
            width = size
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x, seq=None):
        if self.gru is not None:
            x = torch.cat([x, self.gru(seq)[1][-1]], dim=1)
        return self.head(x).squeeze(1)


def batch_inputs(net, Z, seqs, rows):
    x = torch.from_numpy(Z[rows])
    if net.gru is None:
        return (x,)
    s = seqs(rows) if callable(seqs) else seqs[rows].astype(np.float32)
    return x, torch.from_numpy(s)


@torch.no_grad()
def logits(net, Z, seqs):
    net.eval()
    return np.concatenate([net(*batch_inputs(net, Z, seqs, np.arange(start, min(start + PREDICT_BATCH, len(Z))))).numpy()
                           for start in range(0, len(Z), PREDICT_BATCH)])


def log_loss(y, z):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def train(name, Z, seqs, y, valid, fit_checks, curves):
    torch.manual_seed(SEED)
    net = Net(Z.shape[1], sequence=name == "gru")
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    fit_rows, valid_rows = np.flatnonzero(~valid), np.flatnonzero(valid)
    target = torch.from_numpy(y.astype(np.float32))
    best, best_state, best_epoch, started = np.inf, None, 0, time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        tic, total = time.time(), 0.0
        net.train()
        for rows in np.array_split(rng.permutation(fit_rows), len(fit_rows) // BATCH):
            rows.sort()
            loss = loss_fn(net(*batch_inputs(net, Z, seqs, rows)), target[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        valid_loss = log_loss(y[valid_rows], logits(net, Z[valid_rows], seqs[valid_rows] if seqs is not None else None))
        scheduler.step(valid_loss)
        curves.append({"model": name, "epoch": epoch, "train_log_loss": total / len(fit_rows), "valid_log_loss": valid_loss,
                       "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - tic, 1)})
        print(f"  {name} epoch {epoch}: train {total / len(fit_rows):.5f}, valid {valid_loss:.5f} ({time.time()-tic:.0f}s)", flush=True)
        if valid_loss < best - 1e-5:
            best, best_state, best_epoch = valid_loss, copy.deepcopy(net.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)
    torch.save(best_state, RESULTS / "models" / f"{name}.pt")
    fit_checks.append({"model": name, "rows": len(fit_rows), "valid_rows": len(valid_rows), "positives": int(y[fit_rows].sum()),
                       "inputs": Z.shape[1], "best_epoch": best_epoch, "valid_log_loss": best,
                       "parameters": sum(p.numel() for p in net.parameters()), "fit_seconds": round(time.time() - started, 1)})
    return net


def neural_scores(net, scaler, chunk, cube, cells):
    X, quiet = design(chunk, FEATURE_SET)
    cell, history = cells.get_indexer(chunk.h3_cell), season_index(chunk.history_end_date)
    z = logits(net, scaler(X), lambda rows: gather(cube, cell[rows], history[rows]))
    return expit(z + np.log(QUIET_SAMPLE_RATE) * quiet).astype(np.float32)


def main():
    global RESULTS, MAX_EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="cap (smoke test)")
    args = parser.parse_args()
    RESULTS = args.results
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    source_paths = [FIRMS, CWFIS, WEATHER_DAILY, STATIONS, LIGHTNING, REFERENCE_MODEL]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    context = Context(regions)
    labels = Labels(context, regions)
    labels.sources = None  # the grids hold everything later steps need
    drivers = Drivers(context, regions)
    sequences = Sequences(context, labels, drivers)

    print("Building training sample (2018-2021)", flush=True)
    sample, seqs, sample_stats = training_sample(regions, all_regions, context, labels, drivers, sequences)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:  # evaluation needs only 2022-2023
        del labels.grids[key]
    drivers._cache.clear()
    sequences._key, sequences._cube = None, None
    gc.collect()
    y = sample[LABEL].to_numpy()
    X, _ = design(sample, FEATURE_SET)
    valid = (sample.history.to_numpy() // VALID_BLOCK) % VALID_EVERY == 0
    del sample
    columns = FEATURE_SETS[FEATURE_SET] + ["quiet_stratum"]
    fit_checks, curves = [], []
    models = {"catboost_spread_noday": catboost_model()}
    models["catboost_spread_noday"].load_model(str(REFERENCE_MODEL))
    scaler = Scaler(X)
    Z = scaler(X)
    del X
    MAX_EPOCHS = args.max_epochs
    for name in NEURAL:
        models[name] = train(name, Z, seqs if name == "gru" else None, y, valid, fit_checks, curves)
        pd.DataFrame(curves).to_csv(RESULTS / "training_curves.csv", index=False)
    del Z, seqs
    gc.collect()

    rows, thresholds, daily, splits = [], {}, {}, []
    names = (*RULES, *MODELS)
    for region in regions:
        cells = context.grids[region]["cells"]
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            tic = time.time()
            g = labels.grids[region, year]
            frame = load(region, year, with_cells=True)
            n = len(frame)
            cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
            scores = {name: np.empty(n, np.float32) for name in names}
            for start in range(0, n, CHUNK):
                chunk = build(context, labels, drivers, region, year, frame.iloc[start:start + CHUNK].reset_index(drop=True))
                cube = sequences.cube(region, year)
                part = slice(start, start + len(chunk))
                scores["rule_snpp_yesterday"][part] = (chunk.local_count_1d > 0).to_numpy(np.float32)
                scores["rule_pair_yesterday"][part] = chunk.pair_yesterday.to_numpy(np.float32)
                for name in MODELS:
                    if name in NEURAL:
                        scores[name][part] = neural_scores(models[name], scaler, chunk, cube, cells)
                    else:
                        scores[name][part] = predict(models[name], chunk, FEATURE_SET)
                del chunk
            del frame
            assert n == len(cells) * (SEASON_DAYS - FIRST_TARGET_DAY), "expected a full cell x day grid"
            truth = g["label_full"][:, FIRST_TARGET_DAY:]
            near = ((g["disk1"] @ g["label_full"].astype(np.float32)) > 0)[:, FIRST_TARGET_DAY:]
            yesterday = shift(g["pair"], 1)[:, FIRST_TARGET_DAY:]
            volume = yesterday.sum(axis=0)
            for name, s in scores.items():
                score = to_grid(cell, target, s, len(cells))
                for method in METHODS:
                    if method == "rule_volume" and name in RULES:
                        continue
                    if method == "rule_volume":
                        flagged = volume_flags(score, volume).astype(np.float32)
                    for definition in DEFINITIONS:
                        tag = (region, definition, name, method)
                        if method == "f1_2022":
                            spread = sparse_row_max(g["disk1"], score)
                            inputs = (score, truth, truth, score) if definition == "res7_exact" else (score, truth, near, spread)
                            if year == SELECTION_YEAR:
                                flat = [a.ravel() for a in inputs]
                                candidates = candidate_thresholds(flat[0])
                                thresholds[tag] = float(candidates[int(np.argmax(curve(*flat, candidates)[2]))])
                            threshold = thresholds[tag]
                        else:
                            spread = sparse_row_max(g["disk1"], flagged)
                            inputs = (flagged, truth, truth, flagged) if definition == "res7_exact" else (flagged, truth, near, spread)
                            threshold = 0.5
                        counts = daily_counts(*inputs, threshold)
                        if year == EVALUATION_YEAR:
                            daily[tag] = counts
                            if definition == "res7_exact":
                                splits.append({"region": region, "model": name, "method": method,
                                               **breakdown(inputs[0] >= threshold, truth, yesterday)})
                        total = counts.sum(axis=0)
                        row = {"region": region, "year": year, "definition": definition, "model": name, "method": method,
                               "threshold": threshold, "precision": total[1] / max(total[0], 1),
                               "recall": total[3] / max(total[2], 1), "f1": float(f1_from_counts(total)),
                               "flagged": int(total[0]), "positives": int(total[2])}
                        if definition == "res7_exact" and method == "f1_2022" and name in MODELS:
                            row["average_precision"] = float(average_precision_score(truth.ravel(), score.ravel()))
                        rows.append(row)
            latest = [r for r in rows if r["region"] == region and r["year"] == year and r["method"] == "f1_2022"]
            print(f"{region} {year} ({time.time()-tic:.0f}s): " + "; ".join(
                f"{r['model']} {r['f1']:.3f}" for r in latest if r["definition"] == "res7_exact") + " | margin: " + "; ".join(
                f"{r['model']} {r['f1']:.3f}" for r in latest if r["definition"] == "res7_margin1"), flush=True)
            del scores
            gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    # The rules and Experiment 21's loaded model must reproduce Experiment 21 exactly.
    previous = pd.read_csv(EXPERIMENT_16 / "metrics_by_year.csv")
    check = metrics.loc[metrics.model.isin([*RULES, "catboost_spread_noday"])].merge(
        previous, on=["region", "year", "definition", "model", "method"], suffixes=("", "_16"))
    expected = len(regions) * 2 * len(DEFINITIONS) * (len(RULES) + len(METHODS))
    assert len(check) == expected and np.allclose(check.f1, check.f1_16, atol=1e-9), "Experiment 21 not reproduced"

    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["definition", "method", "model"])[["f1", "precision", "recall", "average_precision"]].median().reset_index()
    pooled = {key[1:]: np.sum([daily[(r, *key[1:])] for r in regions], axis=0) for key in daily if key[0] == regions[0]}
    total = lambda d, k, m: pooled[d, m, k].sum(axis=0)
    summary["bc_wide_f1"] = [float(f1_from_counts(total(*key))) for key in zip(summary.definition, summary.method, summary.model)]
    summary["bc_wide_precision"] = [total(*key)[1] / total(*key)[0] for key in zip(summary.definition, summary.method, summary.model)]
    summary["bc_wide_recall"] = [total(*key)[3] / total(*key)[2] for key in zip(summary.definition, summary.method, summary.model)]
    summary.to_csv(RESULTS / "summary_2023.csv", index=False)
    pd.DataFrame(splits).to_csv(RESULTS / "new_vs_continuing_2023.csv", index=False)

    dates = pd.date_range(f"{EVALUATION_YEAR}-05-01", periods=SEASON_DAYS)[FIRST_TARGET_DAY:]
    weeks = dates.isocalendar().week.to_numpy()
    unique_weeks = np.unique(weeks)
    week_member = (weeks[None, :] == unique_weeks[:, None]).astype(np.float64)
    draws = np.random.default_rng(SEED).integers(0, len(unique_weeks), size=(BOOTSTRAP, len(unique_weeks)))
    resample = np.stack([np.bincount(d, minlength=len(unique_weeks)) for d in draws]).astype(np.float64)
    boot = lambda counts: f1_from_counts(resample @ (week_member @ counts))
    boot_rows = []
    for definition in DEFINITIONS:
        for scope in [*regions, "BC-wide"]:
            get = (lambda key: pooled[key]) if scope == "BC-wide" else (lambda key: daily[(scope, *key)])
            for baseline in ("rule_pair_yesterday", "catboost_spread_noday"):
                for model in names:
                    for method in METHODS:
                        key = (definition, model, method)
                        base = (definition, baseline, "f1_2022" if baseline in RULES else method)
                        if model in (*RULES, baseline) or (scope == "BC-wide" and key not in pooled) or (
                                scope != "BC-wide" and (scope, *key) not in daily):
                            continue
                        difference = boot(get(key)) - boot(get(base))
                        boot_rows.append({"scope": scope, "definition": definition, "model": model, "method": method,
                                          "baseline": baseline,
                                          "difference": float(f1_from_counts(get(key).sum(0)) - f1_from_counts(get(base).sum(0))),
                                          "ci_low": float(np.quantile(difference, .025)), "ci_high": float(np.quantile(difference, .975)),
                                          "share_above": float((difference > 0).mean())})
    pd.DataFrame(boot_rows).to_csv(RESULTS / "bootstrap_2023.csv", index=False)
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "label": "S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell, plus one-day gap filling",
        "training_label": "2021: the full label; 2018-2020: gap-filled S-NPP only (stopgap until FIRMS NOAA-20 arrives)",
        "inputs": FEATURE_SETS[FEATURE_SET] + ["quiet_stratum"], "sequence_channels": CHANNELS, "sequence_days": SEQUENCE_DAYS,
        "scaler": scaler.describe(columns),
        "neural": {"hidden": HIDDEN, "gru_hidden": GRU_HIDDEN, "dropout": DROPOUT, "batch": BATCH, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "validation": f"history days whose {VALID_BLOCK}-day block index is a multiple of {VALID_EVERY}",
                   "threads": torch.get_num_threads()},
        "methods": {"f1_2022": "threshold maximizing 2022 F1 per Fire Centre, definition and model",
                    "rule_volume": "each day, flag as many cells per Fire Centre as the two-satellite rule"},
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023, against rule_pair_yesterday and catboost_spread_noday",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(), flush=True)
    print(pd.DataFrame(boot_rows).query("scope == 'BC-wide'").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
