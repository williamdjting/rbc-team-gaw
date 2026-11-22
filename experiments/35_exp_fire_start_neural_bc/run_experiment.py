"""Experiment 34's fire-start forecast with an MLP and an MLP + GRU instead of CatBoost.

Fire starts, quiet cell-days, training draw, thresholds, scoring, watch lists, onset groups and
bootstrap are Experiment 34's (its scoring function is reused unchanged). The networks are
Experiment 23's regularised MLP and MLP + 14-day GRU (lr 1e-4, dropout 0.3, AdamW weight decay
0.1, early stopping on held-out weeks of the training seasons), trained on Experiment 34's
fire-start rows with Experiment 21's spread_noday inputs (the inputs of Experiment 34's best
model). The input scaling is refit on these rows, and the output bias starts at the training
base rate (fire starts are about 0.4% of rows; Experiment 23 started at 50/50). 2024 is never loaded.

Forecasts:
  start_mlp, start_gru: the networks above, trained on fire starts;
  start_drivers: Experiment 34's fire-start CatBoost (loaded; must reproduce Experiment 34);
  catboost_spread_noday: Experiment 21's next-day CatBoost (loaded; must reproduce Experiment 34);
  mlp: Experiment 22's next-day MLP (loaded, Experiment 22's scaling);
  nearby_activity, lightning, random: Experiment 34's simple rankings (must reproduce Experiment 34).

    python run_experiment.py                      # full run
    python run_experiment.py --regions Southeast  # smoke test
    python run_experiment.py --resume             # load saved networks, skip finished region-years
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for name, path in (("experiment_18", "23_exp_regularised_neural_bc/run_experiment.py"),
                   ("experiment_20", "34_exp_fire_start_bc/run_experiment.py")):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / path)
    sys.modules[name] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
import experiment_17  # noqa: E402  (registered by experiment_18)
import experiment_18  # noqa: E402
import experiment_20  # noqa: E402
from experiment_18 import DROPOUT, EXPERIMENT_17, LR, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, experiment_17_scaler  # noqa: E402
from experiment_17 import (BATCH, FEATURE_SET, SEQUENCE_DAYS, VALID_BLOCK, VALID_EVERY, Scaler, Sequences,  # noqa: E402
                           batch_inputs, gather, logits, log_loss)
from experiment_20 import (DEFINITIONS, EXPERIMENT_16_MODEL, SIMPLE, START, WATCH_SIZES, bootstrap,  # noqa: E402
                           score_region_year, start_grids)
from experiment_16 import CWFIS, FIRMS, LIGHTNING, STATIONS, WEATHER_DAILY, Drivers, Labels, build  # noqa: E402
from experiment_13 import STATIC_DIR, StaticLayers, f1_from_counts  # noqa: E402
from experiment_12 import ONSETS, onset_categories  # noqa: E402
from experiment_08 import (QUIET, Context, FEATURE_SETS, QUIET_SAMPLE_RATE, REFERENCE, SEED, SELECTION_YEAR,  # noqa: E402
                           EVALUATION_YEAR, TRAIN_YEARS, catboost_model, design, digest, load, quiet_stratum, season_index)

EXPERIMENT_20 = EXPERIMENTS / "34_exp_fire_start_bc/results"
TRAINED = {"start_mlp": False, "start_gru": True}  # name -> uses the 14-day sequence
LOADED_NETS = {"mlp": EXPERIMENT_17 / "models/mlp.pt"}
CATBOOSTS = {"start_drivers": ("start_drivers", EXPERIMENT_20 / "models/start_drivers.cbm"),
             "catboost_spread_noday": ("spread_noday", EXPERIMENT_16_MODEL)}
SCORED = (*TRAINED, *LOADED_NETS, *CATBOOSTS, *SIMPLE)
REPRODUCED = (*CATBOOSTS, *SIMPLE)
QUIET_COLUMNS = [c for c in QUIET if c not in FEATURE_SETS[FEATURE_SET]]


class Network:
    """A network with its input scaling, callable like Experiment 13's predict on a feature chunk."""

    def __init__(self, net, scaler, sequences, context, start_model):
        self.net, self.scaler, self.sequences, self.context, self.start_model = net, scaler, sequences, context, start_model
        self.region = None  # set before each region is scored

    def __call__(self, chunk):
        X, quiet = design(chunk, FEATURE_SET)
        cells = self.context.grids[self.region]["cells"]
        cube = self.sequences.cube(self.region, int(chunk.target_date.iloc[0].year)) if self.net.gru is not None else None
        cell, history = cells.get_indexer(chunk.h3_cell), season_index(chunk.history_end_date)
        z = logits(self.net, self.scaler(X), lambda rows: gather(cube, cell[rows], history[rows]))
        # Experiment 22's next-day MLP learned quiet rows at the 1% draw rate; a constant shift on quiet rows
        if not self.start_model:
            z = z + np.log(QUIET_SAMPLE_RATE) * quiet
        return expit(z).astype(np.float32)


_predict = experiment_20.predict


def predict(model, chunk, feature_set):
    return model(chunk) if isinstance(model, Network) else _predict(model, chunk, feature_set)


def training_sample(regions, all_regions, context, labels, drivers, sequences, expected_rows):
    """Experiment 34's rows (quiet cell-days: every fire start plus the 1% draw), with each row's sequence.

    To bound memory (the first full run was killed at 8.1 GB here), each training year's label grids
    are dropped once used, tables are kept as float32, and the sequences go straight into one array
    sized from Experiment 34's row counts instead of a list joined at the end.
    """
    tables, stats, filled = [], [], 0
    out = np.empty((expected_rows, SEQUENCE_DAYS, len(experiment_17.CHANNELS)), np.float16)
    for region in regions:
        index = all_regions.index(region)
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            quiet, start = start_grids(labels.grids[region, year], year)
            cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
            quiet = quiet[cell, target] & quiet_stratum(frame)
            start = start[cell, target] & quiet
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            keep = quiet & (start | drawn)
            part = frame.loc[keep].reset_index(drop=True).assign(**{START: start[keep].astype(np.int8)})
            part = build(context, labels, drivers, region, year, part)
            history = season_index(part.history_end_date)
            seq = gather(sequences.cube(region, year), cells.get_indexer(part.h3_cell), history).astype(np.float16)
            out[filled:filled + len(seq)] = seq
            filled += len(seq)
            columns = FEATURE_SETS[FEATURE_SET] + QUIET_COLUMNS
            tables.append(part[columns].astype(np.float32).assign(**{START: part[START].to_numpy(np.int8), "history": history}))
            stats.append({"region": region, "year": year, "fire_starts": int(start.sum()), "kept_rows": int(keep.sum())})
            print(f"  {region} {year}: starts {start.sum():,}; kept {keep.sum():,}", flush=True)
            del frame, part, seq, labels.grids[region, year]
            sequences._key, sequences._cube = None, None
            gc.collect()
    assert filled == len(out), "not Experiment 34's training rows"
    return pd.concat(tables, ignore_index=True), out, pd.DataFrame(stats)


def train(name, Z, seqs, y, valid, fit_checks, curves):
    """Experiment 23's training loop."""
    torch.manual_seed(SEED)
    net = experiment_18.Net(Z.shape[1], sequence=seqs is not None)
    rate = y[~valid].mean()  # start at the base rate, not 50/50: about 0.4% of rows are fire starts
    nn.init.constant_(net.head[-1].bias, float(np.log(rate / (1 - rate))))
    weights = [p for p in net.parameters() if p.ndim > 1]
    biases = [p for p in net.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": weights, "weight_decay": WEIGHT_DECAY},
                                   {"params": biases, "weight_decay": 0.0}], lr=LR)
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
        print(f"  {name} epoch {epoch}: train {total / len(fit_rows):.6f}, valid {valid_loss:.6f} ({time.time()-tic:.0f}s)", flush=True)
        if valid_loss < best - 1e-5:
            best, best_state, best_epoch = valid_loss, copy.deepcopy(net.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)
    torch.save(best_state, RESULTS / "models" / f"{name}.pt")
    fit_checks.append({"model": name, "rows": len(fit_rows), "valid_rows": len(valid_rows), "positives": int(y[fit_rows].sum()),
                       "valid_positives": int(y[valid_rows].sum()), "inputs": Z.shape[1], "best_epoch": best_epoch,
                       "epochs_run": epoch, "valid_log_loss": best, "parameters": sum(p.numel() for p in net.parameters()),
                       "fit_seconds": round(time.time() - started, 1)})
    return net


def main():
    global RESULTS, MAX_EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="cap (smoke test)")
    parser.add_argument("--resume", action="store_true", help="load saved networks and skip finished region-years")
    args = parser.parse_args()
    RESULTS, MAX_EPOCHS = args.results, args.max_epochs
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    (RESULTS / "parts").mkdir(exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    source_paths = [FIRMS, CWFIS, WEATHER_DAILY, STATIONS, LIGHTNING, ONSETS / "onset_cells.csv", ONSETS / "onset_clusters.csv",
                    *sorted(STATIC_DIR.glob("*.parquet")), *LOADED_NETS.values(), *[p for _, p in CATBOOSTS.values()]]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    labels = Labels(context, regions)
    drivers = Drivers(context, regions)
    sequences = Sequences(context, labels, drivers)
    static = StaticLayers()
    onsets = onset_categories()

    models = {}
    for name, (feature_set, path) in CATBOOSTS.items():
        models[name] = catboost_model()
        models[name].load_model(str(path))
    scaler_17 = experiment_17_scaler()
    for name, path in LOADED_NETS.items():
        net = experiment_17.Net(len(scaler_17.mean) + int(scaler_17.missing.sum()), sequence=False)
        net.load_state_dict(torch.load(path))
        models[name] = Network(net, scaler_17, sequences, context, start_model=False)
    scaler_path = RESULTS / "models/scaler.pkl"
    if args.resume:
        scaler = pickle.loads(scaler_path.read_bytes())
        n_inputs = len(scaler.mean) + int(scaler.missing.sum())
        for name, sequence in TRAINED.items():
            net = experiment_18.Net(n_inputs, sequence)
            net.load_state_dict(torch.load(RESULTS / "models" / f"{name}.pt"))
            models[name] = Network(net, scaler, sequences, context, start_model=True)
        print("Loaded saved networks", flush=True)
    else:
        print("Building training sample (quiet cell-days, 2018-2021)", flush=True)
        reference = pd.read_csv(EXPERIMENT_20 / "training_sample.csv").set_index(["region", "year"])
        sample, seqs, sample_stats = training_sample(regions, all_regions, context, labels, drivers, sequences,
                                                     int(reference.loc[regions].kept_rows.sum()))
        sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
        if not args.regions:
            assert all(reference.loc[(r.region, r.year), "kept_rows"] == r.kept_rows and
                       reference.loc[(r.region, r.year), "fire_starts"] == r.fire_starts for r in sample_stats.itertuples()), \
                "not Experiment 34's training rows"
        sequences._key, sequences._cube = None, None
        y = sample[START].to_numpy()
        X, quiet = design(sample, FEATURE_SET)
        assert quiet.all()
        valid = (sample.history.to_numpy() // VALID_BLOCK) % VALID_EVERY == 0
        del sample
        scaler = Scaler(X)
        scaler_path.write_bytes(pickle.dumps(scaler))
        Z = scaler(X)
        del X
        gc.collect()
        fit_checks, curves = [], []
        for name, sequence in TRAINED.items():
            net = train(name, Z, seqs if sequence else None, y, valid, fit_checks, curves)
            models[name] = Network(net, scaler, sequences, context, start_model=True)
            pd.DataFrame(curves).to_csv(RESULTS / "training_curves.csv", index=False)
        pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
        del Z, seqs
    for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:
        del labels.grids[key]
    labels.sources = None
    gc.collect()

    # Experiment 34's scoring, with the networks behind its predict()
    experiment_20.predict = predict
    experiment_20.MODELS, experiment_20.SCORED = {}, SCORED
    experiment_20.LOADED = {**{name: FEATURE_SET for name in (*TRAINED, *LOADED_NETS)}, **{n: f for n, (f, _) in CATBOOSTS.items()}}
    thresholds, parts = {}, []
    for region in regions:
        for model in models.values():
            if isinstance(model, Network):
                model.region = region
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            path = RESULTS / "parts" / f"{region.lower().replace(' ', '_')}_{year}.pkl"
            if args.resume and path.exists():
                out = pickle.loads(path.read_bytes())
                print(f"{region} {year}: loaded finished part", flush=True)
            else:
                out = score_region_year(region, all_regions.index(region), year, context, labels, drivers, static, models,
                                        thresholds, onsets if year == EVALUATION_YEAR else None)
                out["thresholds"] = {k: v for k, v in thresholds.items() if k[0] == region}
                path.write_bytes(pickle.dumps(out))
            thresholds.update(out["thresholds"])
            parts.append(out)
            sequences._key, sequences._cube = None, None
            gc.collect()

    metrics = pd.DataFrame([r for p in parts for r in p["rows"]])
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    previous = pd.read_csv(EXPERIMENT_20 / "metrics_by_year.csv")
    check = metrics.loc[metrics.model.isin(REPRODUCED)].merge(previous, on=["region", "year", "definition", "model"], suffixes=("", "_20"))
    assert len(check) == len(regions) * 2 * len(DEFINITIONS) * len(REPRODUCED) and np.allclose(check.f1, check.f1_20, atol=1e-9), \
        "Experiment 34 not reproduced"
    print(f"Experiment 34 reproduced: {len(check)} rows", flush=True)
    coverage = pd.DataFrame([p["coverage"] for p in parts])
    coverage.to_csv(RESULTS / "coverage.csv", index=False)
    daily = {k: v for p in parts if p["coverage"]["year"] == EVALUATION_YEAR for k, v in p["daily"].items()}

    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["definition", "model"])[["f1", "precision", "recall", "average_precision"]].median()
    summary = summary.add_prefix("median_").reset_index()
    for column, f in (("bc_wide_f1", f1_from_counts), ("bc_wide_precision", lambda t: t[1] / max(t[0], 1)),
                      ("bc_wide_recall", lambda t: t[3] / max(t[2], 1))):
        summary[column] = [float(f(np.sum([daily[r, d, m] for r in regions], axis=0).sum(0)))
                           for d, m in zip(summary.definition, summary.model)]
    summary.to_csv(RESULTS / "summary_2023.csv", index=False)

    watch = pd.DataFrame([r for p in parts for r in p["watch"]])
    watch.to_csv(RESULTS / "watch_list.csv", index=False)
    pooled = watch.loc[watch.year.eq(EVALUATION_YEAR)].groupby(["model", "list_size"])[
        ["fire_starts", "listed", "caught_next_day", "caught_next_day_margin", "caught_within_7d", "listed_start_within_7d"]].sum(min_count=1)
    for column in ("caught_next_day", "caught_next_day_margin", "caught_within_7d"):
        pooled[f"share_{column}"] = pooled[column] / pooled.fire_starts
    pooled["hit_rate_next_day"] = pooled.caught_next_day / pooled.listed
    pooled["hit_rate_within_7d"] = pooled.listed_start_within_7d / pooled.listed
    pooled.reset_index().to_csv(RESULTS / "watch_list_bc_2023.csv", index=False)
    pd.DataFrame([r for p in parts for r in p["groups"]]).to_csv(RESULTS / "onset_groups_2023.csv", index=False)
    boot = pd.concat([bootstrap(daily, regions, list(SCORED), baseline) for baseline in ("start_drivers", "catboost_spread_noday")])
    boot.to_csv(RESULTS / "bootstrap_2023.csv", index=False)

    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "fire_start": "Experiment 34's", "training_rows": "Experiment 34's (every fire start plus Experiment 21's 1% draw of quiet cell-days)",
        "inputs": FEATURE_SETS[FEATURE_SET], "scaler": scaler.describe(FEATURE_SETS[FEATURE_SET] + ["quiet_stratum"]),
        "networks": {"trained": TRAINED, "sequence_days": SEQUENCE_DAYS, "lr": LR, "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
                     "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch": BATCH, "settings": "Experiment 23's"},
        "loaded": {**{n: str(p.relative_to(EXPERIMENTS)) for n, p in LOADED_NETS.items()},
                   **{n: str(p.relative_to(EXPERIMENTS)) for n, (_, p) in CATBOOSTS.items()}},
        "watch_lists": f"top {list(WATCH_SIZES)} quiet cells per Fire Centre per day",
        "bootstrap": "week-block resamples of 2023, against start_drivers and against catboost_spread_noday",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(), flush=True)
    print(pooled.reset_index().query("list_size == 50").round(4).to_string(), flush=True)
    print(boot.query("scope == 'BC-wide'").round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
