"""Experiment 22's networks retrained with a lower learning rate and strong regularisation.

In Experiment 22 both networks reached their best held-out loss after one epoch, then overfit the
training seasons. Everything except the training settings is Experiment 22's: label, training draw,
inputs, input scaling (rebuilt from Experiment 22's manifest), architecture, thresholds, scoring and
bootstrap. 2024 is never loaded.

Changes from Experiment 22:
- learning rate 1e-4 (was 1e-3);
- dropout 0.3 after every hidden layer and on the GRU summary (was 0.1 after the first two layers);
- AdamW weight decay 0.1 on weights, none on biases (was 1e-4 on everything);
- up to 40 epochs with early-stopping patience 5 (was 20 and 3), since smaller steps need more epochs.

Models:
- catboost_spread_noday: Experiment 21's model, loaded; must reproduce Experiment 21.
- mlp: Experiment 22's MLP, loaded; must reproduce Experiment 22.
- mlp_reg, gru_reg: Experiment 22's MLP and MLP + GRU, retrained with the settings above.
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
from sklearn.metrics import average_precision_score
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_17", EXPERIMENTS / "22_exp_neural_models_bc/run_experiment.py")
experiment_17 = sys.modules["experiment_17"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_17)
from experiment_17 import (BATCH, CHANNELS, CHUNK, EXPERIMENT_16, FEATURE_SET, GRU_HIDDEN, HIDDEN, METHODS,  # noqa: E402
                           REFERENCE_MODEL, SEQUENCE_DAYS, VALID_BLOCK, VALID_EVERY, Scaler, Sequences,
                           batch_inputs, logits, log_loss, neural_scores, training_sample)
from experiment_16 import LIGHTNING, STATIONS, WEATHER_DAILY, Drivers, breakdown, build, volume_flags  # noqa: E402
from experiment_15 import LABEL, RULES, DEFINITIONS, Labels, to_grid  # noqa: E402
from experiment_14 import CWFIS, FIRMS, shift  # noqa: E402
from experiment_13 import BOOTSTRAP, daily_counts, f1_from_counts  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve  # noqa: E402
from experiment_08 import (Context, FEATURE_SETS, REFERENCE, SEASON_DAYS, SEED, SELECTION_YEAR, EVALUATION_YEAR,  # noqa: E402
                           TRAIN_YEARS, catboost_model, design, digest, load, predict, season_index, sparse_row_max)

EXPERIMENT_17 = EXPERIMENTS / "22_exp_neural_models_bc/results"
MODELS = ("catboost_spread_noday", "mlp", "mlp_reg", "gru_reg")
NEURAL = ("mlp", "mlp_reg", "gru_reg")
TRAINED = ("mlp_reg", "gru_reg")
BASELINES = ("rule_pair_yesterday", "catboost_spread_noday", "mlp")
LR, WEIGHT_DECAY, DROPOUT, MAX_EPOCHS, PATIENCE = 1e-4, 0.1, 0.3, 40, 5


def experiment_17_scaler():
    """Experiment 22's input scaling, rebuilt from its manifest so its MLP scores exactly as before."""
    described = json.loads((EXPERIMENT_17 / "manifest.json").read_text())["scaler"]
    assert list(described) == FEATURE_SETS[FEATURE_SET] + ["quiet_stratum"]
    scaler = Scaler.__new__(Scaler)
    scaler.log, scaler.missing = (np.array([v[k] for v in described.values()]) for k in ("signed_log", "missing_flag"))
    scaler.mean, scaler.std = (np.array([v[k] for v in described.values()], np.float32) for k in ("mean", "std"))
    return scaler


class Net(nn.Module):
    """Experiment 22's network with dropout after every hidden layer and on the GRU summary."""

    def __init__(self, n_inputs, sequence):
        super().__init__()
        self.gru = nn.GRU(len(CHANNELS), GRU_HIDDEN, batch_first=True) if sequence else None
        self.gru_dropout = nn.Dropout(DROPOUT)
        layers, width = [], n_inputs + (GRU_HIDDEN if sequence else 0)
        for size in HIDDEN:
            layers += [nn.Linear(width, size), nn.SiLU(), nn.Dropout(DROPOUT)]
            width = size
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x, seq=None):
        if self.gru is not None:
            x = torch.cat([x, self.gru_dropout(self.gru(seq)[1][-1])], dim=1)
        return self.head(x).squeeze(1)


def train(name, Z, seqs, y, valid, fit_checks, curves):
    torch.manual_seed(SEED)
    net = Net(Z.shape[1], sequence=name == "gru_reg")
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
        print(f"  {name} epoch {epoch}: train {total / len(fit_rows):.5f}, valid {valid_loss:.5f} ({time.time()-tic:.0f}s)", flush=True)
        if valid_loss < best - 1e-5:
            best, best_state, best_epoch = valid_loss, copy.deepcopy(net.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)
    torch.save(best_state, RESULTS / "models" / f"{name}.pt")
    fit_checks.append({"model": name, "rows": len(fit_rows), "valid_rows": len(valid_rows), "positives": int(y[fit_rows].sum()),
                       "inputs": Z.shape[1], "best_epoch": best_epoch, "epochs_run": epoch, "valid_log_loss": best,
                       "parameters": sum(p.numel() for p in net.parameters()), "fit_seconds": round(time.time() - started, 1)})
    return net


def main():
    global RESULTS, MAX_EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="cap (smoke test)")
    parser.add_argument("--load-models", action="store_true", help="score the networks saved in --results instead of training")
    args = parser.parse_args()
    RESULTS, MAX_EPOCHS = args.results, args.max_epochs
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    reference_mlp = EXPERIMENT_17 / "models/mlp.pt"
    source_paths = [FIRMS, CWFIS, WEATHER_DAILY, STATIONS, LIGHTNING, REFERENCE_MODEL, reference_mlp]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    labels = Labels(context, regions)
    labels.sources = None  # the grids hold everything later steps need
    drivers = Drivers(context, regions)
    sequences = Sequences(context, labels, drivers)

    scaler = experiment_17_scaler()
    n_inputs = len(scaler.mean) + int(scaler.missing.sum())
    models = {"catboost_spread_noday": catboost_model()}
    models["catboost_spread_noday"].load_model(str(REFERENCE_MODEL))
    models["mlp"] = experiment_17.Net(n_inputs, sequence=False)
    models["mlp"].load_state_dict(torch.load(reference_mlp))
    if args.load_models:
        # Resume after an interrupted run: skip the training sample (the memory peak) and score the saved networks.
        for name in TRAINED:
            models[name] = Net(n_inputs, sequence=name == "gru_reg")
            models[name].load_state_dict(torch.load(RESULTS / "models" / f"{name}.pt"))
        curves = pd.read_csv(RESULTS / "training_curves.csv")
        fit_checks = [{"model": name, "best_epoch": int(c.epoch[c.valid_log_loss.idxmin()]), "epochs_run": int(c.epoch.max()),
                       "valid_log_loss": float(c.valid_log_loss.min()), "fit_seconds": float(c.seconds.sum())}
                      for name, c in curves.groupby("model", sort=False)]
        for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:  # evaluation needs only 2022-2023
            del labels.grids[key]
    else:
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
        if regions == all_regions:  # same sample, so refitting must give Experiment 22's scaling
            refit = Scaler(X)
            assert all(np.array_equal(getattr(refit, k), getattr(scaler, k)) for k in ("log", "missing", "mean", "std"))
            del refit
        Z = scaler(X)
        del X
        fit_checks, curves = [], []
        for name in TRAINED:
            models[name] = train(name, Z, seqs if name == "gru_reg" else None, y, valid, fit_checks, curves)
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
    # The rules and the loaded models must reproduce the experiments they come from.
    for source, loaded in ((EXPERIMENT_16, [*RULES, "catboost_spread_noday"]), (EXPERIMENT_17, ["mlp"])):
        check = metrics.loc[metrics.model.isin(loaded)].merge(pd.read_csv(source / "metrics_by_year.csv"),
                                                              on=["region", "year", "definition", "model", "method"], suffixes=("", "_old"))
        expected = len(regions) * 2 * len(DEFINITIONS) * sum(1 if m in RULES else len(METHODS) for m in loaded)
        assert len(check) == expected and np.allclose(check.f1, check.f1_old, atol=1e-9), f"{source.parent.name} not reproduced"

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
            for baseline in BASELINES:
                for model in TRAINED:
                    for method in METHODS:
                        key = (definition, model, method)
                        base = (definition, baseline, "f1_2022" if baseline in RULES else method)
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
        "scaling": "Experiment 22's, rebuilt from its manifest",
        "neural": {"hidden": HIDDEN, "gru_hidden": GRU_HIDDEN, "dropout": DROPOUT, "batch": BATCH, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "weight_decay_on": "weights only (not biases)", "max_epochs": MAX_EPOCHS,
                   "patience": PATIENCE, "validation": f"history days whose {VALID_BLOCK}-day block index is a multiple of {VALID_EVERY}",
                   "threads": torch.get_num_threads()},
        "experiment_17_neural": {"dropout": experiment_17.DROPOUT, "lr": experiment_17.LR, "weight_decay": experiment_17.WEIGHT_DECAY,
                                 "max_epochs": experiment_17.MAX_EPOCHS, "patience": experiment_17.PATIENCE},
        "methods": {"f1_2022": "threshold maximizing 2022 F1 per Fire Centre, definition and model",
                    "rule_volume": "each day, flag as many cells per Fire Centre as the two-satellite rule"},
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023, against {', '.join(BASELINES)}",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(), flush=True)
    print(pd.DataFrame(boot_rows).query("scope == 'BC-wide'").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
