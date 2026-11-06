"""Retrain the next-day forecast on Experiment 19's multi-satellite, gap-filled label.

Label (evaluation, 2022-2023): a VIIRS detection from S-NPP (FIRMS) or NOAA-20 (CWFIS) in the
H3 res-7 cell, plus one-day gap filling (positive the day before and the day after). Training
uses a stopgap until FIRMS NOAA-20 for 2018-2020 arrives: 2021 has the full label; 2018-2020 use
gap-filled S-NPP only (NOAA-20 is absent in 2018 and partial in 2019-2020).
Forecast inputs never use gap filling: only detections seen up to the day before.
- Rules: S-NPP detection yesterday (the old rule); S-NPP or NOAA-20 detection yesterday.
- CatBoost (Experiments 05-07 and 13 settings) on Experiment 13's features:
  catboost_08 (Experiment 13's model, trained on the old label, loaded);
  catboost_stopgap (2018-2021, stopgap label);
  catboost_2021 (2021 only, full label);
  catboost_2021_n20 (2021 only, full label, plus NOAA-20 and two-satellite inputs).
Sampling as Experiment 13 (every active row and every positive of the new label, 1% of quiet
negatives, log(0.01) correction). Scored on the new label, exact cell and one-cell margin
(neighbourhoods within each Fire Centre grid, as Experiment 19), thresholds maximize 2022 F1 per
Fire Centre and apply unchanged to 2023, week-block bootstrap. 2024 is never loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for name, path in (("experiment_13", "18_exp_static_layers_watchlist_bc/run_experiment.py"),
                   ("experiment_14", "19_exp_multisensor_labels_bc/analyze_labels.py")):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / path)
    sys.modules[name] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # experiment_13 also caches BCWS arrays so features can be chunked
from experiment_14 import CWFIS, FIRMS, grid, load_sources, neighbourhood, shift  # noqa: E402
from experiment_13 import BOOTSTRAP, daily_counts, f1_from_counts  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve  # noqa: E402
from experiment_08 import (Context, FEATURE_SETS, MAX_ABS_LOG_ODDS, QUIET_SAMPLE_RATE, REFERENCE, SEASON_DAYS,  # noqa: E402
                           SEED, SELECTION_YEAR, EVALUATION_YEAR, TRAIN_YEARS, catboost_model, design, digest, load,
                           predict, quiet_stratum, raw_scores, season_index, sparse_row_max, window)

EXPERIMENT_08_MODEL = EXPERIMENTS / "13_exp_fire_context_bc/results/models/catboost__all.cbm"
LABEL = "target_viirs_filled"
N20 = ["n20_cell_1d", "n20_cell_3d", "n20_disk1_1d", "n20_disk1_3d", "pair_cell_days_7d", "pair_disk1_1d", "pair_disk1_days_7d"]
FEATURE_SETS["all_n20"] = FEATURE_SETS["all"] + N20
MODELS = {"catboost_08": ("all", None), "catboost_stopgap": ("all", TRAIN_YEARS),
          "catboost_2021": ("all", (2021,)), "catboost_2021_n20": ("all_n20", (2021,))}
RULES = ("rule_snpp_yesterday", "rule_pair_yesterday")
DEFINITIONS = ("res7_exact", "res7_margin1")
CHUNK = 1_000_000


def filled(a):
    return a | (shift(a, 1) & shift(a, -1))


class Labels:
    """Per region and year: grids (cells x 153) of S-NPP, NOAA-20, observed pair, and labels."""

    def __init__(self, context, regions):
        self.sources = load_sources()
        self.grids = {}
        for region in regions:
            cells = context.grids[region]["cells"]
            disk1 = neighbourhood(list(cells))
            for year in range(2018, 2024):
                snpp = grid(self.sources.get(("snpp_firms", year)), cells)
                n20 = grid(self.sources.get(("noaa20", year)), cells)
                pair = snpp | n20
                label = filled(pair) if year >= 2021 else filled(snpp)  # stopgap before 2021
                self.grids[region, year] = {"snpp": snpp, "n20": n20, "pair": pair, "label": label,
                                            "label_full": filled(pair), "disk1": disk1}

    def add(self, region, year, frame, cells):
        """Adds LABEL (the stopgap training label; the full label from 2021) and N20 inputs."""
        g = self.grids[region, year]
        cell = cells.get_indexer(frame.h3_cell)
        history, target = season_index(frame.history_end_date), season_index(frame.target_date)
        frame[LABEL] = g["label"][cell, target].astype(np.int8)
        n20, pair, disk1 = g["n20"].astype(np.float32), g["pair"].astype(np.float32), g["disk1"]
        daily = {"n20_cell_1d": n20, "n20_cell_3d": window(n20, 3), "n20_disk1_1d": disk1 @ n20,
                 "n20_disk1_3d": window(disk1 @ n20, 3), "pair_cell_days_7d": window(pair, 7),
                 "pair_disk1_1d": disk1 @ pair, "pair_disk1_days_7d": window(((disk1 @ pair) > 0).astype(np.float32), 7)}
        for name, values in daily.items():
            frame[name] = values[cell, history]
        frame["pair_yesterday"] = g["pair"][cell, history].astype(np.float32)
        return frame


def training_sample(regions, all_regions, context, labels):
    """Experiment 13's draw, also keeping every positive of the new label."""
    parts, stats = [], []
    for region in regions:
        index = all_regions.index(region)
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            g = labels.grids[region, year]
            positive = g["label"][cells.get_indexer(frame.h3_cell), season_index(frame.target_date)]
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            quiet = quiet_stratum(frame)
            keep = ~quiet | positive | drawn
            part = context.features(region, year, frame.loc[keep].reset_index(drop=True))
            part = labels.add(region, year, part, cells)
            parts.append(part[FEATURE_SETS["all_n20"] + [LABEL]].assign(region=region, year=year))
            stats.append({"region": region, "year": year, "rows": len(frame), "kept_rows": int(keep.sum()),
                          "old_label_positives": int(frame.target_recorded_detection.sum()), "new_label_positives": int(positive.sum()),
                          "quiet_new_positives": int((quiet & positive).sum())})
            print(f"  {region} {year}: kept {keep.sum():,}; new-label positives {positive.sum():,} "
                  f"(old {int(frame.target_recorded_detection.sum()):,})", flush=True)
            del frame
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def fit(name, X, y, columns, fit_checks):
    model = catboost_model()
    tic = time.time()
    model.fit(X, y)
    largest = float(np.abs(raw_scores(model, X)).max())
    assert largest < MAX_ABS_LOG_ODDS, f"{name} diverged"
    fit_checks.append({"model": name, "rows": len(y), "positives": int(y.sum()), "inputs": X.shape[1],
                       "max_abs_training_log_odds": largest, "fit_seconds": round(time.time() - tic, 1)})
    pd.DataFrame({"feature": columns, "importance": model.get_feature_importance()}).sort_values(
        "importance", ascending=False).to_csv(RESULTS / f"feature_importance__{name}.csv", index=False)
    model.save_model(str(RESULTS / "models" / f"{name}.cbm"))
    print(f"Fitted {name}: {len(y):,} rows, {int(y.sum()):,} positives, {time.time()-tic:.0f}s", flush=True)
    return model


def to_grid(cell, target, values, n_cells):
    out = np.zeros((n_cells, SEASON_DAYS), np.float32)
    out[cell, target] = values
    return out[:, FIRST_TARGET_DAY:]


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    RESULTS = args.results
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    source_paths = [FIRMS, CWFIS, EXPERIMENT_08_MODEL]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    labels = Labels(context, regions)

    print("Building training sample (2018-2021)", flush=True)
    sample, sample_stats = training_sample(regions, all_regions, context, labels)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    fit_checks = []
    models = {"catboost_08": catboost_model()}
    models["catboost_08"].load_model(str(EXPERIMENT_08_MODEL))
    for name, (feature_set, years) in MODELS.items():
        if years is None:
            continue
        rows = sample.year.isin(years).to_numpy()
        X, _ = design(sample.loc[rows], feature_set)
        models[name] = fit(name, X, sample.loc[rows, LABEL].to_numpy(), FEATURE_SETS[feature_set] + ["quiet_stratum"], fit_checks)
    del sample

    rows, thresholds, daily = [], {}, {}
    for region in regions:
        cells = context.grids[region]["cells"]
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            tic = time.time()
            g = labels.grids[region, year]
            frame = load(region, year, with_cells=True)
            n = len(frame)
            cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
            scores = {name: np.empty(n, np.float32) for name in (*RULES, *MODELS)}
            for start in range(0, n, CHUNK):
                chunk = context.features(region, year, frame.iloc[start:start + CHUNK].reset_index(drop=True))
                chunk = labels.add(region, year, chunk, cells)
                part = slice(start, start + len(chunk))
                scores["rule_snpp_yesterday"][part] = (chunk.local_count_1d > 0).to_numpy(np.float32)
                scores["rule_pair_yesterday"][part] = chunk.pair_yesterday.to_numpy(np.float32)
                for name, (feature_set, _) in MODELS.items():
                    scores[name][part] = predict(models[name], chunk, feature_set)
                del chunk
            del frame
            assert n == len(cells) * (SEASON_DAYS - FIRST_TARGET_DAY), "expected a full cell x day grid"
            truth = g["label_full"][:, FIRST_TARGET_DAY:]
            near = ((g["disk1"] @ g["label_full"].astype(np.float32)) > 0)[:, FIRST_TARGET_DAY:]
            for name, s in scores.items():
                score = to_grid(cell, target, s, len(cells))
                spread = sparse_row_max(g["disk1"], score)
                for definition in DEFINITIONS:
                    inputs = (score, truth, truth, score) if definition == "res7_exact" else (score, truth, near, spread)
                    tag = (region, definition, name)
                    if year == SELECTION_YEAR:
                        flat = [a.ravel() for a in inputs]
                        candidates = candidate_thresholds(flat[0])
                        thresholds[tag] = float(candidates[int(np.argmax(curve(*flat, candidates)[2]))])
                    counts = daily_counts(*inputs, thresholds[tag])
                    if year == EVALUATION_YEAR:
                        daily[tag] = counts
                    total = counts.sum(axis=0)
                    row = {"region": region, "year": year, "definition": definition, "model": name,
                           "threshold": thresholds[tag], "precision": total[1] / max(total[0], 1),
                           "recall": total[3] / max(total[2], 1), "f1": float(f1_from_counts(total)),
                           "flagged": int(total[0]), "positives": int(total[2])}
                    if definition == "res7_exact" and name in MODELS:
                        row["average_precision"] = float(average_precision_score(truth.ravel(), score.ravel()))
                    rows.append(row)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            print(f"{region} {year} ({time.time()-tic:.0f}s): " + "; ".join(
                f"{r['model']} {r['f1']:.3f}" for r in latest if r["definition"] == "res7_exact") + " | margin: " + "; ".join(
                f"{r['model']} {r['f1']:.3f}" for r in latest if r["definition"] == "res7_margin1"), flush=True)
            del scores

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["definition", "model"])[["f1", "precision", "recall", "average_precision"]].median().reset_index()
    names = (*RULES, *MODELS)
    pooled = {(d, m): np.sum([daily[r, d, m] for r in regions], axis=0) for d in DEFINITIONS for m in names}
    summary["bc_wide_f1"] = [float(f1_from_counts(pooled[d, m].sum(axis=0))) for d, m in zip(summary.definition, summary.model)]
    summary.to_csv(RESULTS / "summary_2023.csv", index=False)

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
            get = (lambda m: pooled[definition, m]) if scope == "BC-wide" else (lambda m: daily[scope, definition, m])
            for model in names:
                if model == "rule_pair_yesterday":
                    continue
                difference = boot(get(model)) - boot(get("rule_pair_yesterday"))
                boot_rows.append({"scope": scope, "definition": definition, "model": model, "baseline": "rule_pair_yesterday",
                                  "difference": float(f1_from_counts(get(model).sum(0)) - f1_from_counts(get("rule_pair_yesterday").sum(0))),
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
        "inputs": "Experiment 13 features (S-NPP); catboost_2021_n20 adds NOAA-20 and two-satellite inputs; no gap filling in inputs",
        "models": {k: {"feature_set": v[0], "training_years": None if v[1] is None else list(v[1])} for k, v in MODELS.items()},
        "n20_features": N20, "margin": "neighbourhoods within each Fire Centre grid",
        "thresholds": "maximize 2022 F1 per Fire Centre, definition and model; applied unchanged to 2023",
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023, against rule_pair_yesterday",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(), flush=True)
    print(pd.DataFrame(boot_rows).query("scope == 'BC-wide'").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
