"""Our saved models scored the way CanadaFireSat scores (Porta et al. 2025, arXiv 2506.08690).

CanadaFireSat reports 60.3% F1 for 2023, while our F1 on the full BC grid is 0.03-0.12. Their F1 is
measured on a sample, not on every place and day, so this experiment reproduces their protocol
with our saved models (no retraining) to put both numbers on the same footing.

Their protocol, as applied here:
- Forecast dates on a fixed 8-day grid (May 8, 16, ... Sep 22), each predicting days t .. t+7.
  The unit is a res-7 cell (about 5 km2; their patches are 2.64 x 2.64 km, about 7 km2).
- Positives: every cell-date whose label is 1. Three labels:
    burn_8d: an S-NPP or NOAA-20 detection in the cell on t .. t+7 (closest to their NBAC
      "the patch burns in the window", spread of existing fires included);
    new_burn_8d: a new satellite fire, Experiment 34's start (a detection after 7 days with none in
      the cell or its six neighbours), in the cell on any of t .. t+7;
    ignition_8d: a BCWS wildfire starts in the cell on t .. t+7 (Experiment 36's label);
    big_lightning_8d: a lightning wildfire of >= 10 ha starts there (Experiment 37's label).
- Test negatives: per Fire Centre and year, 2 per positive, drawn from cell-dates between the
  year's first and last positive date, leaving out cells that burned (satellite or BCWS) in
  earlier years and, for 2023, cells drawn as negatives in 2022; spread evenly over the deciles
  of fire weather index (FWI) of that pool.
- Test Hard negatives (2023): 2 per positive within each Fire Centre x land cover (dominant fuel
  share) x month, matched to the positives' FWI deciles, so weather and land cover cannot
  separate the classes.
- Threshold: maximizes F1 on the 2022 sample drawn the same way (one threshold for BC).
- Five negative draws (seeds); means and spreads are reported.
The same cell-dates are also scored on the full grid (no sampling), the way Experiments 36-37 do.

Scored: Experiment 36's ign_full, Experiment 37's ign_full and ign_lightning, Experiment 21's
next-day satellite model, Experiment 34's satellite fire-start model, and simple rankings by FWI,
recent satellite fire next to the cell (7 days) and lightning over 14 days, plus "everything is
fire" (F1 = 2p / (1 + p) at positive share p).

    python run_experiment.py                      # score, then evaluate
    python run_experiment.py --regions Southeast  # smoke test
    python run_experiment.py --resume             # reuse finished region-year score tables
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_22", EXPERIMENTS / "36_exp_bcws_ignitions_bc/run_experiment.py")
experiment_22 = sys.modules["experiment_22"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_22)
import experiment_15  # noqa: E402  (registered by experiment_22)
import experiment_20  # noqa: E402
from experiment_22 import EXPERIMENT_16_MODEL, EXPERIMENT_20_MODEL, LongLightning, calibrated, full_build, ignition_grid  # noqa: E402
from experiment_16 import Drivers, Labels  # noqa: E402
from experiment_13 import STATIC, StaticLayers  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY  # noqa: E402
from experiment_08 import (Context, REFERENCE, SEASON_DAYS, SEED, SELECTION_YEAR, EVALUATION_YEAR, TRAIN_YEARS,  # noqa: E402
                           catboost_model, digest, predict)
from experiment_03 import CELL, NEIGHBOUR, TARGET, feature_path, slug  # noqa: E402

WINDOW = 8  # days t .. t+7, CanadaFireSat's horizon
ISSUE_DAYS = np.arange(FIRST_TARGET_DAY, SEASON_DAYS - WINDOW + 1, WINDOW)  # May 8 .. Sep 22; t+7 <= Sep 30
EXPERIMENT_22_MODELS = EXPERIMENTS / "36_exp_bcws_ignitions_bc/results/models"
EXPERIMENT_23_MODELS = EXPERIMENTS / "37_exp_lightning_ignitions_bc/results/models"
CATBOOST = {  # name: (feature set, model file, trained on the 1% draw of Experiments 36-37)
    "exp22_ign_full": ("ign_full", EXPERIMENT_22_MODELS / "ign_full.cbm", True),
    "exp23_ign_full": ("ign_full", EXPERIMENT_23_MODELS / "ign_full.cbm", True),
    "exp23_ign_lightning": ("ign_lightning", EXPERIMENT_23_MODELS / "ign_lightning.cbm", True),
    "exp16_spread": ("spread_noday", EXPERIMENT_16_MODEL, False),
    "exp20_start": ("start_drivers", EXPERIMENT_20_MODEL, False),
}
RULES = ("fwi", "recent_fire_nearby", "lightning_14d")
SCORED = (*CATBOOST, *RULES)
LABELS = ("burn_8d", "new_burn_8d", "ignition_8d", "big_lightning_8d")
FUELS = ["fuel_share_conifer", "fuel_share_deciduous", "fuel_share_grass", "fuel_share_mixedwood",
         "fuel_share_non_fuel", "fuel_share_water"]
NEGATIVES_PER_POSITIVE = 2
SEEDS = range(5)
CHUNK = 250_000


# ---------------------------------------------------------------- score tables

def ahead(grid):
    """(cells x issue dates): grid is true on any of days t .. t+7."""
    cs = np.concatenate([np.zeros((grid.shape[0], 1), np.int32), grid.cumsum(axis=1, dtype=np.int32)], axis=1)
    return (cs[:, ISSUE_DAYS + WINDOW] - cs[:, ISSUE_DAYS]) > 0


def load_days(region, year):
    """Experiment 03's loader, reading only the forecast dates on the 8-day grid."""
    dates = [pd.Timestamp(f"{year}-05-01") + pd.Timedelta(days=int(d)) for d in ISSUE_DAYS]
    columns = ["target_date", "history_end_date", TARGET, "h3_cell", *[c for c in CELL + NEIGHBOUR if c != "day_of_year"]]
    frame = pd.read_parquet(feature_path(region, year), columns=columns, filters=[("target_date", "in", dates)])
    assert frame.history_end_date.lt(frame.target_date).all(), "Feature history overlaps target date"
    frame["day_of_year"] = frame.target_date.dt.dayofyear.astype("int16")
    return frame.reset_index(drop=True)


def burned_cells(labels, incidents, cells, region, year):
    """Cells with a satellite detection (S-NPP or NOAA-20) or a BCWS wildfire in the season."""
    return labels.grids[region, year]["pair"].any(axis=1) | ignition_grid(incidents, cells, year)[0].any(axis=1)


def score_region_year(region, year, context, labels, drivers, static, long_lightning, models, incidents, prior_burned):
    tic = time.time()
    cells = context.grids[region]["cells"]
    frame = load_days(region, year)
    assert len(frame) == len(cells) * len(ISSUE_DAYS), "expected every cell on every forecast date"
    cell = cells.get_indexer(frame.h3_cell)
    issue = np.searchsorted(ISSUE_DAYS, (frame.target_date - pd.Timestamp(f"{year}-05-01")).dt.days.to_numpy())
    out = {name: np.empty(len(frame), np.float32) for name in [*SCORED, "wx_fwi"]}
    for begin in range(0, len(frame), CHUNK):
        chunk = full_build(context, labels, drivers, static, long_lightning, region, year,
                           frame.iloc[begin:begin + CHUNK].reset_index(drop=True))
        part = slice(begin, begin + len(chunk))
        for name, (feature_set, _, sampled) in CATBOOST.items():
            out[name][part] = calibrated(models[name], chunk, feature_set) if sampled else predict(models[name], chunk, feature_set)
        out["wx_fwi"][part] = chunk.wx_fwi.to_numpy(np.float32)
        out["fwi"][part] = chunk.wx_fwi.fillna(-1).to_numpy(np.float32)
        out["recent_fire_nearby"][part] = chunk.pair_disk1_days_7d.to_numpy(np.float32)
        out["lightning_14d"][part] = (np.minimum(chunk.ltg_3x3_14d, 9999) * 1e4 + np.minimum(chunk.ltg_9x9_14d, 9999)).to_numpy(np.float32)
        del chunk
        gc.collect()
    big = incidents.loc[incidents.FIRE_CAUSE.eq("Lightning") & incidents.CURRENT_SIZE.ge(10)]
    truth = {"burn_8d": ahead(labels.grids[region, year]["pair"]),
             "new_burn_8d": ahead(experiment_20.start_grids(labels.grids[region, year], year)[1]),
             "ignition_8d": ahead(ignition_grid(incidents, cells, year)[0]),
             "big_lightning_8d": ahead(ignition_grid(big, cells, year)[0])}
    fuels = static.values(region, cells, year)[:, [STATIC.index(f) for f in FUELS]]
    table = pd.DataFrame({"region": region, "year": year, "cell": cell.astype(np.int32), "issue": issue.astype(np.int16),
                          "month": frame.target_date.dt.month.to_numpy(np.int8),
                          "land_cover": np.where(np.isnan(fuels).all(axis=1), -1, np.nanargmax(np.nan_to_num(fuels, nan=-1), axis=1))[cell].astype(np.int8),
                          "prior_burned": prior_burned[cell],
                          **{name: t[cell, issue] for name, t in truth.items()}, **out})
    print(f"{region} {year} ({time.time()-tic:.0f}s): {len(table):,} cell-dates; positives "
          + ", ".join(f"{name} {int(table[name].sum()):,}" for name in LABELS), flush=True)
    return table


# ---------------------------------------------------------------- the CanadaFireSat sample

def fwi_rank_bins(fwi, rng, bins=10):
    """Equal-count FWI bins with random tie-breaking (missing FWI ranks lowest)."""
    order = np.lexsort((rng.random(len(fwi)), np.nan_to_num(fwi, nan=-np.inf)))
    rank = np.empty(len(fwi), np.int64)
    rank[order] = np.arange(len(fwi))
    return rank * bins // max(len(fwi), 1)


def draw_test(data, label, rng, excluded):
    """Indices of positives and of 2-per-positive FWI-stratified negatives; and the negative cells."""
    positives, negatives, negative_cells = [], [], {}
    for region, d in data.groupby("region", sort=True):
        pos = d[label].to_numpy()
        if not pos.any():
            continue
        days = d.issue.to_numpy()
        first, last = days[pos].min(), days[pos].max()
        pool = ~pos & (days >= first) & (days <= last) & ~d.prior_burned.to_numpy()
        if region in excluded:
            pool &= ~np.isin(d.cell.to_numpy(), excluded[region])
        pool_index = d.index.to_numpy()[pool]
        bins = fwi_rank_bins(d.wx_fwi.to_numpy()[pool], rng)
        wanted = NEGATIVES_PER_POSITIVE * int(pos.sum())
        per_bin = np.full(10, wanted // 10) + (np.arange(10) < wanted % 10)
        chosen = [rng.choice(pool_index[bins == b], min(per_bin[b], (bins == b).sum()), replace=False) for b in range(10)]
        chosen = np.concatenate(chosen)
        positives.append(d.index.to_numpy()[pos])
        negatives.append(chosen)
        negative_cells[region] = np.unique(data.loc[chosen, "cell"].to_numpy())
    return np.concatenate(positives), np.concatenate(negatives), negative_cells


def draw_hard(data, label, rng):
    """Negatives matched to the positives' FWI deciles within Fire Centre x land cover x month."""
    positives, negatives = [], []
    for _, d in data.groupby(["region", "land_cover", "month"], sort=True):
        pos = d[label].to_numpy()
        if not pos.any():
            continue
        fwi = d.wx_fwi.to_numpy()
        pool = ~pos & ~d.prior_burned.to_numpy()
        edges = np.unique(np.nanquantile(fwi[pos], np.linspace(0, 1, 11))) if np.isfinite(fwi[pos]).any() else np.array([])
        bin_of = lambda x: np.where(np.isnan(x), -1, np.searchsorted(edges[1:-1], x, side="right"))
        pos_bins, pool_bins = bin_of(fwi[pos]), bin_of(fwi[pool])
        pool_index = d.index.to_numpy()[pool]
        for b, count in zip(*np.unique(pos_bins, return_counts=True)):
            candidates = pool_index[pool_bins == b]
            negatives.append(rng.choice(candidates, min(NEGATIVES_PER_POSITIVE * count, len(candidates)), replace=False))
        positives.append(d.index.to_numpy()[pos])
    return np.concatenate(positives), np.concatenate(negatives)


def best_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))[:-1]
    return float(thresholds[np.argmax(f1)]), float(f1.max())


def scores_at(y, score, threshold):
    flagged = score >= threshold
    tp = int((flagged & y).sum())
    precision, recall = tp / max(flagged.sum(), 1), tp / max(y.sum(), 1)
    return {"f1": 2 * precision * recall / max(precision + recall, 1e-12), "precision": precision, "recall": recall,
            "prauc": float(average_precision_score(y, score))}


def evaluate(data):
    selection, evaluation = data.loc[data.year.eq(SELECTION_YEAR)], data.loc[data.year.eq(EVALUATION_YEAR)]
    rows, sizes = [], []
    for label in LABELS:
        for seed in SEEDS:
            rng = np.random.default_rng([SEED, 24, seed, LABELS.index(label)])
            pos22, neg22, negative_cells = draw_test(selection, label, rng, {})
            pos23, neg23, _ = draw_test(evaluation, label, rng, negative_cells)
            hard_pos, hard_neg = draw_hard(evaluation, label, rng)
            tune = data.loc[np.concatenate([pos22, neg22])]
            sets = {"test": data.loc[np.concatenate([pos23, neg23])], "test_hard": data.loc[np.concatenate([hard_pos, hard_neg])]}
            for name, s in (("val_2022", tune), *sets.items()):
                sizes.append({"label": label, "seed": seed, "set": name, "positives": int(s[label].sum()),
                              "negatives": int((~s[label]).sum()), "positive_share": float(s[label].mean())})
            y_tune = tune[label].to_numpy()
            for model in SCORED:
                threshold, f1_tune = best_threshold(y_tune, tune[model].to_numpy())
                for name, s in sets.items():
                    y, score = s[label].to_numpy(), s[model].to_numpy()
                    rows.append({"label": label, "seed": seed, "set": name, "model": model, "threshold": threshold,
                                 "f1_val_2022": f1_tune, **scores_at(y, score, threshold),
                                 "f1_best_threshold": best_threshold(y, score)[1], "positive_share": float(y.mean())})
            for name, s in sets.items():
                p = float(s[label].mean())
                rows.append({"label": label, "seed": seed, "set": name, "model": "everything_is_fire", "threshold": np.nan,
                             "f1": 2 * p / (1 + p), "precision": p, "recall": 1.0, "prauc": p, "positive_share": p})
    # the same cell-dates without sampling: every cell on every forecast date
    grid_rows = []
    for label in LABELS:
        y_tune, y = selection[label].to_numpy(), evaluation[label].to_numpy()
        for model in SCORED:
            threshold, _ = best_threshold(y_tune, selection[model].to_numpy())
            grid_rows.append({"label": label, "set": "full_grid", "model": model, "threshold": threshold,
                              **scores_at(y, evaluation[model].to_numpy(), threshold), "positive_share": float(y.mean())})
        p = float(y.mean())
        grid_rows.append({"label": label, "set": "full_grid", "model": "everything_is_fire", "f1": 2 * p / (1 + p),
                          "precision": p, "recall": 1.0, "prauc": p, "positive_share": p})
    return pd.DataFrame(rows), pd.DataFrame(sizes), pd.DataFrame(grid_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--resume", action="store_true", help="reuse finished region-year score tables")
    args = parser.parse_args()
    results = args.results
    started = time.time()
    (results / "scores").mkdir(parents=True, exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    paths = {(r, y): results / "scores" / f"{slug(r)}_{y}.parquet" for r in regions for y in (SELECTION_YEAR, EVALUATION_YEAR)}
    model_paths = [path for _, path, _ in CATBOOST.values()]
    hashes = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in model_paths}

    todo = [key for key, path in paths.items() if not (args.resume and path.exists())]
    if todo:
        context = Context(regions)
        incidents = context.incidents
        labels = Labels(context, regions)
        todo_regions = sorted({r for r, _ in todo})
        prior = {}
        for region in todo_regions:
            cells = context.grids[region]["cells"]
            burned = {y: burned_cells(labels, incidents, cells, region, y) for y in (*TRAIN_YEARS, SELECTION_YEAR)}
            prior[region, SELECTION_YEAR] = np.logical_or.reduce([burned[y] for y in TRAIN_YEARS])
            prior[region, EVALUATION_YEAR] = prior[region, SELECTION_YEAR] | burned[SELECTION_YEAR]
        for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:
            del labels.grids[key]
        labels.sources = None
        gc.collect()
        drivers = Drivers(context, regions)
        static = StaticLayers()
        long_lightning = LongLightning(drivers)
        models = {}
        for name, (_, path, _) in CATBOOST.items():
            models[name] = catboost_model()
            models[name].load_model(str(path))
        for region, year in todo:
            table = score_region_year(region, year, context, labels, drivers, static, long_lightning, models,
                                      incidents, prior[region, year])
            table.to_parquet(paths[region, year], index=False)
            del table
            gc.collect()
        del context, labels, drivers, static, long_lightning
        gc.collect()

    data = pd.concat([pd.read_parquet(p) for p in paths.values()], ignore_index=True)
    metrics, sizes, grid = evaluate(data)
    metrics.to_csv(results / "protocol_metrics_by_seed.csv", index=False)
    sizes.to_csv(results / "sample_sizes.csv", index=False)
    grid.to_csv(results / "full_grid_metrics.csv", index=False)
    summary = metrics.groupby(["label", "set", "model"], sort=False).agg(
        f1=("f1", "mean"), f1_sd=("f1", "std"), precision=("precision", "mean"), recall=("recall", "mean"),
        prauc=("prauc", "mean"), f1_best_threshold=("f1_best_threshold", "mean"), positive_share=("positive_share", "mean")).reset_index()
    summary.to_csv(results / "protocol_summary.csv", index=False)

    assert hashes == {str(p.relative_to(EXPERIMENTS)): digest(p) for p in model_paths}
    (results / "manifest.json").write_text(json.dumps({
        "source": "Porta, Dalsasso, McCarty, Tuia: CanadaFireSat, arXiv 2506.08690 (v2, Nov 2025)",
        "model_hashes": hashes, "regions": regions, "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR,
        "holdout_loaded": False, "unit": "res-7 H3 cell (about 5 km2)",
        "forecast_dates": [str((pd.Timestamp("2023-05-01") + pd.Timedelta(days=int(d))).strftime("%b %d")) for d in ISSUE_DAYS],
        "window_days": WINDOW,
        "labels": {"burn_8d": "S-NPP or NOAA-20 detection in the cell on t .. t+7",
                   "new_burn_8d": "Experiment 34 fire start (gap-filled S-NPP or NOAA-20 detection with none in the cell or "
                                  "its six neighbours on the previous 7 days) in the cell on t .. t+7",
                   "ignition_8d": "BCWS wildfire (type Fire) ignites in the cell on t .. t+7",
                   "big_lightning_8d": "BCWS lightning wildfire with CURRENT_SIZE >= 10 ha ignites in the cell on t .. t+7"},
        "test_negatives": "per Fire Centre and year, 2 per positive, between the first and last positive date, excluding cells "
                          "burned (satellite or BCWS) in 2018..y-1 and, for 2023, the 2022 negative cells; equal numbers per FWI decile",
        "test_hard_negatives": "2 per positive per Fire Centre x dominant fuel class x month, matched to the positives' FWI deciles; "
                               "cells burned in 2018..2022 excluded",
        "threshold": "maximizes F1 on the 2022 sample drawn like the test set, one threshold per model and label",
        "seeds": list(SEEDS), "runtime_seconds": round(time.time() - started)}, indent=2), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(sizes.groupby(["label", "set"])[["positives", "negatives", "positive_share"]].mean().round(3).to_string(), flush=True)
    for label in LABELS:
        part = summary.loc[summary.label.eq(label)].pivot(index="model", columns="set", values=["f1", "prauc"])
        part.columns = [f"{a}_{b}" for a, b in part.columns]
        full = grid.loc[grid.label.eq(label)].set_index("model")[["f1", "prauc"]].add_suffix("_full_grid")
        print(f"\n{label}\n" + part.join(full).sort_values("f1_test", ascending=False).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
