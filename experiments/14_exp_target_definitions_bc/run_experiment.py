"""How the target definition sets the F1 ceiling, with Experiment 13's fire-context models.

Scores the previous-day rule and four models under four definitions of a correct forecast:
- res7_exact: the Experiment 02-07 and 13 target (a detection in this H3 res-7 cell tomorrow);
- res7_margin1: res-7 forecasts; a flagged cell counts as correct if a detection falls in
  it or a neighbouring cell, and a detected cell counts as caught if it or a neighbour is
  flagged (neighbourhood verification);
- res6_exact: forecasts and detections on H3 res-6 cells (about 7 res-7 cells each);
  a res-6 cell's score is its highest res-7 child score;
- res6_margin1: res-6 with the one-cell margin.
Models: Experiment 13's gradient boosting and CatBoost on all fire-context features (trained
on the strict target), and the same two retrained on "a detection in the cell or a
neighbour tomorrow". Thresholds maximize 2022 F1 per Fire Centre, definition and model and
are applied unchanged to 2023. 2024 is never loaded.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import h3
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXPERIMENT_08 = EXPERIMENTS / "13_exp_fire_context_bc"
spec = importlib.util.spec_from_file_location("experiment_08", EXPERIMENT_08 / "run_experiment.py")
experiment_08 = sys.modules["experiment_08"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_08)
from experiment_08 import (Context, DETECTIONS, BCWS, MAX_ABS_LOG_ODDS, QUIET_SAMPLE_RATE, REFERENCE,  # noqa: E402
                           SEED, SELECTION_YEAR, EVALUATION_YEAR, TARGET, TRAIN_YEARS, catboost_model, design,
                           digest, gradient_boosting, load, predict, quiet_stratum, raw_scores, season_index, slug,
                           sparse_row_max)

RESULTS = HERE / "results"
FEATURE_SET = "all"
DEFINITIONS = ("res7_exact", "res7_margin1", "res6_exact", "res6_margin1")
DISK_TARGET = "target_disk1"
FIRST_TARGET_DAY = 7  # May 8
CANDIDATES = 1 - np.geomspace(1e-7, 0.05, 4000)  # score quantiles covering the top 5% of rows


# ---------------------------------------------------------------- spatial helpers

def neighbourhood(cells, columns, rings=1):
    """Sparse (cells x columns) map from each cell to the columns within `rings` rings, itself included."""
    position = {c: i for i, c in enumerate(columns)}
    rows, cols = [], []
    for i, cell in enumerate(cells):
        for other in h3.grid_disk(cell, rings):
            j = position.get(other)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(cells), len(columns)))


def membership(children_parent, parents):
    """Sparse (parents x children) map from each parent to its children."""
    position = {p: i for i, p in enumerate(parents)}
    rows = [position[p] for p in children_parent]
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, np.arange(len(rows)))),
                         shape=(len(parents), len(children_parent)))


class Geometry:
    """Res-7 and res-6 neighbourhoods for every Fire Centre, plus res-6 daily detections."""

    def __init__(self, context, regions):
        universe = list(context.universe)
        universe_parent = [h3.cell_to_parent(c, 6) for c in universe]
        self.universe6 = list(dict.fromkeys(sorted(universe_parent)))
        to_parent = membership(universe_parent, self.universe6)
        self.count6 = {year: to_parent @ a["count"] for year, a in context.arrays.items()}
        self.regions = {}
        for region in regions:
            cells = list(context.grids[region]["cells"])
            parent = [h3.cell_to_parent(c, 6) for c in cells]
            cells6 = list(dict.fromkeys(sorted(parent)))
            self.regions[region] = {
                "disk7": neighbourhood(cells, cells),
                "children": membership(parent, cells6),
                "disk6": neighbourhood(cells6, cells6),
                "near6": neighbourhood(cells6, self.universe6),
                "own6": neighbourhood(cells6, self.universe6, rings=0)}


def disk_target(context, region, year, frame):
    """1 if any detection (any Fire Centre) falls in the cell or a ring-1 neighbour on the target day."""
    g = context.grids[region]
    disk = g["disk1"] @ context.arrays[year]["count"]
    cell = g["cells"].get_indexer(frame.h3_cell)
    return (disk[cell, season_index(frame.target_date)] > 0).astype(np.int8)


# ---------------------------------------------------------------- metrics

def to_grid(frame, cells, values):
    grid = np.zeros((len(cells), experiment_08.SEASON_DAYS), np.float32)
    grid[cells.get_indexer(frame.h3_cell), season_index(frame.target_date)] = values
    return grid[:, FIRST_TARGET_DAY:]


def curve(score, truth, precision_hit, recall_score, thresholds):
    """Precision, recall and F1 at each threshold.

    A flag is correct when precision_hit is true; a positive is caught when its recall_score
    (its own score, or the highest score in its neighbourhood) reaches the threshold."""
    flagged = len(score) - np.searchsorted(np.sort(score), thresholds, side="left")
    correct = np.sort(score[precision_hit])
    correct = len(correct) - np.searchsorted(correct, thresholds, side="left")
    caught = np.sort(recall_score[truth])
    caught = len(caught) - np.searchsorted(caught, thresholds, side="left")
    precision = correct / np.maximum(flagged, 1)
    recall = caught / max(int(truth.sum()), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, flagged


def definition_inputs(geometry, region, year, scores, y, near7):
    """For each definition: flattened score, truth, precision hit and recall score per model."""
    g = geometry.regions[region]
    count6 = geometry.count6[year]
    own6 = (g["own6"] @ count6)[:, FIRST_TARGET_DAY:] > 0
    near6 = (g["near6"] @ count6)[:, FIRST_TARGET_DAY:] > 0
    y6 = sparse_row_max(g["children"], y.astype(np.float32)) > 0
    assert (own6 >= y6).all()  # res-6 detections in any centre include this centre's
    out = {}
    for name, score in scores.items():
        score6 = sparse_row_max(g["children"], score)
        out[("res7_exact", name)] = (score.ravel(), y.ravel(), y.ravel(), score.ravel())
        out[("res7_margin1", name)] = (score.ravel(), y.ravel(), near7.ravel(),
                                       sparse_row_max(g["disk7"], score).ravel())
        out[("res6_exact", name)] = (score6.ravel(), y6.ravel(), y6.ravel(), score6.ravel())
        out[("res6_margin1", name)] = (score6.ravel(), y6.ravel(), near6.ravel(),
                                       sparse_row_max(g["disk6"], score6).ravel())
    return out


def candidate_thresholds(score):
    if np.isin(score, (0, 1)).all():
        return np.array([0.5])
    return np.unique(np.quantile(score, CANDIDATES))


# ---------------------------------------------------------------- main

def training_sample(regions, context):
    """Experiment 03's draw, keeping every row positive for the cell-or-neighbour target."""
    parts, stats = [], []
    for index, region in enumerate(regions):
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            disk = disk_target(context, region, year, frame)
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            quiet = quiet_stratum(frame)
            keep = ~quiet | (disk == 1) | drawn
            part = context.features(region, year, frame.loc[keep].reset_index(drop=True))
            part[DISK_TARGET] = disk[keep]
            parts.append(part[experiment_08.FEATURE_SETS[FEATURE_SET] + [TARGET, DISK_TARGET]])
            stats.append({"region": region, "year": year, "rows": len(frame), "strict_positives": int(frame[TARGET].sum()),
                          "disk_positives": int(disk.sum()), "quiet_disk_positives": int((quiet & (disk == 1)).sum()),
                          "kept_rows": int(keep.sum())})
            print(f"  {region} {year}: kept {keep.sum():,}; cell-or-neighbour positives {disk.sum():,} "
                  f"(strict {frame[TARGET].sum():,})", flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def main():
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    manifest_08 = json.loads((EXPERIMENT_08 / "results/manifest.json").read_text())
    assert not manifest_08["holdout_loaded"]
    model_paths = {("gradient_boosting", "strict"): EXPERIMENT_08 / "results/models/gradient_boosting__all.joblib",
                   ("catboost", "strict"): EXPERIMENT_08 / "results/models/catboost__all.cbm"}
    source_paths = [DETECTIONS, BCWS, EXPERIMENT_08 / "results/metrics_by_year.csv", *model_paths.values()]
    hashes = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    context = Context(regions)
    geometry = Geometry(context, regions)
    print(f"Geometry ready: {len(geometry.universe6):,} detected res-6 cells", flush=True)

    models = {("gradient_boosting", "strict"): joblib.load(model_paths[("gradient_boosting", "strict")])}
    catboost = catboost_model()
    catboost.load_model(str(model_paths[("catboost", "strict")]))
    models[("catboost", "strict")] = catboost

    print("Building cell-or-neighbour training sample (2018-2021)", flush=True)
    sample, sample_stats = training_sample(regions, context)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    X, _ = design(sample, FEATURE_SET)
    y_disk = sample[DISK_TARGET].to_numpy()
    fit_checks = []
    for name in ("gradient_boosting", "catboost"):
        model = gradient_boosting() if name == "gradient_boosting" else catboost_model()
        tic = time.time()
        model.fit(X, y_disk)
        largest = float(np.abs(raw_scores(model, X)).max())
        assert largest < MAX_ABS_LOG_ODDS, f"{name} diverged: max |log-odds| {largest:.0f}"
        fit_checks.append({"model": name, "training_target": "cell_or_neighbour", "training_rows": len(X),
                           "training_positives": int(y_disk.sum()), "max_abs_training_log_odds": largest,
                           "fit_seconds": round(time.time() - tic, 1)})
        print(f"Fitted {name} on the cell-or-neighbour target in {time.time()-tic:.0f}s", flush=True)
        models[(name, "cell_or_neighbour")] = model
        if name == "catboost":
            model.save_model(str(RESULTS / "models/catboost__cell_or_neighbour.cbm"))
        else:
            joblib.dump(model, RESULTS / "models/gradient_boosting__cell_or_neighbour.joblib")
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    del sample, X

    reference = pd.read_csv(REFERENCE / "results/all_region_metrics.csv")
    rows, thresholds = [], {}
    for region in regions:
        cells = context.grids[region]["cells"]
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = context.features(region, year, load(region, year, with_cells=True))
            y = to_grid(frame, cells, frame[TARGET].to_numpy()) > 0
            near7 = to_grid(frame, cells, disk_target(context, region, year, frame)) > 0
            assert (near7 >= y).all()
            scores = {("rule_local_1d", "fixed"): to_grid(frame, cells, (frame.local_count_1d > 0).to_numpy(np.float32))}
            for key, model in models.items():
                scores[key] = to_grid(frame, cells, predict(model, frame, FEATURE_SET))
            for (definition, key), (score, truth, hit, recall_score) in definition_inputs(
                    geometry, region, year, scores, y, near7).items():
                tag = (region, definition, *key)
                if year == SELECTION_YEAR:
                    candidates = candidate_thresholds(score)
                    _, _, f1, _ = curve(score, truth, hit, recall_score, candidates)
                    thresholds[tag] = float(candidates[int(np.argmax(f1))])
                precision, recall, f1, flagged = curve(score, truth, hit, recall_score, np.array([thresholds[tag]]))
                row = {"region": region, "year": year, "definition": definition, "model": key[0],
                       "training_target": key[1], "threshold": thresholds[tag], "precision": precision[0],
                       "recall": recall[0], "f1": f1[0], "flagged": int(flagged[0]), "positives": int(truth.sum()),
                       "rows": len(score)}
                if year == EVALUATION_YEAR:
                    _, _, every, _ = curve(score, truth, hit, recall_score, candidate_thresholds(score))
                    row["best_f1_any_2023_threshold"] = float(every.max())
                rows.append(row)
            # The strict rule must reproduce Experiment 02's previous-day rule exactly.
            mine = [r for r in rows if r["region"] == region and r["year"] == year and r["definition"] == "res7_exact"
                    and r["model"] == "rule_local_1d"][0]
            theirs = reference.loc[reference.region.eq(region) & reference.year.eq(year) & reference.rule.eq("local_1d")].iloc[0]
            assert mine["positives"] == theirs.positive_cell_days and abs(mine["f1"] - theirs.f1) < 1e-9, (region, year)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            print(f"{region} {year}: " + "; ".join(f"{r['definition']}/{r['model'][:8]}/{r['training_target'][:6]} "
                                                   f"F1={r['f1']:.3f}" for r in latest), flush=True)
            del frame, scores

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.pivot_table(index=["definition", "model", "training_target"], values=["f1", "precision", "recall"],
                                     aggfunc="median").reset_index()
    summary.to_csv(RESULTS / "summary_2023.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS),
        "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "definitions": {
            "res7_exact": "Experiment 02-07 and 13 target: detection in this res-7 cell on the target day",
            "res7_margin1": "res-7 forecasts; precision counts flags with a detection (any centre) in the cell or a "
                            "ring-1 neighbour; recall counts detected cells with a flag in the cell or an in-centre "
                            "ring-1 neighbour",
            "res6_exact": "res-6 cells; score = highest res-7 child score; truth = any child detection (this centre)",
            "res6_margin1": "res-6 with the same one-cell margin at res-6"},
        "models": {"strict": "Experiment 13 all-features models, trained on the strict target",
                   "cell_or_neighbour": "same features, settings and sampling rule, retrained on a detection in the "
                                        "cell or a ring-1 neighbour (any centre); positives of that target all kept"},
        "feature_set": experiment_08.FEATURE_SETS[FEATURE_SET],
        "thresholds": "per Fire Centre, definition and model, maximize 2022 F1 over score quantiles covering the "
                      "top 5% of rows; applied unchanged to 2023; best_f1_any_2023_threshold is an after-the-fact reference",
        "checks": "strict rule reproduces Experiment 02 F1 on every region-year; source hashes unchanged; no 2024 rows",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
