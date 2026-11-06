"""Forecast lead time: where do learned models pull away from persistence?

Same rows, features (Experiment 13, all groups), sampling and quiet-row correction as
Experiments 03-07 and 13-14, with history ending on day e and the target moved to day e + L for
leads L = 1, 2, 3, 5, 7. One CatBoost model per lead. The baseline is the best of six
persistence rules (cell or cell + in-centre neighbours, active over the past 1, 3 or 7
days), chosen on 2022 per Fire Centre, lead and definition. Scored on the exact res-7 cell
and with Experiment 14's one-cell margin. Thresholds maximize 2022 F1 and are applied
unchanged to 2023. 2024 is never loaded.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXPERIMENT_09 = EXPERIMENTS / "14_exp_target_definitions_bc"
spec = importlib.util.spec_from_file_location("experiment_09", EXPERIMENT_09 / "run_experiment.py")
experiment_09 = sys.modules["experiment_09"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_09)
from experiment_09 import candidate_thresholds, curve, neighbourhood  # noqa: E402
from experiment_08 import (BCWS, Context, DETECTIONS, FEATURE_SETS, MAX_ABS_LOG_ODDS, QUIET_SAMPLE_RATE,  # noqa: E402
                           REFERENCE, SEASON_DAYS, SEED, SELECTION_YEAR, EVALUATION_YEAR, TARGET, TRAIN_YEARS,
                           catboost_model, design, digest, load, predict, quiet_stratum, raw_scores, season_index)

RESULTS = HERE / "results"
LEADS = (1, 2, 3, 5, 7)
FEATURE_SET = "all"
DEFINITIONS = ("res7_exact", "res7_margin1")
RULES = {"cell_1d": ["local_count_1d"], "cell_3d": ["local_count_3d"], "cell_7d": ["local_count_7d"],
         "cell_or_neighbour_1d": ["local_count_1d", "neighbour_count_1d"],
         "cell_or_neighbour_3d": ["local_count_3d", "neighbour_count_3d"],
         "cell_or_neighbour_7d": ["local_count_7d", "neighbour_count_7d"]}
FIRST_HISTORY_DAY = 6  # May 7: the first history end in the Experiment 02 tables


def target_grid(frame, cells):
    """N x 153 recorded-detection target indexed by target day (Experiment 02's in-centre target)."""
    grid = np.zeros((len(cells), SEASON_DAYS), bool)
    grid[cells.get_indexer(frame.h3_cell), season_index(frame.target_date)] = frame[TARGET].to_numpy() == 1
    return grid


def lead_targets(frame, cells, grid):
    """Per lead: target for each row (history end e -> detection on e + L), -1 past September 30."""
    cell, history = cells.get_indexer(frame.h3_cell), season_index(frame.history_end_date)
    out = {}
    for lead in LEADS:
        day = history + lead
        valid = day < SEASON_DAYS
        target = np.full(len(frame), -1, np.int8)
        target[valid] = grid[cell[valid], day[valid]]
        out[lead] = target
    return out


def to_history_grid(frame, cells, values):
    grid = np.zeros((len(cells), SEASON_DAYS), np.float32)
    grid[cells.get_indexer(frame.h3_cell), season_index(frame.history_end_date)] = values
    return grid


def training_sample(regions, context):
    """Experiment 03's draw; every row positive at any lead is kept, then each lead uses its own rule."""
    parts, stats = [], []
    for index, region in enumerate(regions):
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            targets = lead_targets(frame, cells, target_grid(frame, cells))
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            quiet = quiet_stratum(frame)
            keep = ~quiet | drawn | np.any([t == 1 for t in targets.values()], axis=0)
            part = context.features(region, year, frame.loc[keep].reset_index(drop=True))
            part = part[FEATURE_SETS[FEATURE_SET]].assign(region=region, year=year, drawn=drawn[keep])
            for lead, target in targets.items():
                part[f"target_lead{lead}"] = target[keep]
            parts.append(part)
            stats.append({"region": region, "year": year, "rows": len(frame), "kept_rows": int(keep.sum()),
                          **{f"positives_lead{lead}": int((t == 1).sum()) for lead, t in targets.items()}})
            print(f"  {region} {year}: kept {keep.sum():,} of {len(frame):,}", flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def main():
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    source_paths = [DETECTIONS, BCWS] + [REFERENCE / "regions" / r.lower().replace(" ", "_") / "data/features" / f"features_{y}.parquet"
                                         for r in regions for y in (*TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR)]
    hashes = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    context = Context(regions)
    disk7 = {region: neighbourhood(list(context.grids[region]["cells"]), list(context.grids[region]["cells"]))
             for region in regions}

    print("Building training sample (2018-2021)", flush=True)
    sample, sample_stats = training_sample(regions, context)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    quiet = quiet_stratum(sample)
    X, _ = design(sample, FEATURE_SET)
    models, fit_checks = {}, []
    for lead in LEADS:
        target = sample[f"target_lead{lead}"].to_numpy()
        # This lead's own keep rule on the shared sample: active, drawn, or positive at this lead.
        use = (target >= 0) & (~quiet | sample.drawn.to_numpy() | (target == 1))
        model = catboost_model()
        tic = time.time()
        model.fit(X[use], target[use])
        largest = float(np.abs(raw_scores(model, X[use])).max())
        assert largest < MAX_ABS_LOG_ODDS, f"lead {lead} diverged: max |log-odds| {largest:.0f}"
        fit_checks.append({"lead_days": lead, "training_rows": int(use.sum()), "training_positives": int((target[use] == 1).sum()),
                           "max_abs_training_log_odds": largest, "fit_seconds": round(time.time() - tic, 1)})
        model.save_model(str(RESULTS / "models" / f"catboost__lead{lead}.cbm"))
        models[lead] = model
        print(f"Fitted lead {lead}: {use.sum():,} rows, {(target[use] == 1).sum():,} positives, {time.time()-tic:.0f}s", flush=True)
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    del sample, X

    rows, thresholds = [], {}
    for region in regions:
        g, cells = context.grids[region], context.grids[region]["cells"]
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = context.features(region, year, load(region, year, with_cells=True))
            truth_grid = target_grid(frame, cells)
            detected_disk = (g["disk1"] @ context.arrays[year]["count"]) > 0  # any centre, by target day
            scores = {("catboost", "-"): None}
            scores |= {("rule", name): to_history_grid(frame, cells, (frame[cols].sum(axis=1) > 0).to_numpy(np.float32))
                       for name, cols in RULES.items()}
            for lead in LEADS:
                last = SEASON_DAYS - lead  # history ends FIRST_HISTORY_DAY .. last - 1
                truth = truth_grid[:, FIRST_HISTORY_DAY + lead:]
                near = detected_disk[:, FIRST_HISTORY_DAY + lead:]
                assert (near >= truth).all()
                lead_scores = {key: grid[:, FIRST_HISTORY_DAY:last] for key, grid in scores.items() if grid is not None}
                lead_scores[("catboost", "-")] = to_history_grid(frame, cells, predict(models[lead], frame, FEATURE_SET))[:, FIRST_HISTORY_DAY:last]
                for (kind, name), score in lead_scores.items():
                    neighbourhood_max = experiment_09.sparse_row_max(disk7[region], score)
                    for definition in DEFINITIONS:
                        inputs = ((score.ravel(), truth.ravel(), truth.ravel(), score.ravel()) if definition == "res7_exact"
                                  else (score.ravel(), truth.ravel(), near.ravel(), neighbourhood_max.ravel()))
                        tag = (region, lead, definition, kind, name)
                        if year == SELECTION_YEAR:
                            candidates = candidate_thresholds(inputs[0])
                            thresholds[tag] = float(candidates[int(np.argmax(curve(*inputs, candidates)[2]))])
                        precision, recall, f1, flagged = curve(*inputs, np.array([thresholds[tag]]))
                        row = {"region": region, "year": year, "lead_days": lead, "definition": definition, "kind": kind,
                               "name": name, "threshold": thresholds[tag], "precision": precision[0], "recall": recall[0],
                               "f1": f1[0], "flagged": int(flagged[0]), "positives": int(truth.sum())}
                        if year == EVALUATION_YEAR and kind == "catboost":
                            row["best_f1_any_2023_threshold"] = float(curve(*inputs, candidate_thresholds(inputs[0]))[2].max())
                        rows.append(row)
            summary = [r for r in rows if r["region"] == region and r["year"] == year and r["definition"] == "res7_margin1"
                       and (r["kind"] == "catboost" or r["name"] == "cell_1d")]
            print(f"{region} {year} (margin): " + "; ".join(f"L{r['lead_days']} {r['kind'][:4]} {r['f1']:.3f}" for r in summary), flush=True)
            del frame

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    selection = metrics.loc[metrics.year.eq(SELECTION_YEAR) & metrics.kind.eq("rule")]
    chosen = selection.loc[selection.groupby(["region", "lead_days", "definition"]).f1.idxmax(), ["region", "lead_days", "definition", "name"]]
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    best_rule = evaluation.merge(chosen.assign(kind="rule"), on=["region", "lead_days", "definition", "kind", "name"])
    model = evaluation.loc[evaluation.kind.eq("catboost")]
    yesterday = evaluation.loc[evaluation.name.eq("cell_1d")]
    comparison = (model.set_index(["region", "lead_days", "definition"])[["f1", "precision", "recall", "best_f1_any_2023_threshold"]]
                  .add_prefix("model_")
                  .join(best_rule.set_index(["region", "lead_days", "definition"])[["name", "f1", "precision", "recall"]].add_prefix("best_rule_"))
                  .join(yesterday.set_index(["region", "lead_days", "definition"])[["f1"]].add_prefix("yesterday_rule_")))
    comparison["model_minus_best_rule"] = comparison.model_f1 - comparison.best_rule_f1
    comparison.reset_index().to_csv(RESULTS / "comparison_2023.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS),
        "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "leads": list(LEADS), "lead": "history ends on day e (UTC acquisition day); target is a recorded detection on day e + L",
        "definitions": {"res7_exact": "detection in the cell on the target day",
                        "res7_margin1": "Experiment 14 one-cell margin on the target day"},
        "model": "CatBoost (Experiments 05-06 settings), Experiment 13 all-features inputs, one model per lead",
        "rules": RULES, "rule_selection": "best 2022 F1 per Fire Centre, lead and definition",
        "sampling": "Experiment 03 draw; each lead keeps active rows, drawn quiet rows and its own positives; quiet-row log-odds + log(0.01)",
        "thresholds": "maximize 2022 F1 per Fire Centre, lead, definition and model; applied unchanged to 2023",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    print(comparison.groupby(["lead_days", "definition"])[["model_f1", "best_rule_f1", "yesterday_rule_f1", "model_minus_best_rule"]]
          .median().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
