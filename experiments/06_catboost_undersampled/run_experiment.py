"""CatBoost with quiet-negative undersampling (Experiment 06).

The fair CatBoost comparison with Experiments 03-04: the identical training sample
(every positive, every row with activity in the past seven days, and the same ~1%
random draw of quiet negatives), unweighted CatBoost, and the 0.01 odds correction
on every quiet row at prediction. Evaluation uses every 2022 and 2023 row.
CatBoost settings and helpers come from Experiment 05. 2024 is never loaded.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import Pool

HERE = Path(__file__).resolve().parent
EXPERIMENT_05 = HERE.parent / "05_catboost_full_data"
spec = importlib.util.spec_from_file_location("experiment_05", EXPERIMENT_05 / "run_experiment.py")
experiment_05 = sys.modules["experiment_05"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_05)
from experiment_05 import (load, design, quiet_stratum, category_counts, balance_table, fit, evaluate,  # noqa: E402
                           comparison, summary, regions, source_hashes, FEATURE_SETS, INPUTS, QUIET, TARGET, SEED,
                           TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR, REGIONS, CATBOOST_PARAMS, CATEGORIES)

RESULTS = HERE / "results"
QUIET_SAMPLE_RATE = 0.01
# Training rows kept by Experiment 03 (its RESULTS.html); the same draw must reproduce them.
EXPERIMENT_03_KEPT_ROWS = {"cell_only": 1_514_696, "cell_plus_neighbours": 1_758_947}


def training_samples():
    """Experiment 03's sample: one draw per row, shared by both feature sets, seeded by (SEED, year, region index)."""
    parts = {name: [] for name in FEATURE_SETS}
    before = {name: dict.fromkeys(CATEGORIES, 0) for name in FEATURE_SETS}
    after = {name: dict(counts) for name, counts in before.items()}
    rows = 0
    for index, region in enumerate(REGIONS):
        for year in TRAIN_YEARS:
            frame = load(region, year)
            rows += len(frame)
            positive = frame[TARGET].eq(1).to_numpy()
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            for feature_set in FEATURE_SETS:
                quiet = quiet_stratum(frame, feature_set)
                keep = ~quiet | positive | drawn
                for key, value in category_counts(quiet, positive).items():
                    before[feature_set][key] += value
                for key, value in category_counts(quiet[keep], positive[keep]).items():
                    after[feature_set][key] += value
                parts[feature_set].append(frame.loc[keep])
            print(f"  {region} {year}: kept {len(parts['cell_only'][-1]):,} / "
                  f"{len(parts['cell_plus_neighbours'][-1]):,} of {len(frame):,} rows (cell only / + neighbours)",
                  flush=True)
            del frame
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}, rows, before, after


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    regions()
    hashes = source_hashes()

    print("Building undersampled training samples (2018-2021, six Fire Centres)", flush=True)
    samples, rows, before, after = training_samples()
    models, fits, balance, sampling = {}, [], [], {}
    for feature_set, train in samples.items():
        assert after[feature_set]["quiet_positive"] + after[feature_set]["active_positive"] == \
            before[feature_set]["quiet_positive"] + before[feature_set]["active_positive"], "a positive was dropped"
        balance.append(balance_table(feature_set, before[feature_set], after[feature_set], rows, len(train)))
        sampling[feature_set] = {"source_rows": rows, "kept_rows": len(train),
                                 "matches_experiment_03": len(train) == EXPERIMENT_03_KEPT_ROWS[feature_set]}
        X, _ = design(train, feature_set)
        pool = Pool(X, train[TARGET].to_numpy(), feature_names=INPUTS[feature_set])
        models[("unweighted", feature_set)], details = fit(pool, "unweighted", feature_set, RESULTS)
        fits.append(details)
        del X, pool
    del samples
    balance = pd.concat(balance, ignore_index=True)
    balance.to_csv(RESULTS / "catboost_class_balance_before_after.csv", index=False)
    pd.DataFrame(fits).to_csv(RESULTS / "catboost_fit_checks.csv", index=False)

    metrics, predictions, largest = evaluate(models, quiet_rate=QUIET_SAMPLE_RATE)
    metrics.insert(0, "experiment", "06_catboost_undersampled")
    metrics.to_csv(RESULTS / "catboost_metrics_by_year.csv", index=False)
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    evaluation.to_csv(RESULTS / "catboost_metrics_2023.csv", index=False)
    predictions.to_parquet(RESULTS / "catboost_predictions_2023.parquet", index=False)
    del predictions
    own = evaluation
    full_data = EXPERIMENT_05 / "results/catboost_metrics_2023.csv"
    if full_data.exists():
        own = pd.concat([pd.read_csv(full_data), evaluation], ignore_index=True)
    compared = comparison(own)
    compared.to_csv(RESULTS / "catboost_comparison_2023.csv", index=False)

    assert hashes == source_hashes(), "Source files changed during the run"
    thresholds = evaluation.set_index(["region", "model", "feature_set"]).threshold
    (RESULTS / "catboost_manifest.json").write_text(json.dumps({
        "experiment": "06_catboost_undersampled", "source_hashes": hashes, "regions": list(REGIONS),
        "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR,
        "holdout_loaded": False,
        "target": "target_recorded_detection: at least one FIRMS detection recorded in the cell on the target day; "
                  "zero is not confirmed absence; positives are recorded hotspot activity, not ignitions",
        "feature_sets": INPUTS, "quiet_definition": QUIET,
        "day_of_year": "derived from target_date; quiet_stratum derived from the set's own counts",
        "training": "one pooled model per configuration across all six Fire Centres",
        "class_handling": "quiet-negative undersampling, CatBoost without class weights",
        "sampling": {"rule": "keep all positives and all active rows (activity in the past 7 days: cell for "
                             "cell_only, cell or in-region neighbours for cell_plus_neighbours); retain quiet "
                             "negatives with probability 0.01 using Experiment 03's per-row draw",
                     "quiet_sample_rate": QUIET_SAMPLE_RATE, "seed": SEED, "counts": sampling},
        "score_correction": "quiet rows: odds = p / (1 - p); corrected = odds * 0.01 / (1 + odds * 0.01); "
                            "active rows unchanged; sampling correction, not fitted calibration",
        "class_balance": balance.to_dict(orient="records"),
        "fits": fits, "catboost_params": CATBOOST_PARAMS,
        "max_abs_evaluation_log_odds": largest,
        "threshold_rule": "per Fire Centre and configuration, maximize corrected 2022 F1; applied unchanged to 2023",
        "thresholds": {" | ".join(k): v for k, v in thresholds.items()},
        "evaluation": "every 2022 and 2023 row; evaluation data not undersampled",
        "comparison": "Experiments 03-04 metrics parsed from their RESULTS.html (rounded to 3 decimals); "
                      "Experiment 05 from its results; same 2023 rows checked via Experiment 03 TP+FN per centre",
        "checks": "past-only history, no 2024 rows, every positive kept, four categories sum before and after "
                  "sampling, Experiment 02 previous-day rule reproduced on every evaluated region-year, "
                  "no saturated log-odds, source hashes unchanged",
        "seed": SEED, "runtime_seconds": round(time.time() - started),
        "versions": {"catboost": catboost.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]}},
        indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s. Results: {RESULTS}", flush=True)
    print(summary(compared).to_string(), flush=True)


if __name__ == "__main__":
    main()
