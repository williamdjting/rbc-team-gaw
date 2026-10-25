"""CatBoost on every 2018-2021 training row (Experiment 05).

Same feature sets, next-day recorded-activity target and 2018-2021 / 2022 / 2023
split as Experiments 03-04, but no quiet-negative undersampling and no quiet-row
odds correction. Two class-handling variants are fitted per feature set:
unweighted and auto_class_weights="Balanced". One pooled model per configuration
across all six Fire Centres, evaluated per centre. 2024 is never loaded.

Shared CatBoost, evaluation and comparison helpers live here; Experiment 06
imports them.
"""
from __future__ import annotations

import html
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXPERIMENT_02 = EXPERIMENTS / "02_hotspot_persistence_bc"
EXPERIMENT_03 = EXPERIMENTS / "03_exp_learned_baselines_bc"
EXPERIMENT_04 = EXPERIMENTS / "04_exp_gnn_bc"
spec = importlib.util.spec_from_file_location("experiment_03", EXPERIMENT_03 / "run_experiment.py")
experiment_03 = sys.modules["experiment_03"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_03)
from experiment_03 import (load, feature_path, slug, digest, design, quiet_stratum, threshold_metrics,  # noqa: E402
                           best_f1_threshold, FEATURE_SETS, INPUTS, QUIET, TARGET, SEED, TRAIN_YEARS,
                           SELECTION_YEAR, EVALUATION_YEAR, PREDICT_BATCH, MAX_ABS_LOG_ODDS)

RESULTS = HERE / "results"
# Sorted display names; the index seeds Experiment 03's per-region sampling draw.
REGIONS = ("Cariboo", "Coastal", "Kamloops", "Northwest", "Prince George", "Southeast")
# Fixed, untuned settings shared by Experiments 05 and 06.
CATBOOST_PARAMS = {"loss_function": "Logloss", "iterations": 300, "learning_rate": 0.1, "depth": 6,
                   "l2_leaf_reg": 3.0, "border_count": 254, "boost_from_average": True,
                   "random_seed": SEED, "thread_count": -1, "allow_writing_files": False}
VARIANTS = {"unweighted": {}, "balanced": {"auto_class_weights": "Balanced"}}
CATEGORIES = ("active_positive", "active_negative", "quiet_positive", "quiet_negative")


def regions():
    for region in REGIONS:
        manifest = json.loads((EXPERIMENT_02 / "regions" / slug(region) / "results/manifest.json").read_text())
        assert manifest["region"] == region and not manifest["holdout_loaded"], region
    return list(REGIONS)


def source_hashes():
    years = [*TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR]
    paths = [feature_path(r, y) for r in REGIONS for y in years]
    paths += [EXPERIMENT_02 / "regions" / slug(r) / "results/metrics_by_year.csv" for r in REGIONS]
    paths += [EXPERIMENT_03 / "RESULTS.html", EXPERIMENT_04 / "RESULTS.html"]
    return {str(p.relative_to(EXPERIMENTS)): digest(p) for p in paths}


def category_counts(quiet, positive):
    return {"active_positive": int((~quiet & positive).sum()), "active_negative": int((~quiet & ~positive).sum()),
            "quiet_positive": int((quiet & positive).sum()), "quiet_negative": int((quiet & ~positive).sum())}


def balance_table(configuration, before, after, training_rows, fitted_rows):
    assert sum(before.values()) == training_rows, f"{configuration}: categories do not sum to training rows"
    assert sum(after.values()) == fitted_rows, f"{configuration}: categories do not sum to fitted rows"
    table = pd.DataFrame({"configuration": configuration, "category": CATEGORIES,
                          "before_count": [before[c] for c in CATEGORIES],
                          "after_count": [after[c] for c in CATEGORIES]})
    table["before_percent"] = 100 * table.before_count / training_rows
    table["after_percent"] = 100 * table.after_count / fitted_rows
    print(f"{configuration}: {training_rows:,} training rows before, {fitted_rows:,} passed to CatBoost", flush=True)
    print(table.drop(columns="configuration").to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return table


def fill(frame, feature_set, X, y, start):
    """Write one region-year into column-major arrays (CatBoost then shares the memory)."""
    end = start + len(frame)
    for j, column in enumerate(FEATURE_SETS[feature_set]):
        X[start:end, j] = frame[column].to_numpy(dtype=np.float32)
    X[start:end, -1] = quiet_stratum(frame, feature_set)
    y[start:end] = frame[TARGET].to_numpy()
    return end


def full_training_pool(feature_set):
    rows = sum(pq.ParquetFile(feature_path(r, year)).metadata.num_rows for r in REGIONS for year in TRAIN_YEARS)
    X = np.empty((rows, len(INPUTS[feature_set])), dtype=np.float32, order="F")
    y = np.empty(rows, dtype=np.int8)
    counts, start = dict.fromkeys(CATEGORIES, 0), 0
    for region in REGIONS:
        for year in TRAIN_YEARS:
            frame = load(region, year)
            for key, value in category_counts(quiet_stratum(frame, feature_set), frame[TARGET].eq(1).to_numpy()).items():
                counts[key] += value
            start = fill(frame, feature_set, X, y, start)
            del frame
    assert start == rows
    pool = Pool(X, y, feature_names=INPUTS[feature_set])
    pool.quantize(border_count=CATBOOST_PARAMS["border_count"])
    del X
    return pool, rows, counts


def fit(pool, variant, feature_set, results=RESULTS):
    model = CatBoostClassifier(**CATBOOST_PARAMS, **VARIANTS[variant], verbose=50)
    tic = time.time()
    model.fit(pool)
    seconds = time.time() - tic
    print(f"Fitted {variant} / {feature_set} in {seconds:.0f}s", flush=True)
    (results / "models").mkdir(parents=True, exist_ok=True)
    model.save_model(str(results / "models" / f"catboost_{variant}__{feature_set}.cbm"))
    return model, {"variant": variant, "feature_set": feature_set, "training_rows": pool.num_row(),
                   "training_positives": int(np.sum(pool.get_label())), "fit_seconds": round(seconds, 1),
                   "class_weights": model.get_all_params().get("class_weights"),
                   "tree_count": model.tree_count_}


def correct_quiet(p_positive, quiet, rate):
    """Undo quiet-negative sampling on the odds scale; active rows are unchanged."""
    p_positive = p_positive.astype(np.float64)
    odds = p_positive / (1 - p_positive)
    corrected_odds = odds * rate
    return np.where(quiet, corrected_odds / (1 + corrected_odds), p_positive)


def score(model, frame, feature_set, quiet_rate=None):
    """Positive-class probability (quiet rows corrected when quiet_rate is set) and max |raw log-odds|."""
    scores, largest = [], 0.0
    for start in range(0, len(frame), PREDICT_BATCH):
        X, quiet = design(frame.iloc[start:start+PREDICT_BATCH], feature_set)
        raw = model.predict(X, prediction_type="RawFormulaVal")
        largest = max(largest, float(np.abs(raw).max()))
        # Probability from the raw log-odds; equal to predict_proba(X)[:, 1].
        p = 1 / (1 + np.exp(-raw.astype(np.float64)))
        scores.append(p if quiet_rate is None else correct_quiet(p, quiet, quiet_rate))
    return np.concatenate(scores).astype(np.float32), largest


def check_rows(region, year, frame):
    """Same rows and targets as Experiment 02: its previous-day rule must reproduce exactly."""
    rules = pd.read_csv(EXPERIMENT_02 / "regions" / slug(region) / "results/metrics_by_year.csv")
    reference = rules.loc[rules.year.eq(year) & rules.rule.eq("local_1d")].iloc[0]
    check = threshold_metrics(frame[TARGET].to_numpy(), frame.local_count_1d.to_numpy(dtype=np.float32), 1)
    assert all(check[k] == reference[k] for k in ("rows", "positive_cell_days", "tp", "fp", "fn")), (region, year)


def evaluate(models, quiet_rate=None):
    """Score every 2022 and 2023 row; choose thresholds on 2022 F1 per centre and apply them to 2023."""
    rows, thresholds, predictions, largest = [], {}, [], {key: 0.0 for key in models}
    for region in REGIONS:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = load(region, year, with_cells=year == EVALUATION_YEAR)
            check_rows(region, year, frame)
            y = frame[TARGET].to_numpy()
            if year == EVALUATION_YEAR:
                kept = frame[["target_date", "h3_cell", TARGET]].copy()
                kept.insert(0, "region", region)
                for feature_set in FEATURE_SETS:
                    kept[f"quiet__{feature_set}"] = quiet_stratum(frame, feature_set)
            for (variant, feature_set), model in models.items():
                values, peak = score(model, frame, feature_set, quiet_rate)
                largest[(variant, feature_set)] = max(largest[(variant, feature_set)], peak)
                key = (region, variant, feature_set)
                if year == SELECTION_YEAR:
                    thresholds[key] = best_f1_threshold(y, values)
                rows.append({"region": region, "year": year,
                             "role": "threshold_selection" if year == SELECTION_YEAR else "development_evaluation",
                             "model": f"catboost_{variant}", "feature_set": feature_set,
                             **threshold_metrics(y, values, thresholds[key])})
                if year == EVALUATION_YEAR:
                    kept[f"catboost_{variant}__{feature_set}"] = values
            if year == EVALUATION_YEAR:
                predictions.append(kept)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            print(f"{region} {year}: " + "; ".join(f"{r['model'][9:]}/{r['feature_set'][:9]} AP={r['average_precision']:.3f} "
                                                   f"F1={r['f1']:.3f}" for r in latest), flush=True)
            del frame
    for key, peak in largest.items():
        assert peak < MAX_ABS_LOG_ODDS, f"{key} saturated: max |log-odds| {peak:.0f}"
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["region"] = predictions.region.astype("category")
    return pd.DataFrame(rows), predictions, {f"{v}__{f}": round(p, 2) for (v, f), p in largest.items()}


def html_tables(path):
    tables = {}
    for table in re.findall(r"<table.*?</table>", path.read_text(encoding="utf-8"), re.S):
        cells = [[html.unescape(re.sub(r"<.*?>", "", c)).strip() for c in re.findall(r"<t[hd].*?</t[hd]>", row, re.S)]
                 for row in re.findall(r"<tr.*?</tr>", table, re.S)]
        tables[tuple(cells[0])] = pd.DataFrame(cells[1:], columns=cells[0])
    return tables


def number(text):
    text = text.replace(",", "")
    return float(text[:-1]) / 100 if text.endswith("%") else float(text)


def reference_metrics():
    """Published 2023 metrics of Experiments 03-04 (their results/ folders are not in the repository)."""
    names = {"Logistic regression": "logistic_regression", "Gradient boosting": "gradient_boosting",
             "cell only": "cell_only", "cell + neighbours": "cell_plus_neighbours"}
    detail = next(t for h, t in html_tables(EXPERIMENT_03 / "RESULTS.html").items() if "Correct flags" in h)
    rows = []
    for _, r in detail.iterrows():
        model, feature_set = r.Model.split(", ")
        tp, fp, fn = (int(number(r[c])) for c in ("Correct flags", "False flags", "Missed"))
        rows.append({"experiment": "03_learned_baselines", "region": r["Fire Centre"], "model": names[model],
                     "feature_set": names[feature_set], "average_precision": number(r["Average precision"]),
                     "threshold": number(r["Threshold (2022)"]), "precision": number(r.Precision),
                     "recall": number(r.Recall), "f1": number(r.F1), "tp": tp, "fp": fp, "fn": fn})
    combined = next(t for h, t in html_tables(EXPERIMENT_04 / "RESULTS.html").items() if "Source · model" in h)
    for _, r in combined.iterrows():
        source, model, feature_set = r["Source · model"].split(" · ")
        if source == "04_gnn":
            rows.append({"experiment": source, "region": r["Fire Centre"], "model": model, "feature_set": feature_set,
                         "average_precision": number(r["Average precision"]), "precision": number(r.Precision),
                         "recall": number(r.Recall), "f1": number(r.F1)})
    return pd.DataFrame(rows)


def comparison(own):
    """2023 metrics of this run beside Experiments 03-04 (and 05 when Experiment 06 calls this)."""
    reference = reference_metrics()
    positives = own.groupby("region").positive_cell_days.first()
    exp03 = reference.loc[reference.experiment.eq("03_learned_baselines")]
    # Same 2023 rows: every Experiment 03 model's flagged + missed positives equal this run's positives.
    assert (exp03.tp + exp03.fn).eq(exp03.region.map(positives)).all(), "2023 positives differ from Experiment 03"
    own = own[["experiment", "region", "model", "feature_set", "average_precision", "threshold",
               "precision", "recall", "f1", "tp", "fp", "fn", "tn"]]
    table = pd.concat([reference, own], ignore_index=True)
    table["positive_cell_days"] = table.region.map(positives)
    return table.sort_values(["region", "experiment", "model", "feature_set"]).reset_index(drop=True)


def summary(table):
    pivot = table.assign(name=table.experiment.str[:2] + " " + table.model + " / " + table.feature_set)
    return pivot.pivot(index="name", columns="region", values="average_precision").round(3)


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    regions()
    hashes = source_hashes()

    models, fits, balance = {}, [], []
    for feature_set in FEATURE_SETS:
        print(f"Building full training pool for {feature_set} (2018-2021, six Fire Centres)", flush=True)
        pool, rows, counts = full_training_pool(feature_set)
        balance.append(balance_table(feature_set, counts, counts, rows, pool.num_row()))
        for variant in VARIANTS:
            models[(variant, feature_set)], details = fit(pool, variant, feature_set)
            fits.append(details)
        del pool
    balance = pd.concat(balance, ignore_index=True)
    balance.to_csv(RESULTS / "catboost_class_balance_before_after.csv", index=False)
    pd.DataFrame(fits).to_csv(RESULTS / "catboost_fit_checks.csv", index=False)

    metrics, predictions, largest = evaluate(models)
    metrics.insert(0, "experiment", "05_catboost_full_data")
    metrics.to_csv(RESULTS / "catboost_metrics_by_year.csv", index=False)
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    evaluation.to_csv(RESULTS / "catboost_metrics_2023.csv", index=False)
    predictions.to_parquet(RESULTS / "catboost_predictions_2023.parquet", index=False)
    keys = ["region", "target_date", "h3_cell", TARGET] + [f"quiet__{f}" for f in FEATURE_SETS]
    for variant in VARIANTS:
        evaluation.loc[evaluation.model.eq(f"catboost_{variant}")].to_csv(
            RESULTS / f"catboost_metrics_2023_{variant}.csv", index=False)
        predictions[keys + [f"catboost_{variant}__{f}" for f in FEATURE_SETS]].to_parquet(
            RESULTS / f"catboost_predictions_2023_{variant}.parquet", index=False)
    del predictions
    compared = comparison(evaluation)
    compared.to_csv(RESULTS / "catboost_comparison_2023.csv", index=False)

    assert hashes == source_hashes(), "Source files changed during the run"
    thresholds = evaluation.set_index(["region", "model", "feature_set"]).threshold
    (RESULTS / "catboost_manifest.json").write_text(json.dumps({
        "experiment": "05_catboost_full_data", "source_hashes": hashes, "regions": list(REGIONS),
        "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR,
        "holdout_loaded": False,
        "target": "target_recorded_detection: at least one FIRMS detection recorded in the cell on the target day; "
                  "zero is not confirmed absence; positives are recorded hotspot activity, not ignitions",
        "feature_sets": INPUTS, "quiet_definition": QUIET,
        "day_of_year": "derived from target_date; quiet_stratum derived from the set's own counts",
        "training": "one pooled model per configuration across all six Fire Centres, every 2018-2021 row",
        "class_handling": {"unweighted": "no weights, no sampling, no odds correction",
                           "balanced": "auto_class_weights='Balanced' (negative weight 1, positive weight "
                                       "= negatives / positives); scores are not probability-calibrated"},
        "sampling": "none; quiet-row odds correction not applied",
        "class_balance": balance.to_dict(orient="records"),
        "fits": fits, "catboost_params": CATBOOST_PARAMS,
        "max_abs_evaluation_log_odds": largest,
        "threshold_rule": "per Fire Centre and configuration, maximize 2022 F1; applied unchanged to 2023",
        "thresholds": {" | ".join(k): v for k, v in thresholds.items()},
        "comparison": "Experiments 03-04 metrics parsed from their RESULTS.html (rounded to 3 decimals); "
                      "same 2023 rows checked via Experiment 03 TP+FN per centre",
        "checks": "past-only history, no 2024 rows, four categories sum to training rows, Experiment 02 "
                  "previous-day rule reproduced on every evaluated region-year, no saturated log-odds, "
                  "source hashes unchanged",
        "seed": SEED, "runtime_seconds": round(time.time() - started),
        "versions": {"catboost": catboost.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]}},
        indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s. Results: {RESULTS}", flush=True)
    print(summary(compared).to_string(), flush=True)


if __name__ == "__main__":
    main()
