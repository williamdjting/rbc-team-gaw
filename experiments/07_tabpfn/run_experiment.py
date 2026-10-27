"""TabPFN on the active stratum, CatBoost on quiet rows (Experiment 07).

TabPFN is an in-context model: the "training" rows are a context passed in at every
prediction, capped at a few thousand to tens of thousands of rows. Experiment 06's
1.5M-row sample is far beyond that, and scoring every 2022-2023 row (~68M) is not
feasible, so this run is a hybrid:

- Active rows (activity in the past 7 days, as in Experiments 03-06) are scored by
  TabPFN. Its context is a uniform random draw of 2018-2021 active rows, so scores
  are on the active population's scale and need no odds correction. Several draws
  are averaged.
- Quiet rows (~99.9% of rows) keep Experiment 06's CatBoost scores, with its 0.01
  quiet-row odds correction.

Thresholds and metrics follow Experiments 05-06: same feature sets, split, every
2022 and 2023 row evaluated, per-centre 2022 F1 threshold applied unchanged to
2023. CatBoost models and helpers come from Experiments 05-06. 2024 is never loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from tabpfn import TabPFNClassifier

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXPERIMENT_05 = EXPERIMENTS / "05_catboost_full_data"
EXPERIMENT_06 = EXPERIMENTS / "06_catboost_undersampled"
spec = importlib.util.spec_from_file_location("experiment_05", EXPERIMENT_05 / "run_experiment.py")
experiment_05 = sys.modules["experiment_05"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_05)
from experiment_05 import (load, design, quiet_stratum, category_counts, threshold_metrics, best_f1_threshold,  # noqa: E402
                           check_rows, score, comparison, summary, regions, source_hashes, FEATURE_SETS, INPUTS,
                           QUIET, TARGET, SEED, TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR, REGIONS, CATEGORIES)

RESULTS = HERE / "results"
QUIET_SAMPLE_RATE = 0.01  # Experiment 06's rate; its CatBoost quiet scores are corrected with it.
# quiet_stratum is constant (0) on active rows, so TabPFN sees the feature set without it.
TABPFN_INPUTS = {name: columns for name, columns in FEATURE_SETS.items()}
SETTINGS = {
    "full": {"context_rows": 10_000, "contexts": 3, "n_estimators": 4, "predict_batch": 5_000},
    "quick": {"context_rows": 2_000, "contexts": 1, "n_estimators": 1, "predict_batch": 5_000},
}


def device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def active_training_rows():
    """Every 2018-2021 active row per feature set, with the four-category counts of the source rows."""
    parts = {name: [] for name in FEATURE_SETS}
    counts = {name: dict.fromkeys(CATEGORIES, 0) for name in FEATURE_SETS}
    rows = 0
    for region in REGIONS:
        for year in TRAIN_YEARS:
            frame = load(region, year)
            rows += len(frame)
            positive = frame[TARGET].eq(1).to_numpy()
            for feature_set in FEATURE_SETS:
                quiet = quiet_stratum(frame, feature_set)
                for key, value in category_counts(quiet, positive).items():
                    counts[feature_set][key] += value
                parts[feature_set].append(frame.loc[~quiet, TABPFN_INPUTS[feature_set] + [TARGET]])
            print(f"  {region} {year}: {len(parts['cell_only'][-1]):,} / {len(parts['cell_plus_neighbours'][-1]):,} "
                  f"active of {len(frame):,} rows (cell only / + neighbours)", flush=True)
            del frame
    active = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    for feature_set, frame in active.items():
        assert sum(counts[feature_set].values()) == rows, f"{feature_set}: categories do not sum to training rows"
        assert len(frame) == counts[feature_set]["active_positive"] + counts[feature_set]["active_negative"]
    return active, rows, counts


def check_against_experiment_06(counts):
    """Experiment 06 kept every active row, so its active counts must equal these."""
    path = EXPERIMENT_06 / "results/catboost_class_balance_before_after.csv"
    if not path.exists():
        return False
    reference = pd.read_csv(path).set_index(["configuration", "category"]).before_count
    for feature_set, values in counts.items():
        for category, value in values.items():
            assert reference[(feature_set, category)] == value, (feature_set, category)
    return True


def contexts(active, settings):
    """Uniform random context draws from the active rows; seeded by (SEED, feature set, draw)."""
    drawn, balance = {}, []
    for index, (feature_set, frame) in enumerate(active.items()):
        size = min(settings["context_rows"], len(frame))
        for draw in range(settings["contexts"]):
            rows = np.random.default_rng([SEED, index, draw]).choice(len(frame), size=size, replace=False)
            context = frame.iloc[np.sort(rows)]
            drawn[(feature_set, draw)] = (context[TABPFN_INPUTS[feature_set]].to_numpy(dtype=np.float32),
                                          context[TARGET].to_numpy(dtype=np.int64))
            positives = int(context[TARGET].sum())
            balance.append({"feature_set": feature_set, "draw": draw, "context_rows": size,
                            "context_positives": positives, "context_prevalence": positives / size,
                            "active_rows": len(frame), "active_prevalence": float(frame[TARGET].mean())})
    balance = pd.DataFrame(balance)
    print(balance.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return drawn, balance


def fit_tabpfn(drawn, settings, target_device):
    models, fits = {}, []
    for (feature_set, draw), (X, y) in drawn.items():
        model = TabPFNClassifier(n_estimators=settings["n_estimators"], device=target_device,
                                 random_state=SEED + draw, ignore_pretraining_limits=True)
        tic = time.time()
        model.fit(X, y)
        seconds = time.time() - tic
        assert list(model.classes_) == [0, 1], model.classes_
        models[(feature_set, draw)] = model
        fits.append({"feature_set": feature_set, "draw": draw, "context_rows": len(y),
                     "context_positives": int(y.sum()), "fit_seconds": round(seconds, 1)})
        print(f"Fitted TabPFN {feature_set} draw {draw} on {len(y):,} rows in {seconds:.0f}s", flush=True)
    return models, fits


def tabpfn_score(models, X, feature_set, settings, timing):
    """Positive-class probability averaged over the context draws, predicted in batches."""
    draws = [model for (name, _), model in models.items() if name == feature_set]
    total = np.zeros(len(X), dtype=np.float64)
    for model in draws:
        for start in range(0, len(X), settings["predict_batch"]):
            tic = time.time()
            batch = X[start:start+settings["predict_batch"]]
            total[start:start+len(batch)] += model.predict_proba(batch)[:, 1]
            timing["rows"] += len(batch)
            timing["seconds"] += time.time() - tic
    return (total / len(draws)).astype(np.float32)


def evaluate(models, catboost_models, settings):
    """Hybrid scores on every 2022 and 2023 row; thresholds from 2022 F1 per centre, applied to 2023."""
    rows, thresholds, predictions = [], {}, []
    timing, scored = {"rows": 0, "seconds": 0.0}, {name: {"active": 0, "quiet": 0} for name in FEATURE_SETS}
    for region in REGIONS:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = load(region, year, with_cells=year == EVALUATION_YEAR)
            check_rows(region, year, frame)
            y = frame[TARGET].to_numpy()
            if year == EVALUATION_YEAR:
                kept = frame[["target_date", "h3_cell", TARGET]].copy()
                kept.insert(0, "region", region)
            for feature_set in FEATURE_SETS:
                quiet = quiet_stratum(frame, feature_set)
                catboost_values, _ = score(catboost_models[feature_set], frame, feature_set, QUIET_SAMPLE_RATE)
                values = catboost_values.copy()
                X = frame.loc[~quiet, TABPFN_INPUTS[feature_set]].to_numpy(dtype=np.float32)
                values[~quiet] = tabpfn_score(models, X, feature_set, settings, timing)
                scored[feature_set]["active"] += int((~quiet).sum())
                scored[feature_set]["quiet"] += int(quiet.sum())
                key = (region, feature_set)
                if year == SELECTION_YEAR:
                    thresholds[key] = best_f1_threshold(y, values)
                rows.append({"region": region, "year": year,
                             "role": "threshold_selection" if year == SELECTION_YEAR else "development_evaluation",
                             "model": "tabpfn_hybrid", "feature_set": feature_set,
                             **threshold_metrics(y, values, thresholds[key])})
                # Active-stratum-only AP: isolates TabPFN against CatBoost on the rows TabPFN scores.
                if (~quiet).any() and y[~quiet].any():
                    rows[-1]["active_ap_tabpfn"] = average_precision_score(y[~quiet], values[~quiet])
                    rows[-1]["active_ap_catboost_06"] = average_precision_score(y[~quiet], catboost_values[~quiet])
                if year == EVALUATION_YEAR:
                    kept[f"quiet__{feature_set}"] = quiet
                    kept[f"tabpfn_hybrid__{feature_set}"] = values
            if year == EVALUATION_YEAR:
                predictions.append(kept)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            rate = timing["rows"] / max(timing["seconds"], 1e-9)
            print(f"{region} {year}: " + "; ".join(
                f"{r['feature_set'][:9]} AP={r['average_precision']:.3f} F1={r['f1']:.3f} "
                f"activeAP tabpfn={r.get('active_ap_tabpfn', float('nan')):.3f} "
                f"catboost={r.get('active_ap_catboost_06', float('nan')):.3f}" for r in latest)
                + f" [{rate:,.0f} TabPFN rows/s]", flush=True)
            del frame
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["region"] = predictions.region.astype("category")
    return pd.DataFrame(rows), predictions, scored, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small context, one draw, one estimator (smoke test)")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--context-rows", type=int, help="override the context size per draw")
    parser.add_argument("--contexts", type=int, help="override the number of context draws")
    args = parser.parse_args()
    settings = dict(SETTINGS["quick" if args.quick else "full"])
    if args.context_rows:
        settings["context_rows"] = args.context_rows
    if args.contexts:
        settings["contexts"] = args.contexts
    target_device = device(args.device)
    print(f"TabPFN on {target_device} with {settings}", flush=True)

    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    regions()
    hashes = source_hashes()
    catboost_models = {}
    for feature_set in FEATURE_SETS:
        path = EXPERIMENT_06 / "results/models" / f"catboost_unweighted__{feature_set}.cbm"
        assert path.exists(), f"Run Experiment 06 first: {path} is missing"
        catboost_models[feature_set] = CatBoostClassifier().load_model(str(path))

    print("Collecting active training rows (2018-2021, six Fire Centres)", flush=True)
    active, rows, counts = active_training_rows()
    matches_06 = check_against_experiment_06(counts)
    drawn, context_balance = contexts(active, settings)
    del active
    context_balance.to_csv(RESULTS / "tabpfn_context_balance.csv", index=False)
    models, fits = fit_tabpfn(drawn, settings, target_device)
    pd.DataFrame(fits).to_csv(RESULTS / "tabpfn_fit_checks.csv", index=False)

    metrics, predictions, scored, timing = evaluate(models, catboost_models, settings)
    metrics.insert(0, "experiment", "07_tabpfn")
    metrics.to_csv(RESULTS / "tabpfn_metrics_by_year.csv", index=False)
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    evaluation.to_csv(RESULTS / "tabpfn_metrics_2023.csv", index=False)
    predictions.to_parquet(RESULTS / "tabpfn_predictions_2023.parquet", index=False)
    del predictions
    own = [evaluation]
    for path in (EXPERIMENT_05 / "results/catboost_metrics_2023.csv", EXPERIMENT_06 / "results/catboost_metrics_2023.csv"):
        if path.exists():
            own.insert(0, pd.read_csv(path))
    compared = comparison(pd.concat(own, ignore_index=True))
    compared.to_csv(RESULTS / "tabpfn_comparison_2023.csv", index=False)

    assert hashes == source_hashes(), "Source files changed during the run"
    thresholds = evaluation.set_index(["region", "model", "feature_set"]).threshold
    (RESULTS / "tabpfn_manifest.json").write_text(json.dumps({
        "experiment": "07_tabpfn", "source_hashes": hashes, "regions": list(REGIONS),
        "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR,
        "holdout_loaded": False,
        "target": "target_recorded_detection: at least one FIRMS detection recorded in the cell on the target day; "
                  "zero is not confirmed absence; positives are recorded hotspot activity, not ignitions",
        "feature_sets": {"tabpfn": TABPFN_INPUTS, "catboost_quiet_rows": INPUTS}, "quiet_definition": QUIET,
        "model": "hybrid: TabPFN on active rows, Experiment 06 unweighted CatBoost on quiet rows",
        "training": "one pooled TabPFN context per draw and configuration across all six Fire Centres",
        "context": {"rule": "uniform random draw without replacement from every 2018-2021 active row; "
                            "no class rebalancing, so active-row scores need no correction",
                    "seed": SEED, **settings, "balance": context_balance.to_dict(orient="records")},
        "training_source_counts": {"rows": rows, "categories": counts, "matches_experiment_06": matches_06},
        "score_correction": "active rows: TabPFN probability averaged over draws, unchanged; quiet rows: "
                            "Experiment 06 CatBoost with odds * 0.01 correction",
        "evaluated_rows": scored,
        "tabpfn_throughput_rows_per_second": round(timing["rows"] / max(timing["seconds"], 1e-9)),
        "fits": fits, "device": target_device,
        "threshold_rule": "per Fire Centre and configuration, maximize hybrid 2022 F1; applied unchanged to 2023",
        "thresholds": {" | ".join(k): v for k, v in thresholds.items()},
        "evaluation": "every 2022 and 2023 row; evaluation data not undersampled",
        "comparison": "Experiments 03-04 parsed from their RESULTS.html; Experiments 05-06 from their results; "
                      "same 2023 rows checked via Experiment 03 TP+FN per centre",
        "checks": "past-only history, no 2024 rows, four categories sum to training rows, active counts match "
                  "Experiment 06, Experiment 02 previous-day rule reproduced on every evaluated region-year, "
                  "source hashes unchanged",
        "seed": SEED, "runtime_seconds": round(time.time() - started),
        "versions": {"tabpfn": version("tabpfn"), "torch": torch.__version__, "catboost": version("catboost"),
                     "pandas": pd.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}},
        indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s. Results: {RESULTS}", flush=True)
    print(summary(compared).to_string(), flush=True)


if __name__ == "__main__":
    main()
