"""Learned cell-only and cell+neighbour baselines on Experiment 02 features.

Steps 1-2 of the planned comparison in hotspot-forecasting-design.md. Uses the same
per-Fire-Centre feature tables, next-day recorded-activity target and
2018-2021 / 2022 / 2023 split as Experiment 02; fitted models replace the fixed
rules. One model per configuration is trained on all six centres pooled and
evaluated per centre. 2024 is never loaded.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "02_hotspot_persistence_bc"
RESULTS = HERE / "results"
TRAIN_YEARS = range(2018, 2022)
SELECTION_YEAR, EVALUATION_YEAR = 2022, 2023
# Quiet negatives are sampled; positives and rows with recent activity are all kept.
# Models train unweighted and quiet-row log-odds are shifted by log(rate) afterwards.
# (Inverse-rate weights instead make boosting's first Newton steps overshoot to
# saturation at a ~0.03% base rate; scikit-learn has no step cap.)
QUIET_SAMPLE_RATE = 0.01
SEED = 42
PREDICT_BATCH = 1_000_000
CELL = ["local_count_1d", "local_count_3d", "local_count_7d", "days_since_previous_detection",
        "history_left_censored", "day_of_year"]
NEIGHBOUR = ["neighbour_count_1d", "neighbour_count_3d", "neighbour_count_7d", "in_region_neighbours"]
FEATURE_SETS = {"cell_only": CELL, "cell_plus_neighbours": CELL + NEIGHBOUR}
# The quiet stratum must be a function of each set's own inputs for the shift to be exact.
QUIET = {"cell_only": ["local_count_7d"], "cell_plus_neighbours": ["local_count_7d", "neighbour_count_7d"]}
INPUTS = {name: columns + ["quiet_stratum"] for name, columns in FEATURE_SETS.items()}
MODELS = ("logistic_regression", "gradient_boosting")
TARGET = "target_recorded_detection"
MAX_ABS_LOG_ODDS = 50


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slug(name):
    return name.lower().replace(" ", "_")


def feature_path(region, year):
    return REFERENCE / "regions" / slug(region) / "data/features" / f"features_{year}.parquet"


def load(region, year, with_cells=False):
    assert year < 2024, "2024 is the untouched final test year"
    columns = ["target_date", "history_end_date", TARGET, *[c for c in CELL + NEIGHBOUR if c != "day_of_year"]]
    frame = pd.read_parquet(feature_path(region, year), columns=columns + (["h3_cell"] if with_cells else []))
    assert frame.target_date.dt.year.eq(year).all()
    assert frame.history_end_date.lt(frame.target_date).all(), "Feature history overlaps target date"
    frame["day_of_year"] = frame.target_date.dt.dayofyear.astype("int16")
    return frame


def quiet_stratum(frame, feature_set):
    return frame[QUIET[feature_set]].eq(0).all(axis=1).to_numpy()


def design(frame, feature_set):
    quiet = quiet_stratum(frame, feature_set)
    X = np.column_stack([frame[FEATURE_SETS[feature_set]].to_numpy(dtype=np.float32), quiet]).astype(np.float32)
    return X, quiet


def logistic_regression(columns):
    counts = [i for i, c in enumerate(columns) if "count" in c]
    prepare = ColumnTransformer([
        ("log_counts", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), counts),
        # Censored age becomes 0; history_left_censored carries the distinction.
        ("log_age", make_pipeline(SimpleImputer(strategy="constant", fill_value=0),
                                  FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
         [columns.index("days_since_previous_detection")]),
        ("season", SplineTransformer(n_knots=6, degree=3), [columns.index("day_of_year")])],
        remainder="passthrough")
    return make_pipeline(prepare, StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def gradient_boosting():
    # Fixed, untuned settings; missing history age is handled natively.
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=SEED)


def training_samples(regions):
    parts, stats = {name: [] for name in FEATURE_SETS}, []
    for index, region in enumerate(regions):
        for year in TRAIN_YEARS:
            frame = load(region, year)
            positive = frame[TARGET].eq(1).to_numpy()
            # One draw per row, shared by both feature sets.
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            row = {"region": region, "year": year, "rows": len(frame), "positives": int(positive.sum())}
            for feature_set in FEATURE_SETS:
                quiet = quiet_stratum(frame, feature_set)
                keep = ~quiet | positive | drawn
                part = frame.loc[keep, CELL + NEIGHBOUR + [TARGET]]
                assert int(part[TARGET].sum()) == row["positives"]
                parts[feature_set].append(part)
                row |= {f"{feature_set}_active_rows": int((~quiet).sum()),
                        f"{feature_set}_quiet_positive_rows": int((quiet & positive).sum()),
                        f"{feature_set}_sampled_quiet_negatives": int((quiet & ~positive & drawn).sum()),
                        f"{feature_set}_kept_rows": int(keep.sum())}
            stats.append(row)
            print(f"  {region} {year}: kept {row['cell_only_kept_rows']:,} / {row['cell_plus_neighbours_kept_rows']:,} "
                  f"of {len(frame):,} rows (cell only / + neighbours)", flush=True)
            del frame
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}, pd.DataFrame(stats)


def predict(model, frame, feature_set):
    scores = []
    for start in range(0, len(frame), PREDICT_BATCH):
        X, quiet = design(frame.iloc[start:start+PREDICT_BATCH], feature_set)
        # Exact prior shift undoing the quiet-negative sampling (quiet is a model input).
        scores.append(expit(model.decision_function(X) + np.log(QUIET_SAMPLE_RATE) * quiet))
    return np.concatenate(scores).astype(np.float32)


def best_f1_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def threshold_metrics(y, score, threshold):
    positive, prediction = y.astype(bool), score >= threshold
    tp = int((positive & prediction).sum())
    fp = int((~positive & prediction).sum())
    fn = int((positive & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ap = float(average_precision_score(y, score))
    return {"rows": len(y), "positive_cell_days": int(positive.sum()), "prevalence": float(positive.mean()),
            "average_precision": ap, "ap_over_prevalence": ap / positive.mean(), "threshold": threshold,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": int(len(y) - tp - fp - fn), "predicted_positive_cell_days": tp + fp}


def downsampled_curve(y, score, points=400):
    precision, recall, _ = precision_recall_curve(y, score)
    keep = np.unique(np.linspace(0, len(recall) - 1, min(points, len(recall))).astype(int))
    return recall[keep], precision[keep]


def main():
    started = time.time()
    for folder in (RESULTS / "models", RESULTS / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)
    reference_manifest = json.loads((REFERENCE / "results/manifest.json").read_text())
    assert reference_manifest["all_six_complete"] and not reference_manifest["holdout_loaded"]
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    rules = pd.read_csv(REFERENCE / "results/all_region_metrics.csv")
    years = [*TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR]
    source_paths = [feature_path(r, y) for r in regions for y in years] + [REFERENCE / "results/all_region_metrics.csv"]
    hashes = {str(p.relative_to(REFERENCE.parent)): digest(p) for p in source_paths}

    print("Building training samples (2018-2021, six Fire Centres)", flush=True)
    samples, sample_stats = training_samples(regions)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)

    fitted, coefficients, fit_checks = {}, [], []
    for feature_set, train in samples.items():
        X, quiet = design(train, feature_set)
        y = train[TARGET].to_numpy()
        print(f"{feature_set}: {len(train):,} training rows, {int(y.sum()):,} positives ({y.mean():.2%})", flush=True)
        for name in MODELS:
            model = logistic_regression(INPUTS[feature_set]) if name == "logistic_regression" else gradient_boosting()
            tic = time.time()
            model.fit(X, y)
            raw = model.decision_function(X)
            largest = float(np.abs(raw).max())
            assert largest < MAX_ABS_LOG_ODDS, f"{name}/{feature_set} diverged: max |log-odds| {largest:.0f}"
            fit_checks.append({"model": name, "feature_set": feature_set, "training_rows": len(train),
                               "training_positives": int(y.sum()), "max_abs_training_log_odds": largest,
                               "training_average_precision": float(average_precision_score(y, raw)),
                               "fit_seconds": round(time.time() - tic, 1)})
            print(f"Fitted {name} / {feature_set} in {time.time()-tic:.0f}s; max |log-odds| {largest:.1f}", flush=True)
            fitted[(name, feature_set)] = model
            joblib.dump(model, RESULTS / "models" / f"{name}__{feature_set}.joblib")
            if name == "logistic_regression":
                names = model[0].get_feature_names_out(INPUTS[feature_set])
                coefficients += [{"feature_set": feature_set, "feature": n, "coefficient": float(c)}
                                 for n, c in zip(names, model[-1].coef_[0])]
    pd.DataFrame(coefficients).to_csv(RESULTS / "logistic_coefficients.csv", index=False)
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    del samples

    rows, curves, thresholds = [], [], {}
    for region in regions:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = load(region, year, with_cells=year == EVALUATION_YEAR)
            y = frame[TARGET].to_numpy()
            # Same rows and targets as Experiment 02: its previous-day rule must reproduce exactly.
            reference = rules.loc[rules.region.eq(region) & rules.year.eq(year) & rules.rule.eq("local_1d")].iloc[0]
            check = threshold_metrics(y, frame.local_count_1d.to_numpy(dtype=np.float32), 1)
            assert all(check[k] == reference[k] for k in ("rows", "positive_cell_days", "tp", "fp", "fn")), region
            predictions = frame[["target_date", "h3_cell", TARGET]].copy() if year == EVALUATION_YEAR else None
            for (name, feature_set), model in fitted.items():
                score = predict(model, frame, feature_set)
                key = (region, name, feature_set)
                if year == SELECTION_YEAR:
                    thresholds[key] = best_f1_threshold(y, score)
                rows.append({"region": region, "year": year,
                             "role": "threshold_selection" if year == SELECTION_YEAR else "development_evaluation",
                             "model": name, "feature_set": feature_set, **threshold_metrics(y, score, thresholds[key])})
                if year == EVALUATION_YEAR:
                    predictions[f"{name}__{feature_set}"] = score
                    recall, precision = downsampled_curve(y, score)
                    curves.append(pd.DataFrame({"region": region, "model": name, "feature_set": feature_set,
                                                "recall": recall, "precision": precision}))
            if predictions is not None:
                predictions.to_parquet(RESULTS / "predictions" / f"{slug(region)}_{EVALUATION_YEAR}.parquet", index=False)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            print(f"{region} {year}: " + "; ".join(f"{r['model'][:8]}/{r['feature_set'][:9]} AP={r['average_precision']:.3f}"
                                                   for r in latest), flush=True)
            del frame, predictions

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RESULTS / "pr_curves_2023.csv", index=False)

    # Rule references from Experiment 02: the 2022-selected rule is the fair comparison;
    # the best 2023 rule AP is an optimistic, after-the-fact reference.
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    rule_2023 = rules.loc[rules.year.eq(EVALUATION_YEAR) & rules.rule.ne("always_no_detection")]
    selected = pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").set_index("region")
    comparison = evaluation.pivot(index="region", columns=["model", "feature_set"], values=["average_precision", "f1"])
    comparison.columns = [f"{metric}__{model}__{feature_set}" for metric, model, feature_set in comparison.columns]
    comparison["selected_rule"] = selected.rule
    comparison["selected_rule_f1"] = selected.f1
    comparison["selected_rule_average_precision"] = selected.average_precision
    comparison["best_rule_average_precision_2023"] = rule_2023.groupby("region").average_precision.max()
    comparison["best_rule_for_average_precision_2023"] = rule_2023.loc[rule_2023.groupby("region").average_precision.idxmax()].set_index("region").rule
    comparison.reset_index().to_csv(RESULTS / "comparison_2023.csv", index=False)

    assert hashes == {str(p.relative_to(REFERENCE.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS),
        "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "target": "recorded detection on next UTC acquisition day; zero is not confirmed no fire",
        "feature_sets": INPUTS,
        "day_of_year": "derived from target_date; quiet_stratum derived from the set's own counts",
        "training": "one pooled model per configuration across all six Fire Centres",
        "sampling": {"rule": "keep all positives and all rows with activity in the past 7 days (cell, or cell and "
                             "neighbours, per feature set); sample remaining quiet negatives; train unweighted; "
                             "add log(rate) to quiet-row log-odds at prediction",
                     "quiet_sample_rate": QUIET_SAMPLE_RATE, "seed": SEED,
                     "source_rows": int(sample_stats.rows.sum()),
                     "kept_rows": {name: int(sample_stats[f"{name}_kept_rows"].sum()) for name in FEATURE_SETS},
                     "why_not_weights": "inverse-rate weights made gradient boosting diverge (first Newton steps "
                                        "saturate at a ~0.03% base rate; scikit-learn has no max step)"},
        "models": {"logistic_regression": "log1p counts and age, day-of-year spline (6 knots), standardized, C=1.0",
                   "gradient_boosting": gradient_boosting().get_params()},
        "threshold_rule": "per Fire Centre and configuration, maximize 2022 F1; applied unchanged to 2023",
        "checks": "past-only history, no 2024 rows, every positive kept in training, no saturated training log-odds, "
                  "Experiment 02 previous-day rule reproduced on every evaluated region-year, source hashes unchanged",
        "runtime_seconds": round(time.time() - started),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}},
        indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s. Results: {RESULTS}", flush=True)
    print(comparison.filter(like="average_precision").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
