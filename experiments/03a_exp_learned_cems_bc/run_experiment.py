"""Learned baselines with shortlisted CEMS Fire Weather features (fwinx, ffmcode, drtcode, infsinx).

Extends Experiment 03 by joining the 4 shortlisted CEMS weather indices onto each H3 cell
at `history_end_date` (strictly prior to `target_date`). Uses the exact same rows, targets,
2018-2021 / 2022 / 2023 split, 1% quiet-negative sampling, and model hyperparameters as
Experiment 03 so results compare one-to-one against both Experiment 02 rules and
Experiment 03 FIRMS-only models. 2024 is never loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cems_shared import (  # noqa: E402
    ensure_base_h3_grids,
    load_firms_with_cems_and_grid,
    load_teammate_baselines_from_html,
)

import h3
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE_02 = HERE.parent / "02_hotspot_persistence_bc"
REFERENCE_03 = HERE.parent / "03_exp_learned_baselines_bc"
DATASETS_DIR = ROOT / "datasets"
RESULTS = HERE / "results"

TRAIN_YEARS = range(2018, 2022)
SELECTION_YEAR, EVALUATION_YEAR = 2022, 2023
QUIET_SAMPLE_RATE = 0.01
SEED = 42
PREDICT_BATCH = 1_000_000

CELL = [
    "local_count_1d",
    "local_count_3d",
    "local_count_7d",
    "days_since_previous_detection",
    "history_left_censored",
    "day_of_year",
]
NEIGHBOUR = [
    "neighbour_count_1d",
    "neighbour_count_3d",
    "neighbour_count_7d",
    "in_region_neighbours",
]
CEMS = ["fwinx", "ffmcode", "drtcode", "infsinx"]

FEATURE_SETS = {
    "cell_only": CELL + CEMS,
    "cell_plus_neighbours": CELL + NEIGHBOUR + CEMS,
}
QUIET = {
    "cell_only": ["local_count_7d"],
    "cell_plus_neighbours": ["local_count_7d", "neighbour_count_7d"],
}
INPUTS = {name: columns + ["quiet_stratum"] for name, columns in FEATURE_SETS.items()}
MODELS = ("logistic_regression", "gradient_boosting")
TARGET = "target_recorded_detection"
MAX_ABS_LOG_ODDS = 50


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def feature_path(region: str, year: int) -> Path:
    return REFERENCE_02 / "regions" / slug(region) / "data/features" / f"features_{year}.parquet"


def load_cems_lookup() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load 2018-2023 CEMS weather table and unique 0.25° BC grid coordinates."""
    cems_path = DATASETS_DIR / "dataset_CEMS_18_23.csv"
    if not cems_path.exists():
        cems_path = DATASETS_DIR / "dataset_CEMS_18_24.csv"
    if not cems_path.exists():
        raise FileNotFoundError(
            f"Missing CEMS dataset in {DATASETS_DIR}. Run scripts/check_CEMS_coverage.py first."
        )

    cems = pd.read_csv(
        cems_path,
        usecols=["valid_time", "latitude", "longitude_deg"] + CEMS,
    )
    cems["history_end_date"] = pd.to_datetime(cems["valid_time"]).dt.normalize()
    cems = cems[cems["history_end_date"].dt.year < 2024].drop(columns=["valid_time"]).copy()
    for col in CEMS:
        cems[col] = cems[col].astype(np.float32)

    grid = (
        cems[["latitude", "longitude_deg"]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude_deg"])
        .reset_index(drop=True)
    )
    grid["cems_cell_id"] = np.arange(len(grid), dtype=np.int32)

    cems = cems.merge(grid, on=["latitude", "longitude_deg"], how="left")
    cems = cems.drop(columns=["latitude", "longitude_deg"])
    return cems, grid, cems_path


def map_region_h3_to_cems(region: str, cems_grid: pd.DataFrame) -> pd.DataFrame:
    """Map every H3 cell in a Fire Centre to its nearest valid 0.25° CEMS land grid cell."""
    h3_grid_path = REFERENCE_02 / "regions" / slug(region) / "data/h3_grid.parquet"
    if h3_grid_path.exists():
        h3_cells = pd.read_parquet(h3_grid_path, columns=["h3_cell"])["h3_cell"].unique()
    else:
        h3_cells = pd.read_parquet(feature_path(region, 2018), columns=["h3_cell"])["h3_cell"].unique()

    coords = np.array([h3.cell_to_latlng(c) for c in h3_cells], dtype=np.float64)
    cos_lat = np.cos(np.deg2rad(54.0))
    cems_xy = np.column_stack([cems_grid["latitude"].to_numpy(), cems_grid["longitude_deg"].to_numpy() * cos_lat])
    h3_xy = np.column_stack([coords[:, 0], coords[:, 1] * cos_lat])

    tree = cKDTree(cems_xy)
    _, idx = tree.query(h3_xy, k=1)

    return pd.DataFrame({
        "h3_cell": h3_cells,
        "cems_cell_id": cems_grid["cems_cell_id"].to_numpy()[idx],
    })


def load(
    region: str,
    year: int,
    h3_to_cems: pd.DataFrame,
    cems_year: pd.DataFrame,
    with_cells: bool = False,
) -> pd.DataFrame:
    assert year < 2024, "2024 is the untouched final test year"
    columns = ["target_date", "history_end_date", "h3_cell", TARGET, *[c for c in CELL + NEIGHBOUR if c != "day_of_year"]]
    frame = pd.read_parquet(feature_path(region, year), columns=columns)
    assert frame.target_date.dt.year.eq(year).all()
    assert frame.history_end_date.lt(frame.target_date).all(), "Feature history overlaps target date"
    frame["day_of_year"] = frame.target_date.dt.dayofyear.astype("int16")

    # Join CEMS weather strictly on history_end_date (prior day t-1) and nearest CEMS grid cell
    frame = frame.merge(h3_to_cems, on="h3_cell", how="left")
    frame = frame.merge(cems_year, on=["cems_cell_id", "history_end_date"], how="left")
    assert not frame[CEMS].isna().any().any(), f"Missing CEMS weather values in {region} {year}"

    drop_cols = ["cems_cell_id"] + ([] if with_cells else ["h3_cell"])
    return frame.drop(columns=drop_cols)


def quiet_stratum(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    return frame[QUIET[feature_set]].eq(0).all(axis=1).to_numpy()


def design(frame: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    quiet = quiet_stratum(frame, feature_set)
    X = np.column_stack([frame[FEATURE_SETS[feature_set]].to_numpy(dtype=np.float32), quiet]).astype(np.float32)
    return X, quiet


def logistic_regression(columns: list[str]):
    log_cols = [
        i for i, c in enumerate(columns)
        if ("count" in c) or (c in ("fwinx", "drtcode", "infsinx"))
    ]
    prepare = ColumnTransformer([
        ("log_vars", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), log_cols),
        ("log_age", make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        ), [columns.index("days_since_previous_detection")]),
        ("season", SplineTransformer(n_knots=6, degree=3), [columns.index("day_of_year")]),
    ], remainder="passthrough")
    return make_pipeline(prepare, StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def gradient_boosting():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )


def training_samples(regions: list[str], region_maps: dict[str, pd.DataFrame], cems_by_year: dict[int, pd.DataFrame]):
    parts, stats = {name: [] for name in FEATURE_SETS}, []
    for index, region in enumerate(regions):
        for year in TRAIN_YEARS:
            frame = load(region, year, region_maps[region], cems_by_year[year])
            positive = frame[TARGET].eq(1).to_numpy()
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            row = {"region": region, "year": year, "rows": len(frame), "positives": int(positive.sum())}
            for feature_set in FEATURE_SETS:
                quiet = quiet_stratum(frame, feature_set)
                keep = ~quiet | positive | drawn
                part = frame.loc[keep, CELL + NEIGHBOUR + CEMS + [TARGET]]
                assert int(part[TARGET].sum()) == row["positives"]
                parts[feature_set].append(part)
                row |= {
                    f"{feature_set}_active_rows": int((~quiet).sum()),
                    f"{feature_set}_quiet_positive_rows": int((quiet & positive).sum()),
                    f"{feature_set}_sampled_quiet_negatives": int((quiet & ~positive & drawn).sum()),
                    f"{feature_set}_kept_rows": int(keep.sum()),
                }
            stats.append(row)
            print(
                f"  {region} {year}: kept {row['cell_only_kept_rows']:,} / {row['cell_plus_neighbours_kept_rows']:,} "
                f"of {len(frame):,} rows (cell+CEMS / +neighbours+CEMS)",
                flush=True,
            )
            del frame
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}, pd.DataFrame(stats)


def predict(model, frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    scores = []
    for start in range(0, len(frame), PREDICT_BATCH):
        X, quiet = design(frame.iloc[start : start + PREDICT_BATCH], feature_set)
        scores.append(expit(model.decision_function(X) + np.log(QUIET_SAMPLE_RATE) * quiet))
    return np.concatenate(scores).astype(np.float32)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    positive, prediction = y.astype(bool), score >= threshold
    tp = int((positive & prediction).sum())
    fp = int((~positive & prediction).sum())
    fn = int((positive & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ap = float(average_precision_score(y, score))
    return {
        "rows": len(y),
        "positive_cell_days": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "average_precision": ap,
        "ap_over_prevalence": ap / positive.mean(),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(len(y) - tp - fp - fn),
        "predicted_positive_cell_days": tp + fp,
    }


def downsampled_curve(y: np.ndarray, score: np.ndarray, points: int = 400):
    precision, recall, _ = precision_recall_curve(y, score)
    keep = np.unique(np.linspace(0, len(recall) - 1, min(points, len(recall))).astype(int))
    return recall[keep], precision[keep]


def main():
    started = time.time()
    for folder in (RESULTS / "models", RESULTS / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)

    ensure_base_h3_grids()
    reference_manifest = json.loads((REFERENCE_02 / "results/manifest.json").read_text())
    assert reference_manifest["all_six_complete"] and not reference_manifest["holdout_loaded"]
    regions = sorted(pd.read_csv(REFERENCE_02 / "results/selected_rules_2023.csv").region)
    rules = pd.read_csv(REFERENCE_02 / "results/all_region_metrics.csv")

    print("Loading joined FIRMS+CEMS dataset (`firms_with_cems_18_24.csv`) and CEMS lookup (2018-2023)...", flush=True)
    cems_df, cems_grid, cems_paths = load_firms_with_cems_and_grid()
    cems_by_year = {y: grp.copy() for y, grp in cems_df.groupby(cems_df["history_end_date"].dt.year)}
    del cems_df
    region_maps = {r: map_region_h3_to_cems(r, cems_grid) for r in regions}

    years = [*TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR]
    source_paths = [feature_path(r, y) for r in regions for y in years] + [
        REFERENCE_02 / "results/all_region_metrics.csv",
        *cems_paths,
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}

    print("Building training samples with CEMS features (2018-2021, six Fire Centres)", flush=True)
    samples, sample_stats = training_samples(regions, region_maps, cems_by_year)
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
            fit_checks.append({
                "model": name,
                "feature_set": feature_set,
                "training_rows": len(train),
                "training_positives": int(y.sum()),
                "max_abs_training_log_odds": largest,
                "training_average_precision": float(average_precision_score(y, raw)),
                "fit_seconds": round(time.time() - tic, 1),
            })
            print(f"Fitted {name} / {feature_set} in {time.time() - tic:.0f}s; max |log-odds| {largest:.1f}", flush=True)
            fitted[(name, feature_set)] = model
            joblib.dump(model, RESULTS / "models" / f"{name}__{feature_set}.joblib")
            if name == "logistic_regression":
                names = model[0].get_feature_names_out(INPUTS[feature_set])
                coefficients += [
                    {"feature_set": feature_set, "feature": n, "coefficient": float(c)}
                    for n, c in zip(names, model[-1].coef_[0])
                ]
    pd.DataFrame(coefficients).to_csv(RESULTS / "logistic_coefficients.csv", index=False)
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    del samples

    rows, curves, thresholds = [], [], {}
    for region in regions:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = load(region, year, region_maps[region], cems_by_year[year], with_cells=year == EVALUATION_YEAR)
            y = frame[TARGET].to_numpy()
            reference = rules.loc[rules.region.eq(region) & rules.year.eq(year) & rules.rule.eq("local_1d")].iloc[0]
            check = threshold_metrics(y, frame.local_count_1d.to_numpy(dtype=np.float32), 1)
            assert all(check[k] == reference[k] for k in ("rows", "positive_cell_days", "tp", "fp", "fn")), region
            predictions = frame[["target_date", "h3_cell", TARGET]].copy() if year == EVALUATION_YEAR else None
            for (name, feature_set), model in fitted.items():
                score = predict(model, frame, feature_set)
                key = (region, name, feature_set)
                if year == SELECTION_YEAR:
                    thresholds[key] = best_f1_threshold(y, score)
                rows.append({
                    "region": region,
                    "year": year,
                    "role": "threshold_selection" if year == SELECTION_YEAR else "development_evaluation",
                    "model": name,
                    "feature_set": feature_set,
                    **threshold_metrics(y, score, thresholds[key]),
                })
                if year == EVALUATION_YEAR:
                    predictions[f"{name}__{feature_set}"] = score
                    recall, precision = downsampled_curve(y, score)
                    curves.append(pd.DataFrame({
                        "region": region,
                        "model": name,
                        "feature_set": feature_set,
                        "recall": recall,
                        "precision": precision,
                    }))
            if predictions is not None:
                predictions.to_parquet(RESULTS / "predictions" / f"{slug(region)}_{EVALUATION_YEAR}.parquet", index=False)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            print(
                f"{region} {year}: "
                + "; ".join(f"{r['model'][:8]}/{r['feature_set'][:9]} AP={r['average_precision']:.3f}" for r in latest),
                flush=True,
            )
            del frame, predictions

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RESULTS / "pr_curves_2023.csv", index=False)

    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    rule_2023 = rules.loc[rules.year.eq(EVALUATION_YEAR) & rules.rule.ne("always_no_detection")]
    selected = pd.read_csv(REFERENCE_02 / "results/selected_rules_2023.csv").set_index("region")
    comparison = evaluation.pivot(index="region", columns=["model", "feature_set"], values=["average_precision", "f1"])
    comparison.columns = [f"{metric}__{model}__{feature_set}" for metric, model, feature_set in comparison.columns]
    comparison["selected_rule"] = selected.rule
    comparison["selected_rule_f1"] = selected.f1
    comparison["selected_rule_average_precision"] = selected.average_precision
    comparison["best_rule_average_precision_2023"] = rule_2023.groupby("region").average_precision.max()
    comparison["best_rule_for_average_precision_2023"] = (
        rule_2023.loc[rule_2023.groupby("region").average_precision.idxmax()].set_index("region").rule
    )

    # Also compare directly against Experiment 03 FIRMS-only models (from CSV or teammate's RESULTS.html)
    exp03_comp_path = REFERENCE_03 / "results/comparison_2023.csv"
    if exp03_comp_path.exists():
        exp03_comp = pd.read_csv(exp03_comp_path).set_index("region")
        for col in exp03_comp.columns:
            if col.startswith(("average_precision__", "f1__")):
                comparison[f"exp03_{col}"] = exp03_comp[col]
    else:
        teammate_df = load_teammate_baselines_from_html()
        exp03_html = teammate_df.loc[teammate_df["source"].eq("03_learned_baselines")]
        if len(exp03_html):
            piv = exp03_html.pivot(index="region", columns=["model", "feature_set"], values=["average_precision", "f1"])
            for metric, model, fset in piv.columns:
                comparison[f"exp03_{metric}__{model}__{fset}"] = piv[(metric, model, fset)]

    comparison.reset_index().to_csv(RESULTS / "comparison_2023.csv", index=False)

    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(
        json.dumps(
            {
                "source_hashes": hashes,
                "regions": regions,
                "training_years": list(TRAIN_YEARS),
                "selection_year": SELECTION_YEAR,
                "evaluation_year": EVALUATION_YEAR,
                "holdout_loaded": False,
                "target": "recorded detection on next UTC acquisition day; zero is not confirmed no fire",
                "cems_features": CEMS,
                "cems_alignment": "nearest 0.25° BC land grid cell joined on history_end_date (t-1)",
                "feature_sets": INPUTS,
                "training": "one pooled model per configuration across all six Fire Centres",
                "sampling": {
                    "quiet_sample_rate": QUIET_SAMPLE_RATE,
                    "seed": SEED,
                    "source_rows": int(sample_stats.rows.sum()),
                    "kept_rows": {name: int(sample_stats[f"{name}_kept_rows"].sum()) for name in FEATURE_SETS},
                },
                "runtime_seconds": round(time.time() - started),
                "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Done in {time.time() - started:.0f}s. Results: {RESULTS}", flush=True)
    print(comparison.filter(like="average_precision").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()

