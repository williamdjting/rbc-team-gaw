"""Past-only features and fixed rules for next-day recorded hotspot activity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FEATURES = DATA / "features"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
YEARS = range(2018, 2024)


def preceding_sum(values, window):
    """Row d contains only rows [d-window, d); early rows are incomplete."""
    cumulative = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int64),
                            values.cumsum(axis=0, dtype=np.int64)])
    days = np.arange(len(values))
    return cumulative[days] - cumulative[np.maximum(0, days - window)]


def history_age(values):
    """Days since previous recorded detection; NaN means no prior in-season record."""
    age = np.full(values.shape, np.nan, dtype=np.float32)
    last_seen = np.full(values.shape[1], -1, dtype=np.int32)
    for day in range(len(values)):
        known = last_seen >= 0
        age[day, known] = day - last_seen[known]
        last_seen[values[day] > 0] = day
    return age


def check_temporal_logic():
    toy = np.array([[2, 0], [0, 1], [4, 0], [0, 0], [1, 3],
                    [0, 0], [0, 0], [5, 0], [0, 2], [0, 0]])
    for window in (1, 3, 7):
        actual = preceding_sum(toy, window)
        for day in range(len(toy)):
            np.testing.assert_array_equal(actual[day], toy[max(0, day-window):day].sum(axis=0))
        changed = toy.copy()
        changed[7:] = 100
        np.testing.assert_array_equal(actual[:8], preceding_sum(changed, window)[:8])
    age = history_age(toy)
    assert np.isnan(age[0]).all() and age[1, 0] == 1 and age[2, 1] == 1
    assert age[7, 0] == 3 and age[8, 0] == 1


def metrics(y, score, year, name):
    positive = y.astype(bool)
    prediction = score > 0
    tp = int((positive & prediction).sum())
    fp = int((~positive & prediction).sum())
    fn = int((positive & ~prediction).sum())
    tn = int((~positive & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    prevalence = float(positive.mean())
    ap = float(average_precision_score(y, score))
    return {"year": year, "role": "training_diagnostic" if year <= 2021 else
            "rule_selection" if year == 2022 else "development_evaluation",
            "rule": name, "rows": len(y), "positive_cell_days": int(positive.sum()),
            "prevalence": prevalence, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "average_precision": ap, "ap_over_prevalence": ap / prevalence,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "predicted_positive_cell_days": tp + fp}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    check_temporal_logic()
    for folder in (FEATURES, RESULTS, FIGURES):
        folder.mkdir(exist_ok=True)
    source_paths = [DATA / "kamloops_h3_grid.parquet"] + [DATA / f"cell_date_{y}.parquet" for y in YEARS]
    hashes = {p.name: digest(p) for p in source_paths}
    grid = pd.read_parquet(source_paths[0])
    cells = sorted(grid.h3_cell)
    position = {cell: i for i, cell in enumerate(cells)}
    rows, columns = [], []
    for cell, index in position.items():
        for neighbour in sorted(h3.grid_disk(cell, 1)):
            if neighbour != cell and neighbour in position:
                rows.append(index)
                columns.append(position[neighbour])
    adjacency = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, columns)),
                           shape=(len(cells), len(cells)))
    assert adjacency.diagonal().sum() == 0 and (adjacency != adjacency.T).nnz == 0
    neighbour_counts = np.asarray(adjacency.sum(axis=1)).ravel()
    pd.DataFrame({"h3_cell": cells, "in_region_neighbours": neighbour_counts}).to_csv(
        RESULTS / "neighbour_counts.csv", index=False)

    all_metrics, annual_checks, selected_daily, sample_features = [], [], [], []
    example_cells = json.loads((DATA / "manifest.json").read_text())["sample_cells"]
    selected_rule = None
    for year in YEARS:
        dates = pd.date_range(f"{year}-05-01", f"{year}-09-30")
        frame = pd.read_parquet(DATA / f"cell_date_{year}.parquet")
        assert frame.observation_coverage.eq("unknown").all()
        assert not frame.duplicated(["acq_date", "h3_cell"]).any()
        matrix = frame.pivot(index="acq_date", columns="h3_cell", values="detection_count")
        matrix = matrix.reindex(index=dates, columns=cells)
        assert matrix.notna().all().all() and matrix.ge(0).all().all()
        counts = matrix.to_numpy(dtype=np.int32)
        neighbour_daily = (adjacency @ counts.T).T
        for index in (0, len(cells)//2, len(cells)-1):
            indices = adjacency.getrow(index).indices
            np.testing.assert_array_equal(neighbour_daily[:, index], counts[:, indices].sum(axis=1))
        usable_dates = dates[7:]
        features = pd.DataFrame({"target_date": np.repeat(usable_dates, len(cells)),
                                 "h3_cell": np.tile(cells, len(usable_dates))})
        features["history_end_date"] = features.target_date - pd.Timedelta(days=1)
        for window in (1, 3, 7):
            features[f"local_count_{window}d"] = preceding_sum(counts, window)[7:].ravel().astype("int32")
            features[f"neighbour_count_{window}d"] = preceding_sum(neighbour_daily, window)[7:].ravel().astype("int32")
        features["days_since_previous_detection"] = history_age(counts)[7:].ravel()
        features["history_left_censored"] = features.days_since_previous_detection.isna()
        features["in_region_neighbours"] = np.tile(neighbour_counts, len(usable_dates)).astype("int8")
        features["target_recorded_detection"] = (counts[7:].ravel() > 0).astype("int8")
        features["observation_coverage"] = pd.Categorical(["unknown"] * len(features))
        # No coverage-supported true fire-absence target is fabricated here.
        features["coverage_supported_target"] = pd.array([pd.NA] * len(features), dtype="Int8")
        assert features.history_end_date.lt(features.target_date).all()
        assert features.target_date.dt.year.eq(year).all()
        assert len(features) == len(cells) * 146
        assert features.coverage_supported_target.isna().all()
        features.to_parquet(FEATURES / f"features_{year}.parquet", index=False)
        sample_features.append(features.loc[features.h3_cell.isin(example_cells)].copy())
        y = features.target_recorded_detection.to_numpy()
        annual = []
        scores = {"always_no_detection": np.zeros(len(y), dtype=np.int32)}
        for window in (1, 3, 7):
            scores[f"local_{window}d"] = features[f"local_count_{window}d"].to_numpy()
            scores[f"local_or_neighbour_{window}d"] = (
                features[f"local_count_{window}d"] + features[f"neighbour_count_{window}d"]).to_numpy()
        for name, score in scores.items():
            annual.append(metrics(y, score, year, name))
        all_metrics.extend(annual)
        if year == 2022:
            # Select before looking at 2023. Rules are fixed; no model is fitted.
            selected_rule = sorted((m for m in annual if m["rule"] != "always_no_detection"),
                                   key=lambda m: (-m["f1"], m["rule"]))[0]["rule"]
        if year == 2023:
            assert selected_rule is not None
            prediction = scores[selected_rule] > 0
            for day, date in enumerate(usable_dates):
                lo, hi = day * len(cells), (day+1) * len(cells)
                actual, pred = y[lo:hi].astype(bool), prediction[lo:hi]
                selected_daily.append({"date": str(date.date()), "observed_positive_cells": int(actual.sum()),
                                       "predicted_positive_cells": int(pred.sum()),
                                       "true_positive_cells": int((actual & pred).sum())})
            predictions = features[["target_date", "h3_cell", "target_recorded_detection"]].copy()
            predictions["predicted_recorded_detection"] = prediction
            predictions["rule"] = pd.Categorical([selected_rule] * len(predictions))
            predictions.to_parquet(RESULTS / "predictions_2023.parquet", index=False)
        annual_checks.append({"year": year, "rows": len(features), "positives": int(y.sum()),
                              "first_target": str(usable_dates[0].date()),
                              "last_target": str(usable_dates[-1].date()), "excluded_warmup_days": 7})
        print(f"{year}: {len(features):,} feature rows, {int(y.sum()):,} positive recorded-activity targets", flush=True)
        del features, scores, frame, matrix

    results = pd.DataFrame(all_metrics)
    results.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.DataFrame(annual_checks).to_csv(RESULTS / "feature_checks.csv", index=False)
    pd.concat(sample_features, ignore_index=True).to_parquet(RESULTS / "sample_features.parquet", index=False)
    pd.DataFrame(selected_daily).to_csv(RESULTS / "selected_rule_daily_2023.csv", index=False)

    comparison = results.loc[results.year.isin([2022, 2023])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        comparison.pivot(index="rule", columns="year", values=metric).plot.bar(ax=ax)
        ax.set(title=metric.replace("_", " ").title(), xlabel="", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=70, labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=.4)
    fig.suptitle("Next-day recorded hotspot activity: fixed persistence rules")
    fig.savefig(FIGURES / "persistence_metrics.png", dpi=180)
    plt.close(fig)
    daily = pd.DataFrame(selected_daily)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    for column, label in (("observed_positive_cells", "Cells with recorded detections"),
                          ("predicted_positive_cells", "Predicted cells"),
                          ("true_positive_cells", "Correctly predicted positive cells")):
        ax.plot(pd.to_datetime(daily.date), daily[column], label=label, linewidth=1)
    ax.set(title=f"2023 development evaluation: {selected_rule} (selected on 2022)", ylabel="Cells/day")
    ax.legend()
    ax.grid(linestyle=":", alpha=.4)
    fig.savefig(FIGURES / "persistence_daily_2023.png", dpi=180)
    plt.close(fig)

    chosen = results.loc[(results.year == 2023) & (results.rule == selected_rule)].iloc[0]
    assert hashes == {p.name: digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "selected_rule": selected_rule, "selection_year": 2022,
        "evaluation_year": 2023, "holdout_loaded": False,
        "target": "next UTC date recorded detection; not coverage-supported fire absence",
        "features": "local/neighbour past 1,3,7 days; censored days since previous detection",
        "selection_metric": "2022 F1, alphabetical tie-break", "checks": annual_checks,
        "versions": {"h3": h3.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__}}, indent=2), encoding="utf-8")
    print(f"Selected {selected_rule}; 2023 precision={chosen.precision:.4f}, recall={chosen.recall:.4f}, F1={chosen.f1:.4f}")
    from build_report import main as build_report
    build_report()


if __name__ == "__main__":
    main()
