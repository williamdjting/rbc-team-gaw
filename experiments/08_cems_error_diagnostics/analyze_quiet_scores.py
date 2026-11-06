"""Quiet-cell score separation and fixed daily alert budgets; no model fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiment as diagnostic

BUDGETS = [10, 50, 100, 500, 1000]
EDGES = np.linspace(-12, 0, 97)


def daily_budgets(y, scores, budgets):
    """Expected TP under uniform selection within a cutoff-score tie.

    All higher-score candidates are selected. Fractional TP is an expectation,
    not a realizable fractional alert; labels never decide which tied rows win.
    """
    y, scores = np.asarray(y, dtype=int), np.asarray(scores)
    if not len(y):
        return [dict(budget=k, candidates=0, positives=0, alerts=0,
                     expected_tp=0., random_tp=0., cutoff=np.nan,
                     cutoff_tie_size=0, slots_in_tie=0) for k in budgets]
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=y, minlength=len(values))[::-1]
    values, counts = values[::-1], counts[::-1]
    cumulative_n, cumulative_tp = counts.cumsum(), positives.cumsum()
    result = []
    for budget in budgets:
        k = min(budget, len(y))
        index = np.searchsorted(cumulative_n, k)
        above_n = cumulative_n[index - 1] if index else 0
        above_tp = cumulative_tp[index - 1] if index else 0.
        slots = k - above_n
        result.append(dict(budget=budget, candidates=len(y), positives=int(y.sum()), alerts=k,
                           expected_tp=above_tp + slots * positives[index] / counts[index],
                           random_tp=k * y.mean(), cutoff=float(values[index]),
                           cutoff_tie_size=int(counts[index]), slots_in_tie=int(slots)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=diagnostic.HERE / "results/baseline")
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    args = parser.parse_args()
    if any(k <= 0 for k in args.budgets):
        parser.error("Budgets must be positive")
    budgets = sorted(set(args.budgets))
    directory = args.input
    manifest = json.loads((directory / "manifest.json").read_text())
    metrics = pd.read_csv(directory / "activity_metrics.csv")
    summaries, histograms, daily = [], [], []
    for region in manifest["regions"]:
        print(f"Quiet scores: {region}", flush=True)
        frame = pd.read_parquet(diagnostic.base.feature_path(region, 2023), columns=diagnostic.KEY +
                                [diagnostic.TARGET, "local_count_7d", "neighbour_count_7d"])
        quiet = (frame.local_count_7d.eq(0) & frame.neighbour_count_7d.eq(0)).to_numpy()
        for variant in manifest["variants"]:
            if manifest["mode"] == "saved":
                scores, threshold = diagnostic.saved(region, frame, variant)
            else:
                predictions = pd.read_parquet(directory / f"predictions_{diagnostic.base.slug(region)}_{variant}_2023.parquet")
                if len(predictions) != len(frame):
                    raise ValueError("Prediction row count mismatch")
                joined = diagnostic.checked_merge(frame[diagnostic.KEY + [diagnostic.TARGET]],
                    predictions.rename(columns={diagnostic.TARGET: "saved_target"}), diagnostic.KEY, "one_to_one")
                if not joined[diagnostic.TARGET].eq(joined.saved_target).all():
                    raise ValueError("Prediction labels differ")
                scores = joined.score.to_numpy()
                row = metrics.loc[metrics.region.eq(region) & metrics.variant.eq(variant) & metrics.activity_group.eq("all")]
                if len(row) != 1:
                    raise ValueError("Expected one saved selection threshold")
                threshold = float(row.iloc[0].threshold)
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError("Invalid probabilities")
            q = frame.loc[quiet, ["target_date", diagnostic.TARGET]].copy()
            q["score"] = scores[quiet]
            for label in [0, 1]:
                selected = q.loc[q[diagnostic.TARGET].eq(label), "score"].to_numpy()
                quantiles = np.quantile(selected, [0, .1, .5, .9, .99, 1]) if len(selected) else [np.nan] * 6
                summaries.append(dict(region=region, variant=variant, label=label, rows=len(selected),
                                      threshold=threshold, above_threshold=int((selected >= threshold).sum()),
                                      **dict(zip(["min", "p10", "median", "p90", "p99", "max"], quantiles))))
                counts, _ = np.histogram(np.log10(np.maximum(selected, 1e-12)), bins=EDGES)
                for lower, upper, count in zip(EDGES[:-1], EDGES[1:], counts):
                    histograms.append(dict(region=region, variant=variant, label=label,
                                           log10_lower=lower, log10_upper=upper, count=int(count)))
            # Each region-day has its own quiet-cell candidate pool and budget.
            for date, day in q.groupby("target_date", sort=True):
                for row in daily_budgets(day[diagnostic.TARGET].to_numpy(), day.score.to_numpy(), budgets):
                    daily.append(dict(region=region, variant=variant, target_date=date, **row))
    pd.DataFrame(summaries).to_csv(directory / "quiet_score_summary.csv", index=False)
    pd.DataFrame(histograms).to_csv(directory / "quiet_score_histograms.csv", index=False)
    daily = pd.DataFrame(daily)
    daily.to_csv(directory / "quiet_daily_budgets.csv", index=False)
    outputs = []
    for region, group in [(r, g) for r, g in daily.groupby("region")] + [("All centres", daily)]:
        summary = group.groupby(["variant", "budget"])[["alerts", "positives", "expected_tp", "random_tp"]].sum().reset_index()
        summary["region"] = region
        summary["expected_fp"] = summary.alerts - summary.expected_tp
        summary["recall"] = summary.expected_tp / summary.positives.replace(0, np.nan)
        summary["precision"] = summary.expected_tp / summary.alerts.replace(0, np.nan)
        summary["random_recall"] = summary.random_tp / summary.positives.replace(0, np.nan)
        summary["lift_over_random"] = summary.expected_tp / summary.random_tp.replace(0, np.nan)
        outputs.append(summary)
    pd.concat(outputs, ignore_index=True).to_csv(directory / "quiet_budget_summary.csv", index=False)
    (directory / "quiet_analysis_manifest.json").write_text(json.dumps(dict(
        budgets_per_region_per_day=budgets, scope="quiet cells only", evaluation_year=2023,
        ranking="descending corrected score; uniform random tie expectation",
        random_baseline="same region-day quiet candidate pool and alert count",
        note="Descriptive 2023 analysis; budgets not selected by optimizing 2023 performance",
        source_manifest_sha256=diagnostic.base.digest(directory / "manifest.json"),
        code_sha256=diagnostic.base.digest(Path(__file__))), indent=2), encoding="utf-8")
    print("Wrote quiet score distributions and daily-budget diagnostics")


if __name__ == "__main__":
    main()
