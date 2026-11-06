"""Is the F1 gap to the previous-day rule real, and is it the threshold or the scores?

Reads Experiment 13's saved 2023 predictions. For each Fire Centre and key model:
- F1 at the 2022 threshold, and the week-block bootstrap 95% interval of
  model F1 minus rule F1 (whole calendar weeks resampled, since fires span days);
- "rule volume": each day, flag the k highest-scoring cells, where k is the number of
  cells the rule flags that day (cells detected yesterday). k is known on t-1, so this
  is a usable operating point that changes which cells are flagged, not how many.
Writes results/f1_checks_2023.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
spec = importlib.util.spec_from_file_location("experiment_03", HERE.parent / "03_exp_learned_baselines_bc/run_experiment.py")
experiment_03 = sys.modules["experiment_03"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_03)
MODELS = [("gradient_boosting", "base"), ("gradient_boosting", "all"), ("catboost", "all")]
BOOTSTRAP, SEED = 2000, 42
TARGET = experiment_03.TARGET


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def daily_counts(day, y, flag):
    frame = pd.DataFrame({"day": day, "tp": y & flag, "fp": ~y & flag, "fn": y & ~flag})
    return frame.groupby("day")[["tp", "fp", "fn"]].sum()


def main():
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    evaluation = metrics.loc[metrics.role.eq("development_evaluation") & metrics.threshold_rule.eq("2022")]
    rng = np.random.default_rng(SEED)
    rows = []
    for region in sorted(evaluation.region.unique()):
        predictions = pd.read_parquet(RESULTS / "predictions" / f"{experiment_03.slug(region)}_2023.parquet")
        history = experiment_03.load(region, 2023, with_cells=True)
        assert (history.h3_cell.to_numpy() == predictions.h3_cell.to_numpy()).all()
        assert (history.target_date.to_numpy() == predictions.target_date.to_numpy()).all()
        y = predictions[TARGET].to_numpy() == 1
        day = predictions.target_date.to_numpy()
        rule = history.local_count_1d.to_numpy() > 0
        rule_daily = daily_counts(day, y, rule)
        weeks = pd.to_datetime(rule_daily.index).isocalendar().week.to_numpy()
        unique_weeks = np.unique(weeks)
        draws = rng.integers(0, len(unique_weeks), size=(BOOTSTRAP, len(unique_weeks)))
        week_index = {w: np.flatnonzero(weeks == w) for w in unique_weeks}

        def resampled_f1(daily):
            weekly = np.stack([daily.to_numpy()[week_index[w]].sum(axis=0) for w in unique_weeks])
            totals = weekly[draws].sum(axis=1)
            return f1(totals[:, 0], totals[:, 1], totals[:, 2])

        rule_boot = resampled_f1(rule_daily)
        rule_f1 = f1(*rule_daily.sum().to_numpy())
        for model, feature_set in MODELS:
            score = predictions[f"{model}__{feature_set}"].to_numpy()
            threshold = evaluation.loc[evaluation.region.eq(region) & evaluation.model.eq(model)
                                       & evaluation.feature_set.eq(feature_set), "threshold"].iloc[0]
            model_daily = daily_counts(day, y, score >= threshold)
            model_f1 = f1(*model_daily.sum().to_numpy())
            difference = resampled_f1(model_daily) - rule_boot
            # Rule volume: per day, the k = (cells flagged by the rule) highest-scoring cells.
            order = pd.DataFrame({"day": day, "score": score}).groupby("day").score.rank(method="first", ascending=False)
            k = pd.Series(rule, index=day).groupby(level=0).transform("sum").to_numpy()
            volume_daily = daily_counts(day, y, order.to_numpy() <= k)
            volume_f1 = f1(*volume_daily.sum().to_numpy())
            volume_difference = resampled_f1(volume_daily) - rule_boot
            rows.append({"region": region, "model": model, "feature_set": feature_set, "rule_f1": rule_f1,
                         "model_f1_2022_threshold": model_f1, "difference": model_f1 - rule_f1,
                         "difference_ci_low": np.quantile(difference, .025),
                         "difference_ci_high": np.quantile(difference, .975),
                         "share_bootstrap_above_rule": float((difference > 0).mean()),
                         "model_f1_rule_volume": volume_f1, "rule_volume_difference": volume_f1 - rule_f1,
                         "rule_volume_ci_low": np.quantile(volume_difference, .025),
                         "rule_volume_ci_high": np.quantile(volume_difference, .975)})
        print(f"{region}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "f1_checks_2023.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
