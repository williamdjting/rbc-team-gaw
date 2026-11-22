"""How strongly does each input, on its own, relate to big lightning wildfires starting soon?

Label (Experiment 37's): a BCWS lightning wildfire of 10 ha or more starts in the res-7 cell on any
of days t .. t+6. Experiment 36's all-cause label is shown for comparison. Inputs are the values
the models see for target day t (history day t-1): station weather, lightning (1 to 14 days),
Experiment 13's BCWS incident inputs and Experiment 18's static layers. Satellite fire-history
inputs are left out (these fires mostly start in quiet cells) to keep this light enough to run
beside the main experiments.

Rows: 2018-2022 only (2023 is the evaluation year and stays untouched), every positive cell-day
of either label plus a 0.5% random draw of the rest, weighted back to the full grid. Per input:
- weighted Pearson correlation with the label (small for any rare label, so compare inputs with
  each other rather than with 1);
- univariate AUC (0.5 = no information; below 0.5 = fires where the input is low);
- lift: the label's rate among the 10% of cell-days most fire-prone by this input alone,
  divided by the overall rate.
Writes analysis/feature_correlations.csv.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_22", EXPERIMENTS / "36_exp_bcws_ignitions_bc/run_experiment.py")
experiment_22 = sys.modules["experiment_22"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_22)
from experiment_22 import LongLightning, ignition_grid, label_window  # noqa: E402
from experiment_16 import Drivers, LIGHTNING_FEATURES, WEATHER  # noqa: E402
from experiment_13 import STATIC, StaticLayers  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY  # noqa: E402
from experiment_08 import GROUPS, REFERENCE, SEASON_DAYS, SEED, bcws_features, load_bcws, unit_xyz  # noqa: E402
from experiment_03 import slug  # noqa: E402

YEARS = (2018, 2019, 2020, 2021, 2022)
NEGATIVE_RATE = 0.005
OUT = HERE / "analysis"


class Grids:
    """The part of Experiment 13's Context that Drivers needs: cells and their centres."""

    def __init__(self, regions):
        self.grids = {}
        for region in regions:
            grid = pd.read_parquet(REFERENCE / "regions" / slug(region) / "data/h3_grid.parquet",
                                   columns=["h3_cell", "center_latitude", "center_longitude"])
            self.grids[region] = {"cells": pd.Index(grid.h3_cell),
                                  "xyz": unit_xyz(grid.center_latitude.to_numpy(), grid.center_longitude.to_numpy())}


def weighted_corr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def top_decile_lift(x, y, w, direction):
    """Rate of y among the 10% (by weight) of rows most fire-prone by x, over the overall rate."""
    order = np.argsort(-direction * x, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    top = order[cum <= 0.10]
    return np.average(y[top], weights=w[top]) / np.average(y, weights=w)


def main():
    OUT.mkdir(exist_ok=True)
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    grids = Grids(regions)
    drivers = Drivers(grids, regions)
    long_lightning = LongLightning(drivers)
    static = StaticLayers()
    incidents = load_bcws()
    big = incidents.loc[incidents.FIRE_CAUSE.eq("Lightning") & incidents.CURRENT_SIZE.ge(10)]
    names = [*WEATHER, *LIGHTNING_FEATURES, *experiment_22.LIGHTNING_LONG, *GROUPS["bcws"], *STATIC]
    parts = []
    for region_index, region in enumerate(regions):
        cells = grids.grids[region]["cells"]
        for year in YEARS:
            labels = {name: label_window(ignition_grid(frame, cells, year)[0])[:, FIRST_TARGET_DAY:]
                      for name, frame in (("big_lightning_7d", big), ("any_wildfire_7d", incidents))}
            positive = labels["big_lightning_7d"] | labels["any_wildfire_7d"]
            drawn = np.random.default_rng([SEED, year, region_index, 23]).random(positive.shape) < NEGATIVE_RATE
            cell, day = np.nonzero(positive | drawn)
            history = day + FIRST_TARGET_DAY - 1  # inputs for target day t are indexed by history day t-1
            weight = np.where(positive[cell, day], 1.0, 1.0 / NEGATIVE_RATE)
            frame = {"region": region, "year": year, "weight": weight,
                     **{name: label[cell, day] for name, label in labels.items()}}
            arrays = drivers.arrays(region, year, np.zeros((len(cells), SEASON_DAYS), bool))
            arrays |= long_lightning.arrays(region, year)
            arrays |= bcws_features(incidents, year, grids.grids[region]["xyz"])
            for name in [*WEATHER, *LIGHTNING_FEATURES, *experiment_22.LIGHTNING_LONG, *GROUPS["bcws"]]:
                frame[name] = arrays[name][cell, history]
            table = static.values(region, cells, year)
            for j, name in enumerate(STATIC):
                frame[name] = table[cell, j]
            parts.append(pd.DataFrame(frame))
            print(f"{region} {year}: {int(labels['big_lightning_7d'].sum()):,} big-lightning positive cell-days, "
                  f"{int(labels['any_wildfire_7d'].sum()):,} any-wildfire; {len(cell):,} rows", flush=True)
            del arrays, labels, positive, drawn
        drivers._cache.clear()
    data = pd.concat(parts, ignore_index=True)

    rows = []
    for name in names:
        for label in ("big_lightning_7d", "any_wildfire_7d"):
            ok = data[name].notna().to_numpy()
            x, y, w = data[name].to_numpy(np.float64)[ok], data[label].to_numpy(np.float64)[ok], data.weight.to_numpy()[ok]
            if np.nanstd(x) == 0 or y.sum() == 0:
                continue
            auc = roc_auc_score(y, x, sample_weight=w)
            rows.append({"input": name, "label": label, "pearson_r": weighted_corr(x, y, w), "auc": auc,
                         "auc_distance_from_random": abs(auc - 0.5),
                         "fires_where_input_is": "high" if auc >= 0.5 else "low",
                         "top10pct_lift": top_decile_lift(x, y, w, 1 if auc >= 0.5 else -1),
                         "coverage": ok.mean(), "positives": int(y.sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "feature_correlations.csv", index=False)
    pd.set_option("display.width", 220)
    wide = result.pivot(index="input", columns="label", values=["auc", "pearson_r", "top10pct_lift"])
    wide.columns = [f"{a}_{b.replace('_7d', '')}" for a, b in wide.columns]
    wide["direction"] = result.loc[result.label.eq("big_lightning_7d")].set_index("input").fires_where_input_is
    order = (wide.auc_big_lightning - 0.5).abs().sort_values(ascending=False).index
    print(wide.loc[order, ["direction", "auc_big_lightning", "top10pct_lift_big_lightning", "pearson_r_big_lightning",
                           "auc_any_wildfire", "top10pct_lift_any_wildfire"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
