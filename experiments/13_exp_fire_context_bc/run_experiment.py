"""Fire-context features on top of Experiment 03's cell + neighbour baseline.

Adds four feature groups built from the raw FIRMS detections and BC Wildfire Service
incidents: FRP and day/night, a wider spatial neighbourhood, fire-complex size and
duration, and known BCWS incidents. Rows, target, 2018-2021 / 2022 / 2023 split,
sampling and quiet-row correction are exactly Experiment 03's; the base feature set
reproduces Experiment 03's gradient boosting. 2024 is never loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import h3
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXPERIMENT_03 = EXPERIMENTS / "03_exp_learned_baselines_bc"
spec = importlib.util.spec_from_file_location("experiment_03", EXPERIMENT_03 / "run_experiment.py")
experiment_03 = sys.modules["experiment_03"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_03)
from experiment_03 import (CELL, NEIGHBOUR, QUIET_SAMPLE_RATE, REFERENCE, SEED, TARGET, TRAIN_YEARS,  # noqa: E402
                           SELECTION_YEAR, EVALUATION_YEAR, MAX_ABS_LOG_ODDS, PREDICT_BATCH, digest, slug,
                           load, feature_path, gradient_boosting, best_f1_threshold, threshold_metrics,
                           downsampled_curve)

DETECTIONS = EXPERIMENTS / "firms_summer_2018_2024/data/firms_h3_detections.csv"
BCWS = HERE / "data/bcws_incidents_2018_2023.csv"
RESULTS = HERE / "results"
SAMPLE_CACHE = HERE / "data/training_sample.parquet"
SEASON_DAYS = 153  # May 1 - September 30; index 0 is May 1
MAX_RING = 5
# BCWS records are used one day later than FIRMS: features for target day t see incidents
# whose ignition (and out) dates are on or before t-2, guarding against back-dated entries.
BCWS_LAG_DAYS = 1
BCWS_CAP_KM = 200.0
BCWS_MISSING_OUT_ACTIVE_DAYS = 30
BCWS_MAX_ACTIVE_DAYS = 180  # caps implausible out dates (some records run for years)
EARTH_KM = 6371.0

BASE = CELL + NEIGHBOUR
GROUPS = {
    "frp_daynight": ["frp_sum_1d", "frp_max_1d", "frp_sum_7d", "night_count_1d",
                     "disk1_frp_sum_1d", "disk1_night_count_1d"],
    "wider_area": ["ring2_count_1d", "ring2_count_7d", "ring3_count_1d", "ring3_count_7d",
                   "dist_nearest_1d", "dist_nearest_3d", "active_cells_disk5_3d"],
    "fire_complex": ["complex_size_3d", "complex_frp_3d", "active_days_7d", "disk1_active_days_7d",
                     "disk1_days_since_first_7d"],
    "bcws": ["bcws_dist_active_km", "bcws_n_active_25km", "bcws_nearest_age_days", "bcws_nearest_lightning",
             "bcws_dist_recent_out_km", "bcws_n_new_25km_3d"],
}
NEW = [c for group in GROUPS.values() for c in group]
FEATURE_SETS = {"base": BASE, **{f"base+{g}": BASE + cols for g, cols in GROUPS.items()},
                "firms_context": BASE + GROUPS["frp_daynight"] + GROUPS["wider_area"] + GROUPS["fire_complex"],
                "all": BASE + NEW}
# One quiet definition for every set (Experiment 03's cell + neighbour one); it is a model
# input, so the log(rate) shift stays exact.
QUIET = ["local_count_7d", "neighbour_count_7d"]
CATBOOST_SETS = ("base", "all")
POOLED_THRESHOLD_SETS = ("base", "all")
CATBOOST_PARAMS = {"loss_function": "Logloss", "iterations": 300, "learning_rate": 0.1, "depth": 6,
                   "l2_leaf_reg": 3.0, "border_count": 254, "boost_from_average": True,
                   "random_seed": SEED, "thread_count": -1, "allow_writing_files": False}  # Experiments 05-06


# ---------------------------------------------------------------- daily source arrays

def season_index(dates):
    return (dates - pd.to_datetime(dates.dt.year.astype(str) + "-05-01")).dt.days.to_numpy()


def load_detections():
    frame = pd.read_csv(DETECTIONS, usecols=["acq_date", "h3_cell", "frp", "daynight", "type"])
    frame["acq_date"] = pd.to_datetime(frame.acq_date)
    frame = frame.loc[frame.acq_date.dt.year.between(2018, 2023)].copy()
    assert frame.acq_date.dt.month.between(5, 9).all()
    frame["day"] = season_index(frame.acq_date)
    return frame


def daily_arrays(detections, universe):
    """Per year: U x 153 arrays of detection count, night count, FRP sum/max, vegetation count."""
    detections = detections.assign(u=universe.get_indexer(detections.h3_cell))
    assert (detections.u >= 0).all()
    out = {}
    for year, part in detections.groupby(detections.acq_date.dt.year):
        g = part.assign(night=part.daynight.eq("N"), veg=part.type.eq(0)).groupby(["u", "day"])
        agg = g.agg(count=("frp", "size"), night=("night", "sum"), frp_sum=("frp", "sum"),
                    frp_max=("frp", "max"), veg=("veg", "sum")).reset_index()
        arrays = {}
        for column in ("count", "night", "frp_sum", "frp_max", "veg"):
            a = np.zeros((len(universe), SEASON_DAYS), np.float32)
            a[agg.u.to_numpy(), agg.day.to_numpy()] = agg[column].to_numpy()
            arrays[column] = a
        out[int(year)] = arrays
    return out


def window(a, w):
    """Sum over the w days ending at each column (inclusive), truncated at May 1."""
    cs = np.concatenate([np.zeros((a.shape[0], 1), np.float64), np.cumsum(a, axis=1, dtype=np.float64)], axis=1)
    end = np.arange(1, a.shape[1] + 1)
    return (cs[:, end] - cs[:, np.maximum(end - w, 0)]).astype(np.float32)


def ring_matrices(cells, universe):
    """Sparse (len(cells) x U) maps from each cell to universe cells exactly k rings away."""
    position = {c: i for i, c in enumerate(universe)}
    rings = []
    for k in range(MAX_RING + 1):
        rows, cols = [], []
        for i, cell in enumerate(cells):
            for other in ([cell] if k == 0 else h3.grid_ring(cell, k)):
                j = position.get(other)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        rings.append(sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(cells), len(universe))))
    return rings


def complex_arrays(arrays, universe_adjacency):
    """Per universe cell and history day: size and 3-day FRP of its connected active complex."""
    active = window(arrays["count"], 3) > 0
    frp = window(arrays["frp_sum"], 3)
    size = np.zeros(active.shape, np.float32)
    total = np.zeros(active.shape, np.float32)
    for day in range(active.shape[1]):
        members = np.flatnonzero(active[:, day])
        if not len(members):
            continue
        _, label = connected_components(universe_adjacency[members][:, members], directed=False)
        size[members, day] = np.bincount(label)[label]
        total[members, day] = np.bincount(label, weights=frp[members, day])[label]
    return size, total


def sparse_row_max(matrix, values):
    """Row-wise max of values over each row's nonzero columns (0 for empty rows)."""
    out = np.zeros((matrix.shape[0], values.shape[1]), np.float32)
    starts, ends = matrix.indptr[:-1], matrix.indptr[1:]
    filled = np.flatnonzero(ends > starts)
    if len(filled):
        gathered = values[matrix.indices]
        out[filled] = np.maximum.reduceat(gathered, starts[filled], axis=0)
    return out


# ---------------------------------------------------------------- BCWS incidents

def unit_xyz(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return EARTH_KM * np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def load_bcws():
    frame = pd.read_csv(BCWS)
    assert frame.FIRE_YEAR.between(2018, 2023).all(), "2024 must never be loaded"
    frame = frame.loc[frame.FIRE_TYPE.eq("Fire") & frame.LATITUDE.notna() & frame.LONGITUDE.notna()].copy()
    for column in ("IGNITION_DATE", "FIRE_OUT_DATE"):
        local = pd.to_datetime(frame[column], utc=True).dt.tz_convert("America/Vancouver")
        frame[column] = local.dt.tz_localize(None).dt.normalize()
    frame = frame.loc[frame.IGNITION_DATE >= "2017-01-01"].copy()  # drops impossible ignition dates
    frame.loc[frame.FIRE_OUT_DATE < frame.IGNITION_DATE, "FIRE_OUT_DATE"] = pd.NaT
    frame["lightning"] = frame.FIRE_CAUSE.map({"Lightning": 1.0, "Person": 0.0}).fillna(0.5)
    return frame.reset_index(drop=True)


def bcws_features(incidents, year, cell_xyz):
    """N x 153 arrays indexed by history-end day; incident status known BCWS_LAG_DAYS later."""
    n = len(cell_xyz)
    names = GROUPS["bcws"]
    out = {name: np.full((n, SEASON_DAYS), np.nan, np.float32) for name in names}
    xyz = unit_xyz(incidents.LATITUDE.to_numpy(), incidents.LONGITUDE.to_numpy())
    ignition = incidents.IGNITION_DATE.to_numpy()
    fire_out = incidents.FIRE_OUT_DATE.to_numpy()
    assumed_out = ignition + np.timedelta64(BCWS_MISSING_OUT_ACTIVE_DAYS, "D")
    effective_out = np.minimum(np.where(pd.isna(fire_out), assumed_out, fire_out),
                               ignition + np.timedelta64(BCWS_MAX_ACTIVE_DAYS, "D"))
    lightning = incidents.lightning.to_numpy()
    start = np.datetime64(f"{year}-05-01")
    for day in range(SEASON_DAYS):
        known = start + np.timedelta64(day - BCWS_LAG_DAYS, "D")
        active = (ignition <= known) & (effective_out > known)
        recent_out = (fire_out <= known) & (fire_out > known - np.timedelta64(3, "D"))
        new = (ignition <= known) & (ignition > known - np.timedelta64(3, "D"))
        out["bcws_dist_active_km"][:, day] = BCWS_CAP_KM
        out["bcws_dist_recent_out_km"][:, day] = BCWS_CAP_KM
        out["bcws_n_active_25km"][:, day] = 0
        out["bcws_n_new_25km_3d"][:, day] = 0
        if active.any():
            tree = cKDTree(xyz[active])
            distance, nearest = tree.query(cell_xyz, k=1, distance_upper_bound=BCWS_CAP_KM)
            found = np.isfinite(distance)
            out["bcws_dist_active_km"][found, day] = distance[found]
            index = np.flatnonzero(active)[nearest[found]]
            age = (known - ignition[index]) / np.timedelta64(1, "D")
            out["bcws_nearest_age_days"][found, day] = age
            out["bcws_nearest_lightning"][found, day] = lightning[index]
            out["bcws_n_active_25km"][:, day] = tree.query_ball_point(cell_xyz, r=25.0, return_length=True)
        if recent_out.any():
            distance, _ = cKDTree(xyz[recent_out]).query(cell_xyz, k=1, distance_upper_bound=BCWS_CAP_KM)
            found = np.isfinite(distance)
            out["bcws_dist_recent_out_km"][found, day] = distance[found]
        if new.any():
            out["bcws_n_new_25km_3d"][:, day] = cKDTree(xyz[new]).query_ball_point(cell_xyz, r=25.0,
                                                                                    return_length=True)
    return out


# ---------------------------------------------------------------- per region-year features

class Context:
    def __init__(self, regions):
        tic = time.time()
        detections = load_detections()
        self.universe = pd.Index(np.sort(detections.h3_cell.unique()))
        self.arrays = daily_arrays(detections, self.universe)
        del detections
        position = {c: i for i, c in enumerate(self.universe)}
        rows, cols = [], []
        for i, cell in enumerate(self.universe):
            for other in h3.grid_ring(cell, 1):
                j = position.get(other)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        adjacency = sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(self.universe),) * 2)
        self.complex = {year: complex_arrays(a, adjacency) for year, a in self.arrays.items()}
        self.incidents = load_bcws()
        self.grids = {}
        for region in regions:
            grid = pd.read_parquet(REFERENCE / "regions" / slug(region) / "data/h3_grid.parquet",
                                   columns=["h3_cell", "center_latitude", "center_longitude", "is_border_cell"])
            rings = ring_matrices(grid.h3_cell.tolist(), self.universe)
            self.grids[region] = {"cells": pd.Index(grid.h3_cell), "rings": rings,
                                  "border": grid.is_border_cell.to_numpy(),
                                  "disk1": (rings[0] + rings[1]).tocsr(),
                                  "disk2": (rings[0] + rings[1] + rings[2]).tocsr(),
                                  "xyz": unit_xyz(grid.center_latitude.to_numpy(), grid.center_longitude.to_numpy())}
        print(f"Context ready in {time.time()-tic:.0f}s: {len(self.universe):,} detected cells, "
              f"{len(self.incidents):,} BCWS fires", flush=True)

    def features(self, region, year, frame):
        """Adds every new column to Experiment 03's frame for one region-year; checks it against Experiment 02."""
        g, a = self.grids[region], self.arrays[year]
        rings = g["rings"]
        count, indicator = a["count"], (a["count"] > 0).astype(np.float32)
        local = rings[0] @ count
        disk1_indicator = ((g["disk1"] @ indicator) > 0).astype(np.float32)
        ind3 = (window(indicator, 3) > 0).astype(np.float32)
        daily = {
            "frp_sum_1d": rings[0] @ a["frp_sum"], "frp_max_1d": rings[0] @ a["frp_max"],
            "frp_sum_7d": window(rings[0] @ a["frp_sum"], 7), "night_count_1d": rings[0] @ a["night"],
            "disk1_frp_sum_1d": g["disk1"] @ a["frp_sum"], "disk1_night_count_1d": g["disk1"] @ a["night"],
            "ring2_count_1d": rings[2] @ count, "ring2_count_7d": window(rings[2] @ count, 7),
            "ring3_count_1d": rings[3] @ count, "ring3_count_7d": window(rings[3] @ count, 7),
            "active_cells_disk5_3d": sum(r @ ind3 for r in rings),
            "complex_size_3d": sparse_row_max(g["disk2"], self.complex[year][0]),
            "complex_frp_3d": sparse_row_max(g["disk2"], self.complex[year][1]),
            "active_days_7d": window(rings[0] @ indicator, 7),
            "disk1_active_days_7d": window(disk1_indicator, 7),
        }
        for name, source in (("dist_nearest_1d", indicator), ("dist_nearest_3d", ind3)):
            distance = np.full(local.shape, MAX_RING + 1, np.float32)
            for k in range(MAX_RING, -1, -1):
                distance[(rings[k] @ source) > 0] = k
            daily[name] = distance
        first = np.zeros(local.shape, np.float32)
        for offset in range(6, -1, -1):  # earliest disk-1 detection in the 7 days ending at each day
            shifted = np.zeros_like(disk1_indicator)
            shifted[:, offset:] = disk1_indicator[:, :SEASON_DAYS - offset]
            first[(first == 0) & (shifted > 0)] = offset + 1
        daily["disk1_days_since_first_7d"] = first
        daily |= bcws_features(self.incidents, year, g["xyz"])

        cell = g["cells"].get_indexer(frame.h3_cell)
        assert (cell >= 0).all()
        history = season_index(frame.history_end_date)
        target = season_index(frame.target_date)
        assert (target == history + 1).all() and history.min() >= 6
        for name, values in daily.items():
            frame[name] = np.asarray(values)[cell, history]
        # Rebuilt counts and target must equal Experiment 02's on interior cells. Experiment 02
        # assigns detections to a centre by point location, so its border cells may count fewer.
        interior = ~g["border"][cell]
        for w in (1, 3, 7):
            mine, theirs = window(local, w)[cell, history], frame[f"local_count_{w}d"].to_numpy()
            assert np.array_equal(mine[interior], theirs[interior]) and (mine >= theirs).all(), (region, year, w)
        disk_ring1 = rings[1] @ count
        for w in (1, 3, 7):
            assert (window(disk_ring1, w)[cell, history] >= frame[f"neighbour_count_{w}d"].to_numpy()).all()
        target_rebuilt, target_02 = local[cell, target] > 0, frame[TARGET].to_numpy() == 1
        assert np.array_equal(target_rebuilt[interior], target_02[interior]) and (target_rebuilt >= target_02).all()
        vegetation = (rings[0] @ a["veg"])[cell, target] > 0
        frame["target_vegetation_fire"] = (target_02 & vegetation).astype(np.int8)
        return frame


# ---------------------------------------------------------------- models

def quiet_stratum(frame):
    return frame[QUIET].eq(0).all(axis=1).to_numpy()


def design(frame, feature_set):
    quiet = quiet_stratum(frame)
    X = np.column_stack([frame[FEATURE_SETS[feature_set]].to_numpy(dtype=np.float32), quiet]).astype(np.float32)
    return X, quiet


def catboost_model():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**CATBOOST_PARAMS, verbose=False)


def raw_scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X, prediction_type="RawFormulaVal")


def predict(model, frame, feature_set):
    scores = []
    for start in range(0, len(frame), PREDICT_BATCH):
        X, quiet = design(frame.iloc[start:start + PREDICT_BATCH], feature_set)
        scores.append(expit(raw_scores(model, X) + np.log(QUIET_SAMPLE_RATE) * quiet))
    return np.concatenate(scores).astype(np.float32)


def best_weighted_f1_threshold(y, score, weight):
    precision, recall, thresholds = precision_recall_curve(y, score, sample_weight=weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def best_f1(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(f1[:-1].max())


def training_sample(regions, context):
    """Experiment 03's exact draw (same seeds and keep rule), with the new columns attached."""
    parts, stats = [], []
    for index, region in enumerate(regions):
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            positive = frame[TARGET].eq(1).to_numpy()
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            quiet = quiet_stratum(frame)
            keep = ~quiet | positive | drawn
            part = context.features(region, year, frame.loc[keep].reset_index(drop=True))
            part = part[BASE + NEW + [TARGET, "target_vegetation_fire"]].assign(region=region, year=year)
            parts.append(part)
            stats.append({"region": region, "year": year, "rows": len(frame), "positives": int(positive.sum()),
                          "active_rows": int((~quiet).sum()), "quiet_positive_rows": int((quiet & positive).sum()),
                          "sampled_quiet_negatives": int((quiet & ~positive & drawn).sum()), "kept_rows": int(keep.sum())})
            print(f"  {region} {year}: kept {keep.sum():,} of {len(frame):,}", flush=True)
            del frame
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-sample", action="store_true", help="reuse data/training_sample.parquet")
    args = parser.parse_args()
    started = time.time()
    for folder in (RESULTS / "models", RESULTS / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)
    reference_manifest = json.loads((REFERENCE / "results/manifest.json").read_text())
    assert reference_manifest["all_six_complete"] and not reference_manifest["holdout_loaded"]
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    years = [*TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR]
    source_paths = [feature_path(r, y) for r in regions for y in years] + [
        DETECTIONS, BCWS, EXPERIMENT_03 / "results/metrics_by_year.csv"]
    hashes = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    context = Context(regions)

    if args.reuse_sample and SAMPLE_CACHE.exists():
        sample = pd.read_parquet(SAMPLE_CACHE)
        sample_stats = pd.read_csv(RESULTS / "training_sample.csv")
        print(f"Reusing {len(sample):,} cached training rows", flush=True)
    else:
        print("Building training sample (2018-2021, six Fire Centres)", flush=True)
        sample, sample_stats = training_sample(regions, context)
        sample.to_parquet(SAMPLE_CACHE, index=False)
        sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    y_train = sample[TARGET].to_numpy()
    quiet_train = quiet_stratum(sample)
    # Each sampled quiet negative stands for 1/rate quiet negatives in the full data.
    weight_train = np.where(quiet_train & (y_train == 0), 1 / QUIET_SAMPLE_RATE, 1.0)

    fitted, fit_checks, importances = {}, [], []
    configs = [("gradient_boosting", s) for s in FEATURE_SETS] + [("catboost", s) for s in CATBOOST_SETS]
    for name, feature_set in configs:
        X, _ = design(sample, feature_set)
        model = gradient_boosting() if name == "gradient_boosting" else catboost_model()
        tic = time.time()
        model.fit(X, y_train)
        largest = float(np.abs(raw_scores(model, X)).max())
        assert largest < MAX_ABS_LOG_ODDS, f"{name}/{feature_set} diverged: max |log-odds| {largest:.0f}"
        fit_checks.append({"model": name, "feature_set": feature_set, "inputs": X.shape[1],
                           "training_rows": len(X), "training_positives": int(y_train.sum()),
                           "max_abs_training_log_odds": largest, "fit_seconds": round(time.time() - tic, 1)})
        print(f"Fitted {name} / {feature_set} ({X.shape[1]} inputs) in {time.time()-tic:.0f}s", flush=True)
        fitted[(name, feature_set)] = model
        if name == "catboost":
            model.save_model(str(RESULTS / "models" / f"{name}__{feature_set}.cbm"))
            importances += [{"model": name, "feature_set": feature_set, "feature": f, "importance": float(v)}
                            for f, v in zip(FEATURE_SETS[feature_set] + ["quiet_stratum"], model.get_feature_importance())]
        else:
            joblib.dump(model, RESULTS / "models" / f"{name}__{slug(feature_set).replace('+', '_')}.joblib")
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    pd.DataFrame(importances).to_csv(RESULTS / "feature_importance.csv", index=False)

    # Leave-one-training-year-out scores on the sample give weighted 2018-2021 evidence for
    # thresholds, so they need not rest on 2022 alone (Cariboo had 27 positives in 2022).
    oof = {}
    for feature_set in POOLED_THRESHOLD_SETS:
        X, quiet = design(sample, feature_set)
        scores = np.zeros(len(sample), np.float32)
        for year in TRAIN_YEARS:
            held = sample.year.eq(year).to_numpy()
            model = gradient_boosting().fit(X[~held], y_train[~held])
            scores[held] = expit(model.decision_function(X[held]) + np.log(QUIET_SAMPLE_RATE) * quiet[held])
            print(f"  out-of-year scores {feature_set} {year}", flush=True)
        oof[feature_set] = scores
    del X

    reference_03 = pd.read_csv(EXPERIMENT_03 / "results/metrics_by_year.csv")
    rows, curves, thresholds, selection_scores, oracle = [], [], {}, {}, []
    for region in regions:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = context.features(region, year, load(region, year, with_cells=True))
            y = frame[TARGET].to_numpy()
            role = "threshold_selection" if year == SELECTION_YEAR else "development_evaluation"
            predictions = frame[["target_date", "h3_cell", TARGET, "target_vegetation_fire"]].copy()
            rule = frame.local_count_1d.to_numpy(dtype=np.float32)
            rows.append({"region": region, "year": year, "role": role, "model": "rule_local_1d", "feature_set": "rule",
                         "threshold_rule": "fixed", **threshold_metrics(y, rule, 1)})
            for (name, feature_set), model in fitted.items():
                score = predict(model, frame, feature_set)
                key = (region, name, feature_set)
                if year == SELECTION_YEAR:
                    thresholds[(*key, "2022")] = best_f1_threshold(y, score)
                    if name == "gradient_boosting" and feature_set in POOLED_THRESHOLD_SETS:
                        selection_scores[key] = (y, score)
                variants = ["2022"] + (["pooled_2018_2022"] if (*key, "pooled_2018_2022") in thresholds else [])
                for variant in variants:
                    rows.append({"region": region, "year": year, "role": role, "model": name,
                                 "feature_set": feature_set, "threshold_rule": variant,
                                 **threshold_metrics(y, score, thresholds[(*key, variant)])})
                if year == EVALUATION_YEAR:
                    predictions[f"{name}__{feature_set}"] = score
                    oracle.append({"region": region, "model": name, "feature_set": feature_set,
                                   "best_f1_any_2023_threshold": best_f1(y, score)})
                    if feature_set in ("base", "all", "firms_context"):
                        recall, precision = downsampled_curve(y, score)
                        curves.append(pd.DataFrame({"region": region, "model": name, "feature_set": feature_set,
                                                    "recall": recall, "precision": precision}))
            if year == SELECTION_YEAR:
                # Pooled thresholds: weighted out-of-year 2018-2021 sample scores plus all 2022 rows.
                in_region = sample.region.eq(region).to_numpy()
                for feature_set in POOLED_THRESHOLD_SETS:
                    y22, s22 = selection_scores[(region, "gradient_boosting", feature_set)]
                    thresholds[(region, "gradient_boosting", feature_set, "pooled_2018_2022")] = best_weighted_f1_threshold(
                        np.concatenate([y_train[in_region], y22]), np.concatenate([oof[feature_set][in_region], s22]),
                        np.concatenate([weight_train[in_region], np.ones(len(y22))]))
            else:
                predictions.to_parquet(RESULTS / "predictions" / f"{slug(region)}_{EVALUATION_YEAR}.parquet", index=False)
                rows.append({"region": region, "year": year, "role": "vegetation_target_check", "model": "rule_local_1d",
                             "feature_set": "rule", "threshold_rule": "fixed",
                             **threshold_metrics(frame.target_vegetation_fire.to_numpy(), rule, 1)})
                for (name, feature_set) in fitted:
                    rows.append({"region": region, "year": year, "role": "vegetation_target_check", "model": name,
                                 "feature_set": feature_set, "threshold_rule": "2022",
                                 **threshold_metrics(frame.target_vegetation_fire.to_numpy(),
                                                     predictions[f"{name}__{feature_set}"].to_numpy(),
                                                     thresholds[(region, name, feature_set, "2022")])})
                # The base set reruns Experiment 03's gradient boosting on identical rows: it must match.
                mine = [r for r in rows if r["region"] == region and r["year"] == year and r["model"] == "gradient_boosting"
                        and r["feature_set"] == "base" and r["threshold_rule"] == "2022" and r["role"] == role][0]
                theirs = reference_03.loc[reference_03.region.eq(region) & reference_03.year.eq(year)
                                          & reference_03.model.eq("gradient_boosting")
                                          & reference_03.feature_set.eq("cell_plus_neighbours")].iloc[0]
                assert abs(mine["average_precision"] - theirs.average_precision) < 1e-6, (region, mine, theirs)
            latest = [r for r in rows if r["region"] == region and r["year"] == year and r["role"] == role
                      and r["threshold_rule"] in ("2022", "fixed")]
            print(f"{region} {year}: " + "; ".join(f"{r['model'][:8]}/{r['feature_set']} AP={r['average_precision']:.3f} "
                                                   f"F1={r['f1']:.3f}" for r in latest), flush=True)
            del frame, predictions

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RESULTS / "pr_curves_2023.csv", index=False)
    pd.DataFrame(oracle).to_csv(RESULTS / "best_threshold_f1_2023.csv", index=False)

    evaluation = metrics.loc[metrics.role.eq("development_evaluation")]
    ablation = evaluation.loc[evaluation.model.eq("gradient_boosting") & evaluation.threshold_rule.eq("2022")].pivot(
        index="region", columns="feature_set", values=["average_precision", "f1"])
    ablation.columns = [f"{metric}__{feature_set}" for metric, feature_set in ablation.columns]
    ablation.reset_index().to_csv(RESULTS / "ablation_2023.csv", index=False)

    assert hashes == {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS),
        "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "target": "recorded detection on next UTC acquisition day (Experiment 02 target, unchanged)",
        "vegetation_target_check": "same scores evaluated against a target that ignores FIRMS type 2/3 detections",
        "feature_sets": {k: v + ["quiet_stratum"] for k, v in FEATURE_SETS.items()},
        "feature_groups": GROUPS,
        "feature_notes": {
            "neighbourhood": "rings use every detected H3 cell (any Fire Centre); Experiment 02 neighbour counts "
                             "stay in-region. disk1 = cell + ring 1; disk2 adds ring 2",
            "dist_nearest": f"H3 grid steps to the nearest detection, {MAX_RING + 1} = none within {MAX_RING}",
            "complex": "cells detected in the past 3 days joined by H3 adjacency; max over disk2 of complex size "
                       "and 3-day FRP",
            "bcws": f"FIRE_TYPE == 'Fire'; dates in Pacific time; known {BCWS_LAG_DAYS} day after FIRMS history "
                    f"(target t uses ignition/out dates <= t-{BCWS_LAG_DAYS + 1}); missing out date = active "
                    f"{BCWS_MISSING_OUT_ACTIVE_DAYS} days; activity capped at {BCWS_MAX_ACTIVE_DAYS} days; ignitions before "
                    f"2017 dropped; distances capped at {BCWS_CAP_KM:.0f} km; final fire size "
                    "and perimeters are not used (future information)"},
        "sampling": {"rule": "Experiment 03's exact draw: all positives, all rows active in cell or in-region "
                             "neighbours over 7 days, 1% of quiet negatives; unweighted training; quiet-row "
                             "log-odds + log(rate)", "quiet_sample_rate": QUIET_SAMPLE_RATE, "seed": SEED,
                     "source_rows": int(sample_stats.rows.sum()), "kept_rows": int(sample_stats.kept_rows.sum())},
        "models": {"gradient_boosting": gradient_boosting().get_params(), "catboost": CATBOOST_PARAMS},
        "threshold_rules": {
            "2022": "per Fire Centre and configuration, maximize 2022 F1 (as Experiments 03-07)",
            "pooled_2018_2022": "per Fire Centre, maximize F1 over leave-one-year-out 2018-2021 sample scores "
                                "(quiet negatives weighted 1/rate) plus all 2022 rows"},
        "checks": "rebuilt local counts and target equal Experiment 02 on every row; base gradient boosting "
                  "reproduces Experiment 03 AP on every 2023 region; past-only history; no 2024 rows; no "
                  "saturated training log-odds; source hashes unchanged",
        "runtime_seconds": round(time.time() - started),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                     "h3": h3.__version__}}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s. Results: {RESULTS}", flush=True)
    print(ablation.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
