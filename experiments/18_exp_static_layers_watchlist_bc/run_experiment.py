"""Static land, fuel and industry layers: next-day forecast, quiet-cell watch list, wildfire flag.

Experiments 16-17 found that fires in quiet cells are mostly burns with no BC Wildfire
Service record (flat, low farmland and oil and gas sites, mostly in the Peace region),
older wildfires flaring up, and new, mostly lightning, wildfires; none reach the forecast's
alert threshold. This experiment adds per-cell layers from scripts/build_static_layers.py
(terrain, fuels, Agricultural Land Reserve, oil and gas facilities and wells, recent
cutblocks); facilities, wells and cutblocks count only if dated before each season.
A. Next-day forecast: CatBoost on Experiment 13's features with and without the static
   layers, against the previous-day rule, on the exact cell and the one-cell margin (2022
   thresholds applied to 2023), with week-block bootstrap intervals.
B. Quiet-cell watch list: each day, the N highest-scoring quiet cells (no detection in the
   cell or a neighbour for 7 days) per Fire Centre, ranked by the forecast models, by a model
   trained on quiet rows only (adding Experiment 17's history features), and by two simple
   rankings.
C. Wildfire flag: for each detected cell-day, the chance it is not linked to a BCWS wildfire
   (Experiment 17's after-the-fact link), from the static layers and the detection itself,
   with no BCWS inputs.
Train 2018-2021, select 2022, evaluate 2023; 2024 is never loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_12", EXPERIMENTS / "17_exp_history_nonwildfire_bc/run_experiment.py")
experiment_12 = sys.modules["experiment_12"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_12)
import experiment_08  # noqa: E402  (registered by experiment_12)
from experiment_12 import HISTORY, ONSETS, PERIMETERS, add_history, linked_detections, load_perimeters, onset_categories  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve, neighbourhood, to_grid  # noqa: E402
from experiment_08 import (BCWS, Context, DETECTIONS, FEATURE_SETS, MAX_ABS_LOG_ODDS, MAX_RING, QUIET_SAMPLE_RATE,  # noqa: E402
                           REFERENCE, SEASON_DAYS, SEED, SELECTION_YEAR, EVALUATION_YEAR, TARGET, TRAIN_YEARS,
                           catboost_model, design, digest, load, predict, quiet_stratum, raw_scores, season_index,
                           sparse_row_max, window)

STATIC_DIR = EXPERIMENTS.parent / "datasets/static_layers"
EXPERIMENT_08_MODEL = EXPERIMENTS / "13_exp_fire_context_bc/results/models/catboost__all.cbm"
EXPERIMENT_08_SAMPLE = EXPERIMENTS / "13_exp_fire_context_bc/results/training_sample.csv"
EXPERIMENT_12_METRICS = EXPERIMENTS / "17_exp_history_nonwildfire_bc/results/metrics_by_year.csv"
LAYER_COLUMNS = ["elevation_mean_m", "elevation_std_m", "slope_mean_deg", "fuel_share_conifer", "fuel_share_deciduous",
                 "fuel_share_grass", "fuel_share_mixedwood", "fuel_share_non_fuel", "fuel_share_water",
                 "mixedwood_conifer_percent", "alr_share"]
STATIC = LAYER_COLUMNS + ["og_facilities_cell", "og_facilities_disk1", "og_wells_cell", "og_wells_disk1",
                          "cutblock_ha_1_3y_cell", "cutblock_ha_1_3y_disk1"]
FEATURE_SETS["all_static"] = FEATURE_SETS["all"] + STATIC
FEATURE_SETS["quiet"] = FEATURE_SETS["all"] + HISTORY + STATIC
FORECAST = {"rule_yesterday": None, "catboost_08": "all", "catboost_static": "all_static"}
WATCH = ("catboost_08", "catboost_static", "catboost_quiet", "nearby_activity", "burned_before")
WATCH_SIZES = (10, 25, 50, 100)
WATCH_DAYS = 7
DEFINITIONS = ("res7_exact", "res7_margin1")
CHUNK = 1_500_000
BOOTSTRAP = 2000
FLAG_FEATURES = ["count", "night_share", "non_vegetation_share", "frp_sum", "frp_max", "day_of_season", "complex_size_3d",
                 "complex_frp_3d", "active_days_7d", "active_days_30d", "season_active_days", "prior_share_years_detected",
                 "prior_wildfire_days_per_year", "prior_other_days_per_year", "latitude", "longitude"] + STATIC


# ---------------------------------------------------------------- memory

_bcws_features = experiment_08.bcws_features
_bcws_cache = {}


def cached_bcws_features(incidents, year, cell_xyz):
    """Experiment 13's BCWS arrays for the latest region-year, so features can be built in row chunks."""
    key = (year, cell_xyz.shape[0], float(cell_xyz[0, 0]), float(cell_xyz[-1, 2]))
    if key not in _bcws_cache:
        _bcws_cache.clear()
        _bcws_cache[key] = _bcws_features(incidents, year, cell_xyz)
    return _bcws_cache[key]


experiment_08.bcws_features = cached_bcws_features


# ---------------------------------------------------------------- static layers

def parse_dates(values):
    return pd.to_datetime(pd.Series(values).astype("str").str.rstrip("Z"), errors="coerce").to_numpy("datetime64[ns]")


class StaticLayers:
    """Per-cell layers; facilities, wells and cutblocks as known before each season."""

    def __init__(self):
        layers = [pd.read_parquet(STATIC_DIR / f"h3_{name}.parquet").set_index("h3_cell") for name in ("terrain", "fuels", "alr")]
        self.cells = layers[0].index
        assert all(layer.index.equals(self.cells) for layer in layers)
        self.layers = pd.concat(layers, axis=1)[LAYER_COLUMNS].to_numpy(np.float32)
        facilities = pd.read_parquet(STATIC_DIR / "og_facilities.parquet")
        wells = pd.read_parquet(STATIC_DIR / "og_wells.parquet")
        cutblocks = pd.read_parquet(STATIC_DIR / "cutblocks_by_year.parquet")
        self.facility = (self.cells.get_indexer(facilities.h3_cell), parse_dates(facilities.ACTIVITY_APPROVAL_DATE),
                         parse_dates(facilities.ACTIVITY_CANCEL_DATE))
        self.well = (self.cells.get_indexer(wells.h3_cell), wells.WELL_AUTHORIZATION_GRANT_DATE.to_numpy("datetime64[ns]"))
        self.cutblock = (self.cells.get_indexer(cutblocks.h3_cell), cutblocks.harvest_year.to_numpy(),
                         cutblocks.area_ha.to_numpy(np.float64))
        self.disks, self.values_cache = {}, {}

    def counts(self, year):
        """Facilities approved before the year and not cancelled by May 1, wells authorized before the
        year, and area harvested in the three previous years, per layer cell."""
        before, season = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year}-05-01")

        def per_cell(cell, keep, weight=None):
            keep = keep & (cell >= 0)
            return np.bincount(cell[keep], weights=None if weight is None else weight[keep],
                               minlength=len(self.cells)).astype(np.float32)

        cell, start, end = self.facility
        facilities = per_cell(cell, (np.isnat(start) | (start < before)) & (np.isnat(end) | (end >= season)))
        cell, start = self.well
        wells = per_cell(cell, np.isnat(start) | (start < before))
        cell, harvest, area = self.cutblock
        cut = per_cell(cell, (harvest >= year - 3) & (harvest <= year - 1), area)
        return facilities, wells, cut

    def values(self, key, cells, year):
        """len(cells) x len(STATIC) float32 array; key names the cell list for caching."""
        if (key, year) in self.values_cache:
            return self.values_cache[key, year]
        if key not in self.disks:
            self.disks[key] = (self.cells.get_indexer(cells), neighbourhood(list(cells), list(self.cells)))
        index, disk1 = self.disks[key]
        found = index >= 0
        own = np.full((len(cells), len(LAYER_COLUMNS)), np.nan, np.float32)
        own[found] = self.layers[index[found]]
        columns = []
        for values in self.counts(year):
            mine = np.zeros(len(cells), np.float32)
            mine[found] = values[index[found]]
            columns += [mine, disk1 @ values]
        out = np.column_stack([own, *columns]).astype(np.float32)
        self.values_cache[key, year] = out
        return out


def add_static(static, context, region, year, frame):
    cells = context.grids[region]["cells"]
    table = static.values(region, cells, year)
    index = cells.get_indexer(frame.h3_cell)
    for j, name in enumerate(STATIC):
        frame[name] = table[index, j]
    return frame


# ---------------------------------------------------------------- training

def training_sample(regions, all_regions, context, links, static):
    """Experiment 13's draw (same seeds and keep rule) with history and static columns attached."""
    parts, stats = [], []
    for region in regions:
        index = all_regions.index(region)  # Experiment 13 seeds by position among all six
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            positive = frame[TARGET].eq(1).to_numpy()
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            quiet = quiet_stratum(frame)
            keep = ~quiet | positive | drawn
            part = context.features(region, year, frame.loc[keep].reset_index(drop=True))
            part = add_history(context, links, region, year, part)
            part = add_static(static, context, region, year, part)
            parts.append(part[FEATURE_SETS["quiet"] + [TARGET]].assign(region=region, year=year))
            stats.append({"region": region, "year": year, "rows": len(frame), "kept_rows": int(keep.sum()),
                          "positives": int(positive.sum()), "quiet_positive_rows": int((quiet & positive).sum()),
                          "sampled_quiet_negatives": int((quiet & ~positive & drawn).sum())})
            print(f"  {region} {year}: kept {keep.sum():,}; quiet positives {(quiet & positive).sum():,}", flush=True)
            del frame
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def fit(name, X, y, columns, fit_checks):
    model = catboost_model()
    tic = time.time()
    model.fit(X, y)
    largest = float(np.abs(raw_scores(model, X)).max())
    assert largest < MAX_ABS_LOG_ODDS, f"{name} diverged"
    fit_checks.append({"model": name, "rows": len(y), "positives": int(y.sum()), "inputs": X.shape[1],
                       "max_abs_training_log_odds": largest, "fit_seconds": round(time.time() - tic, 1)})
    pd.DataFrame({"feature": columns, "importance": model.get_feature_importance()}).sort_values(
        "importance", ascending=False).to_csv(RESULTS / f"feature_importance__{name}.csv", index=False)
    model.save_model(str(RESULTS / "models" / f"{name}.cbm"))
    print(f"Fitted {name}: {len(y):,} rows, {int(y.sum()):,} positives, {time.time()-tic:.0f}s", flush=True)
    return model


# ---------------------------------------------------------------- scoring helpers

def f1_from_counts(counts):
    """counts[..., 4] = flagged, correct flags, positives, caught positives."""
    precision = counts[..., 1] / np.maximum(counts[..., 0], 1)
    recall = counts[..., 3] / np.maximum(counts[..., 2], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)


def daily_counts(score, truth, hit, recall_score, threshold):
    flag = score >= threshold
    return np.stack([flag.sum(0), (flag & hit).sum(0), truth.sum(0), (truth & (recall_score >= threshold)).sum(0)],
                    axis=1).astype(np.float64)


def forward_window(a, w):
    """Sum over the w days starting at each column (inclusive), truncated at September 30."""
    return window(a[:, ::-1], w)[:, ::-1]


def ranked_lists(score, quiet, tiebreak, sizes):
    """Per day, the top-N quiet cells by score (ties broken by tiebreak), for each N."""
    listed = {n: np.zeros(score.shape, bool) for n in sizes}
    for day in range(score.shape[1]):
        candidates = np.flatnonzero(quiet[:, day])
        order = candidates[np.lexsort((tiebreak[candidates, day], score[candidates, day]))[::-1]]
        for n in sizes:
            listed[n][order[:n], day] = True
    return listed


def onset_positions(onsets, cells, year):
    part = onsets.loc[onsets.target_date.dt.year.eq(year)]
    index = cells.get_indexer(part.h3_cell)
    day = season_index(part.target_date) - FIRST_TARGET_DAY
    keep = (index >= 0) & (day >= 0)
    return index[keep], day[keep], part.bcws_category.to_numpy()[keep]


# ---------------------------------------------------------------- C. wildfire flag

def detection_table(context, links, static, regions):
    """One row per detected universe cell-day, 2018-2023, with Part C features and label."""
    label = links[0]
    universe = context.universe
    lat_lon = np.array([h3.cell_to_latlng(c) for c in universe], np.float32)
    region_of = np.full(len(universe), "outside grids", object)
    for region in regions:
        index = universe.get_indexer(context.grids[region]["cells"])
        region_of[index[index >= 0]] = region
    parts = []
    for year, a in sorted(context.arrays.items()):
        detected = a["count"] > 0
        u, d = np.nonzero(detected)
        indicator = detected.astype(np.float32)
        size, frp3 = context.complex[year]
        earlier = [y for y in context.arrays if y < year]
        prior_years = np.zeros(len(universe), np.float32)
        prior_wild = np.zeros(len(universe), np.float32)
        prior_other = np.zeros(len(universe), np.float32)
        for y in earlier:  # complete earlier seasons, as in Experiment 17
            days = (context.arrays[y]["count"] > 0).sum(axis=1)
            prior_years += days > 0
            prior_wild += label[y].sum(axis=1)
            prior_other += days - label[y].sum(axis=1)
        count = a["count"][u, d]
        per_year = len(earlier) or np.nan
        frame = pd.DataFrame({
            "u": u, "year": year, "region": region_of[u], "h3_cell": universe[u],
            "date": pd.to_datetime(np.datetime64(f"{year}-05-01") + d.astype("timedelta64[D]")).astype("datetime64[ns]"),
            "count": count, "night_share": a["night"][u, d] / count, "non_vegetation_share": 1 - a["veg"][u, d] / count,
            "frp_sum": a["frp_sum"][u, d], "frp_max": a["frp_max"][u, d], "day_of_season": d,
            "complex_size_3d": size[u, d], "complex_frp_3d": frp3[u, d],
            "active_days_7d": window(indicator, 7)[u, d], "active_days_30d": window(indicator, 30)[u, d],
            "season_active_days": window(indicator, SEASON_DAYS)[u, d],
            # Per-season rates, not counts: a count of earlier seasons would identify the year (NaN in 2018).
            "prior_share_years_detected": prior_years[u] / per_year, "prior_wildfire_days_per_year": prior_wild[u] / per_year,
            "prior_other_days_per_year": prior_other[u] / per_year, "latitude": lat_lon[u, 0], "longitude": lat_lon[u, 1]})
        frame[STATIC] = static.values("universe", universe, year)[u]
        frame["non_wildfire"] = (~label[year][u, d]).astype(np.int8)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def industry_or_farm(frame):
    return ((frame.og_facilities_disk1 > 0) | (frame.og_wells_disk1 > 0) | (frame.alr_share >= 0.5)).to_numpy()


def wildfire_flag(context, links, static, regions, onsets, fit_checks):
    print("Part C: wildfire flag", flush=True)
    table = detection_table(context, links, static, regions)
    train = table.year.isin(TRAIN_YEARS).to_numpy()
    X = table[FLAG_FEATURES].to_numpy(np.float32)
    y = table.non_wildfire.to_numpy()
    model = fit("wildfire_flag", X[train], y[train], FLAG_FEATURES, fit_checks)
    table["p_non_wildfire"] = model.predict_proba(X)[:, 1].astype(np.float32)
    selection = table.year.eq(SELECTION_YEAR).to_numpy()
    # Full precision-recall search: non-wildfire detections are far commoner than forecast positives.
    precision, recall, cuts = precision_recall_curve(y[selection], table.p_non_wildfire.to_numpy()[selection])
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    threshold = float(cuts[int(np.argmax(f1[:-1]))])
    rows = []
    for year, part in table.groupby("year"):
        truth = part.non_wildfire.to_numpy() == 1
        for name, flag in (("wildfire_flag", part.p_non_wildfire.to_numpy() >= threshold),
                           ("industry_or_farm_rule", industry_or_farm(part))):
            tp = int((flag & truth).sum())
            precision, recall = tp / max(int(flag.sum()), 1), tp / max(int(truth.sum()), 1)
            row = {"year": year, "role": "train" if year in TRAIN_YEARS else "selection" if year == SELECTION_YEAR else "evaluation",
                   "method": name, "detected_cell_days": len(part), "non_wildfire_share": truth.mean(),
                   "precision": precision, "recall": recall,
                   "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
            if name == "wildfire_flag":
                row |= {"threshold": threshold, "roc_auc": roc_auc_score(truth, part.p_non_wildfire),
                        "average_precision": average_precision_score(truth, part.p_non_wildfire)}
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "wildfire_flag_metrics.csv", index=False)
    # Independent check: Experiment 16's onset groups come from BCWS ignition points, not perimeters.
    joined = onsets.rename(columns={"target_date": "date"}).astype({"date": "datetime64[ns]"}).merge(
        table[["h3_cell", "date", "year", "p_non_wildfire", "non_wildfire"]], on=["h3_cell", "date"], how="inner")
    joined = joined.loc[~joined.year.isin(TRAIN_YEARS)]
    by_group = joined.groupby(["year", "bcws_category"]).agg(
        onset_cells=("p_non_wildfire", "size"), mean_p_non_wildfire=("p_non_wildfire", "mean"),
        flagged_share=("p_non_wildfire", lambda p: float((p >= threshold).mean())),
        label_non_wildfire_share=("non_wildfire", "mean")).reset_index()
    by_group.to_csv(RESULTS / "wildfire_flag_onsets.csv", index=False)
    table.loc[~train, ["h3_cell", "date", "region", "non_wildfire", "p_non_wildfire"]].to_parquet(
        RESULTS / "wildfire_flag_predictions_2022_2023.parquet", index=False)
    print(metrics.round(3).to_string(), flush=True)
    print(by_group.round(3).to_string(), flush=True)


# ---------------------------------------------------------------- main

def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    RESULTS = args.results
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    assert set(regions) <= set(all_regions)
    source_paths = [DETECTIONS, BCWS, PERIMETERS, EXPERIMENT_08_MODEL, ONSETS / "onset_cells.csv", ONSETS / "onset_clusters.csv",
                    *sorted(STATIC_DIR.glob("*.parquet"))]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    print("Linking detections to BCWS wildfires", flush=True)
    links = linked_detections(context, context.incidents, load_perimeters(context.incidents))
    static = StaticLayers()
    onsets = onset_categories()
    fit_checks = []

    wildfire_flag(context, links, static, regions, onsets, fit_checks)

    print("Building training sample (2018-2021)", flush=True)
    sample, sample_stats = training_sample(regions, all_regions, context, links, static)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    reference = pd.read_csv(EXPERIMENT_08_SAMPLE).set_index(["region", "year"]).kept_rows
    assert all(reference[r.region, r.year] == r.kept_rows for r in sample_stats.itertuples()), "not Experiment 13's draw"
    models = {"catboost_08": catboost_model()}
    models["catboost_08"].load_model(str(EXPERIMENT_08_MODEL))
    y = sample[TARGET].to_numpy()
    X, quiet = design(sample, "all_static")
    models["catboost_static"] = fit("catboost_static", X, y, FEATURE_SETS["all_static"] + ["quiet_stratum"], fit_checks)
    X, _ = design(sample.loc[quiet], "quiet")
    models["catboost_quiet"] = fit("catboost_quiet", X, y[quiet], FEATURE_SETS["quiet"] + ["quiet_stratum"], fit_checks)
    del sample, X

    forecast_rows, thresholds, daily, watch_rows, watch_ap, watch_onsets = [], {}, {}, [], [], []
    for region_index, region in enumerate(regions):
        g = context.grids[region]
        cells = g["cells"]
        disk7 = neighbourhood(list(cells), list(cells))
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            tic = time.time()
            frame = load(region, year, with_cells=True)  # features are built per row chunk to bound memory
            n = len(frame)
            scores = {"rule_yesterday": (frame.local_count_1d > 0).to_numpy(np.float32),
                      **{name: np.empty(n, np.float32) for name in WATCH}}
            for start in range(0, n, CHUNK):
                chunk = context.features(region, year, frame.iloc[start:start + CHUNK].reset_index(drop=True))
                chunk = add_static(static, context, region, year, add_history(context, links, region, year, chunk))
                part = slice(start, start + len(chunk))
                for name in ("catboost_08", "catboost_static", "catboost_quiet"):
                    feature_set = {"catboost_08": "all", "catboost_static": "all_static", "catboost_quiet": "quiet"}[name]
                    scores[name][part] = predict(models[name], chunk, feature_set)
                # Simple rankings: nearest detection in 30 days, then active cells within 5 rings;
                # earlier seasons the cell, then its neighbours, burned.
                scores["nearby_activity"][part] = ((MAX_RING + 1 - chunk.dist_nearest_30d) * 1e4
                                                   + np.minimum(chunk.active_cells_disk5_30d, 9999)).to_numpy(np.float32)
                scores["burned_before"][part] = (chunk.prior_years_detected_cell * 10 + chunk.prior_years_detected_disk1).to_numpy(np.float32)
                del chunk
            grids = {name: to_grid(frame, cells, s) for name, s in scores.items()}
            truth = to_grid(frame, cells, frame[TARGET].to_numpy()) > 0
            quiet = to_grid(frame, cells, quiet_stratum(frame)) > 0
            assert len(frame) == truth.size, "expected a full cell x day grid"
            del frame, scores
            local = g["rings"][0] @ context.arrays[year]["count"]
            near = ((g["disk1"] @ context.arrays[year]["count"]) > 0)[:, FIRST_TARGET_DAY:]
            assert (near >= truth).all()
            future = forward_window(local, WATCH_DAYS)[:, FIRST_TARGET_DAY:] > 0
            assert (future >= truth).all()

            # A. next-day forecast
            for name in FORECAST:
                score = grids[name]
                spread = sparse_row_max(disk7, score)
                for definition in DEFINITIONS:
                    inputs = ((score, truth, truth, score) if definition == "res7_exact" else (score, truth, near, spread))
                    tag = (region, definition, name)
                    if year == SELECTION_YEAR:
                        flat = [a.ravel() for a in inputs]
                        candidates = candidate_thresholds(flat[0])
                        thresholds[tag] = float(candidates[int(np.argmax(curve(*flat, candidates)[2]))])
                    counts = daily_counts(*inputs, thresholds[tag])
                    if year == EVALUATION_YEAR:
                        daily[tag] = counts
                    total = counts.sum(axis=0)
                    row = {"region": region, "year": year, "definition": definition, "model": name,
                           "threshold": thresholds[tag], "precision": total[1] / max(total[0], 1),
                           "recall": total[3] / max(total[2], 1), "f1": float(f1_from_counts(total)),
                           "flagged": int(total[0]), "positives": int(total[2])}
                    if definition == "res7_exact" and name != "rule_yesterday":
                        row["average_precision"] = float(average_precision_score(truth.ravel(), score.ravel()))
                    forecast_rows.append(row)

            # B. quiet-cell watch list
            quiet_positive = truth & quiet
            tiebreak = np.random.default_rng([SEED, year, region_index]).random(truth.shape)
            quiet_cells = quiet.sum(axis=0)
            positives_by_day = quiet_positive.sum(axis=0)
            for n_size in WATCH_SIZES:  # random list of n quiet cells, in expectation
                share = np.minimum(n_size, quiet_cells) / np.maximum(quiet_cells, 1)
                listed = np.minimum(n_size, quiet_cells).sum()
                watch_rows.append({"region": region, "year": year, "model": "random", "list_size": n_size,
                                   "quiet_fires": int(quiet_positive.sum()), "listed_cell_days": int(listed),
                                   "caught_next_day": (share * positives_by_day).sum() / max(quiet_positive.sum(), 1),
                                   "caught_within_7d": np.nan,
                                   "hit_rate_next_day": (share * positives_by_day).sum() / max(listed, 1),
                                   "hit_rate_within_7d": (share * (future & quiet).sum(axis=0)).sum() / max(listed, 1)})
            if year == EVALUATION_YEAR:
                onset_cell, onset_day, onset_group = onset_positions(onsets, cells, year)
                is_quiet_onset = quiet_positive[onset_cell, onset_day]
                onset_cell, onset_day, onset_group = onset_cell[is_quiet_onset], onset_day[is_quiet_onset], onset_group[is_quiet_onset]
            for name in WATCH:
                score = grids[name]
                watch_ap.append({"region": region, "year": year, "model": name, "quiet_rows": int(quiet.sum()),
                                 "quiet_fires": int(quiet_positive.sum()),
                                 "average_precision_quiet": float(average_precision_score(truth[quiet], score[quiet])),
                                 "base_rate": float(truth[quiet].mean())})
                for n_size, listed in ranked_lists(score, quiet, tiebreak, WATCH_SIZES).items():
                    recent = window(listed.astype(np.float32), WATCH_DAYS) > 0  # listed on day t-6 .. t
                    watch_rows.append({"region": region, "year": year, "model": name, "list_size": n_size,
                                       "quiet_fires": int(quiet_positive.sum()), "listed_cell_days": int(listed.sum()),
                                       "caught_next_day": (listed & quiet_positive).sum() / max(quiet_positive.sum(), 1),
                                       "caught_within_7d": (recent & quiet_positive).sum() / max(quiet_positive.sum(), 1),
                                       "hit_rate_next_day": (listed & truth).sum() / max(listed.sum(), 1),
                                       "hit_rate_within_7d": (listed & future).sum() / max(listed.sum(), 1)})
                    if year == EVALUATION_YEAR:
                        caught = recent[onset_cell, onset_day]
                        for group in np.unique(onset_group):
                            mask = onset_group == group
                            watch_onsets.append({"region": region, "model": name, "list_size": n_size, "onset_group": group,
                                                 "quiet_onset_cells": int(mask.sum()), "caught_within_7d": int(caught[mask].sum())})
            latest = [r for r in forecast_rows if r["region"] == region and r["year"] == year and r["definition"] == "res7_margin1"]
            top = [r for r in watch_rows if r["region"] == region and r["year"] == year and r["list_size"] == 50]
            print(f"{region} {year} ({time.time()-tic:.0f}s) margin F1: " + "; ".join(f"{r['model']} {r['f1']:.3f}" for r in latest)
                  + " | top-50 quiet caught in 7d: " + "; ".join(f"{r['model']} {r['caught_within_7d']:.3f}" for r in top
                                                                if r["model"] != "random"), flush=True)
            del grids, truth, quiet, near, future, tiebreak

    forecast = pd.DataFrame(forecast_rows)
    forecast.to_csv(RESULTS / "forecast_metrics.csv", index=False)
    evaluation = forecast.loc[forecast.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["definition", "model"])[["f1", "precision", "recall", "average_precision"]].median().reset_index()
    pooled = {(d, m): np.sum([daily[r, d, m] for r in regions], axis=0) for d in DEFINITIONS for m in FORECAST}
    summary["bc_wide_f1"] = [float(f1_from_counts(pooled[d, m].sum(axis=0))) for d, m in zip(summary.definition, summary.model)]
    summary.to_csv(RESULTS / "forecast_summary_2023.csv", index=False)

    # Week-block bootstrap of F1 differences (whole ISO weeks resampled; same draws for every model).
    dates = pd.date_range(f"{EVALUATION_YEAR}-05-01", periods=SEASON_DAYS)[FIRST_TARGET_DAY:]
    weeks = dates.isocalendar().week.to_numpy()
    unique_weeks = np.unique(weeks)
    week_member = (weeks[None, :] == unique_weeks[:, None]).astype(np.float64)  # weeks x days
    draws = np.random.default_rng(SEED).integers(0, len(unique_weeks), size=(BOOTSTRAP, len(unique_weeks)))
    resample = np.stack([np.bincount(d, minlength=len(unique_weeks)) for d in draws]).astype(np.float64)  # B x weeks

    def boot(counts):
        return f1_from_counts(resample @ (week_member @ counts))

    comparisons = [("catboost_08", "rule_yesterday"), ("catboost_static", "rule_yesterday"), ("catboost_static", "catboost_08")]
    boot_rows = []
    for definition in DEFINITIONS:
        for scope in [*regions, "BC-wide"]:
            get = (lambda m: pooled[definition, m]) if scope == "BC-wide" else (lambda m: daily[scope, definition, m])
            for model, baseline in comparisons:
                difference = boot(get(model)) - boot(get(baseline))
                boot_rows.append({"scope": scope, "definition": definition, "model": model, "baseline": baseline,
                                  "difference": float(f1_from_counts(get(model).sum(0)) - f1_from_counts(get(baseline).sum(0))),
                                  "ci_low": float(np.quantile(difference, .025)), "ci_high": float(np.quantile(difference, .975)),
                                  "share_above": float((difference > 0).mean())})
    pd.DataFrame(boot_rows).to_csv(RESULTS / "forecast_bootstrap_2023.csv", index=False)

    # Experiment 13's model and the rule must reproduce Experiment 17 exactly.
    if EXPERIMENT_12_METRICS.exists() and regions == all_regions:
        previous = pd.read_csv(EXPERIMENT_12_METRICS).query("target == 'recorded'")
        merged = forecast.merge(previous, on=["region", "year", "definition", "model"], suffixes=("", "_12"))
        assert len(merged) == 2 * len(regions) * len(DEFINITIONS) * 2, "missing Experiment 17 rows"
        largest = float((merged.f1 - merged.f1_12).abs().max())
        assert largest < 1e-9, f"does not reproduce Experiment 17 (max F1 gap {largest})"
        fit_checks.append({"model": "reproduces_experiment_12", "max_abs_f1_difference": largest})

    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    pd.DataFrame(watch_rows).to_csv(RESULTS / "watch_list.csv", index=False)
    pd.DataFrame(watch_ap).to_csv(RESULTS / "watch_list_average_precision.csv", index=False)
    pd.DataFrame(watch_onsets).to_csv(RESULTS / "watch_list_onsets_2023.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False, "static_features": STATIC,
        "static_notes": "terrain, fuels and ALR are current layers; facilities count if approved before Jan 1 of the "
                        "season year and not cancelled by May 1; wells if authorized before Jan 1; cutblocks harvested "
                        "in the three previous calendar years; disk1 = the cell and its 6 neighbours",
        "forecast_models": {"catboost_08": "Experiment 13 model (loaded)", "catboost_static": "all + static, Experiment 13's draw"},
        "watch_list": {"quiet": "no detection in the cell or a neighbour in the 7 days to the history end",
                       "catboost_quiet": "all + history + static, trained on quiet rows of Experiment 13's draw",
                       "nearby_activity": "nearest detection within 5 rings in 30 days, then active cells within 5 rings",
                       "burned_before": "earlier seasons with a detection in the cell, then in its neighbours",
                       "list_sizes": list(WATCH_SIZES), "caught_within_7d": "listed on any of the 7 days up to the fire day",
                       "hit_rate_within_7d": "listed cell detected on any of the next 7 days"},
        "wildfire_flag": {"label": "not linked to a BCWS wildfire (Experiment 17's after-the-fact link)",
                          "features": FLAG_FEATURES, "no_bcws_inputs": True,
                          "threshold": "maximizes 2022 F1 for the non-wildfire class"},
        "thresholds": "maximize 2022 F1 per Fire Centre, definition and model; applied unchanged to 2023",
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    print(summary.round(3).to_string(), flush=True)
    print(pd.DataFrame(boot_rows).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
