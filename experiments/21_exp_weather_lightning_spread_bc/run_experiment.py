"""Add weather, lightning and wind-aligned spread inputs to the next-day forecast.

Label, training years, sampling, CatBoost settings and scoring follow Experiment 20: the
S-NPP or NOAA-20 gap-filled label (2018-2020 train on gap-filled S-NPP until FIRMS NOAA-20
arrives), train 2018-2021, thresholds from 2022, evaluate 2023, week-block bootstrap against
"S-NPP or NOAA-20 detection yesterday". 2024 is never loaded.

Every input for target day t is known by the end of UTC day t-1 (the history day h):
- NOAA-20 and two-satellite history (Experiment 20's N20 inputs), NaN before 2021, when the
  CWFIS archive is missing or incomplete, so training years keep inputs consistent with labels.
- Weather: BCWS station values from the 12:00 LST row of local date h (20:00 UTC on day h):
  temperature, humidity, wind speed, 24 h precipitation (noon to noon), FWI system codes, and
  3-day precipitation. Each cell uses its nearest station with a reading that day, among its
  three nearest within 100 km (NaN beyond). Daily maxima and minima are not used: the local day
  ends eight hours into UTC day t.
- Lightning: CanCPLD cloud-to-ground flashes in UTC day h (and h-2..h) in the 0.1 degree cell
  holding the H3 cell centre, its 3x3 and 9x9 blocks, and days since lightning in the 3x3 block.
- Spread: for burning neighbours (ring 1 and ring 2) on day h, the sum of cos(angle between the
  direction to the neighbour and the direction the wind blows from), using the station's noon
  wind direction; positive when fire sits upwind. Also multiplied by wind speed. Burning means
  S-NPP or NOAA-20 from 2021 and S-NPP before, matching the training label.

Models, all trained on 2018-2021 with the stopgap label:
  catboost_stopgap (Experiment 20's model, loaded; reproduces Experiment 20),
  catboost_n20 (Experiment 13 inputs + N20), catboost_drivers (+ weather + lightning),
  catboost_spread (+ spread), catboost_spread_noday (catboost_spread without day_of_year).
Thresholds two ways: maximize 2022 F1 per Fire Centre ("f1_2022"), and flag each day as many
cells per Fire Centre as the two-satellite rule flags that day ("rule_volume"; needs no tuning).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DATASETS = EXPERIMENTS.parent / "datasets"
spec = importlib.util.spec_from_file_location("experiment_15", EXPERIMENTS / "20_exp_retrain_multisensor_label_bc/run_experiment.py")
experiment_15 = sys.modules["experiment_15"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_15)
from experiment_15 import LABEL, N20, RULES, DEFINITIONS, Labels, to_grid  # noqa: E402
from experiment_14 import CWFIS, FIRMS, shift  # noqa: E402
from experiment_13 import BOOTSTRAP, daily_counts, f1_from_counts  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve  # noqa: E402
from experiment_08 import (Context, FEATURE_SETS, QUIET_SAMPLE_RATE, REFERENCE, SEASON_DAYS, SEED,  # noqa: E402
                           SELECTION_YEAR, EVALUATION_YEAR, TRAIN_YEARS, catboost_model, design, digest, load,
                           predict, quiet_stratum, season_index, sparse_row_max, unit_xyz, window)

WEATHER_DAILY = DATASETS / "bcws_weather/station_daily_2018_2023.parquet"
STATIONS = DATASETS / "bcws_weather/stations_2018_2023.parquet"
LIGHTNING = DATASETS / "cancpld_lightning/bc_cg_daily_2018_2023.npz"
EXPERIMENT_15_MODEL = EXPERIMENTS / "20_exp_retrain_multisensor_label_bc/results/models/catboost_stopgap.cbm"
EXPERIMENT_15_METRICS = EXPERIMENTS / "20_exp_retrain_multisensor_label_bc/results/metrics_by_year.csv"
FULL_NOAA20_FROM = 2021
STATION_K, STATION_MAX_KM = 3, 100.0
STATION_VALUES = {"temp_noon": "wx_temp", "rh_noon": "wx_rh", "wind_noon": "wx_wind", "precip_24h_mm": "wx_precip_24h",
                  "ffmc": "wx_ffmc", "dmc": "wx_dmc", "dc": "wx_dc", "bui": "wx_bui", "isi": "wx_isi", "fwi": "wx_fwi"}
WEATHER = [*STATION_VALUES.values(), "wx_precip_3d", "wx_station_km"]
LIGHTNING_FEATURES = ["ltg_cell_1d", "ltg_3x3_1d", "ltg_3x3_3d", "ltg_9x9_1d", "ltg_9x9_3d", "ltg_days_since_3x3"]
LIGHTNING_DAYS_CAP = 15
SPREAD = ["upwind_ring1_1d", "upwind_ring2_1d", "upwind_ring1_x_wind", "upwind_ring2_x_wind"]
FEATURE_SETS["n20"] = FEATURE_SETS["all"] + N20
FEATURE_SETS["drivers"] = FEATURE_SETS["n20"] + WEATHER + LIGHTNING_FEATURES
FEATURE_SETS["spread"] = FEATURE_SETS["drivers"] + SPREAD
FEATURE_SETS["spread_noday"] = [c for c in FEATURE_SETS["spread"] if c != "day_of_year"]
MODELS = {"catboost_stopgap": "all", "catboost_n20": "n20", "catboost_drivers": "drivers",
          "catboost_spread": "spread", "catboost_spread_noday": "spread_noday"}
LOADED = {"catboost_stopgap": EXPERIMENT_15_MODEL}
METHODS = ("f1_2022", "rule_volume")
CHUNK = 500_000  # half of Experiment 20's: the full run was killed by memory pressure at 1,000,000


# ---------------------------------------------------------------- drivers

def box_sum(a, radius):
    """Sum over the (2r+1) x (2r+1) block around each cell of a (days x lat x lon) array."""
    padded = np.pad(a, ((0, 0), (radius + 1, radius), (radius + 1, radius)))
    c = padded.cumsum(1, dtype=np.float64).cumsum(2)
    k = 2 * radius + 1
    return (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]).astype(np.float32)


def day_window(a, w):
    """Sum over the w days ending at each day (axis 0), truncated at May 1."""
    c = np.concatenate([np.zeros_like(a[:1], dtype=np.float64), a.cumsum(0, dtype=np.float64)])
    end = np.arange(1, a.shape[0] + 1)
    return (c[end] - c[np.maximum(end - w, 0)]).astype(np.float32)


def bearing_matrices(cells, ring):
    """Sparse (cells x cells) sin and cos of the bearing from each cell to its ring-k neighbours."""
    position = {c: i for i, c in enumerate(cells)}
    centre = np.array([h3.cell_to_latlng(c) for c in cells])
    rows, cols = [], []
    for i, cell in enumerate(cells):
        for other in h3.grid_ring(cell, ring):
            j = position.get(other)
            if j is not None:
                rows.append(i)
                cols.append(j)
    rows, cols = np.array(rows), np.array(cols)
    east = (centre[cols, 1] - centre[rows, 1]) * np.cos(np.deg2rad(centre[rows, 0]))
    north = centre[cols, 0] - centre[rows, 0]
    norm = np.hypot(east, north)
    shape = (len(cells),) * 2
    return (sp.csr_matrix(((east / norm).astype(np.float32), (rows, cols)), shape=shape),
            sp.csr_matrix(((north / norm).astype(np.float32), (rows, cols)), shape=shape))


class Drivers:
    """Weather, lightning and spread inputs per region-year (cells x 153 by history day)."""

    def __init__(self, context, regions):
        tic = time.time()
        daily = pd.read_parquet(WEATHER_DAILY)
        stations = pd.read_parquet(STATIONS)
        assert daily.date.dt.year.between(2018, 2023).all() and stations.year.between(2018, 2023).all()
        self.weather = {}
        for year in range(2018, 2024):
            st = stations.loc[stations.year.eq(year) & stations.latitude.notna() & stations.longitude.notna()]
            st = st.drop_duplicates("station_code", keep="last").reset_index(drop=True)
            d = daily.loc[daily.year.eq(year)]
            index = pd.Index(st.station_code).get_indexer(d.station_code)
            d, index = d.loc[index >= 0], index[index >= 0]
            day = season_index(d.date)
            values = {}
            for column in [*STATION_VALUES, "wind_dir_noon"]:
                a = np.full((len(st), SEASON_DAYS), np.nan, np.float32)
                a[index, day] = d[column].to_numpy(np.float32)
                values[column] = a
            values["precip_3d"] = window(np.nan_to_num(values["precip_24h_mm"]), 3)
            has = np.zeros((len(st), SEASON_DAYS), bool)
            has[index, day] = True
            self.weather[year] = {"tree": cKDTree(unit_xyz(st.latitude.to_numpy(), st.longitude.to_numpy())),
                                  "values": values, "has": has}
        ltg = np.load(LIGHTNING)
        self.ltg_years = [int(y) for y in ltg["years"]]
        self.ltg = ltg["daily"]
        self.ltg_lat, self.ltg_lon = ltg["lat"], ltg["lon"]
        assert max(self.ltg_years) <= 2023
        self.context = context
        self.geometry = {}
        for region in regions:
            cells = list(context.grids[region]["cells"])
            centre = np.array([h3.cell_to_latlng(c) for c in cells])
            iy = np.clip(np.rint((centre[:, 0] - self.ltg_lat[0]) / 0.1).astype(int), 0, len(self.ltg_lat) - 1)
            ix = np.clip(np.rint((centre[:, 1] - self.ltg_lon[0]) / 0.1).astype(int), 0, len(self.ltg_lon) - 1)
            self.geometry[region] = {"iy": iy, "ix": ix, "rings": [bearing_matrices(cells, k) for k in (1, 2)]}
        self._cache = {}
        print(f"Drivers ready in {time.time()-tic:.0f}s", flush=True)

    def arrays(self, region, year, fire):
        """Dict of cells x 153 arrays; fire is the cells x 153 burning map used for spread."""
        key = (region, year)
        if key in self._cache:
            return self._cache[key]
        self._cache.clear()
        g, geo, w = self.context.grids[region], self.geometry[region], self.weather[year]
        n = len(g["cells"])
        distance, nearest = w["tree"].query(g["xyz"], k=STATION_K, distance_upper_bound=STATION_MAX_KM)
        out = {name: np.full((n, SEASON_DAYS), np.nan, np.float32) for name in [*WEATHER, "wind_dir"]}
        chosen = np.zeros((n, SEASON_DAYS), bool)
        for rank in range(STATION_K):
            ok = np.isfinite(distance[:, rank])
            station = np.where(ok, nearest[:, rank], 0)
            take = ~chosen & ok[:, None] & w["has"][station]
            for column, name in [*STATION_VALUES.items(), ("precip_3d", "wx_precip_3d"), ("wind_dir_noon", "wind_dir")]:
                out[name][take] = w["values"][column][station][take]
            out["wx_station_km"][take] = np.broadcast_to(distance[:, rank:rank + 1], take.shape)[take]
            chosen |= take
        flashes = self.ltg[self.ltg_years.index(year)]
        block1, block4 = box_sum(flashes, 1), box_sum(flashes, 4)
        pick = lambda a: a[:, geo["iy"], geo["ix"]].T
        out["ltg_cell_1d"] = pick(flashes)
        out["ltg_3x3_1d"], out["ltg_3x3_3d"] = pick(block1), pick(day_window(block1, 3))
        out["ltg_9x9_1d"], out["ltg_9x9_3d"] = pick(block4), pick(day_window(block4, 3))
        since = np.full((n, SEASON_DAYS), LIGHTNING_DAYS_CAP, np.float32)
        struck = out["ltg_3x3_1d"] > 0
        for day in range(SEASON_DAYS):
            if day:
                since[:, day] = np.minimum(since[:, day - 1] + 1, LIGHTNING_DAYS_CAP)
            since[struck[:, day], day] = 0
        out["ltg_days_since_3x3"] = since
        theta = np.deg2rad(out.pop("wind_dir"))
        fire = fire.astype(np.float32)
        for ring, (sin_b, cos_b) in zip((1, 2), geo["rings"]):
            aligned = (sin_b @ fire) * np.sin(theta) + (cos_b @ fire) * np.cos(theta)
            out[f"upwind_ring{ring}_1d"] = aligned
            out[f"upwind_ring{ring}_x_wind"] = aligned * out["wx_wind"]
        self._cache[key] = out
        return out

    def add(self, region, year, frame, cells, labels):
        lab = labels.grids[region, year]
        fire = lab["pair"] if year >= FULL_NOAA20_FROM else lab["snpp"]
        arrays = self.arrays(region, year, fire)
        cell, history = cells.get_indexer(frame.h3_cell), season_index(frame.history_end_date)
        for name, values in arrays.items():
            frame[name] = values[cell, history]
        if year < FULL_NOAA20_FROM:
            frame[N20] = np.nan
        return frame


def build(context, labels, drivers, region, year, frame):
    cells = context.grids[region]["cells"]
    frame = context.features(region, year, frame)
    frame = labels.add(region, year, frame, cells)
    return drivers.add(region, year, frame, cells, labels)


def training_sample(regions, all_regions, context, labels, drivers):
    """Experiment 20's draw (every active row, every new-label positive, 1% of quiet negatives)."""
    parts, stats = [], []
    columns = list(dict.fromkeys(c for s in set(MODELS.values()) for c in FEATURE_SETS[s]))
    for region in regions:
        index = all_regions.index(region)
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            positive = labels.grids[region, year]["label"][cells.get_indexer(frame.h3_cell), season_index(frame.target_date)]
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            keep = ~quiet_stratum(frame) | positive | drawn
            part = build(context, labels, drivers, region, year, frame.loc[keep].reset_index(drop=True))
            parts.append(part[columns + [LABEL]].assign(region=region, year=year))
            stats.append({"region": region, "year": year, "rows": len(frame), "kept_rows": int(keep.sum()),
                          "positives": int(positive.sum()), "weather_missing_share": float(part.wx_ffmc.isna().mean())})
            print(f"  {region} {year}: kept {keep.sum():,}; positives {positive.sum():,}; "
                  f"no station within {STATION_MAX_KM:.0f} km {part.wx_ffmc.isna().mean():.1%}", flush=True)
            del frame, part
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


# ---------------------------------------------------------------- scoring

def volume_flags(score, k):
    """Per day (column), flag the k[d] highest-scoring cells."""
    flags = np.zeros(score.shape, bool)
    for day in range(score.shape[1]):
        if k[day]:
            flags[np.argpartition(-score[:, day], k[day] - 1)[:k[day]], day] = True
    return flags


def breakdown(flag, truth, yesterday):
    """Counts separating cells burning yesterday from new ones (exact cell)."""
    return {"new_positives": int((truth & ~yesterday).sum()), "new_caught": int((truth & ~yesterday & flag).sum()),
            "continuing_positives": int((truth & yesterday).sum()), "continuing_caught": int((truth & yesterday & flag).sum()),
            "stopped": int((~truth & yesterday).sum()), "stopped_flagged": int((~truth & yesterday & flag).sum()),
            "fresh_false_alarms": int((~truth & ~yesterday & flag).sum())}


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    RESULTS = experiment_15.RESULTS = args.results
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    source_paths = [FIRMS, CWFIS, WEATHER_DAILY, STATIONS, LIGHTNING, EXPERIMENT_15_MODEL]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    labels = Labels(context, regions)
    drivers = Drivers(context, regions)

    print("Building training sample (2018-2021)", flush=True)
    sample, sample_stats = training_sample(regions, all_regions, context, labels, drivers)
    sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
    fit_checks, models = [], {}
    for name, feature_set in MODELS.items():
        if name in LOADED:
            models[name] = catboost_model()
            models[name].load_model(str(LOADED[name]))
            continue
        X, _ = design(sample, feature_set)
        models[name] = experiment_15.fit(name, X, sample[LABEL].to_numpy(), FEATURE_SETS[feature_set] + ["quiet_stratum"], fit_checks)
        del X
    del sample
    for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:  # evaluation needs only 2022-2023
        del labels.grids[key]
    labels.sources = None
    gc.collect()

    rows, thresholds, daily, splits = [], {}, {}, []
    names = (*RULES, *MODELS)
    for region in regions:
        cells = context.grids[region]["cells"]
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            tic = time.time()
            g = labels.grids[region, year]
            frame = load(region, year, with_cells=True)
            n = len(frame)
            cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
            scores = {name: np.empty(n, np.float32) for name in names}
            for start in range(0, n, CHUNK):
                chunk = build(context, labels, drivers, region, year, frame.iloc[start:start + CHUNK].reset_index(drop=True))
                part = slice(start, start + len(chunk))
                scores["rule_snpp_yesterday"][part] = (chunk.local_count_1d > 0).to_numpy(np.float32)
                scores["rule_pair_yesterday"][part] = chunk.pair_yesterday.to_numpy(np.float32)
                for name, feature_set in MODELS.items():
                    scores[name][part] = predict(models[name], chunk, feature_set)
                del chunk
            del frame
            assert n == len(cells) * (SEASON_DAYS - FIRST_TARGET_DAY), "expected a full cell x day grid"
            truth = g["label_full"][:, FIRST_TARGET_DAY:]
            near = ((g["disk1"] @ g["label_full"].astype(np.float32)) > 0)[:, FIRST_TARGET_DAY:]
            yesterday = shift(g["pair"], 1)[:, FIRST_TARGET_DAY:]
            volume = yesterday.sum(axis=0)
            for name, s in scores.items():
                score = to_grid(cell, target, s, len(cells))
                for method in METHODS:
                    if method == "rule_volume" and name in RULES:
                        continue
                    if method == "rule_volume":
                        flagged = volume_flags(score, volume).astype(np.float32)
                    for definition in DEFINITIONS:
                        tag = (region, definition, name, method)
                        if method == "f1_2022":
                            spread = sparse_row_max(g["disk1"], score)
                            inputs = (score, truth, truth, score) if definition == "res7_exact" else (score, truth, near, spread)
                            if year == SELECTION_YEAR:
                                flat = [a.ravel() for a in inputs]
                                candidates = candidate_thresholds(flat[0])
                                thresholds[tag] = float(candidates[int(np.argmax(curve(*flat, candidates)[2]))])
                            threshold = thresholds[tag]
                        else:
                            spread = sparse_row_max(g["disk1"], flagged)
                            inputs = (flagged, truth, truth, flagged) if definition == "res7_exact" else (flagged, truth, near, spread)
                            threshold = 0.5
                        counts = daily_counts(*inputs, threshold)
                        if year == EVALUATION_YEAR:
                            daily[tag] = counts
                            if definition == "res7_exact":
                                splits.append({"region": region, "model": name, "method": method,
                                               **breakdown(inputs[0] >= threshold, truth, yesterday)})
                        total = counts.sum(axis=0)
                        row = {"region": region, "year": year, "definition": definition, "model": name, "method": method,
                               "threshold": threshold, "precision": total[1] / max(total[0], 1),
                               "recall": total[3] / max(total[2], 1), "f1": float(f1_from_counts(total)),
                               "flagged": int(total[0]), "positives": int(total[2])}
                        if definition == "res7_exact" and method == "f1_2022" and name in MODELS:
                            row["average_precision"] = float(average_precision_score(truth.ravel(), score.ravel()))
                        rows.append(row)
            latest = [r for r in rows if r["region"] == region and r["year"] == year and r["method"] == "f1_2022"]
            print(f"{region} {year} ({time.time()-tic:.0f}s): " + "; ".join(
                f"{r['model']} {r['f1']:.3f}" for r in latest if r["definition"] == "res7_exact") + " | margin: " + "; ".join(
                f"{r['model']} {r['f1']:.3f}" for r in latest if r["definition"] == "res7_margin1"), flush=True)
            del scores
            gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    # Experiment 20's loaded model and both rules must reproduce Experiment 20 exactly.
    previous = pd.read_csv(EXPERIMENT_15_METRICS)
    check = metrics.loc[metrics.method.eq("f1_2022") & metrics.model.isin([*RULES, "catboost_stopgap"])].merge(
        previous, on=["region", "year", "definition", "model"], suffixes=("", "_15"))
    assert len(check) == len(regions) * 2 * len(DEFINITIONS) * 3 and np.allclose(check.f1, check.f1_15, atol=1e-9), "Experiment 20 not reproduced"

    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["definition", "method", "model"])[["f1", "precision", "recall", "average_precision"]].median().reset_index()
    pooled = {key[1:]: np.sum([daily[(r, *key[1:])] for r in regions], axis=0) for key in daily if key[0] == regions[0]}
    summary["bc_wide_f1"] = [float(f1_from_counts(pooled[d, m, k].sum(axis=0))) for d, k, m in
                             zip(summary.definition, summary.method, summary.model)]
    summary["bc_wide_precision"] = [pooled[d, m, k].sum(0)[1] / pooled[d, m, k].sum(0)[0] for d, k, m in
                                    zip(summary.definition, summary.method, summary.model)]
    summary["bc_wide_recall"] = [pooled[d, m, k].sum(0)[3] / pooled[d, m, k].sum(0)[2] for d, k, m in
                                 zip(summary.definition, summary.method, summary.model)]
    summary.to_csv(RESULTS / "summary_2023.csv", index=False)
    splits = pd.DataFrame(splits)
    splits.to_csv(RESULTS / "new_vs_continuing_2023.csv", index=False)

    dates = pd.date_range(f"{EVALUATION_YEAR}-05-01", periods=SEASON_DAYS)[FIRST_TARGET_DAY:]
    weeks = dates.isocalendar().week.to_numpy()
    unique_weeks = np.unique(weeks)
    week_member = (weeks[None, :] == unique_weeks[:, None]).astype(np.float64)
    draws = np.random.default_rng(SEED).integers(0, len(unique_weeks), size=(BOOTSTRAP, len(unique_weeks)))
    resample = np.stack([np.bincount(d, minlength=len(unique_weeks)) for d in draws]).astype(np.float64)
    boot = lambda counts: f1_from_counts(resample @ (week_member @ counts))
    boot_rows = []
    for definition in DEFINITIONS:
        for scope in [*regions, "BC-wide"]:
            get = (lambda key: pooled[key]) if scope == "BC-wide" else (lambda key: daily[(scope, *key)])
            baseline = get((definition, "rule_pair_yesterday", "f1_2022"))
            for model in names:
                for method in METHODS:
                    key = (definition, model, method)
                    if model == "rule_pair_yesterday" or (scope != "BC-wide" and (scope, *key) not in daily) or (scope == "BC-wide" and key not in pooled):
                        continue
                    difference = boot(get(key)) - boot(baseline)
                    boot_rows.append({"scope": scope, "definition": definition, "model": model, "method": method,
                                      "baseline": "rule_pair_yesterday",
                                      "difference": float(f1_from_counts(get(key).sum(0)) - f1_from_counts(baseline.sum(0))),
                                      "ci_low": float(np.quantile(difference, .025)), "ci_high": float(np.quantile(difference, .975)),
                                      "share_above": float((difference > 0).mean())})
    pd.DataFrame(boot_rows).to_csv(RESULTS / "bootstrap_2023.csv", index=False)
    pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "label": "S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell, plus one-day gap filling",
        "training_label": "2021: the full label; 2018-2020: gap-filled S-NPP only (stopgap until FIRMS NOAA-20 arrives)",
        "models": {k: {"feature_set": v, "inputs": FEATURE_SETS[v], "loaded_from": str(LOADED.get(k, ""))} for k, v in MODELS.items()},
        "weather": f"nearest of {STATION_K} stations within {STATION_MAX_KM:.0f} km with a 12:00 LST reading on the history date",
        "lightning": "CanCPLD cloud-to-ground flashes, 0.1 degree, UTC history day",
        "spread": "ring-1/ring-2 burning neighbours weighted by cos(bearing - noon wind-from direction)",
        "n20_inputs": f"NaN before {FULL_NOAA20_FROM}", "methods": {
            "f1_2022": "threshold maximizing 2022 F1 per Fire Centre, definition and model",
            "rule_volume": "each day, flag as many cells per Fire Centre as the two-satellite rule"},
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023, against rule_pair_yesterday",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(), flush=True)
    print(pd.DataFrame(boot_rows).query("scope == 'BC-wide'").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
