"""Final test: the saved models of Experiments 20-23 scored once on 2024.

Nothing is retrained or tuned. Thresholds are recomputed from 2022 exactly as before (per Fire
Centre, model and definition), or follow the two-satellite rule's daily volume, and are applied
unchanged to 2024. The label is Experiment 19's: S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection,
one-day gaps filled. NOAA-21, new in 2024, is left out so the label means what it meant in
training. Every forecast must reproduce its 2022 numbers from Experiments 21 and 22 (Experiment
23's scoring never finished, so its models have no 2022 reference).

2024 inputs come from separate files (see README.md); the 2018-2023 files are untouched. The
loaders of Experiments 13, 20 and 21 stop at 2023 on purpose, so this script extends them: it
replaces Experiment 13's detection and incident loaders (as Experiment 18 replaces its BCWS
features) and adds 2024 to Experiment 20's labels and Experiment 21's weather and lightning.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DATASETS = EXPERIMENTS.parent / "datasets"
spec = importlib.util.spec_from_file_location("experiment_18", EXPERIMENTS / "23_exp_regularised_neural_bc/run_experiment.py")
experiment_18 = sys.modules["experiment_18"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_18)
import experiment_08  # noqa: E402
import experiment_17  # noqa: E402
from experiment_18 import EXPERIMENT_17, experiment_17_scaler  # noqa: E402
from experiment_17 import CHUNK, EXPERIMENT_16, FEATURE_SET, METHODS, REFERENCE_MODEL, Sequences, neural_scores  # noqa: E402
from experiment_16 import LIGHTNING, STATION_VALUES, STATIONS, WEATHER_DAILY, Drivers, breakdown, build, volume_flags  # noqa: E402
from experiment_15 import RULES, DEFINITIONS, Labels, filled, to_grid  # noqa: E402
from experiment_14 import CWFIS, FIRMS, grid, neighbourhood, season_day, shift  # noqa: E402
from experiment_13 import BOOTSTRAP, daily_counts, f1_from_counts  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve  # noqa: E402
from experiment_08 import (BCWS, DETECTIONS, REFERENCE, SEASON_DAYS, SEED, SELECTION_YEAR, Context, catboost_model,  # noqa: E402
                           digest, predict, season_index, sparse_row_max, unit_xyz, window)
from experiment_03 import CELL, NEIGHBOUR, TARGET, slug  # noqa: E402

TEST_YEAR = 2024
EXPERIMENT_18 = EXPERIMENTS / "23_exp_regularised_neural_bc/results"
FEATURES_2024 = HERE / "data/regions"
CWFIS_2024 = DATASETS / "cwfis_hotspots/bc_hotspots_2024.parquet"
WEATHER_2024 = DATASETS / "bcws_weather/station_daily_2024.parquet"
STATIONS_2024 = DATASETS / "bcws_weather/stations_2024.parquet"
LIGHTNING_2024 = DATASETS / "cancpld_lightning/bc_cg_daily_2024.npz"
BCWS_2024 = BCWS.with_name("bcws_incidents_2024.csv")
RESULTS = HERE / "results"
MODELS = ("catboost_spread_noday", "mlp", "gru", "mlp_reg", "gru_reg")
NETWORKS = {"mlp": (experiment_17.Net, False, EXPERIMENT_17 / "models/mlp.pt"),
            "gru": (experiment_17.Net, True, EXPERIMENT_17 / "models/gru.pt"),
            "mlp_reg": (experiment_18.Net, False, EXPERIMENT_18 / "models/mlp_reg.pt"),
            "gru_reg": (experiment_18.Net, True, EXPERIMENT_18 / "models/gru_reg.pt")}
BASELINES = ("rule_pair_yesterday", "catboost_spread_noday")


# ---------------------------------------------------------------- 2024 inputs

def load(region, year, with_cells=False):
    """Experiment 03's load, reading 2024 from this experiment's feature tables."""
    if year != TEST_YEAR:
        return experiment_08.load(region, year, with_cells)
    columns = ["target_date", "history_end_date", TARGET, *[c for c in CELL + NEIGHBOUR if c != "day_of_year"]]
    frame = pd.read_parquet(FEATURES_2024 / slug(region) / f"features_{year}.parquet",
                            columns=columns + (["h3_cell"] if with_cells else []))
    assert frame.target_date.dt.year.eq(year).all()
    assert frame.history_end_date.lt(frame.target_date).all(), "Feature history overlaps target date"
    frame["day_of_year"] = frame.target_date.dt.dayofyear.astype("int16")
    return frame


def load_detections():
    """Experiment 13's detection loader, with 2024."""
    frame = pd.read_csv(DETECTIONS, usecols=["acq_date", "h3_cell", "frp", "daynight", "type"])
    frame["acq_date"] = pd.to_datetime(frame.acq_date)
    frame = frame.loc[frame.acq_date.dt.year.between(2018, TEST_YEAR)].copy()
    assert frame.acq_date.dt.month.between(5, 9).all()
    frame["day"] = season_index(frame.acq_date)
    return frame


def load_bcws():
    """Experiment 13's incident loader, with 2024 (fires burning over from 2023 stay in)."""
    frame = pd.concat([pd.read_csv(BCWS), pd.read_csv(BCWS_2024)], ignore_index=True)
    assert frame.FIRE_YEAR.between(2018, TEST_YEAR).all()
    frame = frame.loc[frame.FIRE_TYPE.eq("Fire") & frame.LATITUDE.notna() & frame.LONGITUDE.notna()].copy()
    for column in ("IGNITION_DATE", "FIRE_OUT_DATE"):
        local = pd.to_datetime(frame[column], utc=True).dt.tz_convert("America/Vancouver")
        frame[column] = local.dt.tz_localize(None).dt.normalize()
    frame = frame.loc[frame.IGNITION_DATE >= "2017-01-01"].copy()  # drops impossible ignition dates
    frame.loc[frame.FIRE_OUT_DATE < frame.IGNITION_DATE, "FIRE_OUT_DATE"] = pd.NaT
    frame["lightning"] = frame.FIRE_CAUSE.map({"Lightning": 1.0, "Person": 0.0}).fillna(0.5)
    return frame.reset_index(drop=True)


experiment_08.load_detections = load_detections  # Context() calls these by name
experiment_08.load_bcws = load_bcws


def load_sources():
    """Experiment 19's S-NPP (FIRMS) and NOAA-20 (CWFIS) sources, with 2024."""
    firms = pd.read_csv(FIRMS, usecols=["acq_date", "h3_cell"], parse_dates=["acq_date"])
    firms = firms.loc[firms.acq_date.dt.year.between(2018, TEST_YEAR)]
    cwfis = pd.concat([pd.read_parquet(p, columns=["rep_date", "satellite", "sensor", "h3_cell"]) for p in (CWFIS, CWFIS_2024)],
                      ignore_index=True)
    cwfis["acq_date"] = cwfis.rep_date.dt.normalize()
    pick = {"snpp_firms": firms, "noaa20": cwfis.loc[cwfis.sensor.eq("VIIRS-I") & cwfis.satellite.eq("NOAA-20")]}
    out = {}
    for name, frame in pick.items():
        frame = frame.assign(day=season_day(frame.acq_date), year=frame.acq_date.dt.year)
        frame = frame.loc[(frame.day >= 0) & (frame.day < SEASON_DAYS)].drop_duplicates(["h3_cell", "day"])
        for year, part in frame.groupby("year"):
            out[name, int(year)] = part[["h3_cell", "day"]]
    return out


class TestLabels(Labels):
    """Experiment 20's label grids, for the given years only (2024 included)."""

    def __init__(self, context, regions, years):
        sources = load_sources()
        self.sources, self.grids = None, {}
        for region in regions:
            cells = context.grids[region]["cells"]
            disk1 = neighbourhood(list(cells))
            for year in years:
                snpp = grid(sources.get(("snpp_firms", year)), cells)
                n20 = grid(sources.get(("noaa20", year)), cells)
                pair = snpp | n20
                label = filled(pair) if year >= 2021 else filled(snpp)
                self.grids[region, year] = {"snpp": snpp, "n20": n20, "pair": pair, "label": label,
                                            "label_full": filled(pair), "disk1": disk1}


class TestDrivers(Drivers):
    """Experiment 21's weather, lightning and spread inputs, plus 2024 station weather and lightning."""

    def __init__(self, context, regions):
        super().__init__(context, regions)
        daily, stations = pd.read_parquet(WEATHER_2024), pd.read_parquet(STATIONS_2024)
        assert daily.year.eq(TEST_YEAR).all() and stations.year.eq(TEST_YEAR).all()
        # The body of Experiment 21's per-year weather loop, for 2024.
        st = stations.loc[stations.latitude.notna() & stations.longitude.notna()]
        st = st.drop_duplicates("station_code", keep="last").reset_index(drop=True)
        index = pd.Index(st.station_code).get_indexer(daily.station_code)
        d, index = daily.loc[index >= 0], index[index >= 0]
        day = season_index(d.date)
        values = {}
        for column in [*STATION_VALUES, "wind_dir_noon"]:
            a = np.full((len(st), SEASON_DAYS), np.nan, np.float32)
            a[index, day] = d[column].to_numpy(np.float32)
            values[column] = a
        values["precip_3d"] = window(np.nan_to_num(values["precip_24h_mm"]), 3)
        has = np.zeros((len(st), SEASON_DAYS), bool)
        has[index, day] = True
        self.weather[TEST_YEAR] = {"tree": cKDTree(unit_xyz(st.latitude.to_numpy(), st.longitude.to_numpy())),
                                   "values": values, "has": has}
        ltg = np.load(LIGHTNING_2024)
        assert [int(y) for y in ltg["years"]] == [TEST_YEAR]
        assert np.array_equal(ltg["lat"], self.ltg_lat) and np.array_equal(ltg["lon"], self.ltg_lon)
        self.ltg = np.concatenate([self.ltg, ltg["daily"]])
        self.ltg_years.append(TEST_YEAR)


# ---------------------------------------------------------------- scoring

def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.results
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    regions = args.regions or sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    source_paths = [FIRMS, CWFIS, CWFIS_2024, WEATHER_DAILY, STATIONS, WEATHER_2024, STATIONS_2024, LIGHTNING,
                    LIGHTNING_2024, BCWS, BCWS_2024, REFERENCE_MODEL, *[path for _, _, path in NETWORKS.values()],
                    *sorted(FEATURES_2024.glob("*/features_2024.parquet"))]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    years = (SELECTION_YEAR, TEST_YEAR)
    context = Context(regions)
    labels = TestLabels(context, regions, years)
    drivers = TestDrivers(context, regions)
    sequences = Sequences(context, labels, drivers)
    gc.collect()

    scaler = experiment_17_scaler()
    n_inputs = len(scaler.mean) + int(scaler.missing.sum())
    models = {"catboost_spread_noday": catboost_model()}
    models["catboost_spread_noday"].load_model(str(REFERENCE_MODEL))
    for name, (net, sequence, path) in NETWORKS.items():
        models[name] = net(n_inputs, sequence=sequence)
        models[name].load_state_dict(torch.load(path))

    rows, thresholds, daily, splits, coverage = [], {}, {}, [], []
    names = (*RULES, *MODELS)
    for region in regions:
        cells = context.grids[region]["cells"]
        for year in years:
            tic = time.time()
            g = labels.grids[region, year]
            coverage.append({"region": region, "year": year, "snpp_cell_days": int(g["snpp"].sum()),
                             "noaa20_cell_days": int(g["n20"].sum()), "label_cell_days": int(g["label_full"].sum())})
            frame = load(region, year, with_cells=True)
            n = len(frame)
            cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
            scores = {name: np.empty(n, np.float32) for name in names}
            for start in range(0, n, CHUNK):
                chunk = build(context, labels, drivers, region, year, frame.iloc[start:start + CHUNK].reset_index(drop=True))
                cube = sequences.cube(region, year)
                part = slice(start, start + len(chunk))
                scores["rule_snpp_yesterday"][part] = (chunk.local_count_1d > 0).to_numpy(np.float32)
                scores["rule_pair_yesterday"][part] = chunk.pair_yesterday.to_numpy(np.float32)
                for name in MODELS:
                    if name in NETWORKS:
                        scores[name][part] = neural_scores(models[name], scaler, chunk, cube, cells)
                    else:
                        scores[name][part] = predict(models[name], chunk, FEATURE_SET)
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
                        if year == TEST_YEAR:
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
            if year == SELECTION_YEAR:  # thresholds come from 2022, so 2022 must match the earlier experiments
                metrics = pd.DataFrame(rows)
                for source, loaded in ((EXPERIMENT_16, [*RULES, "catboost_spread_noday"]), (EXPERIMENT_17, ["mlp", "gru"])):
                    check = metrics.loc[metrics.region.eq(region) & metrics.year.eq(year) & metrics.model.isin(loaded)].merge(
                        pd.read_csv(source / "metrics_by_year.csv"), on=["region", "year", "definition", "model", "method"],
                        suffixes=("", "_old"))
                    expected = len(DEFINITIONS) * sum(1 if m in RULES else len(METHODS) for m in loaded)
                    assert len(check) == expected and np.allclose(check.f1, check.f1_old, atol=1e-9), \
                        f"{region} 2022 does not reproduce {source.parent.name}"
            del scores
            gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "label_coverage.csv", index=False)
    test = metrics.loc[metrics.year.eq(TEST_YEAR)]
    summary = test.groupby(["definition", "method", "model"])[["f1", "precision", "recall", "average_precision"]].median().reset_index()
    pooled = {key[1:]: np.sum([daily[(r, *key[1:])] for r in regions], axis=0) for key in daily if key[0] == regions[0]}
    total = lambda d, k, m: pooled[d, m, k].sum(axis=0)
    summary["bc_wide_f1"] = [float(f1_from_counts(total(*key))) for key in zip(summary.definition, summary.method, summary.model)]
    summary["bc_wide_precision"] = [total(*key)[1] / total(*key)[0] for key in zip(summary.definition, summary.method, summary.model)]
    summary["bc_wide_recall"] = [total(*key)[3] / total(*key)[2] for key in zip(summary.definition, summary.method, summary.model)]
    summary.to_csv(RESULTS / f"summary_{TEST_YEAR}.csv", index=False)
    pd.DataFrame(splits).to_csv(RESULTS / f"new_vs_continuing_{TEST_YEAR}.csv", index=False)

    dates = pd.date_range(f"{TEST_YEAR}-05-01", periods=SEASON_DAYS)[FIRST_TARGET_DAY:]
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
            for baseline in BASELINES:
                for model in MODELS:
                    for method in METHODS:
                        if model == baseline:
                            continue
                        key = (definition, model, method)
                        base = (definition, baseline, "f1_2022" if baseline in RULES else method)
                        difference = boot(get(key)) - boot(get(base))
                        boot_rows.append({"scope": scope, "definition": definition, "model": model, "method": method,
                                          "baseline": baseline,
                                          "difference": float(f1_from_counts(get(key).sum(0)) - f1_from_counts(get(base).sum(0))),
                                          "ci_low": float(np.quantile(difference, .025)), "ci_high": float(np.quantile(difference, .975)),
                                          "share_above": float((difference > 0).mean())})
    pd.DataFrame(boot_rows).to_csv(RESULTS / f"bootstrap_{TEST_YEAR}.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": [2018, 2019, 2020, 2021],
        "selection_year": SELECTION_YEAR, "test_year": TEST_YEAR, "development_year": 2023,
        "label": "S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell, plus one-day gap filling; NOAA-21 not used",
        "models": {name: str(path.relative_to(EXPERIMENTS)) for name, (_, _, path) in NETWORKS.items()}
                  | {"catboost_spread_noday": str(REFERENCE_MODEL.relative_to(EXPERIMENTS))},
        "methods": {"f1_2022": "threshold maximizing 2022 F1 per Fire Centre, definition and model, applied unchanged to 2024",
                    "rule_volume": "each day, flag as many cells per Fire Centre as the two-satellite rule"},
        "checks": "2022 reproduces Experiments 21 (rules, CatBoost) and 17 (MLP, GRU) on every Fire Centre",
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of {TEST_YEAR}, against {', '.join(BASELINES)}",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
