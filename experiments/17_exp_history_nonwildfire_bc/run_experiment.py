"""Longer fire memory and a non-wildfire flag (next-day forecasts).

Experiment 16 found that "quiet" onsets are mostly (a) detections with no BC Wildfire
Service fire on record, recurring at the same places (likely agricultural, slash-pile and
industrial burning), and (b) older wildfires flaring up beyond the 7-day, one-ring history
window. This experiment adds:
- history features: 30-day and season-to-date activity around the cell, prior-year
  activity, and how much of it was linked to a BCWS wildfire;
- a wildfire-linked target: a detection counts only if it falls inside (or within 2 km of)
  the final perimeter of a BCWS wildfire active around that day, or within 10 km of an
  active BCWS wildfire's ignition point. Final perimeters are after-the-fact information,
  so they label past detections and are never forecast inputs; current-season features
  use only the ignition-point link with incidents reported by then.
CatBoost (Experiments 05-06 settings) on Experiment 13's features with and without the
history group, trained on the recorded target and on the wildfire-linked target. Scored on
the exact res-7 cell and with Experiment 14's one-cell margin; thresholds maximize 2022 F1
per Fire Centre and apply unchanged to 2023. 2024 is never loaded.
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
import shapely
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_09", EXPERIMENTS / "14_exp_target_definitions_bc/run_experiment.py")
experiment_09 = sys.modules["experiment_09"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_09)
import experiment_08  # noqa: E402  (registered by experiment_09)
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve, neighbourhood, to_grid  # noqa: E402
from experiment_08 import (BCWS, Context, DETECTIONS, FEATURE_SETS, MAX_ABS_LOG_ODDS, QUIET_SAMPLE_RATE,  # noqa: E402
                           REFERENCE, SEASON_DAYS, SEED, SELECTION_YEAR, EVALUATION_YEAR, TARGET, TRAIN_YEARS,
                           catboost_model, design, digest, load, load_bcws, predict, quiet_stratum, raw_scores,
                           season_index, unit_xyz, window)

RESULTS = HERE / "results"
ONSETS = EXPERIMENTS / "16_exp_new_fire_onsets_bc/results"
EXPERIMENT_08_MODEL = EXPERIMENTS / "13_exp_fire_context_bc/results/models/catboost__all.cbm"
PERIMETERS = HERE / "data/bcws_perimeters_2018_2023.geojson"
LINK_KM, PERIMETER_MARGIN_KM, WINDOW_DAYS = 10.0, 2.0, 3
MISSING_OUT_ACTIVE_DAYS, MAX_ACTIVE_DAYS = 60, 180
WILDFIRE = "target_wildfire"
HISTORY = ["local_count_30d", "disk1_count_30d", "active_cells_disk5_30d", "dist_nearest_30d", "disk1_season_days",
           "recent_wildfire_days_disk1", "recent_nonwildfire_days_disk1", "prior_years_available",
           "prior_years_detected_cell", "prior_years_detected_disk1", "prior_wildfire_days_disk1",
           "prior_nonwildfire_days_disk1"]
FEATURE_SETS["all_history"] = FEATURE_SETS["all"] + HISTORY
MODELS = {"catboost_08": ("all", TARGET), "catboost_history": ("all_history", TARGET),
          "catboost_history_wildfire": ("all_history", WILDFIRE)}
TARGETS = {"recorded": TARGET, "wildfire": WILDFIRE}


def active_window(ignition, out):
    effective_out = np.where(pd.isna(out), ignition + np.timedelta64(MISSING_OUT_ACTIVE_DAYS, "D"), out)
    return ignition, np.minimum(effective_out, ignition + np.timedelta64(MAX_ACTIVE_DAYS, "D"))


def local_km(lon, lat):
    """Equirectangular km around 54 N; adequate for a 2 km margin."""
    return np.column_stack([np.asarray(lon) * 111.32 * np.cos(np.deg2rad(54.0)), np.asarray(lat) * 110.57])


def load_perimeters(incidents):
    data = json.loads(PERIMETERS.read_text())
    frame = pd.DataFrame([f["properties"] for f in data["features"]])
    assert frame.FIRE_YEAR.between(2018, 2023).all()
    geometry = shapely.from_geojson([json.dumps(f["geometry"]) for f in data["features"]])
    geometry = shapely.transform(geometry, lambda xy: local_km(xy[:, 0], xy[:, 1]))
    dates = incidents[["FIRE_YEAR", "FIRE_NUMBER", "IGNITION_DATE", "FIRE_OUT_DATE"]].drop_duplicates(["FIRE_YEAR", "FIRE_NUMBER"])
    frame = frame.merge(dates, on=["FIRE_YEAR", "FIRE_NUMBER"], how="left")
    fallback = pd.to_datetime(frame.FIRE_DATE.str.rstrip("Z"), errors="coerce")
    frame["IGNITION_DATE"] = frame.IGNITION_DATE.fillna(fallback).fillna(pd.to_datetime(frame.FIRE_YEAR.astype(str) + "-05-01"))
    return frame, geometry


def linked_detections(context, incidents, perimeters):
    """Per year, U x 153 booleans for detected cell-days.

    label: inside (or within PERIMETER_MARGIN_KM of) the final perimeter of a fire active within
    WINDOW_DAYS of that day, or within LINK_KM of an active fire's ignition point (after the fact).
    known: the ignition-point link only (no final perimeters), for current-season features."""
    lat_lon = np.array([h3.cell_to_latlng(c) for c in context.universe])
    cells = unit_xyz(lat_lon[:, 0], lat_lon[:, 1])
    cells_km = shapely.points(local_km(lat_lon[:, 1], lat_lon[:, 0]))
    points = unit_xyz(incidents.LATITUDE.to_numpy(), incidents.LONGITUDE.to_numpy())
    ignition, effective_out = active_window(incidents.IGNITION_DATE.to_numpy(), incidents.FIRE_OUT_DATE.to_numpy())
    table, shapes = perimeters
    p_start, p_end = active_window(table.IGNITION_DATE.to_numpy(), table.FIRE_OUT_DATE.to_numpy())
    tree = shapely.STRtree(shapes)
    day = np.timedelta64(1, "D")
    label, known = {}, {}
    for year, arrays in context.arrays.items():
        detected = arrays["count"] > 0
        point_flag = np.zeros(detected.shape, bool)
        perimeter_flag = np.zeros(detected.shape, bool)
        for index in range(SEASON_DAYS):
            date = np.datetime64(f"{year}-05-01") + index * day
            today = np.flatnonzero(detected[:, index])
            if not len(today):
                continue
            active = (ignition <= date + day) & (effective_out >= date - day)
            if active.any():
                distance, _ = cKDTree(points[active]).query(cells[today], k=1, distance_upper_bound=LINK_KM)
                point_flag[today[np.isfinite(distance)], index] = True
            hits = tree.query(cells_km[today], predicate="dwithin", distance=PERIMETER_MARGIN_KM)
            if hits.shape[1]:
                cell_hit, shape_hit = hits
                window = (p_start[shape_hit] <= date + WINDOW_DAYS * day) & (p_end[shape_hit] >= date - WINDOW_DAYS * day)
                perimeter_flag[today[cell_hit[window]], index] = True
        label[year], known[year] = point_flag | perimeter_flag, point_flag
        print(f"  {year}: {label[year].sum() / max(detected.sum(), 1):.0%} of detected cell-days are wildfire-linked "
              f"({point_flag.sum() / max(detected.sum(), 1):.0%} by ignition point alone)", flush=True)
    return label, known


def shifted(values, days):
    """values[:, e - days], zero where that is before May 1."""
    out = np.zeros_like(values)
    out[:, days:] = values[:, :SEASON_DAYS - days]
    return out


def add_history(context, links, region, year, frame):
    """Adds HISTORY columns (past-only) and the wildfire-linked target to Experiment 13's frame."""
    g, arrays = context.grids[region], context.arrays[year]
    rings, disk1 = g["rings"], g["disk1"]
    label, known = links
    count = arrays["count"]
    indicator = (count > 0).astype(np.float32)
    wild = known[year].astype(np.float32)
    other = ((count > 0) & ~known[year]).astype(np.float32)
    ind30 = (window(indicator, 30) > 0).astype(np.float32)
    daily = {"local_count_30d": window(rings[0] @ count, 30), "disk1_count_30d": window(disk1 @ count, 30),
             "active_cells_disk5_30d": sum(r @ ind30 for r in rings),
             "disk1_season_days": window(((disk1 @ indicator) > 0).astype(np.float32), SEASON_DAYS),
             # Days e-29 .. e-2 only: a BCWS link for a detection 2+ days old uses incidents reported by e-1.
             "recent_wildfire_days_disk1": shifted(window(((disk1 @ wild) > 0).astype(np.float32), 28), 2),
             "recent_nonwildfire_days_disk1": shifted(window(((disk1 @ other) > 0).astype(np.float32), 28), 2)}
    distance = np.full((len(g["cells"]), SEASON_DAYS), experiment_08.MAX_RING + 1, np.float32)
    for k in range(experiment_08.MAX_RING, -1, -1):
        distance[(rings[k] @ ind30) > 0] = k
    daily["dist_nearest_30d"] = distance
    earlier = [y for y in context.arrays if y < year]
    per_cell = {"prior_years_available": np.full(len(g["cells"]), len(earlier), np.float32)}
    for name in ("prior_years_detected_cell", "prior_years_detected_disk1", "prior_wildfire_days_disk1",
                 "prior_nonwildfire_days_disk1"):
        per_cell[name] = np.zeros(len(g["cells"]), np.float32)
    for y in earlier:  # complete past seasons: their BCWS records are final
        detected_days = (context.arrays[y]["count"] > 0).sum(axis=1).astype(np.float32)
        wild_days = label[y].sum(axis=1).astype(np.float32)
        per_cell["prior_years_detected_cell"] += (rings[0] @ detected_days) > 0
        per_cell["prior_years_detected_disk1"] += (disk1 @ detected_days) > 0
        per_cell["prior_wildfire_days_disk1"] += disk1 @ wild_days
        per_cell["prior_nonwildfire_days_disk1"] += disk1 @ (detected_days - wild_days)
    cell = g["cells"].get_indexer(frame.h3_cell)
    history, target = season_index(frame.history_end_date), season_index(frame.target_date)
    for name, values in daily.items():
        frame[name] = values[cell, history]
    for name, values in per_cell.items():
        frame[name] = values[cell]
    frame[WILDFIRE] = (frame[TARGET].to_numpy() == 1) & ((rings[0] @ label[year].astype(np.float32))[cell, target] > 0)
    frame[WILDFIRE] = frame[WILDFIRE].astype(np.int8)
    return frame


def training_sample(regions, context, links):
    parts, stats = [], []
    for index, region in enumerate(regions):
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            positive = frame[TARGET].eq(1).to_numpy()
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            keep = ~quiet_stratum(frame) | positive | drawn  # Experiment 13's draw
            part = context.features(region, year, frame.loc[keep].reset_index(drop=True))
            part = add_history(context, links, region, year, part)
            part = part[FEATURE_SETS["all_history"] + [TARGET, WILDFIRE]].assign(drawn=drawn[keep], year=year, region=region)
            parts.append(part)
            stats.append({"region": region, "year": year, "rows": len(frame), "kept_rows": int(keep.sum()),
                          "recorded_positives": int(positive.sum()), "wildfire_positives": int(part[WILDFIRE].sum())})
            print(f"  {region} {year}: kept {keep.sum():,}; wildfire-linked {part[WILDFIRE].sum():,} of {positive.sum():,} positives", flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def onset_categories():
    cells = pd.read_csv(ONSETS / "onset_cells.csv", parse_dates=["date"])
    clusters = pd.read_csv(ONSETS / "onset_clusters.csv")[["cluster", "bcws_category"]]
    return cells.merge(clusters, on="cluster")[["h3_cell", "date", "bcws_category"]].rename(columns={"date": "target_date"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-models", action="store_true", help="load results/models/*.cbm instead of refitting")
    args = parser.parse_args()
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    source_paths = [DETECTIONS, BCWS, PERIMETERS, EXPERIMENT_08_MODEL, ONSETS / "onset_cells.csv", ONSETS / "onset_clusters.csv"]
    hashes = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    context = Context(regions)
    print("Linking detections to BCWS wildfires", flush=True)
    incidents = load_bcws()
    links = linked_detections(context, incidents, load_perimeters(incidents))
    label_link = links[0]
    disk7 = {r: neighbourhood(list(context.grids[r]["cells"]), list(context.grids[r]["cells"])) for r in regions}

    models, fit_checks = {}, []
    loaded = catboost_model()
    loaded.load_model(str(EXPERIMENT_08_MODEL))
    models["catboost_08"] = loaded
    saved = {name: RESULTS / "models" / f"{name}.cbm" for name in MODELS if name != "catboost_08"}
    if args.reuse_models and all(path.exists() for path in saved.values()):
        for name, path in saved.items():
            models[name] = catboost_model()
            models[name].load_model(str(path))
        print("Reusing saved models", flush=True)
    else:
        print("Building training sample (2018-2021)", flush=True)
        sample, sample_stats = training_sample(regions, context, links)
        sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
        quiet = quiet_stratum(sample)
    for name, (feature_set, target_name) in MODELS.items():
        if name in models:
            continue
        y = sample[target_name].to_numpy()
        # Exact sampling correction: every row this target calls positive must be kept.
        use = ~quiet | (y == 1) | sample.drawn.to_numpy()
        X, _ = design(sample.loc[use], feature_set)
        model = catboost_model()
        tic = time.time()
        model.fit(X, y[use])
        largest = float(np.abs(raw_scores(model, X)).max())
        assert largest < MAX_ABS_LOG_ODDS, f"{name} diverged"
        fit_checks.append({"model": name, "feature_set": feature_set, "target": target_name, "rows": int(use.sum()),
                           "positives": int(y[use].sum()), "max_abs_training_log_odds": largest,
                           "fit_seconds": round(time.time() - tic, 1)})
        importance = pd.DataFrame({"feature": FEATURE_SETS[feature_set] + ["quiet_stratum"],
                                   "importance": model.get_feature_importance()})
        importance.to_csv(RESULTS / f"feature_importance__{name}.csv", index=False)
        model.save_model(str(RESULTS / "models" / f"{name}.cbm"))
        models[name] = model
        print(f"Fitted {name}: {use.sum():,} rows, {int(y[use].sum()):,} positives, {time.time()-tic:.0f}s", flush=True)
    if fit_checks:
        pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
        del sample

    onsets = onset_categories()
    rows, thresholds, catches = [], {}, []
    for region in regions:
        g, cells = context.grids[region], context.grids[region]["cells"]
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            frame = context.features(region, year, load(region, year, with_cells=True))
            frame = add_history(context, links, region, year, frame)
            scores = {"rule_yesterday": (frame.local_count_1d > 0).to_numpy(np.float32),
                      "rule_yesterday_bcws": ((frame.local_count_1d > 0) & (frame.bcws_dist_active_km <= LINK_KM)).to_numpy(np.float32)}
            for name, model in models.items():
                scores[name] = predict(model, frame, MODELS[name][0])
            near = {"recorded": (g["disk1"] @ context.arrays[year]["count"]) > 0,
                    "wildfire": (g["disk1"] @ label_link[year].astype(np.float32)) > 0}
            grids = {name: to_grid(frame, cells, s) for name, s in scores.items()}
            for target_key, column in TARGETS.items():
                truth = to_grid(frame, cells, frame[column].to_numpy()) > 0
                hit = near[target_key][:, FIRST_TARGET_DAY:]
                assert (hit >= truth).all()
                for name, grid in grids.items():
                    neighbourhood_max = experiment_09.sparse_row_max(disk7[region], grid)
                    for definition in ("res7_exact", "res7_margin1"):
                        inputs = ((grid.ravel(), truth.ravel(), truth.ravel(), grid.ravel()) if definition == "res7_exact"
                                  else (grid.ravel(), truth.ravel(), hit.ravel(), neighbourhood_max.ravel()))
                        tag = (region, target_key, definition, name)
                        if year == SELECTION_YEAR:
                            candidates = candidate_thresholds(inputs[0])
                            thresholds[tag] = float(candidates[int(np.argmax(curve(*inputs, candidates)[2]))])
                        precision, recall, f1, flagged = curve(*inputs, np.array([thresholds[tag]]))
                        row = {"region": region, "year": year, "target": target_key, "definition": definition, "model": name,
                               "threshold": thresholds[tag], "precision": precision[0], "recall": recall[0], "f1": f1[0],
                               "flagged": int(flagged[0]), "positives": int(truth.sum())}
                        if definition == "res7_exact" and not name.startswith("rule"):
                            row["average_precision"] = float(average_precision_score(truth.ravel(), grid.ravel()))
                        rows.append(row)
            if year == EVALUATION_YEAR:
                # Quiet fires caught, by Experiment 16 onset group (recorded target, exact cell).
                quiet_positive = quiet_stratum(frame) & (frame[TARGET].to_numpy() == 1)
                part = frame.loc[quiet_positive, ["h3_cell", "target_date"]].merge(onsets, how="left", on=["h3_cell", "target_date"])
                part["bcws_category"] = part.bcws_category.fillna("not an Experiment 16 onset")
                for name, s in scores.items():
                    flags = s[quiet_positive] >= thresholds[(region, "recorded", "res7_exact", name)]
                    for category, index in part.groupby("bcws_category").groups.items():
                        catches.append({"region": region, "model": name, "onset_group": category,
                                        "quiet_fires": len(index), "caught": int(flags[index].sum())})
            latest = [r for r in rows if r["region"] == region and r["year"] == year and r["definition"] == "res7_margin1"]
            print(f"{region} {year} (margin): " + "; ".join(f"{r['target'][:4]}/{r['model']} {r['f1']:.3f}" for r in latest), flush=True)
            del frame, grids

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.DataFrame(catches).to_csv(RESULTS / "quiet_catch_2023.csv", index=False)
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["target", "definition", "model"])[["f1", "precision", "recall", "average_precision"]].median()
    summary.reset_index().to_csv(RESULTS / "summary_2023.csv", index=False)
    assert hashes == {str(p.relative_to(EXPERIMENTS)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False, "history_features": HISTORY,
        "history_notes": "30-day windows are truncated at May 1 (detections are May-September only); prior-year "
                         "features count complete earlier seasons from 2018, so 2018 rows have none "
                         "(prior_years_available tells the model); recent wildfire/non-wildfire days use detections "
                         "2-29 days before the history end so every BCWS link uses incidents reported by then",
        "wildfire_link": {
            "label": f"inside or within {PERIMETER_MARGIN_KM:.0f} km of the final perimeter of a BCWS fire active within "
                     f"{WINDOW_DAYS} days of the date, or within {LINK_KM:.0f} km of an active BCWS 'Fire' ignition point "
                     f"(ignition at most 1 day later, not out more than 1 day earlier); after-the-fact, labels and "
                     f"prior-year features only",
            "known": "ignition-point link only, for current-season features",
            "active": f"missing out date: active {MISSING_OUT_ACTIVE_DAYS} days; capped at {MAX_ACTIVE_DAYS} days"},
        "models": {k: {"feature_set": v[0], "target": v[1]} for k, v in MODELS.items()},
        "rules": {"rule_yesterday": "detection in the cell yesterday",
                  "rule_yesterday_bcws": "detection in the cell yesterday and a known active BCWS fire within 10 km"},
        "thresholds": "maximize 2022 F1 per Fire Centre, target, definition and model; applied unchanged to 2023",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
