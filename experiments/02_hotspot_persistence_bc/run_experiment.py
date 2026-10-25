"""Repeat the Kamloops workflow independently for all six BC Fire Centres.

Only execution/storage is batched; dates, target, windows, rules and selection
match Experiment 01. This script deliberately does not create an HTML report.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.sparse import csr_matrix
import shapely
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import unary_union
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = HERE.parent / "01_hotspot_persistence_kamloops"
sys.path.insert(0, str(REFERENCE))
from run_persistence import preceding_sum, history_age, check_temporal_logic
from prepare_daily_grid import polygon_parts

INPUT = ROOT / "experiments/firms_summer_2018_2024/data/firms_h3_detections.csv"
BOUNDARY = ROOT / "data/region_exploration/fire_centres.geojson"
YEARS = range(2018, 2024)
RULES = ["always_no_detection", "local_1d", "local_or_neighbour_1d",
         "local_3d", "local_or_neighbour_3d", "local_7d", "local_or_neighbour_7d"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def histogram_add(histogram, y, scores):
    # Exact grouped counts avoid sorting/retaining millions of zero-score rows.
    for flag in (0, 1):
        values, counts = np.unique(scores[y == flag], return_counts=True)
        for value, count in zip(values, counts):
            histogram.setdefault(int(value), [0, 0])[flag] += int(count)


def histogram_metrics(histogram):
    keys = sorted(histogram, reverse=True)
    negative = np.array([histogram[k][0] for k in keys], dtype=np.int64)
    positive = np.array([histogram[k][1] for k in keys], dtype=np.int64)
    total_positive, rows = int(positive.sum()), int(positive.sum() + negative.sum())
    predicted = np.array(keys) > 0
    tp, fp = int(positive[predicted].sum()), int(negative[predicted].sum())
    fn, tn = total_positive-tp, int(negative.sum())-fp
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    ap = float(np.sum(positive/total_positive * (positive.cumsum() / (positive+negative).cumsum()))) if total_positive else float("nan")
    prevalence = total_positive/rows
    return {"rows": rows, "positive_cell_days": total_positive, "prevalence": prevalence,
            "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "average_precision": ap, "ap_over_prevalence": ap/prevalence if prevalence else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "predicted_positive_cell_days": tp+fp}


def check_metrics():
    check_temporal_logic()
    rng = np.random.default_rng(42)
    y, score = rng.integers(0, 2, 1000), rng.integers(0, 10, 1000)
    hist = {}
    for start in range(0, len(y), 137):
        histogram_add(hist, y[start:start+137], score[start:start+137])
    result = histogram_metrics(hist)
    assert np.isclose(result["average_precision"], average_precision_score(y, score))
    assert result["tp"] == int(((score > 0) & (y > 0)).sum())
    constant = {}
    histogram_add(constant, y, np.zeros(len(y), dtype=int))
    assert np.isclose(histogram_metrics(constant)["average_precision"], y.mean())


def write_batch(writer, path, frame):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


def build_grid(region, output):
    candidates = set()
    for polygon in polygon_parts(region):
        candidates.update(h3.h3shape_to_cells_experimental(
            h3.geo_to_h3shape(mapping(polygon)), 7, contain="bbox_overlap"))
    polygons = {}
    for cell in sorted(candidates):
        polygon = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
        if region.intersects(polygon):
            polygons[cell] = polygon
    uncovered = region.difference(unary_union(list(polygons.values()))).area
    assert uncovered < 1e-9, f"Uncovered region: {uncovered} deg2"
    grid = pd.DataFrame([
        {"h3_cell": cell, "resolution": 7, "center_latitude": h3.cell_to_latlng(cell)[0],
         "center_longitude": h3.cell_to_latlng(cell)[1], "is_border_cell": not region.covers(polygon),
         "geometry_wkt": polygon.wkt} for cell, polygon in polygons.items()])
    grid.to_parquet(output / "h3_grid.parquet", index=False)
    return grid, polygons, uncovered


def draw_outline(ax, region):
    for polygon in polygon_parts(region):
        ax.plot(*polygon.exterior.xy, color="gray", linewidth=.6, zorder=2)
        for ring in polygon.interiors:
            ax.plot(*ring.xy, color="gray", linewidth=.5, zorder=2)
    ax.set_aspect(1/np.cos(np.deg2rad(region.centroid.y)))
    ax.grid(linestyle=":", linewidth=.5, alpha=.4)
    ax.set_axisbelow(True)
    ax.set(xlabel="Longitude", ylabel="Latitude")


def make_figures(name, region, polygons, active, samples, metrics, selected_daily, output):
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.add_collection(PolyCollection([list(p.exterior.coords) for p in polygons.values()],
                                     facecolors="none", edgecolors="#3268ad", linewidths=.12))
    draw_outline(ax, region)
    ax.autoscale()
    ax.set_title(f"{name}: fixed resolution-7 grid ({len(polygons):,} cells)")
    fig.savefig(output / "fixed_h3_grid.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    for ax, year in zip(axes.flat, YEARS):
        dates = pd.date_range(f"{year}-05-01", f"{year}-09-30")
        totals = active.loc[active.acq_date.dt.year.eq(year)].groupby("acq_date").detection_count.sum()
        ax.plot(dates, totals.reindex(dates, fill_value=0), color="#ca601c", linewidth=.8)
        ax.set(title=str(year), ylabel="Recorded detections")
        ax.grid(linestyle=":", alpha=.4)
    fig.suptitle(f"{name}: daily observed activity (panel scales vary)")
    fig.savefig(output / "regional_daily_counts.png", dpi=140)
    plt.close(fig)
    sample_ids = samples.h3_cell.unique()
    fig, axes = plt.subplots(len(sample_ids), 1, figsize=(14, 2*len(sample_ids)), squeeze=False,
                             constrained_layout=True)
    for ax, cell in zip(axes.flat, sample_ids):
        for year in YEARS:
            subset = samples.loc[samples.h3_cell.eq(cell) & samples.acq_date.dt.year.eq(year)]
            ax.plot(subset.acq_date, subset.detection_count, color="#ca601c", linewidth=.8)
        ax.set(title=cell, ylabel="Detections/day")
        ax.grid(linestyle=":", alpha=.4)
    fig.suptitle(f"{name}: sample cells selected from 2018; off-season gaps omitted")
    fig.savefig(output / "sample_cell_timeseries.png", dpi=140)
    plt.close(fig)

    import calendar
    for year in (2018, 2023):
        annual = active.loc[active.acq_date.dt.year.eq(year)]
        august = annual.loc[annual.acq_date.dt.month.eq(8)]
        groups_by_kind = {
            "monthly": [(calendar.month_name[m], annual.loc[annual.acq_date.dt.month.eq(m)].groupby("h3_cell").detection_count.sum()) for m in (6, 7, 8, 9)],
            "weekly": [(f"August {lo}-{hi}", august.loc[august.acq_date.dt.day.between(lo, hi)].groupby("h3_cell").detection_count.sum()) for lo, hi in ((1, 7), (8, 14), (15, 21), (22, 31))]}
        for kind, groups in groups_by_kind.items():
            maximum = max((int(counts.max()) for _, counts in groups if len(counts)), default=1)
            norm = LogNorm(1, max(2, maximum))
            fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True, constrained_layout=True)
            for ax, (label, counts) in zip(axes.flat, groups):
                draw_outline(ax, region)
                if len(counts):
                    ax.add_collection(PolyCollection([list(polygons[c].exterior.coords) for c in counts.index],
                                                     array=counts.to_numpy(dtype=float), norm=norm,
                                                     cmap="Oranges", edgecolors="none"))
                left, bottom, right, top = region.bounds
                ax.set(xlim=(left-.08, right+.08), ylim=(bottom-.08, top+.08), title=f"{label} ({counts.sum():,})")
            fig.colorbar(ScalarMappable(norm=norm, cmap="Oranges"), ax=axes.ravel().tolist(),
                         shrink=.8, label="Recorded detections per H3 cell (log scale)")
            note = "; August 22-31 covers 10 days" if kind == "weekly" else ""
            fig.suptitle(f"{name}, {year}: {kind} observed activity\nShared scale within figure{note}")
            fig.savefig(output / f"{kind}_{year}.png", dpi=140)
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        metrics.loc[metrics.year.isin([2022, 2023])].pivot(index="rule", columns="year", values=metric).plot.bar(ax=ax)
        ax.set(title=metric.title(), xlabel="", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=70, labelsize=8)
    fig.suptitle(f"{name}: fixed persistence rules")
    fig.savefig(output / "persistence_metrics.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    for key in ("observed_positive_cells", "predicted_positive_cells", "true_positive_cells"):
        ax.plot(selected_daily.date, selected_daily[key], label=key.replace("_", " "), linewidth=1)
    ax.set(title=f"{name}: selected rule, 2023 development evaluation", ylabel="Cells/day")
    ax.legend()
    ax.grid(linestyle=":", alpha=.4)
    fig.savefig(output / "persistence_daily_2023.png", dpi=140)
    plt.close(fig)


def run_region(name, region, observations, batch_size, source_hashes):
    slug = name.lower().replace(" ", "_")
    output = HERE / "regions" / slug
    data, results, figures = output / "data", output / "results", output / "figures"
    for folder in (data / "features", results, figures):
        folder.mkdir(parents=True, exist_ok=True)
    print(f"\n{name}: building boundary-derived grid", flush=True)
    grid, polygons, uncovered = build_grid(region, data)
    cells = grid.h3_cell.tolist()
    position = {cell: index for index, cell in enumerate(cells)}
    assert set(observations.h3_cell).issubset(position)
    active = observations.groupby(["acq_date", "h3_cell"]).size().rename("detection_count").reset_index()
    active.to_parquet(data / "active_cell_days.parquet", index=False)
    rows, columns = [], []
    for cell, index in position.items():
        for neighbour in sorted(h3.grid_disk(cell, 1)):
            if neighbour != cell and neighbour in position:
                rows.append(index)
                columns.append(position[neighbour])
    adjacency = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, columns)), shape=(len(cells), len(cells)))
    assert adjacency.diagonal().sum() == 0 and (adjacency != adjacency.T).nnz == 0
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype("int8")
    pd.DataFrame({"h3_cell": cells, "in_region_neighbours": degrees}).to_csv(results / "neighbour_counts.csv", index=False)
    first = active.loc[active.acq_date.dt.year.eq(2018)].groupby("h3_cell").detection_count.sum()
    active_sample = first.sort_values(ascending=False, kind="stable").head(3).index.tolist()
    quiet = sorted(set(cells)-set(first.index))
    sample_ids = active_sample + np.random.default_rng(42).choice(quiet, min(3, len(quiet)), replace=False).tolist()
    metrics_rows, checks, monthly_rows, sample_frames, feature_samples = [], [], [], [], []
    selected = None
    selected_daily = None
    for year in YEARS:
        dates = pd.date_range(f"{year}-05-01", f"{year}-09-30")
        subset = active.loc[active.acq_date.dt.year.eq(year)]
        counts = np.zeros((153, len(cells)), dtype=np.int32)
        day_indices = (subset.acq_date-dates[0]).dt.days.to_numpy()
        cell_indices = subset.h3_cell.map(position).to_numpy(dtype=int)
        counts[day_indices, cell_indices] = subset.detection_count.to_numpy()
        assert int(counts.sum()) == int(observations.acq_date.dt.year.eq(year).sum())
        raw = observations.loc[observations.acq_date.dt.year.eq(year)].groupby("acq_date").size().reindex(dates, fill_value=0)
        np.testing.assert_array_equal(counts.sum(axis=1), raw.to_numpy())
        neighbour_daily = (adjacency @ counts.T).T
        for index in (0, len(cells)//2, len(cells)-1):
            np.testing.assert_array_equal(neighbour_daily[:, index], counts[:, adjacency.getrow(index).indices].sum(axis=1))
        histograms = {rule: {} for rule in RULES}
        daily_writer = feature_writer = prediction_writer = None
        selected_totals = np.zeros((146, 3), dtype=np.int64)
        try:
            for start in range(0, len(cells), batch_size):
                stop = min(start+batch_size, len(cells))
                batch_cells = cells[start:stop]
                local, neighbours = counts[:, start:stop], neighbour_daily[:, start:stop]
                size = stop-start
                daily = pd.DataFrame({"acq_date": np.repeat(dates, size), "h3_cell": np.tile(batch_cells, 153),
                                      "detection_count": local.ravel()})
                daily["has_recorded_detection"] = daily.detection_count.gt(0)
                daily["observation_coverage"] = pd.Categorical.from_codes(np.zeros(len(daily), dtype=np.int8), ["unknown"])
                daily["detection_status"] = pd.Categorical.from_codes(daily.has_recorded_detection.to_numpy(dtype=np.int8), ["no_recorded_detection", "recorded_detection"])
                daily_writer = write_batch(daily_writer, data / f"cell_date_{year}.parquet", daily)
                sample_frames.append(daily.loc[daily.h3_cell.isin(sample_ids)].copy())
                features = pd.DataFrame({"target_date": np.repeat(dates[7:], size), "h3_cell": np.tile(batch_cells, 146)})
                features["history_end_date"] = features.target_date-pd.Timedelta(days=1)
                for window in (1, 3, 7):
                    features[f"local_count_{window}d"] = preceding_sum(local, window)[7:].ravel().astype("int32")
                    features[f"neighbour_count_{window}d"] = preceding_sum(neighbours, window)[7:].ravel().astype("int32")
                features["days_since_previous_detection"] = history_age(local)[7:].ravel()
                features["history_left_censored"] = features.days_since_previous_detection.isna()
                features["in_region_neighbours"] = np.tile(degrees[start:stop], 146)
                features["target_recorded_detection"] = (local[7:].ravel()>0).astype("int8")
                features["observation_coverage"] = pd.Categorical.from_codes(np.zeros(len(features), dtype=np.int8), ["unknown"])
                features["coverage_supported_target"] = pd.array(np.full(len(features), np.nan), dtype="Int8")
                assert features.history_end_date.lt(features.target_date).all()
                assert features.coverage_supported_target.isna().all()
                assert len(features) == size*146 and len(daily) == size*153
                feature_writer = write_batch(feature_writer, data / "features" / f"features_{year}.parquet", features)
                feature_samples.append(features.loc[features.h3_cell.isin(sample_ids)].copy())
                y = features.target_recorded_detection.to_numpy()
                for rule in RULES:
                    if rule == "always_no_detection":
                        score = np.zeros(len(y), dtype=np.int32)
                    else:
                        window = int(rule[-2])
                        score = features[f"local_count_{window}d"].to_numpy().copy()
                        if "neighbour" in rule:
                            score += features[f"neighbour_count_{window}d"].to_numpy()
                    histogram_add(histograms[rule], y, score)
                    if year == 2023 and rule == selected:
                        pred = score>0
                        selected_totals[:, 0] += y.reshape(146, size).sum(axis=1)
                        selected_totals[:, 1] += pred.reshape(146, size).sum(axis=1)
                        selected_totals[:, 2] += (pred & (y>0)).reshape(146, size).sum(axis=1)
                        prediction_frame = features[["target_date", "h3_cell", "target_recorded_detection"]].copy()
                        prediction_frame["predicted_recorded_detection"] = pred
                        prediction_frame["rule"] = selected
                        prediction_writer = write_batch(prediction_writer, results / "predictions_2023.parquet", prediction_frame)
                del daily, features
        finally:
            for writer in (daily_writer, feature_writer, prediction_writer):
                if writer is not None:
                    writer.close()
        assert pq.ParquetFile(data / f"cell_date_{year}.parquet").metadata.num_rows == len(cells)*153
        assert pq.ParquetFile(data / "features" / f"features_{year}.parquet").metadata.num_rows == len(cells)*146
        annual = [{"region": name, "year": year,
                   "role": "training_diagnostic" if year <= 2021 else "rule_selection" if year == 2022 else "development_evaluation",
                   "rule": rule, **histogram_metrics(histograms[rule])} for rule in RULES]
        for row in annual:
            assert row["rows"] == len(cells)*146
            assert row["positive_cell_days"] == int((counts[7:]>0).sum())
        metrics_rows.extend(annual)
        if year == 2022:
            selected = sorted((r for r in annual if r["rule"] != "always_no_detection"), key=lambda r: (-r["f1"], r["rule"]))[0]["rule"]
        if year == 2023:
            assert selected is not None
            selected_daily = pd.DataFrame(selected_totals, columns=["observed_positive_cells", "predicted_positive_cells", "true_positive_cells"])
            selected_daily.insert(0, "date", dates[7:])
            selected_daily.to_csv(results / "selected_rule_daily_2023.csv", index=False)
        for month in range(5, 10):
            daily_sum = int(counts[dates.month==month].sum())
            raw_sum = int((observations.acq_date.dt.year.eq(year) & observations.acq_date.dt.month.eq(month)).sum())
            assert daily_sum == raw_sum
            monthly_rows.append({"region": name, "year": year, "month": month, "raw_detection_count": raw_sum, "daily_table_count": daily_sum})
        checks.append({"region": name, "year": year, "cells": len(cells), "daily_rows": len(cells)*153,
                       "feature_rows": len(cells)*146, "detections": int(counts.sum()),
                       "positive_targets": int((counts[7:]>0).sum()), "zero_detection_rows": int((counts==0).sum())})
        print(f"{name} {year}: {len(cells):,} cells; {len(cells)*146:,} feature rows; {int((counts[7:]>0).sum()):,} positive targets", flush=True)
        del counts, neighbour_daily
        gc.collect()
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(results / "metrics_by_year.csv", index=False)
    pd.DataFrame(checks).to_csv(results / "yearly_checks.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(data / "monthly_reconciliation.csv", index=False)
    sample_daily = pd.concat(sample_frames, ignore_index=True).sort_values(["h3_cell", "acq_date"])
    sample_daily["selection"] = np.where(sample_daily.h3_cell.isin(active_sample), "top_count_2018", "no_detections_2018_seed42")
    sample_daily.to_csv(data / "sample_cell_daily_counts.csv", index=False)
    feature_sample = pd.concat(feature_samples, ignore_index=True).sort_values(["h3_cell", "target_date"])
    feature_sample.to_parquet(results / "sample_features.parquet", index=False)
    if name == "Kamloops":
        # End-to-end regression against the original experiment, not just totals.
        reference = pd.read_csv(REFERENCE / "results/metrics_by_year.csv").sort_values(["year", "rule"])
        current = metrics.sort_values(["year", "rule"])
        for column in ("rows", "positive_cell_days", "precision", "recall", "f1", "average_precision", "tp", "fp", "fn", "tn"):
            np.testing.assert_allclose(current[column], reference[column], rtol=1e-12, atol=1e-14)
        previous = pd.read_parquet(REFERENCE / "results/sample_features.parquet").sort_values(["h3_cell", "target_date"])
        pd.testing.assert_frame_equal(feature_sample.reset_index(drop=True), previous.reset_index(drop=True), check_dtype=False, check_categorical=False)
        print("Kamloops regression passed: metrics and sample features match Experiment 01", flush=True)
    print(f"{name}: saving figures", flush=True)
    make_figures(name, region, polygons, active, sample_daily, metrics, selected_daily, figures)
    chosen = metrics.loc[metrics.year.eq(2023) & metrics.rule.eq(selected)].iloc[0].to_dict()
    (results / "manifest.json").write_text(json.dumps({
        "region": name, "source_hashes": source_hashes, "selected_rule": selected,
        "selection_year": 2022, "evaluation_year": 2023, "holdout_loaded": False,
        "grid_cells": len(cells), "resolution": 7, "uncovered_boundary_area_degrees2": uncovered,
        "grid_rule": "boundary-derived intersecting cells; independent of detections",
        "target": "recorded detection on next UTC acquisition day; zero is not confirmed no fire",
        "coverage": "unknown; coverage_supported_target missing on all rows",
        "features": "past-only 1,3,7 day local/within-region-neighbour counts; censored history age",
        "target_dates": "May 8-September 30, 2018-2023; no cross-season history",
        "rule_selection": "highest 2022 F1 with alphabetical tie-break; no model fitting",
        "sample_cells": sample_ids, "batch_size": batch_size,
        "checks": "full boundary coverage, H3 assignment, daily/monthly totals, neighbour sums, complete unique Cartesian rows by construction, past-only windows, parquet row counts",
        "versions": {"h3": h3.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                     "pyarrow": pa.__version__, "shapely": shapely.__version__, "scipy": scipy.__version__}}, indent=2), encoding="utf-8")
    print(f"FINISHED {name}: {selected}; P={chosen['precision']:.3f}, R={chosen['recall']:.3f}, F1={chosen['f1']:.3f}", flush=True)
    return chosen, metrics, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--regions", nargs="+", help="Optional exact Fire Centre names without 'Fire Centre'")
    parser.add_argument("--resume", action="store_true", help="Reuse completed regions only when source and script hashes match")
    args = parser.parse_args()
    assert args.batch_size > 0
    check_metrics()
    summary_dir = HERE / "results"
    summary_dir.mkdir(exist_ok=True)
    source_paths = [INPUT, BOUNDARY, Path(__file__), REFERENCE / "run_persistence.py", REFERENCE / "prepare_daily_grid.py"]
    hashes = {str(p): digest(p) for p in source_paths}
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    regions = {f["properties"]["MOF_FIRE_CENTRE_NAME"].replace(" Fire Centre", ""):
               unary_union(list(polygon_parts(shapely.make_valid(shape(f["geometry"])))) ) for f in boundary["features"]}
    assert len(regions) == 6 and all(g.is_valid for g in regions.values())
    names = args.regions or sorted(regions)
    assert set(names).issubset(regions), f"Choose from {sorted(regions)}"
    # Kamloops first provides a direct equivalence check before other regions.
    names = sorted(names, key=lambda name: (name != "Kamloops", name))
    pieces = []
    for chunk in pd.read_csv(INPUT, usecols=["latitude", "longitude", "acq_date", "h3_cell"], parse_dates=["acq_date"], chunksize=100_000):
        pieces.append(chunk.loc[chunk.acq_date.dt.year.isin(YEARS) & chunk.acq_date.dt.month.between(5, 9)].copy())
    observations = pd.concat(pieces, ignore_index=True)
    assert observations.acq_date.eq(observations.acq_date.dt.normalize()).all()
    assigned = [h3.latlng_to_cell(lat, lon, 7) for lat, lon in zip(observations.latitude, observations.longitude)]
    np.testing.assert_array_equal(assigned, observations.h3_cell.to_numpy())
    matches = {name: shapely.intersects_xy(region, observations.longitude.to_numpy(), observations.latitude.to_numpy()) for name, region in regions.items()}
    memberships = np.column_stack(list(matches.values())).sum(axis=1)
    audit = []
    for year in YEARS:
        mask = observations.acq_date.dt.year.eq(year).to_numpy()
        audit.append({"year": year, "bc_source_detections": int(mask.sum()),
                      "unassigned_to_cached_centres": int((mask & (memberships==0)).sum()),
                      "multiple_centre_matches": int((mask & (memberships>1)).sum()),
                      "regional_membership_count": int(memberships[mask].sum())})
    pd.DataFrame(audit).to_csv(summary_dir / "boundary_membership_audit.csv", index=False)
    observations.loc[memberships!=1].assign(region_match_count=memberships[memberships!=1]).to_csv(
        summary_dir / "boundary_exceptions.csv", index=False)
    print(f"Development detections: {len(observations):,}; no centre: {(memberships==0).sum():,}; multiple centres: {(memberships>1).sum():,}", flush=True)
    chosen_rows, metric_frames, all_checks = [], [], []
    for name in names:
        region_path = HERE / "regions" / name.lower().replace(" ", "_")
        completed = region_path / "results/manifest.json"
        if args.resume and completed.exists() and json.loads(completed.read_text())["source_hashes"] == hashes:
            manifest = json.loads(completed.read_text())
            metrics = pd.read_csv(region_path / "results/metrics_by_year.csv")
            chosen = metrics.loc[metrics.year.eq(2023) & metrics.rule.eq(manifest["selected_rule"])].iloc[0].to_dict()
            checks = pd.read_csv(region_path / "results/yearly_checks.csv").to_dict("records")
            print(f"Reusing completed {name}", flush=True)
        else:
            chosen, metrics, checks = run_region(name, regions[name], observations.loc[matches[name]], args.batch_size, hashes)
        chosen_rows.append(chosen)
        metric_frames.append(metrics)
        all_checks.extend(checks)
        pd.DataFrame(chosen_rows).to_csv(summary_dir / "selected_rules_2023.csv", index=False)
        pd.concat(metric_frames, ignore_index=True).to_csv(summary_dir / "all_region_metrics.csv", index=False)
        pd.DataFrame(all_checks).to_csv(summary_dir / "all_region_checks.csv", index=False)
        gc.collect()
    assert hashes == {str(p): digest(p) for p in source_paths}
    chosen = pd.DataFrame(chosen_rows).sort_values("region")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    chosen.set_index("region")[["precision", "recall", "f1"]].plot.bar(ax=ax)
    ax.set(title="2023: per-region rule selected using 2022 F1", ylim=(0, 1), xlabel="Fire Centre")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle=":", alpha=.4)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/selected_rules_by_region.png", dpi=160)
    plt.close(fig)
    (summary_dir / "manifest.json").write_text(json.dumps({
        "completed_regions": names, "all_six_complete": set(names)==set(regions), "source_hashes": hashes,
        "scope": "six Fire Centres, not municipal city boundaries; independent regional experiments",
        "development_years": list(YEARS), "selection_year": 2022, "evaluation_year": 2023,
        "holdout_loaded": False, "html_created": False,
        "boundary_note": "cached simplified boundaries; excluded/unassigned and multiple matches audited; border H3 cells can appear in multiple grids",
        "aggregation_note": "regional cell-days are not pooled as unique BC cell-days",
        "target_note": "no recorded detection is an observation proxy, not confirmed fire absence",
        "evaluation_note": "2023 is development evaluation; not untouched final test"}, indent=2), encoding="utf-8")
    print("\nCOMPLETE: all requested regions finished; no HTML report created.", flush=True)
    print(chosen[["region", "rule", "precision", "recall", "f1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
