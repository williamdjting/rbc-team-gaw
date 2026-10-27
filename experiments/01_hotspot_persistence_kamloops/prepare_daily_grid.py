"""Prepare a fixed Kamloops grid and daily recorded-detection counts, not labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = ROOT / "experiments/firms_summer_2018_2024/data/firms_h3_detections.csv"
BOUNDARY = ROOT / "data/region_exploration/fire_centres.geojson"
YEARS = range(2018, 2024)
RESOLUTION = 7


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    else:
        for part in getattr(geometry, "geoms", []):
            yield from polygon_parts(part)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def outline(ax, geometry):
    for polygon in polygon_parts(geometry):
        ax.plot(*polygon.exterior.xy, color="gray", linewidth=.7)
        for ring in polygon.interiors:
            ax.plot(*ring.xy, color="gray", linewidth=.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(geometry.centroid.y)))
    ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.set_axisbelow(True)
    ax.grid(linestyle=":", alpha=.4, linewidth=.6)


def main():
    data = HERE / "data"
    figures = HERE / "figures"
    data.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    hashes_before = {str(p): digest(p) for p in (INPUT, BOUNDARY)}
    features = json.loads(BOUNDARY.read_text(encoding="utf-8"))["features"]
    feature = next(f for f in features
                   if f["properties"]["MOF_FIRE_CENTRE_NAME"] == "Kamloops Fire Centre")
    original = shape(feature["geometry"])
    region = unary_union(list(polygon_parts(shapely.make_valid(original))))
    assert region.is_valid and not region.is_empty

    # Grid membership is determined before reading any detection locations.
    # bbox_overlap supplies conservative candidates; planar intersections then
    # use the same lon/lat boundary representation as the notebook.
    candidates = set()
    for polygon in polygon_parts(region):
        candidates.update(h3.h3shape_to_cells_experimental(
            h3.geo_to_h3shape(mapping(polygon)), RESOLUTION, contain="bbox_overlap"))
    grid_polygons = {
        cell: Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
        for cell in sorted(candidates)
    }
    grid_polygons = {cell: polygon for cell, polygon in grid_polygons.items()
                     if polygon.intersects(region)}
    cells = list(grid_polygons)
    uncovered_area = region.difference(unary_union(list(grid_polygons.values()))).area
    assert uncovered_area < 1e-9, f"Grid does not cover the boundary: {uncovered_area} deg2"
    grid = pd.DataFrame([
        {"h3_cell": cell, "resolution": RESOLUTION,
         "center_latitude": h3.cell_to_latlng(cell)[0],
         "center_longitude": h3.cell_to_latlng(cell)[1],
         "is_border_cell": not region.covers(polygon),
         "geometry_wkt": polygon.wkt}
        for cell, polygon in grid_polygons.items()
    ])
    grid.to_parquet(data / "kamloops_h3_grid.parquet", index=False)
    print(f"Fixed grid: {len(cells):,} cells", flush=True)

    # Stream the source. Holdout records are discarded before spatial summaries.
    pieces = []
    for chunk in pd.read_csv(INPUT, usecols=["latitude", "longitude", "acq_date", "h3_cell"],
                             parse_dates=["acq_date"], chunksize=100_000):
        chunk = chunk.loc[chunk.acq_date.dt.year.isin(YEARS)
                          & chunk.acq_date.dt.month.between(5, 9)].copy()
        chunk = chunk.loc[shapely.intersects_xy(
            region, chunk.longitude.to_numpy(), chunk.latitude.to_numpy())].copy()
        pieces.append(chunk)
    observations = pd.concat(pieces, ignore_index=True)
    assert len(observations), "No development detections found"
    assert observations.acq_date.eq(observations.acq_date.dt.normalize()).all()
    assert set(observations.h3_cell).issubset(grid_polygons), "Detection outside fixed grid"
    assert all(h3.get_resolution(cell) == RESOLUTION for cell in observations.h3_cell.unique())
    assigned = [h3.latlng_to_cell(lat, lon, RESOLUTION)
                for lat, lon in zip(observations.latitude, observations.longitude)]
    assert np.array_equal(assigned, observations.h3_cell.to_numpy()), "H3 assignment mismatch"
    counts = observations.groupby(["acq_date", "h3_cell"]).size().rename("detection_count")
    counts.reset_index().to_parquet(data / "active_cell_days.parquet", index=False)

    # Diagnostic examples are selected using 2018 only, not a future test year.
    totals_2018 = observations.loc[observations.acq_date.dt.year == 2018].groupby("h3_cell").size()
    active_sample = totals_2018.sort_values(ascending=False, kind="stable").head(3).index.tolist()
    quiet_pool = sorted(set(cells) - set(totals_2018.index))
    quiet_sample = np.random.default_rng(42).choice(quiet_pool, size=min(3, len(quiet_pool)),
                                                 replace=False).tolist()
    samples = active_sample + quiet_sample
    sample_frames, yearly_checks = [], []
    monthly_checks = []
    for year in YEARS:
        dates = pd.date_range(f"{year}-05-01", f"{year}-09-30")
        index = pd.MultiIndex.from_product([dates, cells], names=["acq_date", "h3_cell"])
        daily = counts.reindex(index, fill_value=0).astype("int32").reset_index()
        daily["has_recorded_detection"] = daily.detection_count.gt(0)
        # Unknown even on positive days: a detection does not establish complete coverage.
        daily["observation_coverage"] = pd.Categorical(["unknown"] * len(daily))
        daily["detection_status"] = pd.Categorical(np.where(
            daily.has_recorded_detection, "recorded_detection", "no_recorded_detection"))
        expected = observations.loc[observations.acq_date.dt.year == year]
        assert len(daily) == len(cells) * len(dates)
        assert not daily.duplicated(["acq_date", "h3_cell"]).any()
        assert int(daily.detection_count.sum()) == len(expected)
        assert daily.observation_coverage.eq("unknown").all()
        raw_daily = expected.groupby("acq_date").size().reindex(dates, fill_value=0)
        pd.testing.assert_series_equal(daily.groupby("acq_date").detection_count.sum(),
                                       raw_daily, check_names=False, check_dtype=False,
                                       check_freq=False)
        daily.to_parquet(data / f"cell_date_{year}.parquet", index=False)
        sample_frames.append(daily.loc[daily.h3_cell.isin(samples)].copy())
        for month in range(5, 10):
            total = int(daily.loc[daily.acq_date.dt.month == month, "detection_count"].sum())
            raw_total = int(expected.acq_date.dt.month.eq(month).sum())
            assert total == raw_total
            monthly_checks.append({"year": year, "month": month,
                                   "raw_detection_count": raw_total, "daily_table_count": total})
        yearly_checks.append({"year": year, "cells": len(cells), "days": len(dates),
                              "rows": len(daily), "detections": len(expected),
                              "zero_detection_rows": int(daily.detection_count.eq(0).sum())})
        print(f"{year}: {len(daily):,} cell-days, {len(expected):,} detections", flush=True)

    monthly = pd.DataFrame(monthly_checks)
    monthly.to_csv(data / "monthly_reconciliation.csv", index=False)
    expected_monthly = {6: 5, 7: 1782, 8: 9648, 9: 985}
    for month, total in expected_monthly.items():
        actual = monthly.loc[(monthly.year == 2018) & (monthly.month == month), "daily_table_count"].item()
        assert actual == total, f"2018 month {month}: notebook {total}, computed {actual}"
    weekly_checks = []
    for year in (2018, 2023):
        for (start, end), expected_total in zip([(1, 7), (8, 14), (15, 21), (22, 31)],
                                               [1631, 2135, 3460, 2422]):
            mask = (observations.acq_date.dt.year.eq(year)
                    & observations.acq_date.dt.month.eq(8)
                    & observations.acq_date.dt.day.between(start, end))
            total = int(mask.sum())
            if year == 2018:
                assert total == expected_total, "August 2018 notebook reconciliation failed"
            weekly_checks.append({"year": year, "start_day": start, "end_day": end,
                                  "detections": total, "detections_per_day": total / (end-start+1)})
    pd.DataFrame(weekly_checks).to_csv(data / "august_period_counts.csv", index=False)
    pd.DataFrame(yearly_checks).to_csv(data / "yearly_checks.csv", index=False)
    sample_daily = pd.concat(sample_frames, ignore_index=True)
    sample_daily["selection"] = np.where(sample_daily.h3_cell.isin(active_sample),
                                         "top_count_2018", "no_detections_2018_seed42")
    sample_daily.to_csv(data / "sample_cell_daily_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.add_collection(PolyCollection([list(p.exterior.coords) for p in grid_polygons.values()],
                                     facecolors="none", edgecolors="#3268ad", linewidths=.15))
    outline(ax, region)
    ax.autoscale()
    ax.set_title(f"Fixed resolution-7 Kamloops grid ({len(cells):,} intersecting cells)")
    fig.savefig(figures / "fixed_h3_grid.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    for ax, year in zip(axes.flat, YEARS):
        dates = pd.date_range(f"{year}-05-01", f"{year}-09-30")
        series = observations.loc[observations.acq_date.dt.year == year].groupby("acq_date").size()
        ax.plot(dates, series.reindex(dates, fill_value=0), color="#d95f02", linewidth=.8)
        ax.set(title=str(year), ylabel="Recorded detections", xlabel="Date")
        ax.grid(linestyle=":", alpha=.4)
    fig.suptitle("Kamloops daily observed hotspot activity (panel scales vary)")
    fig.savefig(figures / "regional_daily_counts.png", dpi=180)
    plt.close(fig)

    # Separate years on the x-axis by plotting each summer independently.
    fig, axes = plt.subplots(len(samples), 1, figsize=(15, 2.3 * len(samples)),
                             squeeze=False, constrained_layout=True)
    for ax, cell in zip(axes.flat, samples):
        subset = sample_daily.loc[sample_daily.h3_cell == cell]
        for year in YEARS:
            season = subset.loc[subset.acq_date.dt.year == year]
            ax.plot(season.acq_date, season.detection_count, linewidth=.8, color="#d95f02")
        selection = "active in 2018" if cell in active_sample else "no detections in 2018"
        ax.set(title=f"{cell} ({selection})", ylabel="Detections/day")
        ax.grid(linestyle=":", alpha=.4)
    fig.suptitle("Example H3 cells: observed activity, not ignition counts; off-season gaps omitted")
    fig.savefig(figures / "sample_cell_timeseries.png", dpi=180)
    plt.close(fig)

    assert hashes_before == {str(p): digest(p) for p in (INPUT, BOUNDARY)}
    manifest = {
        "source_sha256": hashes_before, "years": list(YEARS), "months": [5, 6, 7, 8, 9],
        "h3_resolution": RESOLUTION, "grid_cells": len(cells),
        "grid_rule": "Boundary-derived bbox-overlap candidates filtered by polygon intersection",
        "detection_rule": "Point centre inside/on Kamloops boundary, no confidence/type filtering",
        "boundary_repaired": not original.is_valid,
        "uncovered_boundary_area_degrees2": uncovered_area,
        "boundary_note": "Cached boundary simplified to 0.002 degrees by notebook 01",
        "coverage": "Unknown for every cell-day; zero counts are not verified negatives",
        "date_basis": "FIRMS acq_date (UTC acquisition date); May-September only",
        "off_season": "Not zero-filled; pre-May history unavailable in this input",
        "holdout": "2024 discarded before aggregation; no holdout outputs",
        "labels_created": False, "sample_cells": samples,
        "checks": "Full grid coverage, H3 assignments, unique complete cell-days, daily/monthly totals, 2018 plot totals, source hashes passed",
        "versions": {"h3": h3.__version__, "shapely": shapely.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "yearly_checks": yearly_checks,
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Checks passed. Tables: {data}\nFigures: {figures}")


if __name__ == "__main__":
    main()
