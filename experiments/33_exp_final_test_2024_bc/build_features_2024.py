"""Build Experiment 02's per-cell feature tables for 2024, the final test year.

Same grid (Experiment 02's h3_grid.parquet, not rebuilt), same Fire Centre assignment of
detections (point in the Fire Centre polygon) and the same columns as
02_hotspot_persistence_bc/regions/<centre>/data/features/features_<year>.parquet. Writes
data/regions/<centre>/features_2024.parquet here, leaving Experiment 02 untouched.

    python build_features_2024.py              # 2024
    python build_features_2024.py --check 2023 # rebuild 2023 into a scratch folder and compare with Experiment 02
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from shapely.geometry import shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_02", EXPERIMENTS / "02_hotspot_persistence_bc/run_experiment.py")
experiment_02 = sys.modules["experiment_02"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_02)
from experiment_02 import BOUNDARY, INPUT, digest, history_age, polygon_parts, preceding_sum, write_batch  # noqa: E402

TEST_YEAR = 2024
GRIDS = EXPERIMENTS / "02_hotspot_persistence_bc/regions"


def fire_centres():
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    return {f["properties"]["MOF_FIRE_CENTRE_NAME"].replace(" Fire Centre", ""):
            unary_union(list(polygon_parts(shapely.make_valid(shape(f["geometry"]))))) for f in boundary["features"]}


def build_region(name, observations, year, out):
    """Experiment 02's run_region, reduced to the feature table of one year."""
    grid = pd.read_parquet(GRIDS / name.lower().replace(" ", "_") / "data/h3_grid.parquet")
    cells = grid.h3_cell.tolist()
    position = {cell: index for index, cell in enumerate(cells)}
    assert set(observations.h3_cell).issubset(position)
    rows, columns = [], []
    for cell, index in position.items():
        for neighbour in sorted(h3.grid_disk(cell, 1)):
            if neighbour != cell and neighbour in position:
                rows.append(index)
                columns.append(position[neighbour])
    adjacency = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, columns)), shape=(len(cells), len(cells)))
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype("int8")
    active = observations.groupby(["acq_date", "h3_cell"]).size().rename("detection_count").reset_index()
    dates = pd.date_range(f"{year}-05-01", f"{year}-09-30").as_unit("ns")  # Experiment 02 wrote nanosecond dates
    counts = np.zeros((153, len(cells)), dtype=np.int32)
    counts[(active.acq_date - dates[0]).dt.days.to_numpy(), active.h3_cell.map(position).to_numpy(dtype=int)] = active.detection_count.to_numpy()
    assert int(counts.sum()) == len(observations)
    neighbour_daily = (adjacency @ counts.T).T
    writer = None
    try:
        for start in range(0, len(cells), 2000):
            stop = min(start + 2000, len(cells))
            local, neighbours, size = counts[:, start:stop], neighbour_daily[:, start:stop], stop - start
            features = pd.DataFrame({"target_date": np.repeat(dates[7:], size), "h3_cell": np.tile(cells[start:stop], 146)})
            features["history_end_date"] = features.target_date - pd.Timedelta(days=1)
            for window in (1, 3, 7):
                features[f"local_count_{window}d"] = preceding_sum(local, window)[7:].ravel().astype("int32")
                features[f"neighbour_count_{window}d"] = preceding_sum(neighbours, window)[7:].ravel().astype("int32")
            features["days_since_previous_detection"] = history_age(local)[7:].ravel()
            features["history_left_censored"] = features.days_since_previous_detection.isna()
            features["in_region_neighbours"] = np.tile(degrees[start:stop], 146)
            features["target_recorded_detection"] = (local[7:].ravel() > 0).astype("int8")
            features["observation_coverage"] = pd.Categorical.from_codes(np.zeros(len(features), dtype=np.int8), ["unknown"])
            features["coverage_supported_target"] = pd.array(np.full(len(features), np.nan), dtype="Int8")
            writer = write_batch(writer, out, features)
    finally:
        if writer is not None:
            writer.close()
    return len(cells), int((counts[7:] > 0).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=int, help="rebuild this development year into a scratch folder and compare")
    args = parser.parse_args()
    year = args.check or TEST_YEAR
    root = HERE / ("scratch_check" if args.check else "data") / "regions"
    pieces = []
    for chunk in pd.read_csv(INPUT, usecols=["latitude", "longitude", "acq_date", "h3_cell"], parse_dates=["acq_date"], chunksize=100_000):
        pieces.append(chunk.loc[chunk.acq_date.dt.year.eq(year) & chunk.acq_date.dt.month.between(5, 9)].copy())
    observations = pd.concat(pieces, ignore_index=True)
    regions = fire_centres()
    assert len(regions) == 6
    summary = []
    for name in sorted(regions):
        inside = shapely.intersects_xy(regions[name], observations.longitude.to_numpy(), observations.latitude.to_numpy())
        out = root / name.lower().replace(" ", "_") / f"features_{year}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        n_cells, positives = build_region(name, observations.loc[inside], year, out)
        row = {"region": name, "year": year, "cells": n_cells, "detections": int(inside.sum()), "positive_cell_days": positives}
        if args.check:
            mine = pd.read_parquet(out)
            theirs = pd.read_parquet(GRIDS / name.lower().replace(" ", "_") / f"data/features/features_{year}.parquet")
            pd.testing.assert_frame_equal(mine, theirs, check_dtype=False)  # string vs object cell ids across pandas versions
            assert all(mine[c].dtype == theirs[c].dtype for c in mine if c != "h3_cell")
            row["matches_experiment_02"] = True
        summary.append(row)
        print(row, flush=True)
    summary = pd.DataFrame(summary)
    if not args.check:
        summary.to_csv(HERE / "data/features_2024_summary.csv", index=False)
        (HERE / "data/features_2024_manifest.json").write_text(json.dumps({
            "year": year, "source": str(INPUT.relative_to(EXPERIMENTS.parent.parent)), "source_sha256": digest(INPUT),
            "grids": "Experiment 02 h3_grid.parquet", "rows_per_cell": 146, "regions": summary.to_dict("records")},
            indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
