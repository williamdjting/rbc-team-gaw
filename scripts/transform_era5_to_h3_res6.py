"""Assign ERA5 weather (2018-2024) to ALL British Columbia H3 resolution 6 cells and save as Parquet.

Outputs (saved to `datasets/interim/` by default):
- `era5_h3_res6_cell_date_18_24.parquet`: 1 row per (`h3_cell`, `valid_time`) for 2018-2024
- `era5_h3_res6_cell_date_18_23.parquet`: 1 row per (`h3_cell`, `valid_time`) for 2018-2023 (dev)
- `era5_h3_res6_cell_date_2024_holdout.parquet`: 1 row per (`h3_cell`, `valid_time`) for 2024 (holdout)
- `era5_h3_res6_grid_mapping.parquet`: lookup table mapping every BC `h3_cell` to its ERA5 `(latitude, longitude_deg)` and `era5_dist_km`
- `era5_h3_res6_manifest.json`: run metadata, coverage audit against FIRMS H3 cells, and row counts
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from shapely import intersects_xy, prepare
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
INTERIM_DIR = DATASETS_DIR / "interim"

DEFAULT_INPUT_CANDIDATES = [
    INTERIM_DIR / "dataset_ERA5_18_24.parquet",
    DATASETS_DIR / "dataset_ERA5_18_24.parquet",
    INTERIM_DIR / "dataset_ERA5_18_24.csv",
    DATASETS_DIR / "dataset_ERA5_18_24.csv",
]

BOUNDARY_CANDIDATES = [
    DATASETS_DIR / "raw" / "bc_legal_boundary.geojson",
    DATASETS_DIR / "bc_legal_boundary.geojson",
    ROOT / "data" / "checks" / "bc_legal_boundary.geojson",
]

FIRMS_SUMMARY_CANDIDATES = [
    INTERIM_DIR / "firms_h3_res6_cell_summary.parquet",
    INTERIM_DIR / "firms_h3_res6_cell_summary.csv",
    INTERIM_DIR / "firms_h3_res6_cell_date_18_24.parquet",
    INTERIM_DIR / "firms_h3_res6_cell_date_18_24.csv",
]

WEATHER_AND_LAND_COLS = [
    "swvl1", "swvl2", "swvl3",
    "lai_lv", "lai_hv", "sp", "slhf", "u10", "v10",
    "t2m", "d2m", "ssrd", "ssr", "str", "tp", "skt", "pev",
    "sde", "evatc", "evabs", "evavt", "snowc",
    "cvh", "cvl", "z", "lsm",
]


def resolve_default_input() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[2]


def get_h3_functions() -> tuple[
    Callable[[float, float, int], str],
    Callable[[str], tuple[float, float]],
    Callable[[str, int], Iterable[str]],
]:
    try:
        import h3
    except ImportError as error:
        raise RuntimeError(
            "The h3 package is required. Install it with: python -m pip install h3"
        ) from error

    if hasattr(h3, "latlng_to_cell"):  # h3-py v4
        return (
            lambda lat, lon, res: h3.latlng_to_cell(lat, lon, res),
            lambda cell: h3.cell_to_latlng(cell),
            lambda cell, k: h3.grid_disk(cell, k),
        )
    if hasattr(h3, "geo_to_h3"):  # h3-py v3
        return (
            lambda lat, lon, res: h3.geo_to_h3(lat, lon, res),
            lambda cell: h3.h3_to_geo(cell),
            lambda cell, k: h3.k_ring(cell, k),
        )
    raise RuntimeError("Unsupported h3 package version: no latlng_to_cell or geo_to_h3 function")


def load_bc_polygon():
    boundary_path = next((p for p in BOUNDARY_CANDIDATES if p.is_file()), None)
    if boundary_path is None:
        raise FileNotFoundError(
            "Could not find bc_legal_boundary.geojson in datasets/raw/, datasets/, or data/checks/."
        )
    boundary_data = json.loads(boundary_path.read_text(encoding="utf-8"))
    bc_polygon = unary_union([shape(f["geometry"]) for f in boundary_data["features"]])
    prepare(bc_polygon)
    return bc_polygon


def build_complete_bc_h3_grid_mapping(
    era5_coords: pd.DataFrame,
    resolution: int,
    max_dist_km: float = 12.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate all BC land H3 cells at `resolution` and map each cell to its nearest
    valid ERA5 0.1° BC land grid point `(latitude, longitude_deg)` using a cKDTree.
    """
    cell_fn, cell_center_fn, disk_fn = get_h3_functions()
    bc_polygon = load_bc_polygon()

    seed_cells = {
        cell_fn(float(lat), float(lon), resolution)
        for lat, lon in zip(era5_coords["latitude"], era5_coords["longitude_deg"])
    }

    firms_cells: set[str] = set()
    firms_path = next((p for p in FIRMS_SUMMARY_CANDIDATES if p.is_file()), None)
    if firms_path is not None:
        if firms_path.suffix == ".parquet":
            firms_df = pd.read_parquet(firms_path, columns=["h3_cell"])
        else:
            firms_df = pd.read_csv(firms_path, usecols=["h3_cell"])
        firms_cells = set(firms_df["h3_cell"].dropna().unique())
        seed_cells |= firms_cells

    candidate_cells: set[str] = set()
    for cell in seed_cells:
        candidate_cells.update(disk_fn(cell, 2))

    cand_list = sorted(candidate_cells)
    centers = [cell_center_fn(c) for c in cand_list]
    h3_lats = np.array([pt[0] for pt in centers], dtype=np.float64)
    h3_lons = np.array([pt[1] for pt in centers], dtype=np.float64)

    inside_bc = intersects_xy(bc_polygon, h3_lons, h3_lats)
    in_firms = np.array([c in firms_cells for c in cand_list], dtype=bool)
    keep_bc = inside_bc | in_firms

    bc_cells = np.array(cand_list)[keep_bc]
    bc_h3_lats = h3_lats[keep_bc]
    bc_h3_lons = h3_lons[keep_bc]

    cos_lat = np.cos(np.deg2rad(54.0))
    era5_xy = np.column_stack([
        era5_coords["latitude"].to_numpy(dtype=np.float64),
        era5_coords["longitude_deg"].to_numpy(dtype=np.float64) * cos_lat,
    ])
    h3_xy = np.column_stack([bc_h3_lats, bc_h3_lons * cos_lat])

    tree = cKDTree(era5_xy)
    distances_deg, indices = tree.query(h3_xy, k=1)
    dist_km = distances_deg * 111.32

    bc_in_firms = in_firms[keep_bc]
    valid_land_cell = (dist_km <= max_dist_km) | bc_in_firms

    mapping_df = pd.DataFrame({
        "h3_cell": bc_cells[valid_land_cell],
        "h3_lat": np.round(bc_h3_lats[valid_land_cell], 6).astype(np.float32),
        "h3_lon": np.round(bc_h3_lons[valid_land_cell], 6).astype(np.float32),
        "latitude": era5_coords["latitude"].to_numpy(dtype=np.float32)[indices[valid_land_cell]],
        "longitude_deg": era5_coords["longitude_deg"].to_numpy(dtype=np.float32)[indices[valid_land_cell]],
        "era5_dist_km": np.round(dist_km[valid_land_cell], 2).astype(np.float32),
    }).sort_values("h3_cell", ignore_index=True)

    final_cell_set = set(mapping_df["h3_cell"])
    firms_covered = len(firms_cells & final_cell_set) if firms_cells else 0

    audit = {
        "seed_era5_h3_cells": int(len(seed_cells - firms_cells)),
        "total_bc_land_h3_cells": int(len(mapping_df)),
        "median_era5_dist_km": float(np.median(mapping_df["era5_dist_km"])),
        "max_era5_dist_km": float(np.max(mapping_df["era5_dist_km"])),
        "firms_unique_h3_cells": int(len(firms_cells)),
        "firms_h3_cells_covered": int(firms_covered),
        "firms_coverage_pct": round(firms_covered / len(firms_cells) * 100.0, 2) if firms_cells else 100.0,
    }
    return mapping_df, audit


def expand_era5_batch_to_bc_h3(
    era5_batch: pd.DataFrame,
    mapping_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Map a batch of ERA5 daily rows onto all BC H3 cells (1 row per h3_cell and valid_time)."""
    era5_batch = era5_batch.copy()
    era5_batch["latitude"] = era5_batch["latitude"].astype(np.float32)
    era5_batch["longitude_deg"] = era5_batch["longitude_deg"].astype(np.float32)
    for col in feature_cols:
        era5_batch[col] = era5_batch[col].astype(np.float32)
    if "year" in era5_batch.columns:
        era5_batch["year"] = era5_batch["year"].astype(np.int16)
    if "month" in era5_batch.columns:
        era5_batch["month"] = era5_batch["month"].astype(np.int8)

    era5_cols = ["valid_time", "latitude", "longitude_deg", *feature_cols]
    for extra in ("year", "month"):
        if extra in era5_batch.columns:
            era5_cols.append(extra)

    merged = mapping_df.merge(
        era5_batch[era5_cols],
        on=["latitude", "longitude_deg"],
        how="inner",
    )
    merged.rename(
        columns={"latitude": "era5_lat", "longitude_deg": "era5_lon"},
        inplace=True,
    )
    ordered = [
        "h3_cell", "valid_time", "h3_lat", "h3_lon",
        "era5_lat", "era5_lon", "era5_dist_km",
        *feature_cols,
    ]
    for extra in ("year", "month"):
        if extra in merged.columns:
            ordered.append(extra)

    return merged[ordered].sort_values(["valid_time", "h3_cell"], ignore_index=True)


def iter_input_chunks(input_path: Path, chunk_size: int):
    """Yield DataFrames in chunks from either a Parquet or CSV input file."""
    if input_path.suffix == ".parquet":
        pf = pq.ParquetFile(input_path)
        for record_batch in pf.iter_batches(batch_size=chunk_size):
            yield record_batch.to_pandas()
    else:
        yield from pd.read_csv(input_path, chunksize=chunk_size)


def run(
    input_path: Path,
    output_dir: Path,
    resolution: int = 6,
    chunk_size: int = 400_000,
    max_dist_km: float = 12.0,
    grid_only: bool = False,
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ERA5 dataset not found: {input_path}")
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")

    if input_path.suffix == ".parquet":
        input_columns = pq.read_schema(input_path).names
    else:
        input_columns = pd.read_csv(input_path, nrows=0).columns.tolist()

    lon_col = "longitude_deg" if "longitude_deg" in input_columns else "longitude"
    feature_cols = [c for c in WEATHER_AND_LAND_COLS if c in input_columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"era5_h3_res{resolution}"
    out_18_24 = output_dir / f"{prefix}_cell_date_18_24.parquet"
    out_18_23 = output_dir / f"{prefix}_cell_date_18_23.parquet"
    out_2024 = output_dir / f"{prefix}_cell_date_2024_holdout.parquet"
    grid_map_out = output_dir / f"{prefix}_grid_mapping.parquet"
    manifest_out = output_dir / f"{prefix}_manifest.json"

    print(f"Discovering unique ERA5 BC land grid points from {input_path.name}...")
    first_chunk = next(iter_input_chunks(input_path, 50_000))[["latitude", lon_col]].copy()
    if lon_col != "longitude_deg":
        first_chunk["longitude_deg"] = ((first_chunk[lon_col] + 180.0) % 360.0) - 180.0
    era5_coords = first_chunk[["latitude", "longitude_deg"]].drop_duplicates().reset_index(drop=True)

    print(f"Building complete BC H3 resolution-{resolution} grid mapping from {len(era5_coords):,} ERA5 land points...")
    mapping_df, audit = build_complete_bc_h3_grid_mapping(
        era5_coords=era5_coords,
        resolution=resolution,
        max_dist_km=max_dist_km,
    )
    mapping_df.to_parquet(grid_map_out, index=False)
    print(
        f"  Total BC land H3 res-{resolution} cells mapped: {audit['total_bc_land_h3_cells']:,} "
        f"(up from {audit['seed_era5_h3_cells']:,} direct point cells)\n"
        f"  FIRMS H3 fire cells covered: {audit['firms_h3_cells_covered']:,} / {audit['firms_unique_h3_cells']:,} "
        f"({audit['firms_coverage_pct']}%)\n"
        f"  Median distance from H3 centroid to ERA5 grid point: {audit['median_era5_dist_km']:.2f} km"
    )

    if grid_only:
        manifest = {
            "input_dataset": str(input_path),
            "h3_resolution": resolution,
            "format": "parquet",
            "unique_era5_grid_points": int(len(era5_coords)),
            "unique_h3_cells": int(len(mapping_df)),
            "coverage_audit": audit,
            "outputs": {"grid_mapping": str(grid_map_out)},
        }
        manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    for fpath in (out_18_24, out_18_23, out_2024):
        if fpath.exists():
            fpath.unlink()

    writer_18_24: pq.ParquetWriter | None = None
    writer_18_23: pq.ParquetWriter | None = None
    writer_2024: pq.ParquetWriter | None = None

    carryover = pd.DataFrame()
    total_input_rows = 0
    rows_18_24 = 0
    rows_18_23 = 0
    rows_2024 = 0

    def write_parquet_batch(expanded_df: pd.DataFrame) -> None:
        nonlocal writer_18_24, writer_18_23, writer_2024, rows_18_24, rows_18_23, rows_2024

        table_all = pa.Table.from_pandas(expanded_df, preserve_index=False)
        if writer_18_24 is None:
            writer_18_24 = pq.ParquetWriter(out_18_24, table_all.schema, compression="snappy")
        writer_18_24.write_table(table_all)
        rows_18_24 += len(expanded_df)

        years = expanded_df["year"] if "year" in expanded_df.columns else pd.to_datetime(expanded_df["valid_time"]).dt.year
        dev_df = expanded_df[years <= 2023]
        if not dev_df.empty:
            table_dev = pa.Table.from_pandas(dev_df, schema=table_all.schema, preserve_index=False)
            if writer_18_23 is None:
                writer_18_23 = pq.ParquetWriter(out_18_23, table_all.schema, compression="snappy")
            writer_18_23.write_table(table_dev)
            rows_18_23 += len(dev_df)

        holdout_df = expanded_df[years == 2024]
        if not holdout_df.empty:
            table_holdout = pa.Table.from_pandas(holdout_df, schema=table_all.schema, preserve_index=False)
            if writer_2024 is None:
                writer_2024 = pq.ParquetWriter(out_2024, table_all.schema, compression="snappy")
            writer_2024.write_table(table_holdout)
            rows_2024 += len(holdout_df)

    try:
        print(f"Streaming {input_path.name} and writing complete BC H3 res-{resolution} Parquet files...")
        for chunk_idx, chunk in enumerate(iter_input_chunks(input_path, chunk_size), start=1):
            total_input_rows += len(chunk)
            if lon_col != "longitude_deg":
                chunk["longitude_deg"] = ((chunk[lon_col] + 180.0) % 360.0) - 180.0

            if not carryover.empty:
                chunk = pd.concat([carryover, chunk], ignore_index=True)

            last_date = chunk["valid_time"].iloc[-1]
            is_last_date = chunk["valid_time"] == last_date
            if is_last_date.all():
                carryover = chunk
                continue

            ready_batch = chunk.loc[~is_last_date]
            carryover = chunk.loc[is_last_date].copy()

            expanded = expand_era5_batch_to_bc_h3(ready_batch, mapping_df, feature_cols)
            write_parquet_batch(expanded)
            print(f"  Processed chunk {chunk_idx}: {total_input_rows:,} ERA5 rows -> {rows_18_24:,} H3 daily rows", flush=True)

        if not carryover.empty:
            expanded = expand_era5_batch_to_bc_h3(carryover, mapping_df, feature_cols)
            write_parquet_batch(expanded)
    finally:
        for writer in (writer_18_24, writer_18_23, writer_2024):
            if writer is not None:
                writer.close()

    manifest = {
        "input_dataset": str(input_path),
        "h3_resolution": resolution,
        "format": "parquet",
        "input_rows": int(total_input_rows),
        "unique_era5_grid_points": int(len(era5_coords)),
        "unique_h3_cells": int(len(mapping_df)),
        "coverage_audit": audit,
        "cell_date_rows_18_24": int(rows_18_24),
        "cell_date_rows_18_23": int(rows_18_23),
        "cell_date_rows_2024_holdout": int(rows_2024),
        "outputs": {
            "cell_date_18_24": str(out_18_24),
            "cell_date_18_23": str(out_18_23),
            "cell_date_2024_holdout": str(out_2024),
            "grid_mapping": str(grid_map_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=resolve_default_input(), help="Path to dataset_ERA5_18_24 CSV/Parquet")
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DIR, help="Directory for H3 Parquet output files")
    parser.add_argument("--resolution", type=int, default=6, help="H3 resolution (default: 6)")
    parser.add_argument("--chunk-size", type=int, default=400_000, help="Reading chunk size")
    parser.add_argument(
        "--max-dist-km",
        type=float,
        default=12.0,
        help="Max distance (km) from an ERA5 land point for coastal H3 cells (default: 12.0)",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only build and save `era5_h3_res6_grid_mapping.parquet` and coverage audit",
    )
    args = parser.parse_args()

    manifest = run(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        max_dist_km=args.max_dist_km,
        grid_only=args.grid_only,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
