"""Assign BC summer FIRMS detections (2018-2024) to H3 resolution 6 cells and aggregate by cell and date (Parquet output).

Outputs (saved to `datasets/interim/` by default):
- `firms_h3_res6_detections_18_24.parquet`: row-level FIRMS detections with `h3_cell` (res 6)
- `firms_h3_res6_cell_date_18_24.parquet`: 1 row per (`h3_cell`, `acq_date`) for 2018-2024
- `firms_h3_res6_cell_date_18_23.parquet`: 1 row per (`h3_cell`, `acq_date`) for 2018-2023 (dev)
- `firms_h3_res6_cell_date_2024_holdout.parquet`: 1 row per (`h3_cell`, `acq_date`) for 2024 (holdout)
- `firms_h3_res6_cell_summary.parquet`: 1 row per `h3_cell` across the full period
- `firms_h3_res6_manifest.json`: run metadata and row counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
INTERIM_DIR = DATASETS_DIR / "interim"

DEFAULT_INPUT_CANDIDATES = [
    DATASETS_DIR / "firms_summer_2018_2024" / "data" / "bc_firms_summer_2018_2024.parquet",
    DATASETS_DIR / "firms_summer_2018_2024" / "data" / "bc_firms_summer_2018_2024.csv",
    DATASETS_DIR / "raw" / "bc_firms_summer_2018_2024.csv",
    DATASETS_DIR / "bc_firms_summer_2018_2024.csv",
]


def resolve_default_input() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[1]


def get_h3_functions() -> tuple[Callable[[float, float, int], str], Callable[[str], tuple[float, float]]]:
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
        )
    if hasattr(h3, "geo_to_h3"):  # h3-py v3
        return (
            lambda lat, lon, res: h3.geo_to_h3(lat, lon, res),
            lambda cell: h3.h3_to_geo(cell),
        )
    raise RuntimeError("Unsupported h3 package version: no latlng_to_cell or geo_to_h3 function")


def assign_cells(
    frame: pd.DataFrame,
    resolution: int,
    cell_fn: Callable[[float, float, int], str],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "acq_date",
    h3_col: str = "h3_cell",
) -> pd.DataFrame:
    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    lon = pd.to_numeric(frame[lon_col], errors="coerce")
    valid = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    if not valid.all():
        raise ValueError(f"Found {int((~valid).sum())} invalid coordinate rows")

    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise").dt.strftime("%Y-%m-%d")
    frame[h3_col] = [
        cell_fn(float(latitude), float(longitude), resolution)
        for latitude, longitude in zip(lat, lon)
    ]
    return frame


def aggregate_cell_date(
    frame: pd.DataFrame,
    cell_center_fn: Callable[[str], tuple[float, float]],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "acq_date",
    h3_col: str = "h3_cell",
) -> pd.DataFrame:
    frame["frp"] = pd.to_numeric(frame["frp"], errors="coerce")
    frame["brightness"] = pd.to_numeric(frame["brightness"], errors="coerce")
    frame["bright_t31"] = pd.to_numeric(frame["bright_t31"], errors="coerce")

    grouped = frame.groupby([h3_col, date_col], sort=True)
    result = grouped.agg(
        detection_count=(h3_col, "size"),
        frp_sum=("frp", "sum"),
        frp_max=("frp", "max"),
        brightness_mean=("brightness", "mean"),
        brightness_max=("brightness", "max"),
        bright_t31_mean=("bright_t31", "mean"),
        latitude_mean=(lat_col, "mean"),
        longitude_mean=(lon_col, "mean"),
    ).reset_index()

    unique_cells = result[h3_col].unique()
    centers = {cell: cell_center_fn(cell) for cell in unique_cells}
    result["h3_lat"] = result[h3_col].map(lambda c: centers[c][0])
    result["h3_lon"] = result[h3_col].map(lambda c: centers[c][1])

    return result


def aggregate_cell_summary(
    cell_date: pd.DataFrame,
    date_col: str = "acq_date",
    h3_col: str = "h3_cell",
) -> pd.DataFrame:
    return cell_date.groupby(h3_col, sort=True).agg(
        observed_dates=(date_col, "nunique"),
        total_detections=("detection_count", "sum"),
        total_frp=("frp_sum", "sum"),
        max_daily_frp=("frp_sum", "max"),
        mean_daily_detections=("detection_count", "mean"),
        latitude_mean=("latitude_mean", "mean"),
        longitude_mean=("longitude_mean", "mean"),
        h3_lat=("h3_lat", "first"),
        h3_lon=("h3_lon", "first"),
    ).reset_index()


def run(input_path: Path, output_dir: Path, resolution: int = 6, chunk_size: int = 200_000) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input FIRMS dataset not found: {input_path}")
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")

    cell_fn, cell_center_fn = get_h3_functions()
    print(f"Assigning FIRMS detections from {input_path} to H3 resolution {resolution}...")

    if input_path.suffix == ".parquet":
        raw_df = pd.read_parquet(input_path)
        detections = assign_cells(raw_df, resolution, cell_fn)
    else:
        input_columns = pd.read_csv(input_path, nrows=0).columns.tolist()
        required = {"latitude", "longitude", "acq_date", "frp", "brightness", "bright_t31"}
        missing = required - set(input_columns)
        if missing:
            raise ValueError(f"Input dataset is missing required columns: {sorted(missing)}")
        detection_parts = []
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            detection_parts.append(assign_cells(chunk, resolution, cell_fn))
        detections = pd.concat(detection_parts, ignore_index=True)

    print("Aggregating detections to 1 row per (h3_cell, acq_date)...")
    cell_date = aggregate_cell_date(detections, cell_center_fn)
    summary = aggregate_cell_summary(cell_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"firms_h3_res{resolution}"

    detections_out = output_dir / f"{prefix}_detections_18_24.parquet"
    cell_date_18_24_out = output_dir / f"{prefix}_cell_date_18_24.parquet"
    cell_date_18_23_out = output_dir / f"{prefix}_cell_date_18_23.parquet"
    cell_date_2024_out = output_dir / f"{prefix}_cell_date_2024_holdout.parquet"
    summary_out = output_dir / f"{prefix}_cell_summary.parquet"
    manifest_out = output_dir / f"{prefix}_manifest.json"

    detections.to_parquet(detections_out, index=False)
    cell_date.to_parquet(cell_date_18_24_out, index=False)

    years = pd.to_datetime(cell_date["acq_date"]).dt.year
    cell_date[years <= 2023].to_parquet(cell_date_18_23_out, index=False)
    cell_date[years == 2024].to_parquet(cell_date_2024_out, index=False)
    summary.to_parquet(summary_out, index=False)

    manifest = {
        "input_dataset": str(input_path),
        "h3_resolution": resolution,
        "format": "parquet",
        "detection_rows": int(len(detections)),
        "cell_date_rows_18_24": int(len(cell_date)),
        "cell_date_rows_18_23": int((years <= 2023).sum()),
        "cell_date_rows_2024_holdout": int((years == 2024).sum()),
        "unique_h3_cells": int(cell_date["h3_cell"].nunique()),
        "outputs": {
            "detections_18_24": str(detections_out),
            "cell_date_18_24": str(cell_date_18_24_out),
            "cell_date_18_23": str(cell_date_18_23_out),
            "cell_date_2024_holdout": str(cell_date_2024_out),
            "cell_summary": str(summary_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=resolve_default_input(), help="Path to BC FIRMS CSV/Parquet")
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DIR, help="Directory for H3 Parquet output files")
    parser.add_argument("--resolution", type=int, default=6, help="H3 resolution (default: 6)")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="CSV reading chunk size")
    args = parser.parse_args()

    manifest = run(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        resolution=args.resolution,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
