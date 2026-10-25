"""Assign FIRMS detections to H3 cells and aggregate them by cell and date.

This script does not create future labels. It preserves the point observations
with an H3 identifier and creates descriptive cell-date features for later
experiments. Install the `h3` Python package before running this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def load_settings(path: Path) -> dict:
    import tomllib
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    resolve = lambda value: (path.parent / value).resolve()
    settings = dict(raw)
    settings["input_dataset"] = resolve(raw["input_dataset"])
    settings["cell_date_output"] = resolve(raw["cell_date_output"])
    settings["cell_summary_output"] = resolve(raw["cell_summary_output"])
    settings["h3_detection_output"] = resolve(raw["h3_detection_output"])
    settings["h3_resolution"] = int(raw["h3_resolution"])
    settings["chunk_size"] = int(raw["chunk_size"])
    return settings


def h3_function() -> Callable[[float, float, int], str]:
    try:
        import h3
    except ImportError as error:
        raise RuntimeError(
            "The h3 package is required. Install it in the experiment environment "
            "with: python -m pip install h3"
        ) from error
    if hasattr(h3, "latlng_to_cell"):  # h3-py v4
        return lambda latitude, longitude, resolution: h3.latlng_to_cell(latitude, longitude, resolution)
    if hasattr(h3, "geo_to_h3"):  # h3-py v3
        return lambda latitude, longitude, resolution: h3.geo_to_h3(latitude, longitude, resolution)
    raise RuntimeError("Unsupported h3 package version: no latlng_to_cell or geo_to_h3 function")


def validate_settings(settings: dict) -> None:
    input_path = settings["input_dataset"]
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    resolution = settings["h3_resolution"]
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    for key in ("latitude_column", "longitude_column", "date_column", "h3_column"):
        if not settings.get(key):
            raise ValueError(f"Missing configuration value: {key}")


def assign_cells(frame: pd.DataFrame, settings: dict, cell_fn: Callable) -> pd.DataFrame:
    lat = pd.to_numeric(frame[settings["latitude_column"]], errors="coerce")
    lon = pd.to_numeric(frame[settings["longitude_column"]], errors="coerce")
    valid = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    if not valid.all():
        raise ValueError(f"Found {int((~valid).sum())} invalid coordinate rows")
    frame = frame.copy()
    frame[settings["date_column"]] = pd.to_datetime(frame[settings["date_column"]], errors="raise").dt.strftime("%Y-%m-%d")
    frame[settings["h3_column"]] = [
        cell_fn(float(latitude), float(longitude), settings["h3_resolution"])
        for latitude, longitude in zip(lat, lon)
    ]
    return frame


def aggregate_cell_date(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    cell = settings["h3_column"]
    date = settings["date_column"]
    frame["frp"] = pd.to_numeric(frame["frp"], errors="coerce")
    frame["brightness"] = pd.to_numeric(frame["brightness"], errors="coerce")
    frame["bright_t31"] = pd.to_numeric(frame["bright_t31"], errors="coerce")
    grouped = frame.groupby([cell, date], sort=True)
    result = grouped.agg(
        detection_count=(cell, "size"),
        frp_sum=("frp", "sum"),
        frp_max=("frp", "max"),
        brightness_mean=("brightness", "mean"),
        brightness_max=("brightness", "max"),
        bright_t31_mean=("bright_t31", "mean"),
        latitude_mean=(settings["latitude_column"], "mean"),
        longitude_mean=(settings["longitude_column"], "mean"),
    ).reset_index()
    return result


def aggregate_cell_summary(cell_date: pd.DataFrame, settings: dict) -> pd.DataFrame:
    cell = settings["h3_column"]
    return cell_date.groupby(cell, sort=True).agg(
        observed_dates=(settings["date_column"], "nunique"),
        total_detections=("detection_count", "sum"),
        total_frp=("frp_sum", "sum"),
        max_daily_frp=("frp_sum", "max"),
        mean_daily_detections=("detection_count", "mean"),
        latitude_mean=("latitude_mean", "mean"),
        longitude_mean=("longitude_mean", "mean"),
    ).reset_index()


def run(settings: dict) -> dict:
    validate_settings(settings)
    cell_fn = h3_function()
    input_columns = pd.read_csv(settings["input_dataset"], nrows=0).columns.tolist()
    required = {settings["latitude_column"], settings["longitude_column"], settings["date_column"], "frp", "brightness", "bright_t31"}
    missing = required - set(input_columns)
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {sorted(missing)}")
    detection_parts = []
    for chunk in pd.read_csv(settings["input_dataset"], chunksize=settings["chunk_size"]):
        detection_parts.append(assign_cells(chunk, settings, cell_fn))
    detections = pd.concat(detection_parts, ignore_index=True)
    cell_date = aggregate_cell_date(detections, settings)
    summary = aggregate_cell_summary(cell_date, settings)
    settings["h3_detection_output"].parent.mkdir(parents=True, exist_ok=True)
    detections.to_csv(settings["h3_detection_output"], index=False)
    cell_date.to_csv(settings["cell_date_output"], index=False)
    summary.to_csv(settings["cell_summary_output"], index=False)
    manifest = {
        "input_dataset": str(settings["input_dataset"]),
        "h3_resolution": settings["h3_resolution"],
        "input_rows": int(sum(len(chunk) for chunk in pd.read_csv(settings["input_dataset"], chunksize=settings["chunk_size"]))),
        "cell_date_rows": len(cell_date),
        "detection_rows": len(detections),
        "unique_h3_cells": int(cell_date[settings["h3_column"]].nunique()),
        "h3_detection_output": str(settings["h3_detection_output"]),
        "cell_date_output": str(settings["cell_date_output"]),
        "cell_summary_output": str(settings["cell_summary_output"]),
        "future_labels_created": False,
    }
    manifest_path = settings["cell_date_output"].parent / "h3_transform_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    args = parser.parse_args()
    print(json.dumps(run(load_settings(args.config.resolve())), indent=2))


if __name__ == "__main__":
    main()

