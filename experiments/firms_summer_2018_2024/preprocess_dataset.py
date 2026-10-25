"""Preprocess a FIRMS archive for the 2018–2024 BC summer experiment.

The pipeline is deliberately independent of the main project scripts. It reads a
ZIP member in chunks, applies date and BC point filters, writes development and
holdout CSVs, and records auditable counts and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data" / "check-dependencies"))
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely import intersects_xy


@dataclass(frozen=True)
class Config:
    input_archives: tuple[Path, ...]
    boundary_geojson: Path
    output_dir: Path
    archive_members: tuple[str, ...]
    years: tuple[int, ...]
    summer_months: tuple[int, ...]
    development_years: tuple[int, ...]
    holdout_years: tuple[int, ...]
    chunk_size: int


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> Config:
    # TOML is in the Python 3.11+ standard library.
    import tomllib
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    resolve = lambda value: (path.parent / value).resolve()
    config = Config(
        input_archives=tuple(resolve(value) for value in values["input_archives"]),
        boundary_geojson=resolve(values["boundary_geojson"]),
        output_dir=resolve(values["output_dir"]),
        archive_members=tuple(values["archive_members"]),
        years=tuple(int(x) for x in values["years"]),
        summer_months=tuple(int(x) for x in values["summer_months"]),
        development_years=tuple(int(x) for x in values["development_years"]),
        holdout_years=tuple(int(x) for x in values["holdout_years"]),
        chunk_size=int(values["chunk_size"]),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if len(config.input_archives) != len(config.archive_members) or not config.input_archives:
        raise ValueError("input_archives and archive_members must be non-empty lists of equal length")
    for archive in config.input_archives:
        if not archive.is_file():
            raise FileNotFoundError(f"Input archive not found: {archive}")
    if not config.boundary_geojson.is_file():
        raise FileNotFoundError(f"Boundary GeoJSON not found: {config.boundary_geojson}")
    if not config.years or not set(config.development_years).issubset(config.years):
        raise ValueError("Development years must be a subset of years")
    if not set(config.holdout_years).issubset(config.years):
        raise ValueError("Holdout years must be a subset of years")
    if set(config.development_years) & set(config.holdout_years):
        raise ValueError("Development and holdout years must not overlap")
    if not config.summer_months or not set(config.summer_months).issubset(range(1, 13)):
        raise ValueError("summer_months must contain months from 1 through 12")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")


def load_boundary(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("features"):
        raise ValueError("Boundary GeoJSON contains no features")
    boundary = unary_union([shape(feature["geometry"]) for feature in payload["features"]])
    if not boundary.is_valid:
        raise ValueError("Boundary geometry is invalid")
    return boundary


def source_columns(archive: Path, member: str) -> list[str]:
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        if member not in names:
            raise KeyError(f"Archive member {member!r} not found; available: {sorted(names)}")
        with zipped.open(member) as stream:
            return list(pd.read_csv(stream, nrows=0).columns)


def iter_filtered_chunks(config: Config, archive: Path, archive_member: str, boundary) -> Iterable[tuple[pd.DataFrame, dict[str, int]]]:
    prepared = prep(boundary)
    with zipfile.ZipFile(archive) as zipped, zipped.open(archive_member) as stream:
        for chunk in pd.read_csv(stream, chunksize=config.chunk_size):
            required = {"latitude", "longitude", "acq_date"}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")
            dates = pd.to_datetime(chunk["acq_date"], errors="coerce")
            if dates.isna().any():
                raise ValueError(f"Found {int(dates.isna().sum())} invalid acquisition dates")
            years = dates.dt.year
            in_years = years.isin(config.years)
            in_summer = dates.dt.month.isin(config.summer_months)
            coordinates_valid = chunk[["latitude", "longitude"]].notna().all(axis=1)
            inside = np.zeros(len(chunk), dtype=bool)
            valid = coordinates_valid.to_numpy()
            inside[valid] = intersects_xy(
                boundary,
                chunk.loc[valid, "longitude"].to_numpy(),
                chunk.loc[valid, "latitude"].to_numpy(),
            )
            study = in_years & in_summer & pd.Series(inside, index=chunk.index)
            selected = chunk.loc[study].copy()
            selected["acq_date"] = dates.loc[study].dt.strftime("%Y-%m-%d")
            counts = {
                "raw_rows": len(chunk),
                "study_year_rows": int(in_years.sum()),
                "study_summer_rows": int((in_years & in_summer).sum()),
                "summer_inside_bc_rows": int(study.sum()),
                "summer_outside_bc_rows": int((in_years & in_summer & ~pd.Series(inside, index=chunk.index)).sum()),
            }
            yield selected, counts


def append_csv(frame: pd.DataFrame, path: Path, header: bool) -> None:
    frame.to_csv(path, mode="w" if header else "a", header=header, index=False)


def run(config: Config, dry_run: bool = False) -> dict:
    boundary = load_boundary(config.boundary_geojson)
    columns_by_archive = {
        str(archive): source_columns(archive, member)
        for archive, member in zip(config.input_archives, config.archive_members)
    }
    merged_path = config.output_dir / f"bc_firms_summer_{min(config.years)}_{max(config.years)}.csv"
    if not dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if merged_path.exists():
            merged_path.unlink()
    totals = {key: 0 for key in ["raw_rows", "study_year_rows", "study_summer_rows", "summer_inside_bc_rows", "summer_outside_bc_rows"]}
    year_counts: dict[int, int] = {}
    quality = {"null_cells": 0, "duplicate_rows": 0, "invalid_coordinates": 0}
    first_write = True
    for archive, archive_member in zip(config.input_archives, config.archive_members):
      for selected, counts in iter_filtered_chunks(config, archive, archive_member, boundary):
        for key, value in counts.items():
            totals[key] += value
        if selected.empty:
            continue
        dates = pd.to_datetime(selected["acq_date"])
        for year, count in dates.dt.year.value_counts().items():
            year_counts[int(year)] = year_counts.get(int(year), 0) + int(count)
        quality["null_cells"] += int(selected.isna().sum().sum())
        quality["duplicate_rows"] += int(selected.duplicated().sum())
        quality["invalid_coordinates"] += int((~np.isfinite(selected[["latitude", "longitude"]])).any(axis=1).sum())
        if not dry_run and not selected.empty:
            append_csv(selected, merged_path, first_write)
            first_write = False
    if dry_run:
        return {"dry_run": True, "source_columns_by_archive": columns_by_archive, "totals": totals}
    year_frame = pd.DataFrame({"year": sorted(year_counts), "summer_inside_bc_rows": [year_counts[y] for y in sorted(year_counts)]})
    year_frame.to_csv(config.output_dir / "year_counts.csv", index=False)
    (config.output_dir / "quality_checks.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_archives": [str(path) for path in config.input_archives],
        "source_sha256": {str(path): sha256(path) for path in config.input_archives},
        "boundary_geojson": str(config.boundary_geojson),
        "boundary_sha256": sha256(config.boundary_geojson),
        "archive_members": config.archive_members,
        "source_columns_by_archive": columns_by_archive,
        "years": config.years,
        "summer_months": config.summer_months,
        "development_years": config.development_years,
        "holdout_years": config.holdout_years,
        "boundary_rule": "point centre inside or on BC polygon; no buffer",
        "totals": totals,
        "year_counts": year_counts,
        "quality_checks": quality,
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(load_config(args.config.resolve()), dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=list))


if __name__ == "__main__":
    main()

