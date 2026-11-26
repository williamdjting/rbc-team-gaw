"""Rebuild ERA5 wind from the 5-hourly monthly archives (00, 05, 10, 15, 20 UTC) at H3 res 7.

Stage 1 (grid): read u10 and v10 from data_1.nc in each monthly zip in datasets/ERA5 dataset/ and write one
parquet per month with the 0.1 degree grid, the valid time, u10, v10 and speed.
Stage 2 (cells): map every res-7 cell in the six Fire Centre grids to its nearest 0.1 degree point, then
write one parquet per year with one row per cell and UTC day:
  wind_speed_day_mean   mean of the five speeds on the day (speed from each snapshot, not averaged vectors)
  wind_speed_day_max    largest of the five speeds
  wind_pm_u, wind_pm_v  afternoon vector mean of the 20 UTC snapshot (13:00 PDT) and the next day's 00 UTC
                        snapshot (17:00 PDT), i.e. the 13:00-17:00 local window
  wind_pm_speed         speed of that afternoon vector mean
  wind_pm_dir_deg       direction the wind blows toward, in degrees clockwise from north

Gusts and rain are not in these files and are not built here.

usage: python build_era5_wind_5h.py REPO_ROOT [--years 2018 ... 2024] [--stage grid|cells|all]
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import h3
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

REGIONS = ["cariboo", "coastal", "kamloops", "northwest", "prince_george", "southeast"]
MONTHS = [5, 6, 7, 8, 9]


def read_zip_wind(zip_path: Path, tmp: Path) -> pd.DataFrame:
    """u10, v10 from data_1.nc inside one monthly archive."""
    with zipfile.ZipFile(zip_path) as z:
        z.extract("data_1.nc", tmp)
    with h5py.File(tmp / "data_1.nc", "r") as h:
        vt = h["valid_time"][:]
        if vt.dtype.kind in "iu":
            vt = pd.to_datetime(vt, unit="s").to_numpy()
        lat = h["latitude"][:]
        lon = h["longitude"][:]
        u = h["u10"][:]
        v = h["v10"][:]
    nt, nlat, nlon = u.shape
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    df = pd.DataFrame({
        "lat": np.tile(lat_g.ravel(), nt),
        "lon": np.tile(lon_g.ravel(), nt),
        "valid_time": np.repeat(pd.to_datetime(vt), nlat * nlon),
        "u10": u.reshape(-1),
        "v10": v.reshape(-1),
    })
    df["speed"] = np.hypot(df.u10, df.v10)
    (tmp / "data_1.nc").unlink()
    return df


def stage_grid(root: Path, years: list[int]) -> None:
    src_dir = root / "datasets/ERA5 dataset"
    out_dir = root / "data/era5_wind_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in sorted(src_dir.glob("*.zip")):
        tmp = Path(tempfile.mkdtemp(prefix="era5wind_"))
        try:
            with zipfile.ZipFile(zip_path) as z:
                if "data_1.nc" not in z.namelist():
                    continue
            df = read_zip_wind(zip_path, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        month = df.valid_time.dt.month.iloc[0]
        year = df.valid_time.dt.year.iloc[0]
        if year not in years or month not in MONTHS:
            continue
        path = out_dir / f"wind_{year}{month:02d}.parquet"
        df.to_parquet(path, index=False)
        print(f"grid {year}-{month:02d}: {len(df):,} rows from {zip_path.name}", flush=True)


def region_cells(root: Path) -> pd.DataFrame:
    grids = [pd.read_parquet(root / f"experiments/02_hotspot_persistence_bc/regions/{r}/data/h3_grid.parquet",
                             columns=["h3_cell"]) for r in REGIONS]
    cells = pd.concat(grids).h3_cell.drop_duplicates().reset_index(drop=True).to_frame("h3_cell")
    ll = np.array([h3.cell_to_latlng(c) for c in cells.h3_cell])
    cells["lat"], cells["lon"] = ll[:, 0], ll[:, 1]
    return cells


def stage_cells(root: Path, years: list[int]) -> None:
    grid_dir = root / "data/era5_wind_grid"
    out_dir = root / "data/era5_wind_h3r7"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = region_cells(root)
    # Nearest 0.1 degree point, with longitude scaled by cos(latitude) so distances are roughly isotropic.
    lat0 = np.radians(cells.lat.mean())
    scale = np.array([1.0, np.cos(lat0)])
    grid_pts = None
    for year in years:
        frames = []
        for month in MONTHS:
            path = grid_dir / f"wind_{year}{month:02d}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            print(f"cells {year}: no grid files, skipped", flush=True)
            continue
        g = pd.concat(frames, ignore_index=True)
        if grid_pts is None:
            pts = g[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
            tree = cKDTree(pts[["lat", "lon"]].to_numpy() * scale)
            _, idx = tree.query(cells[["lat", "lon"]].to_numpy() * scale)
            cells["grid_id"] = idx
            grid_pts = pts
        pid = {(a, b): i for i, (a, b) in enumerate(grid_pts[["lat", "lon"]].to_numpy())}
        g["grid_id"] = [pid[(a, b)] for a, b in zip(g.lat, g.lon)]
        g["date"] = g.valid_time.dt.normalize()
        g["hour"] = g.valid_time.dt.hour
        # Daily features per grid point (small table: grid points x days).
        day = g.groupby(["grid_id", "date"]).agg(
            wind_speed_day_mean=("speed", "mean"), wind_speed_day_max=("speed", "max")).reset_index()
        # Afternoon window: 20 UTC of day d and 00 UTC of day d+1 (13:00 and 17:00 PDT).
        pm20 = g[g.hour == 20][["grid_id", "date", "u10", "v10"]]
        pm00 = g[g.hour == 0][["grid_id", "date", "u10", "v10"]].copy()
        pm00["date"] = pm00.date - pd.Timedelta(days=1)
        pm = pd.concat([pm20, pm00]).groupby(["grid_id", "date"]).agg(
            wind_pm_u=("u10", "mean"), wind_pm_v=("v10", "mean")).reset_index()
        pm["wind_pm_speed"] = np.hypot(pm.wind_pm_u, pm.wind_pm_v)
        # Direction the wind blows toward, clockwise from north.
        pm["wind_pm_dir_deg"] = (np.degrees(np.arctan2(pm.wind_pm_u, pm.wind_pm_v)) + 360) % 360
        gd = day.merge(pm, on=["grid_id", "date"], how="left")
        gd = gd[gd.date.dt.year == year]
        del g

        # Map to cells in chunks, so the cell x day table never has to fit in memory at once.
        path = out_dir / f"wind_{year}.parquet"
        writer = None
        for start in range(0, len(cells), 20_000):
            part = cells.iloc[start:start + 20_000][["h3_cell", "grid_id"]]
            out = part.merge(gd, on="grid_id", how="inner").drop(columns="grid_id")
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
        print(f"cells {year}: {len(cells):,} cells -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2018, 2025)))
    ap.add_argument("--stage", choices=["grid", "cells", "all"], default="all")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.stage in ("grid", "all"):
        stage_grid(root, args.years)
    if args.stage in ("cells", "all"):
        stage_cells(root, args.years)


if __name__ == "__main__":
    main()
