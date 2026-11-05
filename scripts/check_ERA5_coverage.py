"""
Load, standardize, clip to the official British Columbia legal boundary, and combine
all 35 monthly ERA5-Land ZIP/NetCDF files (2018-2024, May-September) into clean daily
BC land datasets (`dataset_ERA5_18_24`, `dataset_ERA5_18_23`, `dataset_ERA5_2024_holdout`).

Mirrors the exact workflow of `scripts/check_CEMS_coverage.py`:
1. Fetch & prepare the official BC Legal Boundary (`datasets/bc_legal_boundary.geojson`).
2. Extract & combine the monthly ERA5-Land archives from `datasets/ERA5 dataset/`:
   - `data_0.nc` (`swvl1`, `swvl2`, `swvl3`)
   - `data_1.nc` (`lai_lv`, `lai_hv`, `sp`, `slhf`, `u10`, `v10`, `t2m`, `d2m`, `ssrd`, `ssr`, `str`, `tp`, `skt`, `pev`)
   - `data_2.nc` (`sde`, `evatc`, `evabs`, `evavt`, `snowc`)
   - Static land/topography fields: `cvh.nc` (`cvh`), `cvl.nc` (`cvl`), `geo.nc` (`z`), `lsm.nc` (`lsm`)
3. Standardize longitude to (-180 to 180) as `longitude_deg` (handling both -180..180 in
   `data_0/1.nc` and 0..360 in `data_2.nc` and static files) and slice to the BC bounding box.
4. Filter grid points to inside the BC legal polygon and on land (`t2m` not null).
5. Aggregate the 5 daily timestamps (00:00, 05:00, 10:00, 15:00, 20:00 UTC) to daily resolution
   (`valid_time` date) to match `dataset_CEMS_18_24.csv` and FIRMS `acq_date`.
6. Print the BC filtering audit summary and save Full (2018-2024), Dev (2018-2023), and
   Holdout (2024) datasets.
"""

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import requests
from shapely import intersects_xy, prepare
from shapely.geometry import shape
from shapely.ops import unary_union
import xarray as xr

# ---------------------------------------------------------
# Dynamic Paths Setup
# ROOT resolves to: /Users/arshdeepkaur/wildfire_detection/rbc-team-gaw
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

ERA5_DIR = DATASETS_DIR / "ERA5 dataset"
if not ERA5_DIR.exists():
    ERA5_DIR = ROOT.parent / "ERA5_BC" / "ERA5_BC_Data"

# BC Bounding Box (South, North, West, East) at 0.1° ERA5-Land resolution
LAT_MIN, LAT_MAX = 48.3, 60.0
LON_MIN, LON_MAX = -139.1, -114.0

TIME_VARS_BY_FILE = {
    "data_0.nc": ["swvl1", "swvl2", "swvl3"],
    "data_1.nc": [
        "lai_lv", "lai_hv", "sp", "slhf", "u10", "v10",
        "t2m", "d2m", "ssrd", "ssr", "str", "tp", "skt", "pev",
    ],
    "data_2.nc": ["sde", "evatc", "evabs", "evavt", "snowc"],
}
STATIC_VARS_BY_FILE = {
    "cvh.nc": "cvh",
    "cvl.nc": "cvl",
    "geo.nc": "z",
    "lsm.nc": "lsm",
}


def open_standardized_bc_nc(nc_path: Path) -> xr.Dataset:
    """Open a NetCDF file, standardize longitude to [-180, 180], sort, and slice to BC bbox."""
    ds = xr.open_dataset(nc_path)
    lon_deg = np.round(((ds["longitude"].values + 180.0) % 360.0) - 180.0, 2)
    lat_deg = np.round(ds["latitude"].values, 2)
    ds = ds.assign_coords(latitude=lat_deg, longitude=lon_deg).sortby(["latitude", "longitude"])
    ds = ds.sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX))
    return ds


# ---------------------------------------------------------
# 1. Fetch & Prepare Official BC Legal Boundary (DataBC)
# ---------------------------------------------------------
boundary_file = DATASETS_DIR / "bc_legal_boundary.geojson"

if not boundary_file.exists():
    print(f"Downloading BC legal boundary from DataBC to {boundary_file}...")
    SERVICE = "https://delivery.maps.gov.bc.ca/arcgis/rest/services/mpcm/bcgwpub/MapServer/537/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    resp = requests.get(SERVICE, params=params, timeout=60)
    resp.raise_for_status()
    boundary_file.write_text(resp.text, encoding="utf-8")

print(f"Loading and preparing BC boundary geometry from {boundary_file.name}...")
boundary_data = json.loads(boundary_file.read_text(encoding="utf-8"))
bc_polygon = unary_union([shape(f["geometry"]) for f in boundary_data["features"]])
prepare(bc_polygon)
print("BC Legal boundary ready.")

# ---------------------------------------------------------
# 2. Discover ERA5 Monthly ZIP Archives (2018-2024)
# ---------------------------------------------------------
zip_files = sorted(ERA5_DIR.glob("*.zip"))
if not zip_files:
    raise FileNotFoundError(f"No ERA5 .zip files found in {ERA5_DIR}")
print(f"Found {len(zip_files)} ERA5 monthly ZIP archives in {ERA5_DIR}")

# ---------------------------------------------------------
# 3 & 4. Build Standardized Grid (-180..180), BC Polygon Mask & Static Features
# ---------------------------------------------------------
print("Filtering 0.1° ERA5 grid coordinates against BC boundary and land mask...")
with tempfile.TemporaryDirectory() as tmp_init:
    tmp_init_path = Path(tmp_init)
    with zipfile.ZipFile(zip_files[0]) as zf:
        zf.extractall(tmp_init_path)

    ds_ref = open_standardized_bc_nc(tmp_init_path / "data_1.nc")
    lats = ds_ref["latitude"].values.astype(np.float32)
    lons_deg = ds_ref["longitude"].values.astype(np.float32)
    lons_360 = np.round(lons_deg % 360.0, 2).astype(np.float32)

    lon_grid, lat_grid = np.meshgrid(lons_deg, lats)
    lon_360_grid, _ = np.meshgrid(lons_360, lats)

    inside_bc_mask = intersects_xy(bc_polygon, lon_grid, lat_grid)
    land_not_null_mask = ~np.isnan(ds_ref["t2m"].values[0])
    valid_grid_mask = inside_bc_mask & land_not_null_mask
    ds_ref.close()

    n_bbox_points = int(inside_bc_mask.size)
    n_valid_points = int(valid_grid_mask.sum())
    valid_lats = lat_grid[valid_grid_mask]
    valid_lons_deg = lon_grid[valid_grid_mask]
    valid_lons_360 = lon_360_grid[valid_grid_mask]

    # Extract static invariant variables (cvh, cvl, z, lsm) on the valid BC land points
    static_values = {}
    for fname, var_name in STATIC_VARS_BY_FILE.items():
        fpath = tmp_init_path / fname
        if fpath.exists():
            ds_stat = open_standardized_bc_nc(fpath)
            arr_2d = np.squeeze(ds_stat[var_name].values).astype(np.float32)
            static_values[var_name] = arr_2d[valid_grid_mask]
            ds_stat.close()

print(
    f"0.1° Grid points in bounding box: {n_bbox_points:,} | "
    f"Valid BC land points kept: {n_valid_points:,} ({n_valid_points / n_bbox_points * 100:.1f}%)"
)

# ---------------------------------------------------------
# 5. Load, Combine & Aggregate All 35 Monthly Files to Daily BC Rows
# ---------------------------------------------------------
monthly_frames = []
total_raw_daily_bbox_rows = 0

with tempfile.TemporaryDirectory() as tmp_work:
    tmp_work_path = Path(tmp_work)
    for idx, zpath in enumerate(zip_files, start=1):
        with zipfile.ZipFile(zpath) as zf:
            for fname in TIME_VARS_BY_FILE:
                zf.extract(fname, tmp_work_path)

        month_vars = {}
        dates = None
        for fname, var_list in TIME_VARS_BY_FILE.items():
            ds = open_standardized_bc_nc(tmp_work_path / fname)
            if dates is None:
                raw_times = pd.to_datetime(ds["valid_time"].values)
                dates = raw_times.floor("D").unique()
                day_codes = (raw_times.floor("D") - dates[0]).days.to_numpy(dtype=np.int32)
                n_days = len(dates)

            for var_name in var_list:
                # Extract only valid BC land cells: shape (n_timestamps, n_valid_points)
                vals = ds[var_name].values[:, valid_grid_mask].astype(np.float32)
                # Aggregate 5 daily timestamps to daily mean per (date, grid_point)
                if len(vals) == n_days * 5:
                    daily_vals = vals.reshape(n_days, 5, n_valid_points).mean(axis=1)
                else:
                    daily_vals = np.zeros((n_days, n_valid_points), dtype=np.float32)
                    counts = np.bincount(day_codes, minlength=n_days).astype(np.float32)[:, None]
                    np.add.at(daily_vals, day_codes, vals)
                    daily_vals /= np.maximum(1.0, counts)
                month_vars[var_name] = daily_vals.ravel()
            ds.close()

        total_raw_daily_bbox_rows += n_days * n_bbox_points
        ym_str = str(dates[0])[:7]
        print(f"  [{idx:2d}/{len(zip_files)}] Processed {ym_str} ({zpath.name}): {n_days} days x {n_valid_points:,} BC cells", flush=True)

        month_df = pd.DataFrame({
            "valid_time": np.repeat(dates, n_valid_points),
            "latitude": np.tile(valid_lats, n_days),
            "longitude": np.tile(valid_lons_360, n_days),
            "longitude_deg": np.tile(valid_lons_deg, n_days),
            **month_vars,
        })
        for var_name, stat_arr in static_values.items():
            month_df[var_name] = np.tile(stat_arr, n_days)

        monthly_frames.append(month_df)

dataset_18_24 = pd.concat(monthly_frames, ignore_index=True)
del monthly_frames

# Sort chronologically and add calendar features matching check_CEMS_coverage.py
dataset_18_24["valid_time"] = pd.to_datetime(dataset_18_24["valid_time"])
dataset_18_24.sort_values(["valid_time", "latitude", "longitude_deg"], inplace=True, ignore_index=True)
dataset_18_24["year"] = dataset_18_24["valid_time"].dt.year.astype(np.int16)
dataset_18_24["month"] = dataset_18_24["valid_time"].dt.month.astype(np.int8)

# ---------------------------------------------------------
# 6. Audit Summary
# ---------------------------------------------------------
print("\n" + "=" * 45)
print("=== ERA5 BC FILTERING SUMMARY (2018-2024) ===")
print("=" * 45)
print(f"Total raw bounding box rows: {total_raw_daily_bbox_rows:,}")
print(
    f"Valid BC rows kept:          {len(dataset_18_24):,} "
    f"({(len(dataset_18_24) / total_raw_daily_bbox_rows * 100):.1f}%)"
)
print(f"Rows removed (ocean/outside):{(total_raw_daily_bbox_rows - len(dataset_18_24)):,}")
print(f"Remaining null values (t2m): {int(dataset_18_24['t2m'].isnull().sum())}")

print("\nValid BC Rows by Year:")
print(dataset_18_24["year"].value_counts().sort_index())

# ---------------------------------------------------------
# 7. Save Valid BC Data (Full 2018-2024, Dev 2018-2023, and 2024 Holdout)
# ---------------------------------------------------------
output_full_parquet = DATASETS_DIR / "dataset_ERA5_18_24.parquet"
output_dev_parquet = DATASETS_DIR / "dataset_ERA5_18_23.parquet"
output_holdout_parquet = DATASETS_DIR / "dataset_ERA5_2024_holdout.parquet"

print(f"\nSaving full 2018-2024 BC ERA5 dataset to {output_full_parquet}...")
dataset_18_24.to_parquet(output_full_parquet, index=False)

dataset_18_24[dataset_18_24["year"] <= 2023].to_parquet(output_dev_parquet, index=False)
print(f"Saved 2018-2023 dev dataset to {output_dev_parquet}")

dataset_18_24[dataset_18_24["year"] == 2024].to_parquet(output_holdout_parquet, index=False)
print(f"Saved 2024 holdout dataset to {output_holdout_parquet}")

output_full_csv = DATASETS_DIR / "dataset_ERA5_18_24.csv"
output_dev_csv = DATASETS_DIR / "dataset_ERA5_18_23.csv"
output_holdout_csv = DATASETS_DIR / "dataset_ERA5_2024_holdout.csv"

print(f"Saving CSV versions to {output_full_csv.name}, {output_dev_csv.name}, {output_holdout_csv.name}...")
dataset_18_24.to_csv(output_full_csv, index=False)
dataset_18_24[dataset_18_24["year"] <= 2023].to_csv(output_dev_csv, index=False)
dataset_18_24[dataset_18_24["year"] == 2024].to_csv(output_holdout_csv, index=False)
print("Saved successfully!")

