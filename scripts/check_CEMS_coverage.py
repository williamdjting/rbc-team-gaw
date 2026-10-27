import json
from pathlib import Path
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
# 2. Load & Combine Your CEMS NetCDF Files (2018-2024)
# ---------------------------------------------------------
print("Loading CEMS NetCDF files...")
# Checks for the files either in datasets/ or project root
nc_file_1 = DATASETS_DIR / "6e1134df5a293fa7621174aea2a2cbc8.nc"
if not nc_file_1.exists():
    nc_file_1 = ROOT.parent / "6e1134df5a293fa7621174aea2a2cbc8.nc"

nc_file_2 = DATASETS_DIR / "303b53fd5f8792b16b03fcf152073514.nc"
if not nc_file_2.exists():
    nc_file_2 = ROOT.parent / "303b53fd5f8792b16b03fcf152073514.nc"

nc_file_3 = DATASETS_DIR / "CEMS2024.nc"
if not nc_file_3.exists():
    nc_file_3 = ROOT.parent / "CEMS2024.nc"

ds1 = xr.open_dataset(nc_file_1)  # 2018-2020
ds2 = xr.open_dataset(nc_file_2)  # 2021-2023
ds3 = xr.open_dataset(nc_file_3)  # 2024
ds = xr.concat([ds1, ds2, ds3], dim="valid_time")

# Convert to DataFrame
df = ds.to_dataframe().reset_index()

# ---------------------------------------------------------
# 3. Standardize Longitude to (-180 to 180)
# (CEMS uses 0–360; BC is around -114° to -139°)
# ---------------------------------------------------------
df["longitude_deg"] = ((df["longitude"] + 180) % 360) - 180

# ---------------------------------------------------------
# 4. Filter Grid Points: Inside BC Polygon & Not Ocean
# ---------------------------------------------------------
print("Filtering grid coordinates against BC boundary...")
# Compute polygon check once per unique lat/lon pair (fast)
unique_grid = df[["latitude", "longitude_deg"]].drop_duplicates().copy()
lats = unique_grid["latitude"].to_numpy()
lons = unique_grid["longitude_deg"].to_numpy()

unique_grid["inside_bc"] = intersects_xy(bc_polygon, lons, lats)

# Merge mask back to full DataFrame
df = df.merge(unique_grid, on=["latitude", "longitude_deg"], how="left")

# Keep only rows inside BC and on land (fwinx is not null)
dataset_18_24 = df[df["inside_bc"] & df["fwinx"].notnull()].copy()

# Clean up helper columns and add calendar features
dataset_18_24.drop(columns=["inside_bc", "surface"], inplace=True, errors="ignore")
dataset_18_24["valid_time"] = pd.to_datetime(dataset_18_24["valid_time"])
dataset_18_24["year"] = dataset_18_24["valid_time"].dt.year
dataset_18_24["month"] = dataset_18_24["valid_time"].dt.month

# ---------------------------------------------------------
# 5. Audit Summary
# ---------------------------------------------------------
print("\n" + "=" * 45)
print("=== CEMS BC FILTERING SUMMARY (2018-2024) ===")
print("=" * 45)
print(f"Total raw bounding box rows: {len(df):,}")
print(f"Valid BC rows kept:          {len(dataset_18_24):,} ({(len(dataset_18_24) / len(df) * 100):.1f}%)")
print(f"Rows removed (ocean/outside):{(len(df) - len(dataset_18_24)):,}")
print(f"Remaining null values:       {dataset_18_24['fwinx'].isnull().sum()}")

print("\nValid BC Rows by Year:")
print(dataset_18_24["year"].value_counts().sort_index())

# ---------------------------------------------------------
# 6. Save Valid BC Data (Full 2018-2024, Dev 2018-2023, and 2024 Holdout)
# ---------------------------------------------------------
output_full = DATASETS_DIR / "dataset_CEMS_18_24.csv"
print(f"\nSaving full 2018-2024 BC dataset to {output_full}...")
dataset_18_24.to_csv(output_full, index=False)

output_dev = DATASETS_DIR / "dataset_CEMS_18_23.csv"
dataset_18_24[dataset_18_24["year"] <= 2023].to_csv(output_dev, index=False)
print(f"Saved 2018-2023 dev dataset to {output_dev}")

output_holdout = DATASETS_DIR / "dataset_CEMS_2024_holdout.csv"
dataset_18_24[dataset_18_24["year"] == 2024].to_csv(output_holdout, index=False)
print(f"Saved 2024 holdout dataset to {output_holdout}")
print("Saved successfully!")