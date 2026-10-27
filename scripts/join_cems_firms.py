"""
Join the 4 shortlisted CEMS Fire Weather features (fwinx, ffmcode, drtcode, infsinx)
with the NASA FIRMS BC summer datasets (2018-2023) based on latitude, longitude, and date.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------
# 1. Setup Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
FIRMS_DATA_DIR = DATASETS_DIR / "firms_summer_2018_2024" / "data"

CEMS_CSV = DATASETS_DIR / "dataset_CEMS_18_24.csv"
if not CEMS_CSV.exists():
    CEMS_CSV = DATASETS_DIR / "dataset_CEMS_18_23.csv"
FIRMS_DETECTIONS_CSV = FIRMS_DATA_DIR / "firms_h3_detections.csv"
FIRMS_H3_DAILY_CSV = FIRMS_DATA_DIR / "firms_h3_cell_date.csv"

OUT_DETECTIONS_18_24 = DATASETS_DIR / "firms_with_cems_18_24.csv"
OUT_DETECTIONS_18_23 = DATASETS_DIR / "firms_with_cems_18_23.csv"
OUT_DETECTIONS_2024 = DATASETS_DIR / "firms_with_cems_2024_holdout.csv"

OUT_H3_DAILY_18_24 = DATASETS_DIR / "firms_h3_cell_date_with_cems_18_24.csv"
OUT_H3_DAILY_18_23 = DATASETS_DIR / "firms_h3_cell_date_with_cems_18_23.csv"
OUT_H3_DAILY_2024 = DATASETS_DIR / "firms_h3_cell_date_with_cems_2024_holdout.csv"

SHORTLISTED_CEMS = ["fwinx", "ffmcode", "drtcode", "infsinx"]


def build_spatial_mapper(cems_grid: pd.DataFrame, target_coords: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Map arbitrary (lat, lon) coordinates to the nearest valid CEMS 0.25° BC land grid cell
    using a fast KD-Tree so even coastal/border points get valid land weather.
    """
    unique_targets = target_coords[[lat_col, lon_col]].drop_duplicates().copy()

    # Convert degrees to approximate radians/cosine-scaled coordinates for accurate BC distance
    cos_lat = np.cos(np.deg2rad(54.0))
    cems_xy = np.column_stack([cems_grid["cems_lat"], cems_grid["cems_lon"] * cos_lat])
    target_xy = np.column_stack([unique_targets[lat_col], unique_targets[lon_col] * cos_lat])

    tree = cKDTree(cems_xy)
    distances_deg, indices = tree.query(target_xy, k=1)

    unique_targets["cems_lat"] = cems_grid["cems_lat"].to_numpy()[indices]
    unique_targets["cems_lon"] = cems_grid["cems_lon"].to_numpy()[indices]
    unique_targets["cems_dist_km"] = (distances_deg * 111.32).round(2)

    return unique_targets


def main():
    # ---------------------------------------------------------
    # 2. Load Shortlisted CEMS Features (2018-2023)
    # ---------------------------------------------------------
    print(f"Loading CEMS dataset from {CEMS_CSV.name}...")
    cems = pd.read_csv(
        CEMS_CSV,
        usecols=["valid_time", "latitude", "longitude_deg"] + SHORTLISTED_CEMS,
    )
    cems = cems.rename(
        columns={
            "valid_time": "date",
            "latitude": "cems_lat",
            "longitude_deg": "cems_lon",
        }
    )
    cems["date"] = pd.to_datetime(cems["date"]).dt.strftime("%Y-%m-%d")

    # Unique CEMS BC land grid coordinates (2,277 cells)
    cems_grid = cems[["cems_lat", "cems_lon"]].drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(cems):,} CEMS rows across {len(cems_grid):,} unique BC grid cells.")

    # ---------------------------------------------------------
    # 3. Join onto Raw FIRMS Detections (2018-2024)
    # ---------------------------------------------------------
    if FIRMS_DETECTIONS_CSV.exists():
        print(f"\nLoading FIRMS detections from {FIRMS_DETECTIONS_CSV.name}...")
        firms = pd.read_csv(FIRMS_DETECTIONS_CSV)

        firms["year"] = pd.to_datetime(firms["acq_date"]).dt.year
        firms_18_24 = firms[firms["year"].between(2018, 2024)].copy()

        print("Mapping FIRMS coordinates to nearest CEMS grid cells...")
        coord_map = build_spatial_mapper(cems_grid, firms_18_24, "latitude", "longitude")
        firms_18_24 = firms_18_24.merge(coord_map, on=["latitude", "longitude"], how="left")

        print("Joining shortlisted CEMS weather features by (date, cems_lat, cems_lon)...")
        firms_joined = firms_18_24.merge(
            cems,
            left_on=["acq_date", "cems_lat", "cems_lon"],
            right_on=["date", "cems_lat", "cems_lon"],
            how="left",
        ).drop(columns=["date"])

        null_weather = firms_joined["fwinx"].isna().sum()
        print(f"Total joined FIRMS detection rows (2018-2024): {len(firms_joined):,}")
        print(f"Missing CEMS values after join: {null_weather}")
        print(f"Median distance to CEMS grid centre: {firms_joined['cems_dist_km'].median():.2f} km")

        firms_joined.drop(columns=["year"]).to_csv(OUT_DETECTIONS_18_24, index=False)
        firms_joined[firms_joined["year"] <= 2023].drop(columns=["year"]).to_csv(OUT_DETECTIONS_18_23, index=False)
        firms_joined[firms_joined["year"] == 2024].drop(columns=["year"]).to_csv(OUT_DETECTIONS_2024, index=False)
        print(f"Saved joined detections to: {OUT_DETECTIONS_18_24.name}, {OUT_DETECTIONS_18_23.name}, {OUT_DETECTIONS_2024.name}")

    # ---------------------------------------------------------
    # 4. Join onto Aggregated H3 Cell-Date Table (2018-2024)
    # ---------------------------------------------------------
    if FIRMS_H3_DAILY_CSV.exists():
        print(f"\nLoading H3 daily cell table from {FIRMS_H3_DAILY_CSV.name}...")
        h3_df = pd.read_csv(FIRMS_H3_DAILY_CSV)
        h3_df["year"] = pd.to_datetime(h3_df["acq_date"]).dt.year
        h3_18_24 = h3_df[h3_df["year"].between(2018, 2024)].copy()

        coord_map_h3 = build_spatial_mapper(cems_grid, h3_18_24, "latitude_mean", "longitude_mean")
        h3_18_24 = h3_18_24.merge(coord_map_h3, on=["latitude_mean", "longitude_mean"], how="left")

        h3_joined = h3_18_24.merge(
            cems,
            left_on=["acq_date", "cems_lat", "cems_lon"],
            right_on=["date", "cems_lat", "cems_lon"],
            how="left",
        ).drop(columns=["date"])

        print(f"Total joined H3 cell-date rows (2018-2024): {len(h3_joined):,}")
        print(f"Missing CEMS values after join: {h3_joined['fwinx'].isna().sum()}")

        h3_joined.drop(columns=["year"]).to_csv(OUT_H3_DAILY_18_24, index=False)
        h3_joined[h3_joined["year"] <= 2023].drop(columns=["year"]).to_csv(OUT_H3_DAILY_18_23, index=False)
        h3_joined[h3_joined["year"] == 2024].drop(columns=["year"]).to_csv(OUT_H3_DAILY_2024, index=False)
        print(f"Saved joined H3 cell-date datasets to: {OUT_H3_DAILY_18_24.name}, {OUT_H3_DAILY_18_23.name}, {OUT_H3_DAILY_2024.name}")

    # ---------------------------------------------------------
    # 5. Quick Summary of Shortlisted CEMS Features During Fires
    # ---------------------------------------------------------
    print("\n=== CEMS Shortlisted Features Summary on Active Fire Detections (2018-2023) ===")
    print(firms_joined[SHORTLISTED_CEMS].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T.round(2))


if __name__ == "__main__":
    main()