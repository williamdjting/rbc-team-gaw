"""Visualize H3 Resolution 6 cells across British Columbia for both ERA5 and FIRMS.

Generates two publication-ready map figures in `data/exploration/`:
1. `h3_res6_bc_snapshot_map.png` (4-panel BC map for a target peak fire date, e.g., 2023-08-18):
   - Panel 1 (FIRMS): Binary Fire vs. No Fire (`1 = Fire Detected`, `0 = No Fire`) per H3 res-6 cell
   - Panel 2 (ERA5):  2m Air Temperature (`t2m_c`, °C) per H3 res-6 cell
   - Panel 3 (ERA5):  Vapor Pressure Deficit (`vpd_kpa`, kPa — atmospheric fire aridity) per H3 res-6 cell
   - Panel 4 (ERA5):  Topsoil Moisture Layer 1 (`swvl1`, m³/m³ — surface dryness) per H3 res-6 cell

2. `h3_res6_bc_firms_binary_and_era5_summer_mean.png`:
   - Side-by-side comparison of multi-year Binary Fire Presence (`Ever Fire vs. Never Fire` on the BC H3 res-6 grid)
     alongside peak-season ERA5 weather on the H3 res-6 hexagon grid.

Works either from the outputs of `transform_era5_to_h3_res6.py` / `transform_firms_to_h3_res6.py`
OR directly from `dataset_ERA5_18_24.csv` and `bc_firms_summer_2018_2024.csv` on the fly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
INTERIM_DIR = DATASETS_DIR / "interim"
OUT_DIR = ROOT / "data" / "exploration"

BOUNDARY_CANDIDATES = [
    DATASETS_DIR / "raw" / "bc_legal_boundary.geojson",
    DATASETS_DIR / "bc_legal_boundary.geojson",
    ROOT / "data" / "checks" / "bc_legal_boundary.geojson",
]

ERA5_H3_CANDIDATES = [
    INTERIM_DIR / "era5_h3_res6_cell_date_18_24.parquet",
    INTERIM_DIR / "era5_h3_res6_cell_date_18_23.parquet",
    INTERIM_DIR / "era5_h3_res6_cell_date_18_24.csv",
    INTERIM_DIR / "era5_h3_res6_cell_date_18_23.csv",
]

ERA5_RAW_CANDIDATES = [
    INTERIM_DIR / "dataset_ERA5_18_24.parquet",
    INTERIM_DIR / "dataset_ERA5_18_23.parquet",
    INTERIM_DIR / "dataset_ERA5_18_24.csv",
    INTERIM_DIR / "dataset_ERA5_18_23.csv",
    DATASETS_DIR / "dataset_ERA5_18_24.parquet",
    DATASETS_DIR / "dataset_ERA5_18_24.csv",
    DATASETS_DIR / "dataset_ERA5_18_23.csv",
]

FIRMS_H3_CANDIDATES = [
    INTERIM_DIR / "firms_h3_res6_cell_date_18_24.parquet",
    INTERIM_DIR / "firms_h3_res6_cell_date_18_23.parquet",
    INTERIM_DIR / "firms_h3_res6_cell_date_18_24.csv",
    INTERIM_DIR / "firms_h3_res6_cell_date_18_23.csv",
]

FIRMS_RAW_CANDIDATES = [
    DATASETS_DIR / "firms_summer_2018_2024" / "data" / "bc_firms_summer_2018_2024.csv",
    DATASETS_DIR / "raw" / "bc_firms_summer_2018_2024.csv",
    DATASETS_DIR / "bc_firms_summer_2018_2024.csv",
]


def get_h3_adapters() -> tuple[
    Callable[[float, float, int], str],
    Callable[[str], list[tuple[float, float]]],
]:
    """Return (latlng_to_cell, cell_to_lonlat_polygon) supporting both h3 v3 and v4."""
    try:
        import h3
    except ImportError as error:
        raise RuntimeError(
            "The h3 package is required. Install it with: python -m pip install h3"
        ) from error

    if hasattr(h3, "latlng_to_cell"):  # h3-py v4
        def to_cell(lat: float, lon: float, res: int) -> str:
            return h3.latlng_to_cell(lat, lon, res)

        def to_poly(cell: str) -> list[tuple[float, float]]:
            # cell_to_boundary returns ((lat, lng), ...); convert to (lon, lat) for plotting
            return [( float(lng), float(lat) ) for lat, lng in h3.cell_to_boundary(cell)]

        return to_cell, to_poly

    if hasattr(h3, "geo_to_h3"):  # h3-py v3
        def to_cell(lat: float, lon: float, res: int) -> str:
            return h3.geo_to_h3(lat, lon, res)

        def to_poly(cell: str) -> list[tuple[float, float]]:
            return [( float(lon), float(lat) ) for lat, lon in h3.h3_to_geo_boundary(cell, geo_json=True)]

        return to_cell, to_poly

    raise RuntimeError("Unsupported h3 version: expected latlng_to_cell (v4) or geo_to_h3 (v3)")


def plot_bc_boundary(ax: plt.Axes) -> None:
    """Overlay the official British Columbia legal boundary if available."""
    boundary_path = next((p for p in BOUNDARY_CANDIDATES if p.is_file()), None)
    if boundary_path is None:
        return
    data = json.loads(boundary_path.read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords:
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                ax.plot(xs, ys, color="#222222", linewidth=0.55, alpha=0.8, zorder=5)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    xs = [pt[0] for pt in ring]
                    ys = [pt[1] for pt in ring]
                    ax.plot(xs, ys, color="#222222", linewidth=0.55, alpha=0.8, zorder=5)


def add_era5_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 2m Temperature (°C) and Vapor Pressure Deficit (VPD, kPa) from t2m and d2m."""
    df = df.copy()
    df["t2m_c"] = df["t2m"] - 273.15
    df["d2m_c"] = df["d2m"] - 273.15
    es_kpa = 0.61078 * np.exp((17.27 * df["t2m_c"]) / (df["t2m_c"] + 237.3))
    ea_kpa = 0.61078 * np.exp((17.27 * df["d2m_c"]) / (df["d2m_c"] + 237.3))
    df["vpd_kpa"] = np.maximum(0.0, es_kpa - ea_kpa)
    return df


def load_era5_h3_for_date(
    target_date: str,
    resolution: int,
    to_cell: Callable[[float, float, int], str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Load ERA5 H3 res-6 rows for `target_date` (from precomputed H3 Parquet/CSV or raw ERA5)."""
    h3_path = next((p for p in ERA5_H3_CANDIDATES if p.is_file()), None)
    if h3_path is not None:
        print(f"Loading ERA5 H3 res-{resolution} data for {target_date} from {h3_path.name}...")
        usecols = ["h3_cell", "valid_time", "t2m", "d2m", "swvl1"]
        if h3_path.suffix == ".parquet":
            day_df = pd.read_parquet(
                h3_path,
                columns=usecols,
                filters=[("valid_time", "==", target_date)],
            )
            if day_df.empty:
                raise ValueError(f"Date {target_date} not found in {h3_path}")
            return add_era5_derived_features(day_df)

        parts = []
        for chunk in pd.read_csv(h3_path, usecols=usecols, chunksize=chunk_size):
            matched = chunk[chunk["valid_time"] == target_date]
            if not matched.empty:
                parts.append(matched)
        if not parts:
            raise ValueError(f"Date {target_date} not found in {h3_path}")
        day_df = pd.concat(parts, ignore_index=True)
        return add_era5_derived_features(day_df)

    raw_path = next((p for p in ERA5_RAW_CANDIDATES if p.is_file()), None)
    if raw_path is None:
        raise FileNotFoundError("Could not find ERA5 dataset in datasets/interim/ or datasets/.")

    print(f"Extracting {target_date} from {raw_path.name} and mapping to full BC H3 res-{resolution} grid...")
    usecols = ["valid_time", "latitude", "longitude_deg", "t2m", "d2m", "swvl1"]
    if raw_path.suffix == ".parquet":
        day_df = pd.read_parquet(
            raw_path,
            columns=usecols,
            filters=[("valid_time", "==", target_date)],
        )
        if day_df.empty:
            raise ValueError(f"Date {target_date} not found in {raw_path}")
    else:
        parts = []
        for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=chunk_size):
            matched = chunk[chunk["valid_time"] == target_date]
            if not matched.empty:
                parts.append(matched)
            if parts and chunk["valid_time"].iloc[-1] > target_date:
                break
        if not parts:
            raise ValueError(f"Date {target_date} not found in {raw_path}")
        day_df = pd.concat(parts, ignore_index=True)

    # Use complete BC H3 grid mapping if available (or build it on the fly) so there are no gaps between 0.1° rows
    for grid_map_path in (
        INTERIM_DIR / f"era5_h3_res{resolution}_grid_mapping.parquet",
        INTERIM_DIR / f"era5_h3_res{resolution}_grid_mapping.csv",
    ):
        if grid_map_path.is_file():
            grid_map = pd.read_parquet(grid_map_path) if grid_map_path.suffix == ".parquet" else pd.read_csv(grid_map_path)
            if {"h3_cell", "latitude", "longitude_deg"}.issubset(grid_map.columns) and len(grid_map) > len(day_df):
                merged = grid_map[["h3_cell", "latitude", "longitude_deg"]].merge(
                    day_df, on=["latitude", "longitude_deg"], how="inner"
                )
                return add_era5_derived_features(merged)

    try:
        from transform_era5_to_h3_res6 import build_complete_bc_h3_grid_mapping
        era5_coords = day_df[["latitude", "longitude_deg"]].drop_duplicates().reset_index(drop=True)
        grid_map, _ = build_complete_bc_h3_grid_mapping(era5_coords, resolution=resolution)
        merged = grid_map[["h3_cell", "latitude", "longitude_deg"]].merge(
            day_df, on=["latitude", "longitude_deg"], how="inner"
        )
        return add_era5_derived_features(merged)
    except Exception:
        day_df["h3_cell"] = [
            to_cell(float(lat), float(lon), resolution)
            for lat, lon in zip(day_df["latitude"], day_df["longitude_deg"])
        ]
        day_df = (
            day_df.groupby("h3_cell", as_index=False)
            .agg({"t2m": "mean", "d2m": "mean", "swvl1": "mean"})
        )
        day_df["valid_time"] = target_date
        return add_era5_derived_features(day_df)


def load_firms_h3(
    resolution: int,
    to_cell: Callable[[float, float, int], str],
) -> pd.DataFrame:
    """Load FIRMS H3 cell-date table (from precomputed H3 Parquet/CSV or raw FIRMS CSV)."""
    h3_path = next((p for p in FIRMS_H3_CANDIDATES if p.is_file()), None)
    if h3_path is not None:
        print(f"Loading FIRMS H3 res-{resolution} cell-date table from {h3_path.name}...")
        cols = ["h3_cell", "acq_date", "detection_count", "frp_sum"]
        if h3_path.suffix == ".parquet":
            return pd.read_parquet(h3_path, columns=cols)
        return pd.read_csv(h3_path, usecols=cols)

    raw_path = next((p for p in FIRMS_RAW_CANDIDATES if p.is_file()), None)
    if raw_path is None:
        raise FileNotFoundError("Could not find FIRMS dataset in datasets/.")

    print(f"Loading raw FIRMS from {raw_path.name} and assigning H3 res-{resolution} cells on the fly...")
    firms = pd.read_csv(raw_path, usecols=["latitude", "longitude", "acq_date", "frp"])
    firms["acq_date"] = pd.to_datetime(firms["acq_date"]).dt.strftime("%Y-%m-%d")
    firms["h3_cell"] = [
        to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(firms["latitude"], firms["longitude"])
    ]
    return (
        firms.groupby(["h3_cell", "acq_date"], as_index=False)
        .agg(detection_count=("h3_cell", "size"), frp_sum=("frp", "sum"))
    )


def draw_h3_choropleth(
    ax: plt.Axes,
    polygons: list[list[tuple[float, float]]],
    values: np.ndarray,
    cmap: str,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render true H3 hexagon polygons on a BC map axis."""
    collection = PolyCollection(
        polygons,
        array=values,
        cmap=cmap,
        edgecolors="none",
        antialiaseds=False,
         zorder=2,
    )
    if vmin is not None or vmax is not None:
        collection.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(collection)
    plot_bc_boundary(ax)

    ax.set_xlim(-139.2, -113.8)
    ax.set_ylim(48.2, 60.2)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(54.0)))
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.45)

    cbar = plt.colorbar(collection, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9.5)


def draw_h3_binary_fire_map(
    ax: plt.Axes,
    bg_polygons: list[list[tuple[float, float]]],
    fire_polygons: list[list[tuple[float, float]]],
    title: str,
    fire_label: str,
    no_fire_label: str = "No Fire (0)",
) -> None:
    """Render binary Fire (1) vs. No-Fire (0) H3 hexagons on a BC map axis."""
    bg_collection = PolyCollection(
        bg_polygons,
        facecolors="#e2e6ea",
        edgecolors="none",
        antialiaseds=False,
        zorder=2,
    )
    ax.add_collection(bg_collection)

    if fire_polygons:
        fire_collection = PolyCollection(
            fire_polygons,
            facecolors="#d90429",
            edgecolors="#7f0000",
            linewidths=0.25,
            antialiaseds=True,
            zorder=4,
        )
        ax.add_collection(fire_collection)

    plot_bc_boundary(ax)
    ax.set_xlim(-139.2, -113.8)
    ax.set_ylim(48.2, 60.2)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(54.0)))
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.45)

    legend_handles = [
        Patch(facecolor="#d90429", edgecolor="#7f0000", label=fire_label),
        Patch(facecolor="#e2e6ea", edgecolor="#adb5bd", label=no_fire_label),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=str,
        default="2023-08-18",
        help="Target date (YYYY-MM-DD) for single-dayERA5 & FIRMS H3 comparison (default: 2023-08-18)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Target year for cumulative seasonal binary fire map (default: 2023)",
    )
    parser.add_argument("--resolution", type=int, default=6, help="H3 resolution (default: 6)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    to_cell, to_poly = get_h3_adapters()

    # 1. Load FIRMS H3 data and ERA5 H3 data for the snapshot date
    firms_h3 = load_firms_h3(args.resolution, to_cell)
    era5_day = load_era5_h3_for_date(args.date, args.resolution, to_cell)

    # Precompute hexagon boundary polygons for all active BC H3 cells
    all_bc_cells = pd.Index(era5_day["h3_cell"]).union(pd.Index(firms_h3["h3_cell"].unique()))
    poly_cache = {cell: to_poly(cell) for cell in all_bc_cells}

    era5_polygons = [poly_cache[c] for c in era5_day["h3_cell"]]

    # Binary fire presence on target date
    firms_on_date = set(firms_h3.loc[firms_h3["acq_date"] == args.date, "h3_cell"])
    day_no_fire_cells = [c for c in era5_day["h3_cell"] if c not in firms_on_date]
    day_no_fire_polys = [poly_cache[c] for c in day_no_fire_cells]
    day_fire_polys = [poly_cache[c] for c in firms_on_date if c in poly_cache]

    # ---------------------------------------------------------
    # Figure 1: 4-Panel BC Map (FIRMS Binary Fire + 3 ERA5 Features on Target Date)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 13.5))
    axes = axes.flatten()

    draw_h3_binary_fire_map(
        axes[0],
        bg_polygons=day_no_fire_polys,
        fire_polygons=day_fire_polys,
        title=f"FIRMS Binary Fire Presence on {args.date}\n(H3 Res {args.resolution}: {len(firms_on_date):,} Active Fire Cells)",
        fire_label=f"Fire Detected (1) [n={len(firms_on_date):,}]",
        no_fire_label=f"No Fire (0) [n={len(day_no_fire_cells):,}]",
    )

    draw_h3_choropleth(
        axes[1],
        polygons=era5_polygons,
        values=era5_day["t2m_c"].to_numpy(),
        cmap="RdYlBu_r",
        title=f"ERA5 2m Air Temperature (t2m_c) on {args.date}",
        cbar_label="Temperature (°C)",
        vmin=float(np.nanpercentile(era5_day["t2m_c"], 2)),
        vmax=float(np.nanpercentile(era5_day["t2m_c"], 98)),
    )

    draw_h3_choropleth(
        axes[2],
        polygons=era5_polygons,
        values=era5_day["vpd_kpa"].to_numpy(),
        cmap="YlOrRd",
        title=f"ERA5 Vapor Pressure Deficit (VPD) on {args.date}\n(Atmospheric Fire Stress / Aridity)",
        cbar_label="VPD (kPa)",
        vmin=0.0,
        vmax=float(np.nanpercentile(era5_day["vpd_kpa"], 98)),
    )

    draw_h3_choropleth(
        axes[3],
        polygons=era5_polygons,
        values=era5_day["swvl1"].to_numpy(),
        cmap="YlGnBu",
        title=f"ERA5 Topsoil Volumetric Moisture (swvl1) on {args.date}\n(Lower = Drier Surface Fuels)",
        cbar_label="Soil Moisture (m³/m³)",
        vmin=0.05,
        vmax=0.45,
    )

    fig.suptitle(
        f"British Columbia H3 Resolution {args.resolution} (~36 km² Hexagons) — FIRMS Fire Presence & Key ERA5 Weather ({args.date})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    snapshot_out = OUT_DIR / f"h3_res{args.resolution}_bc_snapshot_{args.date}.png"
    fig.savefig(snapshot_out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 4-panel snapshot map to: {snapshot_out}")

    # ---------------------------------------------------------
    # Figure 2: Full-Season Binary Fire Map (Year) vs. Multi-Year Binary Fire Map (2018-2023)
    # ---------------------------------------------------------
    firms_h3["year"] = pd.to_datetime(firms_h3["acq_date"]).dt.year
    year_fire_cells = set(firms_h3.loc[firms_h3["year"] == args.year, "h3_cell"])
    dev_fire_cells = set(firms_h3.loc[firms_h3["year"] <= 2023, "h3_cell"])

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7.2))

    year_no_fire_polys = [poly_cache[c] for c in era5_day["h3_cell"] if c not in year_fire_cells]
    year_fire_polys = [poly_cache[c] for c in year_fire_cells if c in poly_cache]
    draw_h3_binary_fire_map(
        axes2[0],
        bg_polygons=year_no_fire_polys,
        fire_polygons=year_fire_polys,
        title=f"FIRMS Binary Fire Map — Summer {args.year} (May–Sep)\n({len(year_fire_cells):,} H3 Res-{args.resolution} Cells Burned)",
        fire_label=f"Fire in {args.year} (1) [n={len(year_fire_cells):,}]",
        no_fire_label="No Fire (0)",
    )

    dev_no_fire_polys = [poly_cache[c] for c in era5_day["h3_cell"] if c not in dev_fire_cells]
    dev_fire_polys = [poly_cache[c] for c in dev_fire_cells if c in poly_cache]
    draw_h3_binary_fire_map(
        axes2[1],
        bg_polygons=dev_no_fire_polys,
        fire_polygons=dev_fire_polys,
        title=f"FIRMS Cumulative Binary Fire Footprint (2018–2023)\n({len(dev_fire_cells):,} Unique H3 Res-{args.resolution} Cells with ≥1 Detection)",
        fire_label=f"Ever Fire 2018–2023 (1) [n={len(dev_fire_cells):,}]",
        no_fire_label="Never Fire 2018–2023 (0)",
    )

    fig2.suptitle(
        f"BC Wildfire Footprint at H3 Resolution {args.resolution} (Binary Fire vs. No Fire)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig2.tight_layout()
    footprint_out = OUT_DIR / f"h3_res{args.resolution}_bc_firms_binary_footprint.png"
    fig2.savefig(footprint_out, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved binary fire footprint map to: {footprint_out}")


if __name__ == "__main__":
    main()
