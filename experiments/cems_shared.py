"""Shared utilities for Experiments 05 and 06 using `datasets/firms_with_cems_18_24.csv`.

Handles:
1. Loading `datasets/firms_with_cems_18_24.csv` (the joined FIRMS + CEMS dataset)
   alongside `datasets/dataset_CEMS_18_24.csv` (to supply CEMS weather on the 99.7%
   of H3 cell-days that have 0 FIRMS detections), strictly filtering out 2024 (`year < 2024`).
2. Automatically downloading `data/region_exploration/fire_centres.geojson` and building
   the regional H3 grids (`02_hotspot_persistence_bc/regions/`) from `firms_with_cems_18_24.csv`
   if they have not been generated locally yet.
3. Parsing the teammate's saved 2023 baseline metrics from `04_exp_gnn_bc/RESULTS.html`
   when `03_exp_learned_baselines_bc/results/` or `04_exp_gnn_bc/results/` are gitignored
   and not on disk, so Experiments 05 and 06 can be run directly without re-running 03 or 04.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENTS_DIR.parent

FIRMS_WITH_CEMS_PATH = ROOT / "datasets/firms_with_cems_18_24.csv"
CEMS_GRID_18_24_PATH = ROOT / "datasets/dataset_CEMS_18_24.csv"
CEMS_GRID_18_23_PATH = ROOT / "datasets/dataset_CEMS_18_23.csv"
FIRE_CENTRES_GEOJSON = ROOT / "data/region_exploration/fire_centres.geojson"
EXPERIMENT_02 = EXPERIMENTS_DIR / "02_hotspot_persistence_bc"
EXPERIMENT_03 = EXPERIMENTS_DIR / "03_exp_learned_baselines_bc"
EXPERIMENT_04 = EXPERIMENTS_DIR / "04_exp_gnn_bc"

CEMS_FEATURES = ["fwinx", "ffmcode", "drtcode", "infsinx"]

CENTRE_URL = (
    "https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/"
    "bcgw_pub_whse_legal_admin_boundaries/MapServer/1/query"
)


def ensure_fire_centres_boundary() -> Path:
    """Ensure `data/region_exploration/fire_centres.geojson` exists locally."""
    if FIRE_CENTRES_GEOJSON.exists():
        return FIRE_CENTRES_GEOJSON
    print(f"Downloading BC Fire Centres boundary to {FIRE_CENTRES_GEOJSON} ...", flush=True)
    FIRE_CENTRES_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    params = urllib.parse.urlencode({
        "where": "1=1",
        "outFields": "*",
        "outSR": 4326,
        "f": "geojson",
        "maxAllowableOffset": 0.002,
    })
    with urllib.request.urlopen(f"{CENTRE_URL}?{params}", timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert len(data.get("features", [])) == 6, "Expected 6 BC Fire Centres from DataBC"
    FIRE_CENTRES_GEOJSON.write_text(json.dumps(data), encoding="utf-8")
    return FIRE_CENTRES_GEOJSON


def ensure_base_h3_grids() -> None:
    """Ensure the 6 Fire Centre H3 grids & daily feature Parquets exist (built from `firms_with_cems_18_24.csv`)."""
    assert FIRMS_WITH_CEMS_PATH.exists(), (
        f"Missing required joined dataset: {FIRMS_WITH_CEMS_PATH}. "
        "Run `scripts/join_cems_firms.py` first."
    )
    selected_rules_csv = EXPERIMENT_02 / "results/selected_rules_2023.csv"
    all_metrics_csv = EXPERIMENT_02 / "results/all_region_metrics.csv"
    if selected_rules_csv.exists() and all_metrics_csv.exists():
        regions = pd.read_csv(selected_rules_csv)["region"].tolist()
        all_parquets_present = all(
            (EXPERIMENT_02 / "regions" / r.lower().replace(" ", "_") / "data/features/features_2023.parquet").exists()
            for r in regions
        )
        if all_parquets_present:
            return

    ensure_fire_centres_boundary()
    print(
        "\n[Auto-Setup] Regional H3 daily feature Parquets not found on disk (gitignored).\n"
        f"Building regional H3 grids and 2018-2023 daily history tables from {FIRMS_WITH_CEMS_PATH.name}...",
        flush=True,
    )
    subprocess.run(
        [sys.executable, str(EXPERIMENT_02 / "run_experiment.py"), "--resume", "--skip-figures", "--batch-size", "10000"],
        check=True,
    )


def load_firms_with_cems_and_grid() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Load `firms_with_cems_18_24.csv` + `dataset_CEMS_18_24.csv` (strictly filtered to 2018-2023).

    Why both files are used together:
    - `datasets/firms_with_cems_18_24.csv` contains every FIRMS hotspot detection in BC
      joined with its 4 CEMS weather features (`fwinx`, `ffmcode`, `drtcode`, `infsinx`).
    - Because 99.7% of H3 cell-days have NO fire (`target_recorded_detection = 0`), those
      quiet/negative cell-days are not in `firms_with_cems_18_24.csv`. The complete 0.25-deg
      BC CEMS lookup (`dataset_CEMS_18_24.csv`) supplies the daily fire weather for all
      quiet cell-days while matching `firms_with_cems_18_24.csv` on active fire days.
    """
    assert FIRMS_WITH_CEMS_PATH.exists(), f"Missing {FIRMS_WITH_CEMS_PATH}"
    grid_path = CEMS_GRID_18_24_PATH if CEMS_GRID_18_24_PATH.exists() else CEMS_GRID_18_23_PATH
    assert grid_path.exists(), f"Missing CEMS grid CSV at {grid_path}"

    # 1. Load `firms_with_cems_18_24.csv` (strictly 2018-2023, excluding 2024 holdout)
    firms_cems = pd.read_csv(
        FIRMS_WITH_CEMS_PATH,
        usecols=["acq_date", "h3_cell", "cems_lat", "cems_lon", *CEMS_FEATURES],
    )
    firms_cems["history_end_date"] = pd.to_datetime(firms_cems["acq_date"]).dt.normalize()
    firms_cems = firms_cems.loc[
        firms_cems["history_end_date"].dt.year.between(2018, 2023)
        & firms_cems["history_end_date"].dt.month.between(5, 9)
    ].copy()
    assert (firms_cems["history_end_date"].dt.year < 2024).all(), "2024 holdout must never be loaded"
    print(
        f"Loaded {len(firms_cems):,} joined FIRMS+CEMS detection rows (2018-2023) from {FIRMS_WITH_CEMS_PATH.name}",
        flush=True,
    )

    # 2. Load 0.25-deg BC CEMS grid for all H3 cells (active + quiet/non-fire days)
    cems_df = pd.read_csv(
        grid_path,
        usecols=["valid_time", "latitude", "longitude_deg", *CEMS_FEATURES],
    )
    cems_df["history_end_date"] = pd.to_datetime(cems_df["valid_time"]).dt.normalize()
    cems_df = cems_df.loc[cems_df["history_end_date"].dt.year.between(2018, 2023)].drop(columns=["valid_time"]).copy()
    assert (cems_df["history_end_date"].dt.year < 2024).all(), "2024 holdout must never be loaded"
    for col in CEMS_FEATURES:
        cems_df[col] = cems_df[col].astype(np.float32)

    cems_grid = (
        cems_df[["latitude", "longitude_deg"]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude_deg"])
        .reset_index(drop=True)
    )
    cems_grid["cems_cell_id"] = np.arange(len(cems_grid), dtype=np.int32)

    # Verify that `firms_with_cems_18_24.csv` CEMS coordinates are an exact subset of `cems_grid`
    sample_check = (
        firms_cems.head(1000)
        .merge(
            cems_df,
            left_on=["cems_lat", "cems_lon", "history_end_date"],
            right_on=["latitude", "longitude_deg", "history_end_date"],
            suffixes=("_firms", "_grid"),
        )
    )
    for col in CEMS_FEATURES:
        assert np.allclose(sample_check[f"{col}_firms"], sample_check[f"{col}_grid"], atol=1e-4)

    cems_df = cems_df.merge(cems_grid, on=["latitude", "longitude_deg"], how="left")
    cems_df = cems_df.drop(columns=["latitude", "longitude_deg"])
    return cems_df, cems_grid, [FIRMS_WITH_CEMS_PATH, grid_path]


def load_teammate_baselines_from_html() -> pd.DataFrame:
    """Parse teammate's 2023 baseline results from `04_exp_gnn_bc/RESULTS.html` if CSVs are gitignored."""
    html_path = EXPERIMENT_04 / "RESULTS.html"
    if not html_path.exists():
        return pd.DataFrame()
    html = html_path.read_text(encoding="utf-8")
    tables = re.findall(r"<table.*?</table>", html, flags=re.S)
    if len(tables) < 6:
        return pd.DataFrame()
    rows = re.findall(r"<tr>(.*?)</tr>", tables[5], flags=re.S)
    parsed = []
    for r in rows[1:]:
        cells = [re.sub(r"<.*?>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r)]
        if len(cells) < 6:
            continue
        region, source_model, ap, prec, rec, f1 = cells[:6]
        parts = [p.strip() for p in source_model.split("·")]
        source = parts[0] if len(parts) > 0 else ""
        model = parts[1] if len(parts) > 1 else ""
        feature_set = parts[2] if len(parts) > 2 else ""
        parsed.append({
            "region": region,
            "year": 2023,
            "source": source,
            "model": model,
            "feature_set": feature_set,
            "average_precision": float(ap),
            "precision": float(prec.rstrip("%")) / 100.0,
            "recall": float(rec.rstrip("%")) / 100.0,
            "f1": float(f1),
            "tp": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        })
    return pd.DataFrame(parsed)

