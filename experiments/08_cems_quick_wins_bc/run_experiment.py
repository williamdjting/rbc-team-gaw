"""Experiment 08: Testing All 5 CEMS & FIRMS Quick Wins Across All 6 BC Fire Centres.

Implements and evaluates all 5 Quick Wins in a single reproducible ablation & diagnostics pipeline
(`2018-2021` train, `2022` / `2021+2022` validation, `2023` evaluation; `2024` strictly untouched):

1. Quick Win 1 — Active vs. Quiet / Neighbour-Spread / Pure-Ignition AP Breakdown:
   Evaluates 2023 Average Precision separately on:
   - `all_rows`: All 34.24M cell-days in 2023
   - `active_local_7d`: Active persistence cell-days (`local_count_7d > 0`)
   - `no_local_7d`: Unburned cell-days (`local_count_7d == 0`), further decomposed into:
     - `neighbour_spread`: `local_count_7d == 0` and `neighbour_count_7d > 0` (spread into unburned cell)
     - `pure_ignition`: `local_count_7d == 0` and `neighbour_count_7d == 0` (new ignition far from active fire)

2. Quick Win 2 — Fire Radiative Power (FRP), Brightness & 2-Hop Spatial Features:
   Adds strictly past-only (`[t-window, t-1]`) intensity and 2-hop spatial features from
   `datasets/firms_h3_cell_date_with_cems_18_24.csv`:
   - `local_frp_sum_1d`, `local_frp_sum_3d`, `local_frp_sum_7d`, `local_frp_max_1d`
   - `local_brightness_max_1d`
   - `neighbour_frp_sum_1d`, `neighbour_frp_sum_3d`
   - `neighbour_2hop_count_1d`, `neighbour_2hop_frp_1d`

3. Quick Win 3 — Engineered CEMS Temporal, Climatological Anomaly & Interaction Features:
   Adds strictly past-only weather build-up, anomalies (relative to 2018-2021 training climatology),
   and fire-weather interaction terms:
   - `fwinx_mean_3d`, `fwinx_max_7d`, `fwinx_delta_3d`
   - `drtcode_delta_7d`, `ffmcode_mean_3d`, `infsinx_max_3d`
   - `fwinx_anom`, `drtcode_anom` (cell-month anomaly vs. 2018-2021 baseline)
   - `fwinx_x_local_1d`, `infsinx_x_neighbour_1d`, `fwinx_x_frp_1d`

4. Quick Win 4 — Multi-Year Validation (`2021 + 2022` OOF + Selection vs. `2022`-only):
   Evaluates F1 threshold selection using out-of-fold 2021 + 2022 validation and shrinkage-stabilized
   thresholds against 2022-only threshold selection (fixing small-sample instability in Cariboo 2022).

5. Quick Win 5 — Multi-Seed Ensemble (5 Seeds) + GNN Blend & Paired Day-Block Bootstrap 95% CIs:
   Trains a 5-seed ensemble (`seeds 42..46`), evaluates a GBDT + GNN blend, and runs 1,000 paired
   day-block bootstrap replicates across the 146 target days of 2023 to report 95% confidence
   intervals and two-sided p-values for every step of the ablation ladder.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import h3
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.special import expit, logit
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS_DIR = HERE.parent
REFERENCE_01 = EXPERIMENTS_DIR / "01_hotspot_persistence_kamloops"
REFERENCE_02 = EXPERIMENTS_DIR / "02_hotspot_persistence_bc"
REFERENCE_03A = EXPERIMENTS_DIR / "03a_exp_learned_cems_bc"
REFERENCE_04A = EXPERIMENTS_DIR / "04a_exp_gnn_cems_bc"
DATASETS_DIR = ROOT / "datasets"
RESULTS = HERE / "results"

sys.path.insert(0, str(EXPERIMENTS_DIR))
from cems_shared import (  # noqa: E402
    CEMS_FEATURES,
    ensure_base_h3_grids,
    load_firms_with_cems_and_grid,
    load_teammate_baselines_from_html,
)

sys.path.insert(0, str(REFERENCE_01))
from run_persistence import preceding_sum  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_YEARS = [2018, 2019, 2020, 2021]
SELECTION_YEAR = 2022
EVALUATION_YEAR = 2023
ALL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
DAYS_FULL = 153   # May 1 .. Sep 30
DAYS_TARGET = 146 # May 8 .. Sep 30 (indices 7..152 of summer; history_end_date is indices 6..151)
BATCH_SIZE = 10000
QUIET_SAMPLE_RATE = 0.01
SEEDS = [42, 43, 44, 45, 46]
BOOTSTRAP_REPLICATES = 1000
TARGET = "target_recorded_detection"

BASE_CELL_COLS = [
    "local_count_1d",
    "local_count_3d",
    "local_count_7d",
    "days_since_previous_detection",
    "history_left_censored",
    "day_of_year",
]
BASE_NEIGHBOUR_COLS = [
    "neighbour_count_1d",
    "neighbour_count_3d",
    "neighbour_count_7d",
    "in_region_neighbours",
]
RAW_CEMS_COLS = ["fwinx", "ffmcode", "drtcode", "infsinx"]

FRP_BRIGHTNESS_COLS = [
    "local_frp_sum_1d",
    "local_frp_sum_3d",
    "local_frp_sum_7d",
    "local_frp_max_1d",
    "local_brightness_max_1d",
    "neighbour_frp_sum_1d",
    "neighbour_frp_sum_3d",
    "neighbour_2hop_count_1d",
    "neighbour_2hop_frp_1d",
]

ENGINEERED_CEMS_COLS = [
    "fwinx_mean_3d",
    "fwinx_max_7d",
    "fwinx_delta_3d",
    "drtcode_delta_7d",
    "ffmcode_mean_3d",
    "infsinx_max_3d",
    "fwinx_anom",
    "drtcode_anom",
    "fwinx_x_local_1d",
    "infsinx_x_neighbour_1d",
    "fwinx_x_frp_1d",
]

MODEL_SPECS = {
    "M0_firms_only": BASE_CELL_COLS + BASE_NEIGHBOUR_COLS,
    "M1_raw_cems": BASE_CELL_COLS + BASE_NEIGHBOUR_COLS + RAW_CEMS_COLS,
    "M2_frp_brightness": BASE_CELL_COLS + BASE_NEIGHBOUR_COLS + RAW_CEMS_COLS + FRP_BRIGHTNESS_COLS,
    "M3_engineered_cems": BASE_CELL_COLS + BASE_NEIGHBOUR_COLS + RAW_CEMS_COLS + ENGINEERED_CEMS_COLS,
    "M4_frp_plus_eng_cems": (
        BASE_CELL_COLS + BASE_NEIGHBOUR_COLS + RAW_CEMS_COLS + FRP_BRIGHTNESS_COLS + ENGINEERED_CEMS_COLS
    ),
}

ALL_FEATURE_COLS = MODEL_SPECS["M4_frp_plus_eng_cems"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_path(region: str, year: int) -> Path:
    return REFERENCE_02 / "regions" / slug(region) / "data/features" / f"features_{year}.parquet"


def preceding_max_3d_or_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Strictly past-only rolling max over [d-window, d) for each row d."""
    out = np.zeros_like(values, dtype=np.float32)
    for w in range(1, window + 1):
        shifted = np.zeros_like(values, dtype=np.float32)
        shifted[w:] = values[:-w]
        out = np.maximum(out, shifted)
    return out


def build_cems_daily_arrays(
    cems_df: pd.DataFrame, n_cems_cells: int
) -> dict[int, dict[str, np.ndarray]]:
    """Precompute (146, n_cems_cells) strictly past-only raw & engineered CEMS arrays per year.

    Index d in 0..145 corresponds to target_date May 8 .. Sep 30 (summer index 7..152),
    whose history_end_date is summer index h = d + 6 (May 7 .. Sep 29).
    """
    # Step 1: Build full (153, n_cems_cells) arrays for May 1 .. Sep 30 of each year 2018-2023
    raw_summer: dict[int, dict[str, np.ndarray]] = {}
    months_summer: dict[int, np.ndarray] = {}
    for year in ALL_YEARS:
        assert year < 2024, "2024 holdout must never be loaded"
        start_date = pd.Timestamp(f"{year}-05-01")
        end_date = pd.Timestamp(f"{year}-09-30")
        sub = cems_df.loc[
            (cems_df["history_end_date"] >= start_date) & (cems_df["history_end_date"] <= end_date)
        ]
        day_idx = (sub["history_end_date"] - start_date).dt.days.to_numpy(dtype=np.int32)
        cell_idx = sub["cems_cell_id"].to_numpy(dtype=np.int32)
        year_arrays = {}
        for col in RAW_CEMS_COLS:
            arr = np.full((DAYS_FULL, n_cems_cells), np.nan, dtype=np.float32)
            arr[day_idx, cell_idx] = sub[col].to_numpy(dtype=np.float32)
            assert not np.isnan(arr).any(), f"Missing CEMS grid values for {col} in {year}"
            year_arrays[col] = arr
        raw_summer[year] = year_arrays
        dates = pd.date_range(start_date, end_date)
        months_summer[year] = dates.month.to_numpy(dtype=np.int32)

    # Step 2: Compute 2018-2021 monthly climatology per CEMS cell for fwinx and drtcode (strictly train years)
    fwinx_clim = np.zeros((10, n_cems_cells), dtype=np.float32)
    drtcode_clim = np.zeros((10, n_cems_cells), dtype=np.float32)
    for month in (5, 6, 7, 8, 9):
        fwinx_stack = np.vstack([raw_summer[y]["fwinx"][months_summer[y] == month] for y in TRAIN_YEARS])
        drtcode_stack = np.vstack([raw_summer[y]["drtcode"][months_summer[y] == month] for y in TRAIN_YEARS])
        fwinx_clim[month] = fwinx_stack.mean(axis=0)
        drtcode_clim[month] = drtcode_stack.mean(axis=0)

    # Step 3: For each year, extract target indices d_summer = 7..152 (history_end_date h = 6..151)
    cems_features_by_year: dict[int, dict[str, np.ndarray]] = {}
    for year in ALL_YEARS:
        fwinx = raw_summer[year]["fwinx"]
        ffmcode = raw_summer[year]["ffmcode"]
        drtcode = raw_summer[year]["drtcode"]
        infsinx = raw_summer[year]["infsinx"]

        # Rolling past-only windows [d_summer - window, d_summer) for d_summer in 7..152
        fwinx_mean_3d = (preceding_sum(fwinx, 3)[7:] / 3.0).astype(np.float32)
        ffmcode_mean_3d = (preceding_sum(ffmcode, 3)[7:] / 3.0).astype(np.float32)
        fwinx_max_7d = preceding_max_3d_or_1d(fwinx, 7)[7:].astype(np.float32)
        infsinx_max_3d = preceding_max_3d_or_1d(infsinx, 3)[7:].astype(np.float32)

        # History end index h = 6..151 (146 days)
        h_idx = np.arange(6, 152)
        h_months = months_summer[year][h_idx]

        fwinx_h = fwinx[h_idx]
        ffmcode_h = ffmcode[h_idx]
        drtcode_h = drtcode[h_idx]
        infsinx_h = infsinx[h_idx]

        # 3-day change in fwinx up to h: fwinx[h] - fwinx[h - 3]
        fwinx_delta_3d = (fwinx[h_idx] - fwinx[h_idx - 3]).astype(np.float32)
        # 7-day build-up in drtcode across [h - 6, h]: drtcode[h] - drtcode[h - 6]
        drtcode_delta_7d = (drtcode[h_idx] - drtcode[h_idx - 6]).astype(np.float32)

        # Cell-month anomalies relative to 2018-2021 climatology
        fwinx_anom = (fwinx_h - fwinx_clim[h_months]).astype(np.float32)
        drtcode_anom = (drtcode_h - drtcode_clim[h_months]).astype(np.float32)

        cems_features_by_year[year] = {
            "fwinx": fwinx_h,
            "ffmcode": ffmcode_h,
            "drtcode": drtcode_h,
            "infsinx": infsinx_h,
            "fwinx_mean_3d": fwinx_mean_3d,
            "fwinx_max_7d": fwinx_max_7d,
            "fwinx_delta_3d": fwinx_delta_3d,
            "drtcode_delta_7d": drtcode_delta_7d,
            "ffmcode_mean_3d": ffmcode_mean_3d,
            "infsinx_max_3d": infsinx_max_3d,
            "fwinx_anom": fwinx_anom,
            "drtcode_anom": drtcode_anom,
        }
    return cems_features_by_year


def build_region_structures(regions: list[str], cems_grid: pd.DataFrame) -> dict[str, dict]:
    """Build per-region H3 cell lists, 1-hop & 2-hop adjacency matrices, CEMS mapping, and Parquet row indices."""
    cos_lat = np.cos(np.deg2rad(54.0))
    cems_xy = np.column_stack([
        cems_grid["latitude"].to_numpy(),
        cems_grid["longitude_deg"].to_numpy() * cos_lat,
    ])
    tree = cKDTree(cems_xy)
    cems_ids = cems_grid["cems_cell_id"].to_numpy(dtype=np.int32)

    structures = {}
    for region in regions:
        grid_path = REFERENCE_02 / "regions" / slug(region) / "data/h3_grid.parquet"
        cells = pd.read_parquet(grid_path, columns=["h3_cell"])["h3_cell"].tolist()
        n_cells = len(cells)
        position = {c: i for i, c in enumerate(cells)}

        # Map each H3 cell to nearest 0.25-deg CEMS cell
        coords = np.array([h3.cell_to_latlng(c) for c in cells], dtype=np.float64)
        h3_xy = np.column_stack([coords[:, 0], coords[:, 1] * cos_lat])
        _, idx = tree.query(h3_xy, k=1)
        cell_to_cems = cems_ids[idx]

        # 1-hop adjacency matrix
        rows_1, cols_1 = [], []
        for c, i in position.items():
            for nb in sorted(h3.grid_disk(c, 1)):
                if nb != c and nb in position:
                    rows_1.append(i)
                    cols_1.append(position[nb])
        adj_1hop = csr_matrix(
            (np.ones(len(rows_1), dtype=np.float32), (rows_1, cols_1)),
            shape=(n_cells, n_cells),
        )

        # 2-hop ring adjacency matrix (strictly distance-2 H3 neighbours in region)
        adj_sq = (adj_1hop @ adj_1hop).tocsr()
        adj_sq.setdiag(0)
        adj_sq = adj_sq - adj_sq.multiply(adj_1hop > 0)
        adj_sq.eliminate_zeros()
        adj_2hop = (adj_sq > 0).astype(np.float32)

        # Precompute exact (day_target_idx, cell_idx) order in features_<year>.parquet
        # Experiment 02 writes batches of size 10,000 cells, each repeated across 146 target days
        day_parts, cell_parts = [], []
        for start in range(0, n_cells, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, n_cells)
            size = stop - start
            day_parts.append(np.repeat(np.arange(DAYS_TARGET, dtype=np.int16), size))
            cell_parts.append(np.tile(np.arange(start, stop, dtype=np.int32), DAYS_TARGET))
        parquet_day_idx = np.concatenate(day_parts)
        parquet_cell_idx = np.concatenate(cell_parts)

        # Verify against 2018 parquet
        check_df = pd.read_parquet(feature_path(region, 2018), columns=["target_date", "h3_cell"])
        expected_day = (check_df["target_date"] - pd.Timestamp("2018-05-08")).dt.days.to_numpy(dtype=np.int16)
        assert np.array_equal(parquet_day_idx, expected_day), f"Day index mismatch in {region}"
        assert check_df["h3_cell"].iloc[0] == cells[parquet_cell_idx[0]]
        assert check_df["h3_cell"].iloc[-1] == cells[parquet_cell_idx[-1]]
        del check_df

        structures[region] = {
            "cells": cells,
            "n_cells": n_cells,
            "position": position,
            "cell_to_cems": cell_to_cems,
            "adj_1hop": adj_1hop,
            "adj_2hop": adj_2hop,
            "parquet_day_idx": parquet_day_idx,
            "parquet_cell_idx": parquet_cell_idx,
        }
    return structures


def load_firms_frp_cell_date() -> tuple[pd.DataFrame, Path]:
    """Load aggregated H3 cell-day FRP and brightness history (2018-2023 May-Sep, 2024 excluded)."""
    frp_path = DATASETS_DIR / "firms_h3_cell_date_with_cems_18_24.csv"
    assert frp_path.exists(), f"Missing {frp_path}"
    df = pd.read_csv(
        frp_path,
        usecols=["h3_cell", "acq_date", "detection_count", "frp_sum", "frp_max", "brightness_max"],
    )
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    df = df.loc[
        df["acq_date"].dt.year.between(2018, 2023) & df["acq_date"].dt.month.between(5, 9)
    ].copy()
    assert (df["acq_date"].dt.year < 2024).all(), "2024 holdout must never be loaded"
    return df, frp_path


def compute_region_year_frp_features(
    frp_year_df: pd.DataFrame,
    region_struct: dict,
    year: int,
) -> dict[str, np.ndarray]:
    """Compute (146, n_cells) past-only FRP, brightness, and 2-hop spatial arrays for one region-year."""
    n_cells = region_struct["n_cells"]
    position = region_struct["position"]
    adj_1hop = region_struct["adj_1hop"]
    adj_2hop = region_struct["adj_2hop"]

    sub = frp_year_df.loc[frp_year_df["h3_cell"].isin(position)]
    counts = np.zeros((DAYS_FULL, n_cells), dtype=np.float32)
    frp_sum = np.zeros((DAYS_FULL, n_cells), dtype=np.float32)
    frp_max = np.zeros((DAYS_FULL, n_cells), dtype=np.float32)
    bright_max = np.zeros((DAYS_FULL, n_cells), dtype=np.float32)

    if len(sub) > 0:
        day_idx = (sub["acq_date"] - pd.Timestamp(f"{year}-05-01")).dt.days.to_numpy(dtype=np.int32)
        cell_idx = sub["h3_cell"].map(position).to_numpy(dtype=np.int32)
        counts[day_idx, cell_idx] = sub["detection_count"].to_numpy(dtype=np.float32)
        frp_sum[day_idx, cell_idx] = np.maximum(0.0, sub["frp_sum"].to_numpy(dtype=np.float32))
        frp_max[day_idx, cell_idx] = np.maximum(0.0, sub["frp_max"].to_numpy(dtype=np.float32))
        bright_max[day_idx, cell_idx] = np.maximum(0.0, sub["brightness_max"].to_numpy(dtype=np.float32))

    # 1-hop and 2-hop neighbour daily sums
    nb_frp_daily = (adj_1hop @ frp_sum.T).T.astype(np.float32)
    nb2_count_daily = (adj_2hop @ counts.T).T.astype(np.float32)
    nb2_frp_daily = (adj_2hop @ frp_sum.T).T.astype(np.float32)

    # Strictly past-only [d - window, d) slices at target days d = 7..152 (history_end_date h = 6..151)
    h_idx = np.arange(6, 152)
    return {
        "local_frp_sum_1d": frp_sum[h_idx],
        "local_frp_sum_3d": preceding_sum(frp_sum, 3)[7:].astype(np.float32),
        "local_frp_sum_7d": preceding_sum(frp_sum, 7)[7:].astype(np.float32),
        "local_frp_max_1d": frp_max[h_idx],
        "local_brightness_max_1d": bright_max[h_idx],
        "neighbour_frp_sum_1d": nb_frp_daily[h_idx],
        "neighbour_frp_sum_3d": preceding_sum(nb_frp_daily, 3)[7:].astype(np.float32),
        "neighbour_2hop_count_1d": nb2_count_daily[h_idx],
        "neighbour_2hop_frp_1d": nb2_frp_daily[h_idx],
    }


def load_enriched_region_year(
    region: str,
    year: int,
    region_struct: dict,
    cems_year_arrays: dict[str, np.ndarray],
    frp_year_df: pd.DataFrame,
    keep_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Load base FIRMS features and attach FRP/brightness/2-hop + raw/engineered CEMS features via O(1) indexing."""
    assert year < 2024, "2024 holdout must never be loaded"
    read_cols = [
        "target_date",
        TARGET,
        "local_count_1d",
        "local_count_3d",
        "local_count_7d",
        "days_since_previous_detection",
        "history_left_censored",
        "neighbour_count_1d",
        "neighbour_count_3d",
        "neighbour_count_7d",
        "in_region_neighbours",
    ]
    frame = pd.read_parquet(feature_path(region, year), columns=read_cols)
    day_idx = region_struct["parquet_day_idx"]
    cell_idx = region_struct["parquet_cell_idx"]

    if keep_mask is not None:
        frame = frame.loc[keep_mask].reset_index(drop=True)
        day_idx = day_idx[keep_mask]
        cell_idx = cell_idx[keep_mask]

    frame["day_of_year"] = frame["target_date"].dt.dayofyear.astype(np.int16)
    cems_cell_idx = region_struct["cell_to_cems"][cell_idx]

    # Attach CEMS raw + temporal/anomaly features
    for col, arr in cems_year_arrays.items():
        frame[col] = arr[day_idx, cems_cell_idx]

    # Attach FRP, brightness, and 2-hop spatial features
    frp_arrays = compute_region_year_frp_features(frp_year_df, region_struct, year)
    for col, arr in frp_arrays.items():
        frame[col] = arr[day_idx, cell_idx]

    # Fire-weather interaction terms
    fwinx_vals = frame["fwinx"].to_numpy(dtype=np.float32)
    infsinx_vals = frame["infsinx"].to_numpy(dtype=np.float32)
    local_1d = frame["local_count_1d"].to_numpy(dtype=np.float32)
    nb_1d = frame["neighbour_count_1d"].to_numpy(dtype=np.float32)
    frp_1d = frame["local_frp_sum_1d"].to_numpy(dtype=np.float32)

    frame["fwinx_x_local_1d"] = fwinx_vals * np.log1p(local_1d)
    frame["infsinx_x_neighbour_1d"] = infsinx_vals * np.log1p(nb_1d)
    frame["fwinx_x_frp_1d"] = fwinx_vals * np.log1p(frp_1d)

    return frame


def quiet_stratum_mask(frame: pd.DataFrame) -> np.ndarray:
    return (frame["local_count_7d"].to_numpy() == 0) & (frame["neighbour_count_7d"].to_numpy() == 0)


def design_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    quiet = quiet_stratum_mask(frame)
    X = np.column_stack([frame[feature_cols].to_numpy(dtype=np.float32), quiet.astype(np.float32)])
    return X, quiet


def make_gbdt(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def predict_gbdt(model: HistGradientBoostingClassifier, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    batch_size = 2_000_000
    scores = []
    for start in range(0, len(frame), batch_size):
        sub = frame.iloc[start : start + batch_size]
        X, quiet = design_matrix(sub, feature_cols)
        scores.append(expit(model.decision_function(X) + np.log(QUIET_SAMPLE_RATE) * quiet))
    return np.concatenate(scores).astype(np.float32)


def predict_gbdt_ensemble(
    models: list[HistGradientBoostingClassifier], frame: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (seed0_scores, mean_ensemble_scores) reusing the design matrix across all seeds."""
    batch_size = 2_000_000
    seed0_parts, ens_parts = [], []
    shift_const = np.log(QUIET_SAMPLE_RATE)
    for start in range(0, len(frame), batch_size):
        sub = frame.iloc[start : start + batch_size]
        X, quiet = design_matrix(sub, feature_cols)
        shift = shift_const * quiet
        s0 = expit(models[0].decision_function(X) + shift).astype(np.float32)
        acc = s0.copy()
        for m in models[1:]:
            acc += expit(m.decision_function(X) + shift).astype(np.float32)
        seed0_parts.append(s0)
        ens_parts.append(acc / len(models))
    return np.concatenate(seed0_parts).astype(np.float32), np.concatenate(ens_parts).astype(np.float32)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    positive, prediction = y.astype(bool), score >= threshold
    tp = int((positive & prediction).sum())
    fp = int((~positive & prediction).sum())
    fn = int((positive & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ap = float(average_precision_score(y, score)) if positive.sum() > 0 else float("nan")
    return {
        "rows": len(y),
        "positive_cell_days": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "average_precision": ap,
        "ap_over_prevalence": ap / positive.mean() if positive.mean() > 0 else float("nan"),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(len(y) - tp - fp - fn),
        "predicted_positive_cell_days": tp + fp,
    }


def collect_sampled_training_table(
    regions: list[str],
    years: list[int],
    region_structs: dict[str, dict],
    cems_by_year: dict[int, dict[str, np.ndarray]],
    frp_by_year: dict[int, pd.DataFrame],
    seed: int = 42,
) -> pd.DataFrame:
    """Collect active rows + all positives + 1% sampled quiet negatives across regions and years."""
    parts = []
    for r_idx, region in enumerate(regions):
        for year in years:
            base = pd.read_parquet(
                feature_path(region, year),
                columns=[TARGET, "local_count_7d", "neighbour_count_7d"],
            )
            positive = base[TARGET].to_numpy() == 1
            quiet = (base["local_count_7d"].to_numpy() == 0) & (base["neighbour_count_7d"].to_numpy() == 0)
            drawn = np.random.default_rng([seed, year, r_idx]).random(len(base)) < QUIET_SAMPLE_RATE
            keep = (~quiet) | positive | drawn
            del base
            part = load_enriched_region_year(
                region, year, region_structs[region], cems_by_year[year], frp_by_year[year], keep_mask=keep
            )
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def daily_binned_counts(
    y: np.ndarray,
    score: np.ndarray,
    day_idx: np.ndarray,
    n_days: int = DAYS_TARGET,
    n_bins: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin daily (y, score) into (n_days, n_bins) positive/negative count matrices for fast day-block bootstrap.

    Bins are spaced uniformly in logit(score) space so both the high-probability tail and
    low-probability quiet tail have high resolution. Bin 0 is highest score; bin n_bins-1 is lowest.
    """
    clipped = np.clip(score.astype(np.float64), 1e-7, 1.0 - 1e-7)
    log_odds = logit(clipped)
    lo, hi = -16.2, 12.0
    scaled = (hi - log_odds) / (hi - lo)
    bin_idx = np.clip((scaled * n_bins).astype(np.int32), 0, n_bins - 1)

    flat_idx = day_idx.astype(np.int32) * n_bins + bin_idx
    pos_mask = y == 1
    pos_counts = np.bincount(flat_idx[pos_mask], minlength=n_days * n_bins).reshape(n_days, n_bins).astype(np.float64)
    neg_counts = np.bincount(flat_idx[~pos_mask], minlength=n_days * n_bins).reshape(n_days, n_bins).astype(np.float64)
    return pos_counts, neg_counts


def ap_from_binned_counts(pos_bins: np.ndarray, neg_bins: np.ndarray) -> np.ndarray:
    """Compute Average Precision for each row of (R, n_bins) positive and negative bin counts."""
    cum_pos = np.cumsum(pos_bins, axis=1)
    cum_tot = cum_pos + np.cumsum(neg_bins, axis=1)
    total_pos = cum_pos[:, -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_at_k = np.where(cum_tot > 0, cum_pos / cum_tot, 0.0)
        weights = np.where(total_pos > 0, pos_bins / total_pos, 0.0)
    return np.sum(weights * precision_at_k, axis=1)


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)

    ensure_base_h3_grids()
    regions = sorted(pd.read_csv(REFERENCE_02 / "results/selected_rules_2023.csv")["region"].tolist())
    rules_df = pd.read_csv(REFERENCE_02 / "results/all_region_metrics.csv")

    print("1. Loading CEMS 0.25° grid & building temporal/climatological anomaly arrays (2018-2023)...", flush=True)
    cems_df, cems_grid, cems_paths = load_firms_with_cems_and_grid()
    cems_by_year = build_cems_daily_arrays(cems_df, len(cems_grid))
    del cems_df

    print("2. Loading FIRMS H3 cell-day FRP & brightness table (2018-2023)...", flush=True)
    frp_df, frp_path = load_firms_frp_cell_date()
    frp_by_year = {y: grp.copy() for y, grp in frp_df.groupby(frp_df["acq_date"].dt.year)}
    del frp_df

    print("3. Building per-region H3 1-hop & 2-hop graph structures and Parquet index maps...", flush=True)
    region_structs = build_region_structures(regions, cems_grid)

    print("4. Building 2018-2021 sampled training datasets (Seeds 42..46)...", flush=True)
    t0 = time.time()
    train_seed42 = collect_sampled_training_table(
        regions, TRAIN_YEARS, region_structs, cems_by_year, frp_by_year, seed=42
    )
    y_train42 = train_seed42[TARGET].to_numpy()
    print(
        f"   Seed 42 training table: {len(train_seed42):,} rows, {int(y_train42.sum()):,} positives "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Fit Ablation Ladder Models M0..M4 on Seed 42
    print("5. Fitting Ablation Ladder Models (M0..M4) on 2018-2021...", flush=True)
    fitted_models: dict[str, HistGradientBoostingClassifier] = {}
    fit_rows = []
    for model_name, cols in MODEL_SPECS.items():
        tic = time.time()
        X_tr, _ = design_matrix(train_seed42, cols)
        clf = make_gbdt(seed=42)
        clf.fit(X_tr, y_train42)
        raw_pred = clf.decision_function(X_tr)
        fit_sec = round(time.time() - tic, 2)
        fitted_models[model_name] = clf
        fit_rows.append({
            "model": model_name,
            "num_features": len(cols) + 1,
            "training_rows": len(train_seed42),
            "training_positives": int(y_train42.sum()),
            "train_ap": float(average_precision_score(y_train42, raw_pred)),
            "max_abs_log_odds": float(np.abs(raw_pred).max()),
            "fit_seconds": fit_sec,
        })
        print(
            f"   Fitted {model_name:22s} ({len(cols)+1:2d} features) in {fit_sec:5.1f}s | "
            f"train AP={fit_rows[-1]['train_ap']:.4f}",
            flush=True,
        )
    del train_seed42
    gc.collect()

    # Fit additional seeds 43..46 for M4_frp_plus_eng_cems (Quick Win 5: 5-seed ensemble)
    print("6. Fitting Seeds 43..46 for Quick Win 5 Multi-Seed Ensemble...", flush=True)
    ensemble_models = [fitted_models["M4_frp_plus_eng_cems"]]
    m4_cols = MODEL_SPECS["M4_frp_plus_eng_cems"]
    for seed in SEEDS[1:]:
        tic = time.time()
        train_s = collect_sampled_training_table(
            regions, TRAIN_YEARS, region_structs, cems_by_year, frp_by_year, seed=seed
        )
        y_s = train_s[TARGET].to_numpy()
        X_s, _ = design_matrix(train_s, m4_cols)
        clf_s = make_gbdt(seed=seed)
        clf_s.fit(X_s, y_s)
        ensemble_models.append(clf_s)
        print(f"   Fitted M4 seed {seed} in {time.time() - tic:.1f}s", flush=True)
        del train_s, X_s, y_s
        gc.collect()

    # Fit 2018-2020 model to generate out-of-fold 2021 validation predictions for Quick Win 4 (Multi-Year Validation)
    print("7. Fitting 2018-2020 model for Quick Win 4 Out-of-Fold 2021 + 2022 Validation...", flush=True)
    train_18_20 = collect_sampled_training_table(
        regions, [2018, 2019, 2020], region_structs, cems_by_year, frp_by_year, seed=42
    )
    y_18_20 = train_18_20[TARGET].to_numpy()
    oof_2021_models: dict[str, HistGradientBoostingClassifier] = {}
    for model_name in ("M1_raw_cems", "M4_frp_plus_eng_cems"):
        cols = MODEL_SPECS[model_name]
        X_sub, _ = design_matrix(train_18_20, cols)
        clf_sub = make_gbdt(seed=42)
        clf_sub.fit(X_sub, y_18_20)
        oof_2021_models[model_name] = clf_sub
    del train_18_20, y_18_20
    gc.collect()

    # Evaluate on 2021 (OOF) and 2022 (Selection) to compute thresholds for Quick Win 4
    # Optimization: F1 thresholds lie in [0.15, 0.65], whereas quiet_stratum rows (shifted by log(0.01) = -4.605)
    # have score < 0.03. Filtering to `(~quiet) | (y == 1)` retains 100% of positives and 100% of rows that
    # can ever exceed any F1 threshold, yielding the exact same F1 threshold 25x faster (~5s total).
    print("8. Scoring 2021 (OOF) and 2022 (Validation) across all 6 Fire Centres...", flush=True)
    t_step8 = time.time()
    ALL_EVAL_MODELS = [
        "M0_firms_only",
        "M1_raw_cems",
        "M2_frp_brightness",
        "M3_engineered_cems",
        "M4_frp_plus_eng_cems",
        "M5_ensemble_5seeds",
        "04a_gnn_cems",
        "M6_gbdt_gnn_blend",
    ]
    thresholds_2022: dict[tuple[str, str], float] = {}
    thresholds_2021_2022: dict[tuple[str, str], float] = {}
    thresholds_shrunk_2022: dict[tuple[str, str], float] = {}

    val_2022_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    val_2021_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    gnn_2022_metrics = pd.read_csv(REFERENCE_04A / "results/metrics_by_year.csv")

    for region in regions:
        r_tic = time.time()
        # 2021 OOF active + positive rows for M1 and M4
        base_2021 = pd.read_parquet(
            feature_path(region, 2021), columns=[TARGET, "local_count_7d", "neighbour_count_7d"]
        )
        keep_2021 = (
            (base_2021["local_count_7d"].to_numpy() > 0)
            | (base_2021["neighbour_count_7d"].to_numpy() > 0)
            | (base_2021[TARGET].to_numpy() == 1)
        )
        del base_2021
        frame_2021 = load_enriched_region_year(
            region, 2021, region_structs[region], cems_by_year[2021], frp_by_year[2021], keep_mask=keep_2021
        )
        y_2021 = frame_2021[TARGET].to_numpy()
        for m_name, clf_oof in oof_2021_models.items():
            s_2021 = predict_gbdt(clf_oof, frame_2021, MODEL_SPECS[m_name])
            val_2021_cache[(region, m_name)] = (y_2021, s_2021)
        del frame_2021

        # 2022 validation active + positive rows for M0..M5
        base_2022 = pd.read_parquet(
            feature_path(region, 2022), columns=[TARGET, "local_count_7d", "neighbour_count_7d"]
        )
        keep_2022 = (
            (base_2022["local_count_7d"].to_numpy() > 0)
            | (base_2022["neighbour_count_7d"].to_numpy() > 0)
            | (base_2022[TARGET].to_numpy() == 1)
        )
        del base_2022
        frame_2022 = load_enriched_region_year(
            region, 2022, region_structs[region], cems_by_year[2022], frp_by_year[2022], keep_mask=keep_2022
        )
        y_2022 = frame_2022[TARGET].to_numpy()
        scores_2022 = {}
        for m_name in ("M0_firms_only", "M1_raw_cems", "M2_frp_brightness", "M3_engineered_cems"):
            cols = MODEL_SPECS[m_name]
            scores_2022[m_name] = predict_gbdt(fitted_models[m_name], frame_2022, cols)
            thresholds_2022[(region, m_name)] = best_f1_threshold(y_2022, scores_2022[m_name])
            val_2022_cache[(region, m_name)] = (y_2022, scores_2022[m_name])

        s_m4_22, s_m5_22 = predict_gbdt_ensemble(ensemble_models, frame_2022, m4_cols)
        scores_2022["M4_frp_plus_eng_cems"] = s_m4_22
        scores_2022["M5_ensemble_5seeds"] = s_m5_22
        thresholds_2022[(region, "M4_frp_plus_eng_cems")] = best_f1_threshold(y_2022, s_m4_22)
        thresholds_2022[(region, "M5_ensemble_5seeds")] = best_f1_threshold(y_2022, s_m5_22)
        val_2022_cache[(region, "M4_frp_plus_eng_cems")] = (y_2022, s_m4_22)
        val_2022_cache[(region, "M5_ensemble_5seeds")] = (y_2022, s_m5_22)

        gnn_row_2022 = gnn_2022_metrics.loc[
            gnn_2022_metrics["region"].eq(region)
            & gnn_2022_metrics["year"].eq(2022)
            & gnn_2022_metrics["model"].eq("gnn")
        ].iloc[0]
        thresholds_2022[(region, "04a_gnn_cems")] = float(gnn_row_2022["threshold"])
        thresholds_2022[(region, "M6_gbdt_gnn_blend")] = float(
            0.75 * thresholds_2022[(region, "M5_ensemble_5seeds")] + 0.25 * thresholds_2022[(region, "04a_gnn_cems")]
        )
        print(f"   Scored 2021+2022 validation for {region:14s} in {time.time() - r_tic:.1f}s", flush=True)
        del frame_2022
        gc.collect()
    print(f"   Completed Step 8 in {time.time() - t_step8:.1f}s!", flush=True)

    # Compute Quick Win 4 thresholds:
    # (a) 2021 + 2022 combined OOF+Validation per region
    # (b) Empirical-Bayes shrinkage of 2022 regional threshold toward pooled BC 2022 threshold
    for m_name in ("M1_raw_cems", "M4_frp_plus_eng_cems", "M5_ensemble_5seeds"):
        base_m = "M4_frp_plus_eng_cems" if m_name == "M5_ensemble_5seeds" else m_name
        pooled_y_2022 = np.concatenate([val_2022_cache[(r, m_name)][0] for r in regions])
        pooled_s_2022 = np.concatenate([val_2022_cache[(r, m_name)][1] for r in regions])
        bc_global_thr_2022 = best_f1_threshold(pooled_y_2022, pooled_s_2022)

        for region in regions:
            y21, s21 = val_2021_cache[(region, base_m)]
            y22, s22 = val_2022_cache[(region, m_name)]
            y_comb = np.concatenate([y21, y22])
            s_comb = np.concatenate([s21, s22])
            thresholds_2021_2022[(region, m_name)] = best_f1_threshold(y_comb, s_comb)

            # Shrinkage weight based on positive cell-days in 2022 (Cariboo had only 27 positives in 2022!)
            pos_2022 = int(y22.sum())
            weight = pos_2022 / (pos_2022 + 200.0)
            thresholds_shrunk_2022[(region, m_name)] = float(
                weight * thresholds_2022[(region, m_name)] + (1.0 - weight) * bc_global_thr_2022
            )

    del val_2021_cache, val_2022_cache
    gc.collect()

    # Step 9: Evaluate all models on 2023 across all 6 Fire Centres + Quick Wins 1, 4, 5
    print("9. Evaluating 2023 across all 6 Fire Centres (Ablation + Quick Wins 1, 4, 5)...", flush=True)
    ablation_rows = []
    subset_ap_rows = []
    multi_year_val_rows = []
    bootstrap_rows = []

    rng_boot = np.random.default_rng(202308)
    # Pre-generate 1,000 paired day-block bootstrap weight vectors over the 146 target days
    boot_weights = rng_boot.multinomial(
        DAYS_TARGET, np.full(DAYS_TARGET, 1.0 / DAYS_TARGET), size=BOOTSTRAP_REPLICATES
    ).astype(np.float64)

    pooled_daily_bins: dict[str, list[ tuple[np.ndarray, np.ndarray] ]] = {m: [] for m in ALL_EVAL_MODELS}

    for region in regions:
        tic = time.time()
        frame_2023 = load_enriched_region_year(
            region, 2023, region_structs[region], cems_by_year[2023], frp_by_year[2023]
        )
        y_2023 = frame_2023[TARGET].to_numpy()
        day_idx_2023 = region_structs[region]["parquet_day_idx"]

        # Verify alignment with Experiment 02 local_1d benchmark
        ref_rule = rules_df.loc[
            rules_df["region"].eq(region) & rules_df["year"].eq(2023) & rules_df["rule"].eq("local_1d")
        ].iloc[0]
        check_rule = threshold_metrics(y_2023, frame_2023["local_count_1d"].to_numpy(dtype=np.float32), 1.0)
        assert all(check_rule[k] == ref_rule[k] for k in ("rows", "positive_cell_days", "tp", "fp", "fn")), region

        scores_2023: dict[str, np.ndarray] = {}
        for m_name in ("M0_firms_only", "M1_raw_cems", "M2_frp_brightness", "M3_engineered_cems"):
            scores_2023[m_name] = predict_gbdt(fitted_models[m_name], frame_2023, MODEL_SPECS[m_name])

        # M4_frp_plus_eng_cems (seed 42) and M5_ensemble_5seeds (5-seed average) in one pass
        scores_2023["M4_frp_plus_eng_cems"], scores_2023["M5_ensemble_5seeds"] = predict_gbdt_ensemble(
            ensemble_models, frame_2023, m4_cols
        )

        # Load saved 04a GNN 2023 predictions (aligned by row order in features_2023.parquet)
        gnn_pred_path = REFERENCE_04A / "results/predictions" / f"{slug(region)}_2023.parquet"
        gnn_pred_df = pd.read_parquet(gnn_pred_path, columns=[TARGET, "gnn"])
        assert np.array_equal(gnn_pred_df[TARGET].to_numpy(), y_2023), f"GNN target alignment mismatch in {region}"
        scores_2023["04a_gnn_cems"] = gnn_pred_df["gnn"].to_numpy(dtype=np.float32)
        del gnn_pred_df

        # M6_gbdt_gnn_blend: 75% M5_ensemble_5seeds + 25% 04a_gnn_cems (probability blend)
        scores_2023["M6_gbdt_gnn_blend"] = (
            0.75 * scores_2023["M5_ensemble_5seeds"] + 0.25 * scores_2023["04a_gnn_cems"]
        ).astype(np.float32)

        # Save predictions for top models
        pred_out = pd.DataFrame({
            "target_date": frame_2023["target_date"],
            TARGET: y_2023,
            "M0_firms_only": scores_2023["M0_firms_only"],
            "M1_raw_cems": scores_2023["M1_raw_cems"],
            "M4_frp_plus_eng_cems": scores_2023["M4_frp_plus_eng_cems"],
            "M5_ensemble_5seeds": scores_2023["M5_ensemble_5seeds"],
            "M6_gbdt_gnn_blend": scores_2023["M6_gbdt_gnn_blend"],
        })
        pred_out.to_parquet(RESULTS / "predictions" / f"{slug(region)}_2023.parquet", index=False)
        del pred_out

        # Record standard 2023 evaluation metrics for every model
        for m_name in ALL_EVAL_MODELS:
            ablation_rows.append({
                "region": region,
                "year": 2023,
                "model": m_name,
                **threshold_metrics(y_2023, scores_2023[m_name], thresholds_2022[(region, m_name)]),
            })

        # Quick Win 1: Active vs. Quiet / Neighbour-Spread / Pure-Ignition AP Breakdown
        local_7d = frame_2023["local_count_7d"].to_numpy()
        nb_7d = frame_2023["neighbour_count_7d"].to_numpy()
        regimes = {
            "all_rows": np.ones(len(y_2023), dtype=bool),
            "active_local_7d": local_7d > 0,
            "no_local_7d": local_7d == 0,
            "neighbour_spread": (local_7d == 0) & (nb_7d > 0),
            "pure_ignition": (local_7d == 0) & (nb_7d == 0),
        }
        for regime_name, mask in regimes.items():
            y_sub = y_2023[mask]
            pos_sub = int(y_sub.sum())
            prev_sub = float(y_sub.mean())
            for m_name in ALL_EVAL_MODELS:
                ap_sub = float(average_precision_score(y_sub, scores_2023[m_name][mask])) if pos_sub > 0 else float("nan")
                subset_ap_rows.append({
                    "region": region,
                    "regime": regime_name,
                    "model": m_name,
                    "rows": int(mask.sum()),
                    "positives": pos_sub,
                    "prevalence": prev_sub,
                    "average_precision": ap_sub,
                    "ap_over_prevalence": ap_sub / prev_sub if prev_sub > 0 else float("nan"),
                })

        # Quick Win 4: Multi-Year Validation (2022-only vs. 2021+2022 OOF+Val vs. Empirical-Bayes Shrunk 2022)
        for m_name in ("M1_raw_cems", "M4_frp_plus_eng_cems", "M5_ensemble_5seeds"):
            for scheme, thr_dict in (
                ("2022_only", thresholds_2022),
                ("2021_plus_2022_oof", thresholds_2021_2022),
                ("2022_shrunk_eb", thresholds_shrunk_2022),
            ):
                tm = threshold_metrics(y_2023, scores_2023[m_name], thr_dict[(region, m_name)])
                multi_year_val_rows.append({
                    "region": region,
                    "model": m_name,
                    "threshold_scheme": scheme,
                    "threshold": tm["threshold"],
                    "precision": tm["precision"],
                    "recall": tm["recall"],
                    "f1": tm["f1"],
                    "tp": tm["tp"],
                    "fp": tm["fp"],
                    "fn": tm["fn"],
                })

        # Quick Win 5: Paired Day-Block Bootstrap 95% CIs on 2023 AP
        boot_aps_region: dict[str, np.ndarray] = {}
        exact_aps_region: dict[str, float] = {}
        for m_name in ALL_EVAL_MODELS:
            p_bins, n_bins = daily_binned_counts(y_2023, scores_2023[m_name], day_idx_2023)
            pooled_daily_bins[m_name].append((p_bins, n_bins))
            exact_ap = float(average_precision_score(y_2023, scores_2023[m_name]))
            full_binned_ap = float(ap_from_binned_counts(p_bins.sum(axis=0, keepdims=True), n_bins.sum(axis=0, keepdims=True))[0])
            assert abs(exact_ap - full_binned_ap) < 1e-3, (
                f"Binned AP approximation error too large in {region} {m_name}: {exact_ap:.5f} vs {full_binned_ap:.5f}"
            )
            boot_p = boot_weights @ p_bins
            boot_n = boot_weights @ n_bins
            boot_aps_region[m_name] = ap_from_binned_counts(boot_p, boot_n)
            exact_aps_region[m_name] = exact_ap

        comparisons = [
            ("M1_raw_cems", "M0_firms_only", "Raw CEMS vs. FIRMS-only"),
            ("M2_frp_brightness", "M1_raw_cems", "+FRP/Brightness/2-hop vs. Raw CEMS"),
            ("M3_engineered_cems", "M1_raw_cems", "+Engineered CEMS vs. Raw CEMS"),
            ("M4_frp_plus_eng_cems", "M1_raw_cems", "+FRP & Engineered CEMS vs. Raw CEMS"),
            ("M5_ensemble_5seeds", "M4_frp_plus_eng_cems", "5-Seed Ensemble vs. Single Seed (M4)"),
            ("M5_ensemble_5seeds", "M0_firms_only", "5-Seed Ensemble (M5) vs. FIRMS-only (M0)"),
            ("M5_ensemble_5seeds", "M1_raw_cems", "5-Seed Ensemble (M5) vs. Exp 03a Raw CEMS (M1)"),
            ("M6_gbdt_gnn_blend", "04a_gnn_cems", "GBDT+GNN Blend (M6) vs. Exp 04a GNN"),
            ("M6_gbdt_gnn_blend", "M0_firms_only", "GBDT+GNN Blend (M6) vs. FIRMS-only (M0)"),
        ]
        for new_m, ref_m, label in comparisons:
            diffs = boot_aps_region[new_m] - boot_aps_region[ref_m]
            point_diff = exact_aps_region[new_m] - exact_aps_region[ref_m]
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            p_one_sided = float(np.mean(diffs <= 0.0)) if point_diff >= 0 else float(np.mean(diffs >= 0.0))
            p_two_sided = min(1.0, 2.0 * p_one_sided)
            bootstrap_rows.append({
                "region": region,
                "comparison": label,
                "model_new": new_m,
                "model_ref": ref_m,
                "ap_new": exact_aps_region[new_m],
                "ap_ref": exact_aps_region[ref_m],
                "ap_delta": point_diff,
                "ap_rel_pct": 100.0 * point_diff / exact_aps_region[ref_m],
                "ci95_low": float(ci_lo),
                "ci95_high": float(ci_hi),
                "p_value_two_sided": p_two_sided,
                "significant_95": bool((ci_lo > 0) or (ci_hi < 0)),
            })

        print(
            f"   {region:14s} ({time.time() - tic:.1f}s): "
            f"M0={exact_aps_region['M0_firms_only']:.4f} -> "
            f"M1={exact_aps_region['M1_raw_cems']:.4f} -> "
            f"M4={exact_aps_region['M4_frp_plus_eng_cems']:.4f} -> "
            f"M5={exact_aps_region['M5_ensemble_5seeds']:.4f} -> "
            f"M6={exact_aps_region['M6_gbdt_gnn_blend']:.4f}",
            flush=True,
        )
        del frame_2023, scores_2023
        gc.collect()

    # Also compute Macro-Average (Mean across 6 Fire Centres) Paired Day-Block Bootstrap CIs
    macro_boot_aps: dict[str, np.ndarray] = {}
    macro_exact_aps: dict[str, float] = {}
    for m_name in ALL_EVAL_MODELS:
        per_region_boots = []
        for p_bins, n_bins in pooled_daily_bins[m_name]:
            per_region_boots.append(ap_from_binned_counts(boot_weights @ p_bins, boot_weights @ n_bins))
        macro_boot_aps[m_name] = np.mean(per_region_boots, axis=0)
        macro_exact_aps[m_name] = float(
            np.mean([r["average_precision"] for r in ablation_rows if r["model"] == m_name])
        )

    for new_m, ref_m, label in comparisons:
        diffs = macro_boot_aps[new_m] - macro_boot_aps[ref_m]
        point_diff = macro_exact_aps[new_m] - macro_exact_aps[ref_m]
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        p_one_sided = float(np.mean(diffs <= 0.0)) if point_diff >= 0 else float(np.mean(diffs >= 0.0))
        p_two_sided = min(1.0, 2.0 * p_one_sided)
        bootstrap_rows.append({
            "region": "Mean (6 Centres)",
            "comparison": label,
            "model_new": new_m,
            "model_ref": ref_m,
            "ap_new": macro_exact_aps[new_m],
            "ap_ref": macro_exact_aps[ref_m],
            "ap_delta": point_diff,
            "ap_rel_pct": 100.0 * point_diff / macro_exact_aps[ref_m],
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi),
            "p_value_two_sided": p_two_sided,
            "significant_95": bool((ci_lo > 0) or (ci_hi < 0)),
        })

    # Save all CSV outputs
    fit_df = pd.DataFrame(fit_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    subset_ap_df = pd.DataFrame(subset_ap_rows)
    multi_year_val_df = pd.DataFrame(multi_year_val_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    fit_df.to_csv(RESULTS / "fit_checks.csv", index=False)
    ablation_df.to_csv(RESULTS / "ablation_2023.csv", index=False)
    subset_ap_df.to_csv(RESULTS / "active_vs_quiet_ap_2023.csv", index=False)
    multi_year_val_df.to_csv(RESULTS / "multi_year_validation_2023.csv", index=False)
    bootstrap_df.to_csv(RESULTS / "bootstrap_significance_2023.csv", index=False)

    # Create wide pivot table for easy inspection
    piv_ap = ablation_df.pivot(index="region", columns="model", values="average_precision")[ALL_EVAL_MODELS]
    piv_ap.loc["Mean (6 Centres)"] = piv_ap.mean(axis=0)
    piv_ap.reset_index().to_csv(RESULTS / "summary_ap_table_2023.csv", index=False)

    manifest = {
        "regions": regions,
        "training_years": TRAIN_YEARS,
        "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR,
        "holdout_loaded": False,
        "models_evaluated": ALL_EVAL_MODELS,
        "feature_counts": {k: len(v) + 1 for k, v in MODEL_SPECS.items()},
        "seeds_ensemble": SEEDS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "runtime_seconds": round(time.time() - started, 1),
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nCompleted Experiment 08 in {time.time() - started:.1f}s!", flush=True)
    print("\n=== 2023 Average Precision (AP) Across All 6 Fire Centres ===", flush=True)
    print(piv_ap.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()

