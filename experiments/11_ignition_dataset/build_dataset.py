"""Build the daily ignition dataset (Week 3, Task 2).

One row per H3 cell (default resolution 6) per target day, May 8 - September 30,
2018-2023, over the six BC Fire Centre grids used in Experiments 02-07.

Column prefixes keep label-only information away from features:
  y_*     targets for day t (from Experiment 10 labels_v2, default label = tiers A+B)
  info_*  retrospective label metadata (final size, tier). NEVER use as features.
  f_*     features using only data through day t-1 (FIRMS detections, earlier BC
          ignitions, calendar, location). Weather, lightning and static layers: Task 4.
Rows with `masked == True` are neither positive nor negative (active fire footprints,
spread/flare-up/static/late-season FIRMS onsets); drop them from training and scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/10_ignition_label_fusion"))
import run_analysis as ra  # noqa: E402

LABELS = ROOT / "experiments/10_ignition_label_fusion/results/labels_v2"
REGIONS = ROOT / "experiments/02_hotspot_persistence_bc/regions"
REGION_NAMES = {"cariboo": "Cariboo", "coastal": "Coastal", "kamloops": "Kamloops", "northwest": "Northwest",
                "prince_george": "Prince George", "southeast": "Southeast"}
YEARS = list(range(2018, 2024))
SPLIT = {2018: "train_extra", 2019: "train", 2020: "train", 2021: "train", 2022: "valid", 2023: "test_dev"}
HISTORY_START = (5, 1)
TARGET_START = (5, 8)
SEASON_END = (9, 30)
MASK_FOOTPRINT, MASK_ONSET = 1, 2


def season_days(year, start):
    first = pd.Timestamp(year, *start)
    last = pd.Timestamp(year, *SEASON_END)
    return pd.date_range(first, last, freq="D")


def load_grid(res):
    parts = []
    for folder, name in REGION_NAMES.items():
        g = pd.read_parquet(REGIONS / folder / "data/h3_grid.parquet", columns=["h3_cell"])
        g["region"] = name
        parts.append(g)
    g7 = pd.concat(parts, ignore_index=True)
    if res == 7:
        # Border res-7 cells appear in several regional grids; keep one row per cell.
        grid = g7.drop_duplicates("h3_cell").rename(columns={"h3_cell": "cell"})
    else:
        g7["cell"] = [h3.cell_to_parent(c, res) for c in g7["h3_cell"]]
        counts = g7.groupby(["cell", "region"]).size().rename("n").reset_index()
        grid = counts.sort_values(["cell", "n"], ascending=[True, False]).drop_duplicates("cell")[["cell", "region"]]
        child = g7.drop_duplicates("h3_cell").groupby("cell").size().rename("children_in_grid")
        grid = grid.merge(child, left_on="cell", right_index=True)
    grid = grid.sort_values("cell").reset_index(drop=True)
    ll = np.array([h3.cell_to_latlng(c) for c in grid["cell"]])
    grid["lat"], grid["lon"] = ll[:, 0].astype("float32"), ll[:, 1].astype("float32")
    return grid


def ring_index(cells, k):
    """Padded index matrix of in-grid cells at exactly distance k (ring), -1 padded."""
    pos = {c: i for i, c in enumerate(cells)}
    rows = [[pos[n] for n in h3.grid_ring(c, k) if n in pos] for c in cells]
    width = max(len(r) for r in rows)
    out = np.full((len(cells), width), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def ring_sum(arr, idx):
    """Sum arr over ring neighbours. arr: (cells, days); idx: (cells, width) padded with -1."""
    padded = np.vstack([arr, np.zeros((1, arr.shape[1]), arr.dtype)])
    return padded[idx].sum(axis=1)


def trailing(arr, window):
    """Sum over days [t-window, t-1] for each t (past only). Column t uses columns < t."""
    c = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=np.int64)
    np.cumsum(arr, axis=1, out=c[:, 1:])
    t = np.arange(arr.shape[1])
    lo = np.maximum(t - window, 0)
    return c[:, t] - c[:, lo]


def days_since(arr):
    """Days since the last positive day strictly before t within the season (NaN if none)."""
    n_cells, n_days = arr.shape
    out = np.full((n_cells, n_days), np.nan, dtype=np.float32)
    last = np.full(n_cells, -1)
    for t in range(n_days):
        has = last >= 0
        out[has, t] = t - last[has]
        last = np.where(arr[:, t] > 0, t, last)
    return out


def to_cell(cells_r7, res):
    return cells_r7 if res == 7 else np.array([h3.cell_to_parent(c, res) for c in cells_r7])


def load_inputs(res):
    ev = pd.read_csv(LABELS / "ignition_events.csv", parse_dates=["date"])
    ev["cell"] = to_cell(ev["h3_cell"].to_numpy(), res)
    fp = pd.read_csv(LABELS / "fire_footprints.csv.gz", parse_dates=["start_date", "end_date"])
    fp["cell"] = to_cell(fp["h3_cell"].to_numpy(), res)
    fp = fp.drop_duplicates(["cell", "fire_id"])
    om = pd.read_csv(LABELS / "onset_mask.csv", parse_dates=["date"])
    om["cell"] = to_cell(om["h3_cell"].to_numpy(), res)
    firms = ra.load_firms()[["h3_cell", "date"]]
    firms["cell"] = to_cell(firms["h3_cell"].to_numpy(), res) if res != 7 else firms["h3_cell"].to_numpy()
    return ev, fp, om, firms


def dense(df, pos, dates, value=None):
    """Count rows of df (columns cell, date) into a (cells, days) array."""
    arr = np.zeros((len(pos), len(dates)), dtype=np.int32)
    d0 = dates[0]
    di = ((df["date"] - d0).dt.days).to_numpy()
    ci = df["cell"].map(pos).to_numpy()
    ok = (~pd.isna(ci)) & (di >= 0) & (di < len(dates))
    w = np.ones(ok.sum(), dtype=np.int32) if value is None else value[ok].astype(np.int32)
    np.add.at(arr, (ci[ok].astype(np.int64), di[ok]), w)
    return arr


def build_year(year, grid, pos, rings, ev, fp, om, firms):
    hist = season_days(year, HISTORY_START)
    tgt_offset = (pd.Timestamp(year, *TARGET_START) - hist[0]).days
    n = len(grid)

    e = ev[ev["in_default_label"] & (ev["date"].dt.year == year)]
    y_all = dense(e, pos, hist)
    y_ge1 = dense(e[e["size_ha"] >= 1], pos, hist)
    y_a = dense(e[e["tier"] == "A"], pos, hist)
    y_l = dense(e[e["cause"] == "Lightning"], pos, hist)
    y_p = dense(e[e["cause"] == "Person"], pos, hist)
    max_size = np.zeros((n, len(hist)), dtype=np.float32)
    ok = e["cell"].map(pos).notna()
    ci = e.loc[ok, "cell"].map(pos).astype(int).to_numpy()
    di = (e.loc[ok, "date"] - hist[0]).dt.days.to_numpy()
    inr = (di >= 0) & (di < len(hist))
    np.maximum.at(max_size, (ci[inr], di[inr]), e.loc[ok, "size_ha"].fillna(0).to_numpy()[inr].astype(np.float32))

    mask = np.zeros((n, len(hist)), dtype=np.int8)
    f = fp[(fp["end_date"] >= hist[0]) & (fp["start_date"] <= hist[-1])]
    for cell, s, t in zip(f["cell"], f["start_date"], f["end_date"]):
        i = pos.get(cell)
        if i is None:
            continue
        a, b = max((s - hist[0]).days, 0), min((t - hist[0]).days, len(hist) - 1)
        mask[i, a:b + 1] |= MASK_FOOTPRINT
    o = om[om["date"].dt.year == year]
    mask[dense(o, pos, hist) > 0] |= MASK_ONSET
    masked = (mask > 0) & (y_all == 0)

    fdet = dense(firms[firms["date"].dt.year == year], pos, hist)
    r1, r2 = rings
    feats = {
        "f_firms_cell_1d": trailing(fdet, 1), "f_firms_cell_7d": trailing(fdet, 7),
        "f_firms_ring1_1d": trailing(ring_sum(fdet, r1), 1), "f_firms_ring1_7d": trailing(ring_sum(fdet, r1), 7),
        "f_firms_ring2_7d": trailing(ring_sum(fdet, r2), 7),
        "f_days_since_firms_cell": days_since(fdet),
        "f_bc_ign_cell_season": trailing(y_all, len(hist)),
        "f_bc_ign_ring1_7d": trailing(y_all + ring_sum(y_all, r1), 7),
        "f_bc_ign_ring2_7d": trailing(ring_sum(y_all, r2), 7),
        "f_bc_ign_ring1_1d": trailing(y_all + ring_sum(y_all, r1), 1),
    }
    sl = slice(tgt_offset, len(hist))
    dates = hist[sl]
    nd = len(dates)
    out = pd.DataFrame({
        "h3_cell": np.repeat(grid["cell"].to_numpy(), nd),
        "date": np.tile(dates.values, n),
        "region": np.repeat(grid["region"].to_numpy(), nd),
        "year": np.int16(year), "split": SPLIT[year],
        "y_ignition": (y_all[:, sl] > 0).ravel().astype(np.int8),
        "y_ignition_ge1ha": (y_ge1[:, sl] > 0).ravel().astype(np.int8),
        "y_n_ignitions": y_all[:, sl].ravel().astype(np.int16),
        "y_n_lightning": y_l[:, sl].ravel().astype(np.int16),
        "y_n_person": y_p[:, sl].ravel().astype(np.int16),
        "info_n_tier_a": y_a[:, sl].ravel().astype(np.int16),
        "info_max_size_ha": max_size[:, sl].ravel(),
        "masked": masked[:, sl].ravel(),
        "mask_reason": mask[:, sl].ravel(),
        "f_lat": np.repeat(grid["lat"].to_numpy(), nd), "f_lon": np.repeat(grid["lon"].to_numpy(), nd),
        "f_doy": np.tile(dates.dayofyear.values.astype(np.int16), n),
    })
    for k, v in feats.items():
        out[k] = v[:, sl].ravel().astype(np.float32 if k.startswith("f_days") else np.int32)
    for c in ("h3_cell", "region", "split"):
        out[c] = out[c].astype("category")
    return out, (y_all[:, sl], fdet)


def checks(year, out, ev, grid, pos, fdet_full, hist_offset):
    e = ev[ev["in_default_label"] & (ev["date"].dt.year == year)]
    in_scope = e[e["cell"].isin(pos) & ra.season_mask((e["date"] - pd.Timestamp("1970-01-01")).dt.days)]
    res = dict(year=year, events_in_scope=len(in_scope), events_total=len(e),
               events_outside_grid=int((~e["cell"].isin(pos)).sum()),
               positives_cell_days=int(out["y_ignition"].sum()), events_counted=int(out["y_n_ignitions"].sum()))
    assert res["events_counted"] == len(in_scope), res
    assert not (out["masked"] & (out["y_ignition"] == 1)).any()
    # Past-only check: recompute f_firms_cell_7d for a sample of rows directly from the raw dense array.
    rng = np.random.default_rng(year)
    nd = out["date"].nunique()
    for r in rng.choice(len(out), 2000, replace=False):
        ci, t = divmod(r, nd)
        t_full = t + hist_offset
        expect = fdet_full[ci, max(t_full - 7, 0):t_full].sum()
        assert out["f_firms_cell_7d"].iat[r] == expect
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=6, choices=[5, 6, 7])
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/ignition_dataset")
    args = p.parse_args()
    out_dir = args.output_dir / f"res{args.resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(args.resolution)
    grid.to_parquet(out_dir / "grid.parquet", index=False)
    pos = {c: i for i, c in enumerate(grid["cell"])}
    rings = (ring_index(grid["cell"].tolist(), 1), ring_index(grid["cell"].tolist(), 2))
    ev, fp, om, firms = load_inputs(args.resolution)
    print(f"grid: {len(grid):,} cells at resolution {args.resolution}", flush=True)

    summaries, check_rows = [], []
    for year in YEARS:
        out, (_, fdet) = build_year(year, grid, pos, rings, ev, fp, om, firms)
        offset = (pd.Timestamp(year, *TARGET_START) - pd.Timestamp(year, *HISTORY_START)).days
        check_rows.append(checks(year, out, ev, grid, pos, fdet, offset))
        out.to_parquet(out_dir / f"ignition_{year}.parquet", index=False)
        s = out.assign(neg=(~out["masked"]) & (out["y_ignition"] == 0)).groupby("region", observed=True).agg(
            rows=("y_ignition", "size"), positives=("y_ignition", "sum"), positives_ge1ha=("y_ignition_ge1ha", "sum"),
            masked=("masked", "sum"), negatives=("neg", "sum"))
        s["year"], s["split"] = year, SPLIT[year]
        summaries.append(s.reset_index())
        print(f"{year}: {len(out):,} rows, {int(out['y_ignition'].sum()):,} positive cell-days, "
              f"{int(out['masked'].sum()):,} masked", flush=True)
    summ = pd.concat(summaries, ignore_index=True)
    summ["prevalence"] = summ["positives"] / (summ["positives"] + summ["negatives"])
    summ.to_csv(out_dir / "summary_by_region_year.csv", index=False)
    by_year = summ.groupby(["year", "split"])[["rows", "positives", "positives_ge1ha", "masked", "negatives"]].sum()
    by_year["prevalence"] = by_year["positives"] / (by_year["positives"] + by_year["negatives"])
    by_year["masked_share"] = by_year["masked"] / by_year["rows"]
    by_year.to_csv(out_dir / "summary_by_year.csv")
    pd.DataFrame(check_rows).to_csv(out_dir / "checks.csv", index=False)
    manifest = dict(resolution=args.resolution, cells=len(grid), years=YEARS, split=SPLIT,
                    target_dates="May 8 - Sep 30", history_start="May 1 (within season; no cross-season history)",
                    label_source=str(LABELS.relative_to(ROOT)), default_label="tiers A+B",
                    mask_reason_bits={"footprint": MASK_FOOTPRINT, "onset": MASK_ONSET},
                    features_past_only="all f_* use data through t-1",
                    label_only_columns=["info_n_tier_a", "info_max_size_ha"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.width", 200)
    print(by_year)
    print(pd.DataFrame(check_rows))


if __name__ == "__main__":
    main()
