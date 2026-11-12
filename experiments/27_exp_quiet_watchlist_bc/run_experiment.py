"""Experiment 27: quiet-cell watch list across all six Fire Centres.

Experiment 18 tested a daily watch list in Kamloops only: each day, the N highest-ranked
quiet cells (no S-NPP detection in the cell or a ring-1 neighbour over the 7 days to the
history day). This experiment repeats that scoring for all six centres, using rankings that
already exist on disk, so no model is trained:

- catboost_13 / gbm_13: Experiment 13 fire-context models (2023 predictions only)
- mlp_lightning: Experiment 25's S-NPP + lightning MLP (2022-2024 full-grid scores)
- mlp_no_lightning: Experiment 26's matched MLP without lightning (2022-2024)
- nearby_activity, burned_before: Experiment 18's simple rankings, rebuilt from FIRMS
- random: expected share for a random list of quiet cells

Experiment 22's two-satellite MLP/GRU cannot be scored here: its weather and lightning
preprocessing inputs are not in this checkout.

Truth is a recorded S-NPP detection in the cell on the target day. On quiet cells this equals
Experiment 25's gap-filled target, because a quiet cell has no detection the day before.
Nothing is tuned: lists are fixed sizes, so every year is scored the same way. 2024 was
already inspected in Experiments 25-26 and is reported as a secondary year.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
RESULTS = HERE / "results"
DETECTIONS = EXPERIMENTS / "firms_summer_2018_2024/data/firms_h3_cell_date.csv"
EXP13 = EXPERIMENTS / "13_exp_fire_context_bc/results/predictions"
EXP25 = ROOT / "data/snpp_lightning_h3r7/evaluation"
EXP26 = EXPERIMENTS / "26_exp_lightning_ablation/results"
ONSETS = EXPERIMENTS / "16_exp_new_fire_onsets_bc/results"

REGIONS = ["Cariboo", "Coastal", "Kamloops", "Northwest", "Prince George", "Southeast"]
YEARS = (2022, 2023, 2024)
SEASON_DAYS, FIRST_TARGET_DAY, TARGET_DAYS = 153, 7, 146  # index 0 = May 1; targets May 8 - Sep 30
MAX_RING, WATCH_DAYS = 5, 7
SIZES = (10, 25, 50, 100)
SEED, BOOTSTRAP = 27, 2000
EXP18_KAMLOOPS_2023 = {"catboost_13": (0.207, 0.273), "nearby_activity": (0.058, 0.078),
                       "burned_before": (0.013, 0.013)}  # (caught next day, caught within 7 days), top 50


def slug(region):
    return region.lower().replace(" ", "_")


def window(a, w):
    """Sum over the w columns ending at each column (inclusive), truncated at column 0."""
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1, dtype=np.float64)], axis=1)
    end = np.arange(1, a.shape[1] + 1)
    return (cs[:, end] - cs[:, np.maximum(end - w, 0)]).astype(np.float32)


def forward_window(a, w):
    return window(a[:, ::-1], w)[:, ::-1]


# ---------------------------------------------------------------- detections and simple rankings

def detection_arrays():
    d = pd.read_csv(DETECTIONS, usecols=["h3_cell", "acq_date", "detection_count"], parse_dates=["acq_date"])
    d = d.loc[d.acq_date.dt.month.between(5, 9)]
    d["day"] = (d.acq_date - pd.to_datetime(d.acq_date.dt.year.astype(str) + "-05-01")).dt.days
    return d


_RINGS: dict = {}


def ring_matrices(cells, universe):
    position = {c: i for i, c in enumerate(universe)}
    mats = []
    for k in range(MAX_RING + 1):
        rows, cols = [], []
        for i, cell in enumerate(cells):
            for other in (h3.grid_ring(cell, k) if k else [cell]):
                j = position.get(other)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        mats.append(sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(cells), len(universe))))
    return mats


def simple_rankings(cells, detections, year):
    """Experiment 18's nearby_activity and burned_before, as cells x target-day grids."""
    universe = pd.Index(sorted(set(detections.h3_cell)))
    key = (len(cells), cells[0], cells[-1])
    if key not in _RINGS:
        _RINGS.clear()
        _RINGS[key] = ring_matrices(cells, universe)
    rings = _RINGS[key]
    disk1 = rings[0] + rings[1]

    def indicator(y):
        part = detections.loc[detections.acq_date.dt.year.eq(y)]
        a = np.zeros((len(universe), SEASON_DAYS), np.float32)
        a[universe.get_indexer(part.h3_cell), part.day.to_numpy()] = 1
        return a

    ind30 = (window(indicator(year), 30) > 0).astype(np.float32)
    distance = np.full((len(cells), SEASON_DAYS), MAX_RING + 1, np.float32)
    active = np.zeros((len(cells), SEASON_DAYS), np.float32)
    for k in range(MAX_RING, -1, -1):
        hits = rings[k] @ ind30
        distance[hits > 0] = k
        active += hits
    nearby = (MAX_RING + 1 - distance) * 1e4 + np.minimum(active, 9999)
    cell_years = np.zeros(len(cells), np.float32)
    disk_years = np.zeros(len(cells), np.float32)
    for y in range(2018, year):
        burned = (indicator(y).sum(axis=1) > 0).astype(np.float32)
        cell_years += (rings[0] @ burned) > 0
        disk_years += (disk1 @ burned) > 0
    burned_before = np.repeat((cell_years * 10 + disk_years)[:, None], TARGET_DAYS, axis=1)
    # Target day t uses history day t-1.
    history = slice(FIRST_TARGET_DAY - 1, SEASON_DAYS - 1)
    return {"nearby_activity": nearby[:, history], "burned_before": burned_before}


# ---------------------------------------------------------------- model scores

def load_exp25(region, year):
    path = EXP25 / str(year) / f"{slug(region)}_snpp_lightning_full_grid.parquet"
    t = pq.read_table(path, columns=["h3_cell", "target_date", "quiet_stratum", "target_snpp_filled", "mlp_score"])
    cells_long = t.column("h3_cell").to_numpy(zero_copy_only=False)
    dates = pd.to_datetime(t.column("target_date").to_numpy(zero_copy_only=False))
    n = len(cells_long) // TARGET_DAYS
    assert n * TARGET_DAYS == len(cells_long)
    cells = pd.Index(cells_long[::TARGET_DAYS])
    assert (cells_long.reshape(n, TARGET_DAYS) == cells.to_numpy()[:, None]).all(), "expected cell-major rows"
    day = (dates - pd.Timestamp(f"{year}-05-01")).days.to_numpy().reshape(n, TARGET_DAYS)
    assert (day == np.arange(FIRST_TARGET_DAY, SEASON_DAYS)[None, :]).all(), "expected May 8 - Sep 30 per cell"
    grid = lambda name: t.column(name).to_numpy(zero_copy_only=False).reshape(n, TARGET_DAYS)
    no_ltg = np.load(EXP26 / f"scores_without_lightning_{slug(region)}_{year}.npy").reshape(n, TARGET_DAYS)
    return cells, grid("quiet_stratum") > 0, grid("target_snpp_filled").astype(bool), {
        "mlp_lightning": grid("mlp_score").astype(np.float32), "mlp_no_lightning": no_ltg}


def load_exp13(region, cells):
    f = pd.read_parquet(EXP13 / f"{slug(region)}_2023.parquet",
                        columns=["target_date", "h3_cell", "target_recorded_detection", "catboost__all", "gradient_boosting__all"])
    row = cells.get_indexer(f.h3_cell)
    assert (row >= 0).all() and len(f) == len(cells) * TARGET_DAYS, "Experiment 13 grid differs from Experiment 25"
    col = (f.target_date - pd.Timestamp("2023-05-01")).dt.days.to_numpy() - FIRST_TARGET_DAY
    out = {}
    for name, column in (("truth", "target_recorded_detection"), ("catboost_13", "catboost__all"),
                         ("gbm_13", "gradient_boosting__all")):
        g = np.zeros((len(cells), TARGET_DAYS), np.float32)
        g[row, col] = f[column].to_numpy()
        out[name] = g
    return out


# ---------------------------------------------------------------- watch list scoring

def ranked_lists(score, quiet, tiebreak):
    listed = {n: np.zeros(score.shape, bool) for n in SIZES}
    for day in range(score.shape[1]):
        candidates = np.flatnonzero(quiet[:, day])
        order = candidates[np.lexsort((tiebreak[candidates, day], score[candidates, day]))[::-1]]
        for n in SIZES:
            listed[n][order[:n], day] = True
    return listed


def onset_groups(region, cells, year):
    c = pd.read_csv(ONSETS / "onset_cells.csv", parse_dates=["date"])
    k = pd.read_csv(ONSETS / "onset_clusters.csv", usecols=["cluster", "bcws_category"])
    c = c.merge(k, on="cluster")
    c = c.loc[c.date.dt.year.eq(year)]
    row = cells.get_indexer(c.h3_cell)
    col = (c.date - pd.Timestamp(f"{year}-05-01")).dt.days.to_numpy() - FIRST_TARGET_DAY
    keep = (row >= 0) & (col >= 0)
    return row[keep], col[keep], c.bcws_category.to_numpy()[keep]


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    detections = detection_arrays()
    rows, ap_rows, onset_rows, checks, daily = [], [], [], [], {}
    for region_index, region in enumerate(REGIONS):
        for year in YEARS:
            tic = time.time()
            cells, quiet, truth, scores = load_exp25(region, year)
            if year == 2023:
                e13 = load_exp13(region, cells)
                truth13 = e13.pop("truth") > 0
                mismatch = int((truth13[quiet] != truth[quiet]).sum())
                checks.append({"region": region, "check": "quiet-cell truth: Exp 13 vs Exp 25", "mismatches": mismatch})
                truth = truth13  # Experiment 18's target
                scores.update(e13)
            scores.update(simple_rankings(list(cells), detections, year))
            quiet_fire = truth & quiet
            # A detection on any of the 7 target days starting today (for the 7-day hit rate).
            future = forward_window(truth.astype(np.float32), WATCH_DAYS) > 0
            tiebreak = np.random.default_rng([SEED, year, region_index]).random(truth.shape)
            n_quiet = quiet.sum(axis=0)
            fires_by_day = quiet_fire.sum(axis=0)
            if year == 2023:
                o_row, o_col, o_group = onset_groups(region, cells, year)
                is_quiet = quiet_fire[o_row, o_col]
                o_row, o_col, o_group = o_row[is_quiet], o_col[is_quiet], o_group[is_quiet]
            for n in SIZES:
                share = np.minimum(n, n_quiet) / np.maximum(n_quiet, 1)
                listed = np.minimum(n, n_quiet).sum()
                rows.append({"region": region, "year": year, "model": "random", "list_size": n,
                             "quiet_fires": int(quiet_fire.sum()), "listed_cell_days": int(listed),
                             "caught_next_day": float((share * fires_by_day).sum() / max(quiet_fire.sum(), 1)),
                             "caught_within_7d": np.nan,
                             "hit_rate_next_day": float((share * fires_by_day).sum() / max(listed, 1)),
                             "hit_rate_within_7d": float((share * (future & quiet).sum(axis=0)).sum() / max(listed, 1))})
            for name, score in scores.items():
                ap_rows.append({"region": region, "year": year, "model": name, "quiet_cell_days": int(quiet.sum()),
                                "quiet_fires": int(quiet_fire.sum()), "base_rate": float(truth[quiet].mean()),
                                "average_precision_quiet": float(average_precision_score(truth[quiet], score[quiet]))})
                for n, listed in ranked_lists(score, quiet, tiebreak).items():
                    recent = window(listed.astype(np.float32), WATCH_DAYS) > 0  # listed on day t-6 .. t
                    caught7 = recent & quiet_fire
                    rows.append({"region": region, "year": year, "model": name, "list_size": n,
                                 "quiet_fires": int(quiet_fire.sum()), "listed_cell_days": int(listed.sum()),
                                 "caught_next_day": float((listed & quiet_fire).sum() / max(quiet_fire.sum(), 1)),
                                 "caught_within_7d": float(caught7.sum() / max(quiet_fire.sum(), 1)),
                                 "hit_rate_next_day": float((listed & truth).sum() / max(listed.sum(), 1)),
                                 "hit_rate_within_7d": float((listed & future).sum() / max(listed.sum(), 1))})
                    daily[region, year, name, n] = np.stack([fires_by_day, caught7.sum(axis=0)], axis=1)
                    if year == 2023:
                        caught = recent[o_row, o_col]
                        for group in np.unique(o_group):
                            mask = o_group == group
                            onset_rows.append({"region": region, "model": name, "list_size": n, "onset_group": group,
                                               "quiet_onset_cells": int(mask.sum()), "caught_within_7d": int(caught[mask].sum())})
            top = {r["model"]: r["caught_within_7d"] for r in rows
                   if r["region"] == region and r["year"] == year and r["list_size"] == 50 and r["model"] != "random"}
            print(f"{region} {year} ({time.time() - tic:.0f}s) quiet fires {int(quiet_fire.sum()):,} | top-50 caught in 7d: "
                  + "; ".join(f"{k} {v:.3f}" for k, v in top.items()), flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "watch_metrics.csv", index=False)
    pd.DataFrame(ap_rows).to_csv(RESULTS / "quiet_average_precision.csv", index=False)
    onsets = pd.DataFrame(onset_rows)
    onsets.to_csv(RESULTS / "watch_onsets_2023.csv", index=False)

    # BC-wide pooled recall within 7 days (sum of caught / sum of quiet fires across centres).
    pooled = (metrics.loc[metrics.model.ne("random")]
              .assign(caught=lambda d: d.caught_within_7d * d.quiet_fires, caught1=lambda d: d.caught_next_day * d.quiet_fires,
                      hits=lambda d: d.hit_rate_next_day * d.listed_cell_days)
              .groupby(["year", "model", "list_size"])[["quiet_fires", "caught", "caught1", "hits", "listed_cell_days"]].sum())
    pooled["caught_within_7d"] = pooled.caught / pooled.quiet_fires
    pooled["caught_next_day"] = pooled.caught1 / pooled.quiet_fires
    pooled["hit_rate_next_day"] = pooled.hits / pooled.listed_cell_days
    rand = (metrics.loc[metrics.model.eq("random")].assign(c=lambda d: d.caught_next_day * d.quiet_fires)
            .groupby(["year", "list_size"])[["c", "quiet_fires"]].sum())
    pooled = pooled.reset_index()
    pooled["random_caught_next_day"] = [rand.loc[(y, n), "c"] / rand.loc[(y, n), "quiet_fires"]
                                        for y, n in zip(pooled.year, pooled.list_size)]
    centres_better = []
    for r in pooled.itertuples():
        per = metrics.loc[metrics.year.eq(r.year) & metrics.list_size.eq(r.list_size)].set_index(["region", "model"]).caught_within_7d
        centres_better.append(int(sum(per[g, r.model] > per[g, "nearby_activity"] for g in REGIONS)))
    pooled["centres_beating_nearby_activity"] = centres_better
    pooled[["year", "model", "list_size", "quiet_fires", "caught_next_day", "caught_within_7d", "hit_rate_next_day",
            "random_caught_next_day", "centres_beating_nearby_activity"]].to_csv(RESULTS / "watch_summary_bc.csv", index=False)

    # Week-block bootstrap of BC-wide recall-within-7-days differences.
    comparisons = [("mlp_lightning", "nearby_activity"), ("mlp_no_lightning", "nearby_activity"),
                   ("mlp_lightning", "mlp_no_lightning"), ("catboost_13", "nearby_activity"),
                   ("mlp_lightning", "catboost_13"), ("mlp_no_lightning", "catboost_13"), ("gbm_13", "catboost_13")]
    boot_rows = []
    rng = np.random.default_rng(SEED)
    for year in YEARS:
        dates = pd.date_range(f"{year}-05-08", f"{year}-09-30")
        week = pd.Index(dates.isocalendar().week.to_numpy())
        weeks = np.unique(week)
        members = [np.flatnonzero(week == w) for w in weeks]
        draws = [np.concatenate([members[i] for i in rng.integers(0, len(weeks), len(weeks))]) for _ in range(BOOTSTRAP)]
        for n in SIZES:
            for a, b in comparisons:
                if (REGIONS[0], year, a, n) not in daily or (REGIONS[0], year, b, n) not in daily:
                    continue
                da = sum(daily[g, year, a, n] for g in REGIONS)
                db = sum(daily[g, year, b, n] for g in REGIONS)
                diff = lambda idx: da[idx, 1].sum() / da[idx, 0].sum() - db[idx, 1].sum() / db[idx, 0].sum()
                values = np.array([diff(d) for d in draws])
                boot_rows.append({"year": year, "list_size": n, "model": a, "baseline": b,
                                  "difference": float(diff(np.arange(len(dates)))),
                                  "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
                                  "share_above_zero": float((values > 0).mean())})
    pd.DataFrame(boot_rows).to_csv(RESULTS / "bootstrap_bc.csv", index=False)

    # Reproduction check against Experiment 18 (Kamloops 2023, top 50).
    k = metrics.loc[metrics.region.eq("Kamloops") & metrics.year.eq(2023) & metrics.list_size.eq(50)].set_index("model")
    for name, (next_day, within) in EXP18_KAMLOOPS_2023.items():
        checks.append({"region": "Kamloops", "check": f"Exp 18 reproduction, {name}, top 50",
                       "exp18_next_day": next_day, "exp27_next_day": round(float(k.loc[name, "caught_next_day"]), 3),
                       "exp18_within_7d": within, "exp27_within_7d": round(float(k.loc[name, "caught_within_7d"]), 3)})
    checks.append({"region": "Kamloops", "check": "Exp 18 quiet fires 2023 = 450", "value": int(k.loc["catboost_13", "quiet_fires"])})
    pd.DataFrame(checks).to_csv(RESULTS / "checks.csv", index=False)

    sources = [DETECTIONS, ONSETS / "onset_cells.csv", ONSETS / "onset_clusters.csv"]
    (RESULTS / "manifest.json").write_text(json.dumps({
        "regions": REGIONS, "years": list(YEARS), "list_sizes": list(SIZES), "watch_days": WATCH_DAYS,
        "quiet": "no S-NPP detection in the cell or an in-region ring-1 neighbour over the 7 days to the history day "
                 "(Experiment 25's quiet_stratum, same definition as Experiments 13 and 18)",
        "caught_within_7d": "quiet fire whose cell was on the list on any of the 7 days up to and including the fire day",
        "rankings": {"catboost_13": "Experiment 13 catboost__all (2023 only)", "gbm_13": "Experiment 13 gradient_boosting__all (2023 only)",
                     "mlp_lightning": "Experiment 25 mlp_score", "mlp_no_lightning": "Experiment 26 scores_without_lightning",
                     "nearby_activity": "Experiment 18: rings to nearest detection in 30 days, then active cells within 5 rings",
                     "burned_before": "Experiment 18: earlier seasons (from 2018) the cell, then its ring-1 disk, had a detection"},
        "not_scored": "Experiment 22 MLP/GRU: weather and lightning preprocessing inputs are missing from this checkout",
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks, pooled over the six centres",
        "sources": [str(p.relative_to(ROOT)) for p in sources],
        "runtime_minutes": round((time.time() - started) / 60, 1)}, indent=2))
    print(f"Done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
