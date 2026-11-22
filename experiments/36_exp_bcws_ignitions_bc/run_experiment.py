"""Predict where a wildfire will start: BC Wildfire Service (BCWS) ignitions as the label.

The label for target day t: a BCWS incident of type "Fire" whose point is in the cell and whose
ignition date (Pacific time) falls on any of days t .. t+6 ("will a wildfire start here in the next
7 days"). Ignitions up to October 6 count, so late-September windows are complete. Satellites see
only about a fifth of these fires; the label covers every reported wildfire, including small ones,
and leaves out unreported burns. Every cell-day is a candidate (no quiet-week condition). Inputs
are known by the end of day t-1; Experiment 13's BCWS inputs only see records dated t-2 or earlier,
so they never see a fire being predicted.

Scored at three sizes of area, since "will a fire start in this area" depends on the area:
res-7 cells (about 5 km2), and their res-6 (about 36 km2) and res-5 (about 250 km2) parents
within each Fire Centre. An area is positive if any of its cells has an ignition; its score is
1 - prod(1 - p) over its cells (probabilities corrected for the 1% draw), or the maximum for
simple rankings.

Models, CatBoost with Experiment 21's settings, trained on 2018-2021 (every ignition plus a 1%
draw of the other cell-days, Experiment 21's seeds):
  ign_drivers: Experiment 21's spread_noday inputs;
  ign_lightning: + lightning over 7 and 14 days (fires can smoulder for days after a strike);
  ign_full: ign_lightning + Experiment 18's static layers.
References on the same cell-days: Experiment 21's next-day satellite model, Experiment 34's
satellite fire-start model, rankings by recent lightning and by distance to an active BCWS
wildfire, and random. Thresholds maximize 2022 F1 per Fire Centre and area size; 2023 is
evaluated; 2024 is not loaded. Per fire: whether its area was listed (or flagged) on any of the 7
forecast days whose window holds the fire, and on the first of them (a full week ahead).

    python run_experiment.py                      # full run
    python run_experiment.py --regions Southeast  # smoke test
    python run_experiment.py --resume             # load saved models, skip finished region-years
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import pickle
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_20", EXPERIMENTS / "34_exp_fire_start_bc/run_experiment.py")
experiment_20 = sys.modules["experiment_20"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_20)
import experiment_15  # noqa: E402  (registered by experiment_20)
from experiment_20 import EXPERIMENT_16_MODEL, START  # noqa: E402
from experiment_16 import CWFIS, FIRMS, LIGHTNING, STATIONS, WEATHER_DAILY, Drivers, Labels, box_sum, build, day_window  # noqa: E402
from experiment_13 import BOOTSTRAP, STATIC, STATIC_DIR, StaticLayers, add_static, daily_counts, f1_from_counts  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve, membership  # noqa: E402
from experiment_08 import (BCWS, Context, FEATURE_SETS, QUIET_SAMPLE_RATE, REFERENCE, SEASON_DAYS, SEED,  # noqa: E402
                           SELECTION_YEAR, EVALUATION_YEAR, TRAIN_YEARS, catboost_model, design, digest, load,
                           predict, raw_scores, season_index, window)

EXPERIMENT_20_MODEL = EXPERIMENTS / "34_exp_fire_start_bc/results/models/start_drivers.cbm"
IGNITION = "bcws_ignition_7d"
HORIZON = 7  # days t .. t+6
LIGHTNING_LONG = ["ltg_3x3_7d", "ltg_3x3_14d", "ltg_9x9_7d", "ltg_9x9_14d"]
FEATURE_SETS["ign_drivers"] = FEATURE_SETS["spread_noday"]
FEATURE_SETS["ign_lightning"] = FEATURE_SETS["spread_noday"] + LIGHTNING_LONG
FEATURE_SETS["ign_full"] = FEATURE_SETS["ign_lightning"] + STATIC
MODELS = {"ign_drivers": "ign_drivers", "ign_lightning": "ign_lightning", "ign_full": "ign_full"}
LOADED = {"catboost_spread_noday": ("spread_noday", EXPERIMENT_16_MODEL), "start_drivers": ("start_drivers", EXPERIMENT_20_MODEL)}
SIMPLE = ("lightning_14d", "near_active_bcws")
SCORED = (*MODELS, *LOADED, *SIMPLE)
RESOLUTIONS = (7, 6, 5)
WATCH_SHARES = (0.001, 0.01, 0.05)  # share of a Fire Centre's areas listed each day
CHUNK = 500_000
BASELINE = "catboost_spread_noday"


# ---------------------------------------------------------------- label and inputs

def ignition_grid(incidents, cells, year):
    """(cells x 153+6) bool from May 1 to October 6: a BCWS wildfire ignited in the cell that day;
    and the per-fire table."""
    days = SEASON_DAYS + HORIZON - 1
    first = pd.Timestamp(f"{year}-05-01")
    part = incidents.loc[incidents.IGNITION_DATE.between(first, first + pd.Timedelta(days=days - 1))]
    cell = cells.get_indexer([h3.latlng_to_cell(lat, lon, 7) for lat, lon in zip(part.LATITUDE, part.LONGITUDE)])
    day = (part.IGNITION_DATE - first).dt.days.to_numpy()
    keep = cell >= 0
    grid = np.zeros((len(cells), days), bool)
    grid[cell[keep], day[keep]] = True
    fires = pd.DataFrame({"cell": cell[keep], "day": day[keep], "cause": part.FIRE_CAUSE.to_numpy()[keep],
                          "size_ha": part.CURRENT_SIZE.to_numpy()[keep]})
    return grid, fires


def label_window(grid):
    """(cells x 153): a wildfire starts in the cell on any of days t .. t+6."""
    cs = np.concatenate([np.zeros((grid.shape[0], 1), np.int32), grid.cumsum(axis=1, dtype=np.int32)], axis=1)
    return (cs[:, HORIZON:HORIZON + SEASON_DAYS] - cs[:, :SEASON_DAYS]) > 0


class LongLightning:
    """Experiment 21's lightning blocks summed over 7 and 14 days, by history day."""

    def __init__(self, drivers):
        self.drivers, self._key, self._out = drivers, None, None

    def arrays(self, region, year):
        if self._key != (region, year):
            d, geo = self.drivers, self.drivers.geometry[region]
            flashes = d.ltg[d.ltg_years.index(year)]
            pick = lambda a: a[:, geo["iy"], geo["ix"]].T
            out = {}
            for radius, name in ((1, "3x3"), (4, "9x9")):
                block = box_sum(flashes, radius)
                for days in (7, 14):
                    out[f"ltg_{name}_{days}d"] = pick(day_window(block, days))
            self._key, self._out = (region, year), out
        return self._out

    def add(self, region, year, frame, cells):
        arrays = self.arrays(region, year)
        cell, history = cells.get_indexer(frame.h3_cell), season_index(frame.history_end_date)
        for name, values in arrays.items():
            frame[name] = values[cell, history]
        return frame


def full_build(context, labels, drivers, static, long_lightning, region, year, frame):
    cells = context.grids[region]["cells"]
    frame = build(context, labels, drivers, region, year, frame)
    return long_lightning.add(region, year, add_static(static, context, region, year, frame), cells)


def calibrated(model, chunk, feature_set):
    """Ignition probability: the model saw every ignition but 1% of other cell-days."""
    out = []
    for begin in range(0, len(chunk), 65_536):
        X, _ = design(chunk.iloc[begin:begin + 65_536], feature_set)
        out.append(expit(raw_scores(model, X) + np.log(QUIET_SAMPLE_RATE)))
    return np.concatenate(out).astype(np.float32)


def training_sample(regions, all_regions, context, labels, drivers, static, long_lightning, incidents):
    parts, stats = [], []
    columns = list(dict.fromkeys(c for s in MODELS.values() for c in FEATURE_SETS[s]))
    for region in regions:
        index = all_regions.index(region)
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            grid, fires = ignition_grid(incidents, cells, year)
            positive = label_window(grid)[cells.get_indexer(frame.h3_cell), season_index(frame.target_date)]
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            keep = positive | drawn
            part = frame.loc[keep].reset_index(drop=True).assign(**{IGNITION: positive[keep].astype(np.int8)})
            part = full_build(context, labels, drivers, static, long_lightning, region, year, part)
            parts.append(part[list(dict.fromkeys(columns + experiment_20.QUIET)) + [IGNITION]].astype(np.float32).assign(region=region, year=year))
            stats.append({"region": region, "year": year, "rows": len(frame), "fires": len(fires),
                          "positive_cell_days": int(positive.sum()), "kept_rows": int(keep.sum())})
            print(f"  {region} {year}: {len(fires):,} fires; positive cell-days {positive.sum():,}; kept {keep.sum():,}", flush=True)
            del frame, part, labels.grids[region, year]
            gc.collect()
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


# ---------------------------------------------------------------- scoring

def area_maps(cells):
    """Per resolution: sparse (areas x cells) membership, or None for the cells themselves."""
    maps = {7: None}
    for resolution in RESOLUTIONS[1:]:
        parent = [h3.cell_to_parent(c, resolution) for c in cells]
        maps[resolution] = membership(parent, sorted(set(parent)))
    return maps


def to_area(m, score, probability):
    if m is None:
        return score
    if probability:  # P(at least one ignition) if cells were independent
        return (1 - np.exp(m @ np.log1p(-np.minimum(score, 1 - 1e-7)))).astype(np.float32)
    starts, ends = m.indptr[:-1], m.indptr[1:]
    return np.maximum.reduceat(score[m.indices], starts, axis=0).astype(np.float32)


def watch_lists(score, tiebreak, sizes):
    """Per day, the n highest-scoring areas for each n in sizes (ties broken at random)."""
    listed = {n: np.zeros(score.shape, bool) for n in sizes}
    top = max(sizes)
    for day in range(score.shape[1]):
        order = np.lexsort((tiebreak[:, day], score[:, day]))[::-1][:top]
        for n in sizes:
            listed[n][order[:n], day] = True
    return listed


def score_region_year(region, region_index, year, context, labels, drivers, static, long_lightning, models, thresholds, incidents):
    tic = time.time()
    cells = context.grids[region]["cells"]
    frame = load(region, year, with_cells=True)
    n = len(frame)
    assert n == len(cells) * (SEASON_DAYS - FIRST_TARGET_DAY), "expected a full cell x day grid"
    cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
    scores = {name: np.empty(n, np.float32) for name in SCORED}
    for begin in range(0, n, CHUNK):
        chunk = full_build(context, labels, drivers, static, long_lightning, region, year,
                           frame.iloc[begin:begin + CHUNK].reset_index(drop=True))
        part = slice(begin, begin + len(chunk))
        for name, feature_set in MODELS.items():
            scores[name][part] = calibrated(models[name], chunk, feature_set)
        for name, (feature_set, _) in LOADED.items():
            scores[name][part] = predict(models[name], chunk, feature_set)
        scores["lightning_14d"][part] = (np.minimum(chunk.ltg_3x3_14d, 9999) * 1e4 + np.minimum(chunk.ltg_9x9_14d, 9999)).to_numpy(np.float32)
        scores["near_active_bcws"][part] = (-chunk.bcws_dist_active_km).to_numpy(np.float32)
        del chunk
    del frame
    grid, fires = ignition_grid(incidents, cells, year)
    truth7 = label_window(grid)[:, FIRST_TARGET_DAY:]
    # per-fire scores use fires inside the scored days, so all 7 forecasts covering them exist when day >= 6
    fires = fires.loc[fires.day.between(FIRST_TARGET_DAY, SEASON_DAYS - 1)].assign(day=lambda f: f.day - FIRST_TARGET_DAY)
    maps = area_maps(cells)
    out = {"rows": [], "daily": {}, "watch": [], "causes": [], "coverage": []}
    rng = np.random.default_rng([SEED, year, region_index])
    for resolution, m in maps.items():
        truth = truth7 if m is None else (m @ truth7.astype(np.float32)) > 0
        unit = fires.cell.to_numpy() if m is None else m.T.tocsr()[fires.cell.to_numpy()].indices  # each fire's area
        out["coverage"].append({"region": region, "year": year, "resolution": resolution, "areas": truth.shape[0],
                                "area_days": int(truth.size), "areas_with_ignition": int(truth.sum()), "fires": len(fires),
                                "base_rate": float(truth.mean())})
        tiebreak = rng.random(truth.shape).astype(np.float32)
        sizes = {share: max(1, math.ceil(share * truth.shape[0])) for share in WATCH_SHARES}
        for name in SCORED:
            score = to_area(m, to_grid(cell, target, scores[name], len(cells)), probability=name in MODELS or name in LOADED)
            tag = (region, f"res{resolution}", name)
            if year == SELECTION_YEAR:
                flat = score.ravel()
                candidates = candidate_thresholds(flat)
                truth_flat = truth.ravel()
                thresholds[tag] = float(candidates[int(np.argmax(curve(flat, truth_flat, truth_flat, flat, candidates)[2]))])
            counts = daily_counts(score, truth, truth, score, thresholds[tag])
            out["daily"][tag] = counts
            total = counts.sum(axis=0)
            out["rows"].append({"region": region, "year": year, "resolution": resolution, "model": name,
                                "threshold": thresholds[tag], "precision": total[1] / max(total[0], 1),
                                "recall": total[3] / max(total[2], 1), "f1": float(f1_from_counts(total)),
                                "flagged": int(total[0]), "positives": int(total[2]),
                                "average_precision": float(average_precision_score(truth.ravel(), score.ravel()))})
            flagged = score >= thresholds[tag]
            lists = watch_lists(score, tiebreak, set(sizes.values()))
            for share, n_listed in sizes.items():
                listed = lists[n_listed]
                out["watch"].append({"region": region, "year": year, "resolution": resolution, "model": name, "share": share,
                                     "listed_per_day": n_listed, "positive_area_days": int(truth.sum()),
                                     "caught": int((listed & truth).sum())})
                if share == 0.01 and len(fires):
                    out["causes"] += per_fire(region, year, resolution, name, fires, unit, listed, flagged)
        for share, n_listed in sizes.items():  # a random list, in expectation
            p = n_listed / truth.shape[0]
            out["watch"].append({"region": region, "year": year, "resolution": resolution, "model": "random", "share": share,
                                 "listed_per_day": n_listed, "positive_area_days": int(truth.sum()), "caught": p * truth.sum()})
    del scores
    rows = {(r["resolution"], r["model"]): r for r in out["rows"]}
    watch = {(r["resolution"], r["model"]): r for r in out["watch"] if r["share"] == 0.01}
    lines = [f"{region} {year} ({time.time()-tic:.0f}s): {len(fires):,} BCWS wildfires in the scored days"]
    for resolution in RESOLUTIONS:
        lines.append(f"  res{resolution} F1: " + "; ".join(f"{m} {rows[resolution, m]['f1']:.3f}" for m in SCORED)
                     + " | AP: " + "; ".join(f"{m} {rows[resolution, m]['average_precision']:.4f}" for m in SCORED)
                     + f" (base {out['coverage'][RESOLUTIONS.index(resolution)]['base_rate']:.5f})"
                     + " | top 1%: " + "; ".join(f"{m} {watch[resolution, m]['caught']/max(watch[resolution, m]['positive_area_days'], 1):.2f}"
                                                         for m in (*SCORED, "random")))
    print("\n".join(lines), flush=True)
    return out


def per_fire(region, year, resolution, name, fires, unit, listed, flagged):
    """Per cause: fires whose area was listed (top 1%) or flagged on any of the 7 forecast days whose
    window holds the fire (t = d-6 .. d), and on the first of them (a full week ahead)."""
    rows, day = [], fires.day.to_numpy()
    in_week = {k: (window(a.astype(np.float32), HORIZON) > 0)[unit, day] for k, a in (("listed", listed), ("flagged", flagged))}
    ahead = day >= HORIZON - 1  # the forecast issued a week ahead was scored
    first = {k: a[unit[ahead], day[ahead] - (HORIZON - 1)] for k, a in (("listed", listed), ("flagged", flagged))}
    for cause in ("Lightning", "Person", "all"):
        mask = np.ones(len(fires), bool) if cause == "all" else fires.cause.to_numpy() == cause
        rows.append({"region": region, "year": year, "resolution": resolution, "model": name, "cause": cause,
                     "fires": int(mask.sum()), "listed_in_week": int(in_week["listed"][mask].sum()),
                     "flagged_in_week": int(in_week["flagged"][mask].sum()), "fires_week_ahead": int(mask[ahead].sum()),
                     "listed_week_ahead": int(first["listed"][mask[ahead]].sum()),
                     "flagged_week_ahead": int(first["flagged"][mask[ahead]].sum())})
    return rows


def to_grid(cell, target, values, n_cells):
    return experiment_15.to_grid(cell, target, values, n_cells)


def bootstrap(daily, regions, names):
    dates = pd.date_range(f"{EVALUATION_YEAR}-05-01", periods=SEASON_DAYS)[FIRST_TARGET_DAY:]
    weeks = dates.isocalendar().week.to_numpy()
    unique_weeks = np.unique(weeks)
    week_member = (weeks[None, :] == unique_weeks[:, None]).astype(np.float64)
    draws = np.random.default_rng(SEED).integers(0, len(unique_weeks), size=(BOOTSTRAP, len(unique_weeks)))
    resample = np.stack([np.bincount(d, minlength=len(unique_weeks)) for d in draws]).astype(np.float64)
    boot = lambda counts: f1_from_counts(resample @ (week_member @ counts))
    rows = []
    for resolution in RESOLUTIONS:
        get = lambda model: np.sum([daily[r, f"res{resolution}", model] for r in regions], axis=0)
        for model in names:
            if model == BASELINE:
                continue
            difference = boot(get(model)) - boot(get(BASELINE))
            rows.append({"resolution": resolution, "model": model, "baseline": BASELINE,
                         "difference": float(f1_from_counts(get(model).sum(0)) - f1_from_counts(get(BASELINE).sum(0))),
                         "ci_low": float(np.quantile(difference, .025)), "ci_high": float(np.quantile(difference, .975)),
                         "share_above": float((difference > 0).mean())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--resume", action="store_true", help="load saved models and skip finished region-years")
    args = parser.parse_args()
    results = experiment_15.RESULTS = args.results
    started = time.time()
    (results / "models").mkdir(parents=True, exist_ok=True)
    (results / "parts").mkdir(exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    source_paths = [BCWS, FIRMS, CWFIS, WEATHER_DAILY, STATIONS, LIGHTNING, EXPERIMENT_16_MODEL, EXPERIMENT_20_MODEL,
                    *sorted(STATIC_DIR.glob("*.parquet"))]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    incidents = context.incidents  # Experiment 13's BCWS loader: type "Fire", Pacific dates, 2018-2023 only
    labels = Labels(context, regions)
    drivers = Drivers(context, regions)
    static = StaticLayers()
    long_lightning = LongLightning(drivers)

    models = {}
    for name, (_, path) in LOADED.items():
        models[name] = catboost_model()
        models[name].load_model(str(path))
    if args.resume:
        for name in MODELS:
            models[name] = catboost_model()
            models[name].load_model(str(results / "models" / f"{name}.cbm"))
        print("Loaded saved models", flush=True)
    else:
        print("Building training sample (every ignition plus 1% of other cell-days, 2018-2021)", flush=True)
        sample, sample_stats = training_sample(regions, all_regions, context, labels, drivers, static, long_lightning, incidents)
        sample_stats.to_csv(results / "training_sample.csv", index=False)
        fit_checks = []
        y = sample[IGNITION].to_numpy().astype(np.int8)
        for name, feature_set in MODELS.items():
            X, _ = design(sample, feature_set)
            models[name] = experiment_15.fit(name, X, y, FEATURE_SETS[feature_set] + ["quiet_stratum"], fit_checks)
            del X
        pd.DataFrame(fit_checks).to_csv(results / "fit_checks.csv", index=False)
        del sample
    for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:
        del labels.grids[key]
    labels.sources = None
    gc.collect()

    thresholds, parts = {}, []
    for region in regions:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            path = results / "parts" / f"{region.lower().replace(' ', '_')}_{year}.pkl"
            if args.resume and path.exists():
                out = pickle.loads(path.read_bytes())
                print(f"{region} {year}: loaded finished part", flush=True)
            else:
                out = score_region_year(region, all_regions.index(region), year, context, labels, drivers, static,
                                        long_lightning, models, thresholds, incidents)
                out["thresholds"] = {k: v for k, v in thresholds.items() if k[0] == region}
                path.write_bytes(pickle.dumps(out))
            thresholds.update(out["thresholds"])
            parts.append(out)
            gc.collect()

    metrics = pd.DataFrame([r for p in parts for r in p["rows"]])
    metrics.to_csv(results / "metrics_by_year.csv", index=False)
    coverage = pd.DataFrame([r for p in parts for r in p["coverage"]])
    coverage.to_csv(results / "coverage.csv", index=False)
    daily = {k: v for p in parts if p["coverage"][0]["year"] == EVALUATION_YEAR for k, v in p["daily"].items()}
    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["resolution", "model"])[["f1", "precision", "recall", "average_precision"]].median()
    summary = summary.add_prefix("median_").reset_index()
    for column, f in (("bc_wide_f1", f1_from_counts), ("bc_wide_precision", lambda t: t[1] / max(t[0], 1)),
                      ("bc_wide_recall", lambda t: t[3] / max(t[2], 1))):
        summary[column] = [float(f(np.sum([daily[r, f"res{res}", m] for r in regions], axis=0).sum(0)))
                           for res, m in zip(summary.resolution, summary.model)]
    summary.to_csv(results / "summary_2023.csv", index=False)
    watch = pd.DataFrame([r for p in parts for r in p["watch"]])
    watch.to_csv(results / "watch_list.csv", index=False)
    pooled = watch.loc[watch.year.eq(EVALUATION_YEAR)].groupby(["resolution", "share", "model"])[
        ["positive_area_days", "caught"]].sum()
    pooled["share_caught"] = pooled.caught / pooled.positive_area_days
    pooled.reset_index().to_csv(results / "watch_list_bc_2023.csv", index=False)
    causes = pd.DataFrame([r for p in parts for r in p["causes"]])
    causes.to_csv(results / "causes.csv", index=False)
    boot = bootstrap(daily, regions, list(SCORED))
    boot.to_csv(results / "bootstrap_2023.csv", index=False)

    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (results / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "label": "BCWS incident of type Fire with its point in the res-7 cell and ignition date (Pacific) on any of days t .. t+6 (ignitions to October 6 count)",
        "areas": {"7": "res-7 cell", "6": "res-6 parent within the Fire Centre", "5": "res-5 parent within the Fire Centre"},
        "area_score": "1 - prod(1 - p) over the area's cells for models (p corrected by log(0.01)); max for simple rankings",
        "training_rows": "every positive cell-day plus a 1% draw of other cell-days, Experiment 21's seeds", "horizon_days": HORIZON,
        "models": {k: {"feature_set": v, "inputs": FEATURE_SETS[v]} for k, v in MODELS.items()},
        "references": {"catboost_spread_noday": str(EXPERIMENT_16_MODEL.relative_to(EXPERIMENTS)),
                       "start_drivers": str(EXPERIMENT_20_MODEL.relative_to(EXPERIMENTS)),
                       "lightning_14d": "ltg_3x3_14d * 1e4 + ltg_9x9_14d", "near_active_bcws": "-bcws_dist_active_km",
                       "random": "expected counts of a random daily list"},
        "watch_lists": f"top {list(WATCH_SHARES)} of each Fire Centre's areas per day",
        "thresholds": "maximize 2022 F1 per Fire Centre, area size and model",
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023, against {BASELINE}",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(), flush=True)
    print(pooled.reset_index().query("share == 0.01").round(4).to_string(), flush=True)
    print(causes.loc[causes.year.eq(EVALUATION_YEAR)].groupby(["resolution", "model", "cause"])[
        ["fires", "listed_in_week", "fires_week_ahead", "listed_week_ahead"]].sum()
          .assign(share_week=lambda c: c.listed_in_week / c.fires, share_ahead=lambda c: c.listed_week_ahead / c.fires_week_ahead)
          .round(3).to_string(), flush=True)
    print(boot.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
