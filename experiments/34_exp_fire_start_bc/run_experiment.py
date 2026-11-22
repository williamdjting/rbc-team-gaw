"""Predict fire starts: the first detection in a cell after a quiet week.

A fire start on target day t: the label (S-NPP or NOAA-20 from 2021, S-NPP before, one-day gaps
filled; Experiment 20's training label, the full label from 2021) is positive in the cell, and
the same source detected nothing in the cell or its ring-1 neighbours on days t-7 .. t-1. The
cell-day must also be in Experiment 13's quiet stratum (no S-NPP count in the cell or its
neighbours over the 7 days); the two differ on a few cell-days at Fire Centre borders. Only such
quiet cell-days can start a fire, so every model is trained and scored on quiet cell-days.
Before 2021 this is Experiment 16's onset definition.

Models, trained on quiet cell-days of 2018-2021 (every start plus Experiment 21's 1% draw of the
rest), with Experiment 21's CatBoost settings:
  start_drivers: Experiment 21's spread_noday inputs (fire history, BCWS incidents, weather, lightning);
  start_static: + Experiment 18's static layers (terrain, fuels, farmland reserve, oil and gas, cutblocks);
  start_static_day: start_static + day_of_year.
References, scored on the same quiet cell-days:
  catboost_spread_noday: Experiment 21's next-day model (loaded; trained on all cell-days);
  nearby_activity: Experiment 18's ranking on 3 days (nearest detection, then active cells within 5 rings);
  lightning: flashes in the 3x3 block over 3 days, then the 9x9 block;
  random (watch lists only, in expectation).
Thresholds maximize 2022 F1 per Fire Centre; 2023 is evaluated; 2024 is not loaded. Metrics: F1
on the exact cell and with a one-cell margin, average precision, daily watch lists of the top N
quiet cells per Fire Centre, Experiment 16's onset groups, and week-block bootstrap intervals
against Experiment 21's model.

    python run_experiment.py                      # full run
    python run_experiment.py --regions Southeast  # smoke test
    python run_experiment.py --resume             # load saved models, skip finished region-years
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_16", EXPERIMENTS / "21_exp_weather_lightning_spread_bc/run_experiment.py")
experiment_16 = sys.modules["experiment_16"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_16)
import experiment_15  # noqa: E402  (registered by experiment_16)
from experiment_16 import (CWFIS, FIRMS, FULL_NOAA20_FROM, LIGHTNING, STATIONS, WEATHER_DAILY, Drivers, Labels,  # noqa: E402
                           build, shift, to_grid)
from experiment_13 import (BOOTSTRAP, STATIC, STATIC_DIR, StaticLayers, add_static, daily_counts, f1_from_counts,  # noqa: E402
                           forward_window, ranked_lists)
from experiment_12 import ONSETS, onset_categories  # noqa: E402
from experiment_09 import FIRST_TARGET_DAY, candidate_thresholds, curve  # noqa: E402
from experiment_08 import (QUIET, Context, FEATURE_SETS, MAX_RING, QUIET_SAMPLE_RATE, REFERENCE, SEASON_DAYS, SEED,  # noqa: E402
                           SELECTION_YEAR, EVALUATION_YEAR, TRAIN_YEARS, catboost_model, design, digest, load, predict,
                           quiet_stratum, season_index, sparse_row_max, window)

EXPERIMENT_16_MODEL = EXPERIMENTS / "21_exp_weather_lightning_spread_bc/results/models/catboost_spread_noday.cbm"
START = "fire_start"
QUIET_DAYS = 7
FEATURE_SETS["start_drivers"] = FEATURE_SETS["spread_noday"]
FEATURE_SETS["start_static"] = FEATURE_SETS["spread_noday"] + STATIC
FEATURE_SETS["start_static_day"] = FEATURE_SETS["start_static"] + ["day_of_year"]
MODELS = {"start_drivers": "start_drivers", "start_static": "start_static", "start_static_day": "start_static_day"}
LOADED = {"catboost_spread_noday": "spread_noday"}
SIMPLE = ("nearby_activity", "lightning")
SCORED = (*MODELS, *LOADED, *SIMPLE)
DEFINITIONS = ("res7_exact", "res7_margin1")
WATCH_SIZES = (10, 25, 50, 100)
WATCH_DAYS = 7
CHUNK = 500_000


def start_grids(g, year):
    """(cells x 153) quiet and fire-start grids by target day."""
    fire = g["pair"] if year >= FULL_NOAA20_FROM else g["snpp"]  # the source of that year's label
    burning_near = ((g["disk1"] @ fire.astype(np.float32)) > 0).astype(np.float32)
    quiet = shift(window(burning_near, QUIET_DAYS) == 0, 1)  # nothing in the disk on t-7 .. t-1
    quiet[:, :QUIET_DAYS] = False  # the window would reach before May 1
    return quiet, g["label"] & quiet


def simple_scores(chunk):
    nearby = (MAX_RING + 1 - chunk.dist_nearest_3d) * 1e4 + np.minimum(chunk.active_cells_disk5_3d, 9999)
    lightning = np.minimum(chunk.ltg_3x3_3d, 9999) * 1e4 + np.minimum(chunk.ltg_9x9_3d, 9999)
    return {"nearby_activity": nearby.to_numpy(np.float32), "lightning": lightning.to_numpy(np.float32)}


def training_sample(regions, all_regions, context, labels, drivers, static):
    """Quiet cell-days of 2018-2021: every fire start plus Experiment 21's 1% draw (same seeds)."""
    parts, stats = [], []
    columns = list(dict.fromkeys(c for s in MODELS.values() for c in FEATURE_SETS[s]))
    for region in regions:
        index = all_regions.index(region)
        cells = context.grids[region]["cells"]
        for year in TRAIN_YEARS:
            frame = load(region, year, with_cells=True)
            quiet, start = start_grids(labels.grids[region, year], year)
            cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
            quiet = quiet[cell, target] & quiet_stratum(frame)
            start = start[cell, target] & quiet
            drawn = np.random.default_rng([SEED, year, index]).random(len(frame)) < QUIET_SAMPLE_RATE
            keep = quiet & (start | drawn)
            part = frame.loc[keep].reset_index(drop=True).assign(**{START: start[keep].astype(np.int8)})
            part = add_static(static, context, region, year, build(context, labels, drivers, region, year, part))
            assert part[START].sum() == start.sum()
            parts.append(part[list(dict.fromkeys(columns + QUIET)) + [START]].assign(region=region, year=year))
            stats.append({"region": region, "year": year, "rows": len(frame), "quiet_rows": int(quiet.sum()),
                          "fire_starts": int(start.sum()), "kept_rows": int(keep.sum())})
            print(f"  {region} {year}: quiet {quiet.sum():,}; starts {start.sum():,}; kept {keep.sum():,}", flush=True)
            del frame, part
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def watch_counts(listed, start, quiet, future, disk1):
    """Counts for one daily watch list (cells x days, booleans)."""
    recent = window(listed.astype(np.float32), WATCH_DAYS) > 0  # listed on t-6 .. t
    near_listed = sparse_row_max(disk1, listed.astype(np.float32)) > 0
    return {"listed": int(listed.sum()), "caught_next_day": int((listed & start).sum()),
            "caught_next_day_margin": int((near_listed & start).sum()), "caught_within_7d": int((recent & start).sum()),
            "listed_start_within_7d": int((listed & future).sum())}


def random_watch_counts(n, start, quiet, future):
    """Expected counts for a list of n quiet cells drawn at random each day."""
    share = np.minimum(n, quiet.sum(0)) / np.maximum(quiet.sum(0), 1)
    missed = np.exp(window(np.log1p(-np.minimum(share, 1 - 1e-12))[None, :], WATCH_DAYS)[0])  # never listed on t-6 .. t
    per_day_future = (future & quiet).sum(0) / np.maximum(quiet.sum(0), 1)
    return {"listed": float(np.minimum(n, quiet.sum(0)).sum()), "caught_next_day": float((share * start.sum(0)).sum()),
            "caught_next_day_margin": np.nan, "caught_within_7d": float(((1 - missed) * start.sum(0)).sum()),
            "listed_start_within_7d": float((np.minimum(n, quiet.sum(0)) * per_day_future).sum())}


def score_region_year(region, region_index, year, context, labels, drivers, static, models, thresholds, onsets):
    tic = time.time()
    cells = context.grids[region]["cells"]
    g = labels.grids[region, year]
    frame = load(region, year, with_cells=True)
    n = len(frame)
    assert n == len(cells) * (SEASON_DAYS - FIRST_TARGET_DAY), "expected a full cell x day grid"
    cell, target = cells.get_indexer(frame.h3_cell), season_index(frame.target_date)
    scores = {name: np.empty(n, np.float32) for name in SCORED}
    for begin in range(0, n, CHUNK):
        chunk = build(context, labels, drivers, region, year, frame.iloc[begin:begin + CHUNK].reset_index(drop=True))
        chunk = add_static(static, context, region, year, chunk)
        part = slice(begin, begin + len(chunk))
        for name, feature_set in {**MODELS, **LOADED}.items():
            scores[name][part] = predict(models[name], chunk, feature_set)
        for name, values in simple_scores(chunk).items():
            scores[name][part] = values
        del chunk
    stratum = to_grid(cell, target, quiet_stratum(frame).astype(np.float32), len(cells)) > 0
    del frame
    quiet, start = (a[:, FIRST_TARGET_DAY:] & stratum for a in start_grids(g, year))
    disk1 = g["disk1"]
    near = (disk1 @ start.astype(np.float32)) > 0  # a start in the cell or a neighbour
    future = forward_window(start.astype(np.float32), WATCH_DAYS) > 0  # a start in the cell on t .. t+6
    tiebreak = np.random.default_rng([SEED, year, region_index]).random(quiet.shape).astype(np.float32)
    out = {"rows": [], "daily": {}, "watch": [], "groups": [],
           "coverage": {"region": region, "year": year, "cell_days": int(quiet.size), "quiet_cell_days": int(quiet.sum()),
                        "fire_starts": int(start.sum()), "label_positives": int(g["label"][:, FIRST_TARGET_DAY:].sum()),
                        "base_rate": float(start.sum() / quiet.sum())}}
    group_cells = group_days = group_names = None
    if onsets is not None:  # Experiment 16's onsets (S-NPP) that are also fire starts here
        part = onsets.loc[onsets.target_date.dt.year.eq(year)]
        index, day = cells.get_indexer(part.h3_cell), season_index(part.target_date) - FIRST_TARGET_DAY
        keep = (index >= 0) & (day >= 0)
        index, day, names = index[keep], day[keep], part.bcws_category.to_numpy()[keep]
        found = start[index, day]
        group_cells, group_days, group_names = index[found], day[found], names[found]
        out["coverage"]["experiment_11_onsets"] = int(keep.sum())
        out["coverage"]["experiment_11_onsets_matched"] = int(found.sum())
    for n_size in WATCH_SIZES:
        out["watch"].append({"region": region, "year": year, "model": "random", "list_size": n_size,
                             "fire_starts": int(start.sum()), **random_watch_counts(n_size, start, quiet, future)})
    for name in SCORED:
        score = to_grid(cell, target, scores.pop(name), len(cells))
        score[~quiet] = -1.0  # every score is non-negative; non-quiet cell-days are never flagged
        spread = sparse_row_max(disk1, score)
        for definition in DEFINITIONS:
            hit, recall_score = (start, score) if definition == "res7_exact" else (near, spread)
            tag = (region, definition, name)
            if year == SELECTION_YEAR:
                candidates = candidate_thresholds(score[quiet])
                f1 = curve(score[quiet], start[quiet], hit[quiet], recall_score[quiet], candidates)[2]
                thresholds[tag] = float(candidates[int(np.argmax(f1))])
            counts = daily_counts(score, start, hit, recall_score, thresholds[tag])
            out["daily"][tag] = counts
            total = counts.sum(axis=0)
            row = {"region": region, "year": year, "definition": definition, "model": name, "threshold": thresholds[tag],
                   "precision": total[1] / max(total[0], 1), "recall": total[3] / max(total[2], 1),
                   "f1": float(f1_from_counts(total)), "flagged": int(total[0]), "fire_starts": int(total[2]),
                   "caught": int(total[3])}
            if definition == "res7_exact":
                row["average_precision"] = float(average_precision_score(start[quiet], score[quiet]))
            out["rows"].append(row)
            if group_cells is not None:
                caught = recall_score[group_cells, group_days] >= thresholds[tag]
                for group in np.unique(group_names):
                    mask = group_names == group
                    out["groups"].append({"region": region, "year": year, "model": name, "measure": f"f1_2022_{definition}",
                                          "onset_group": group, "fire_starts": int(mask.sum()), "caught": int(caught[mask].sum())})
        for n_size, listed in ranked_lists(score, quiet, tiebreak, WATCH_SIZES).items():
            out["watch"].append({"region": region, "year": year, "model": name, "list_size": n_size,
                                 "fire_starts": int(start.sum()), **watch_counts(listed, start, quiet, future, disk1)})
            if group_cells is not None and n_size == 50:
                recent = window(listed.astype(np.float32), WATCH_DAYS) > 0
                caught = recent[group_cells, group_days]
                for group in np.unique(group_names):
                    mask = group_names == group
                    out["groups"].append({"region": region, "year": year, "model": name, "measure": "top50_within_7d",
                                          "onset_group": group, "fire_starts": int(mask.sum()), "caught": int(caught[mask].sum())})
        del score, spread
    exact = {r["model"]: r for r in out["rows"] if r["definition"] == "res7_exact"}
    margin = {r["model"]: r for r in out["rows"] if r["definition"] == "res7_margin1"}
    top50 = {r["model"]: r for r in out["watch"] if r["list_size"] == 50}
    print(f"{region} {year} ({time.time()-tic:.0f}s): {start.sum():,} starts in {quiet.sum():,} quiet cell-days\n"
          f"  F1 exact: " + "; ".join(f"{m} {r['f1']:.3f}" for m, r in exact.items()) + "\n"
          f"  F1 margin: " + "; ".join(f"{m} {r['f1']:.3f}" for m, r in margin.items()) + "\n"
          f"  AP: " + "; ".join(f"{m} {r['average_precision']:.4f}" for m, r in exact.items()) + f" (base rate {start.sum()/quiet.sum():.5f})\n"
          f"  top-50 caught within 7 days: " + "; ".join(f"{m} {r['caught_within_7d']/max(r['fire_starts'],1):.3f}"
                                                         for m, r in top50.items()), flush=True)
    return out


def bootstrap(daily, regions, names, baseline):
    dates = pd.date_range(f"{EVALUATION_YEAR}-05-01", periods=SEASON_DAYS)[FIRST_TARGET_DAY:]
    weeks = dates.isocalendar().week.to_numpy()
    unique_weeks = np.unique(weeks)
    week_member = (weeks[None, :] == unique_weeks[:, None]).astype(np.float64)
    draws = np.random.default_rng(SEED).integers(0, len(unique_weeks), size=(BOOTSTRAP, len(unique_weeks)))
    resample = np.stack([np.bincount(d, minlength=len(unique_weeks)) for d in draws]).astype(np.float64)
    boot = lambda counts: f1_from_counts(resample @ (week_member @ counts))
    rows = []
    for definition in DEFINITIONS:
        for scope in [*regions, "BC-wide"]:
            get = lambda model: (np.sum([daily[r, definition, model] for r in regions], axis=0) if scope == "BC-wide"
                                 else daily[scope, definition, model])
            for model in names:
                if model == baseline:
                    continue
                difference = boot(get(model)) - boot(get(baseline))
                rows.append({"scope": scope, "definition": definition, "model": model, "baseline": baseline,
                             "difference": float(f1_from_counts(get(model).sum(0)) - f1_from_counts(get(baseline).sum(0))),
                             "ci_low": float(np.quantile(difference, .025)), "ci_high": float(np.quantile(difference, .975)),
                             "share_above": float((difference > 0).mean())})
    return pd.DataFrame(rows)


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="*", help="subset of Fire Centres (smoke test)")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--resume", action="store_true", help="load saved models and skip finished region-years")
    args = parser.parse_args()
    RESULTS = experiment_15.RESULTS = args.results
    started = time.time()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    (RESULTS / "parts").mkdir(exist_ok=True)
    all_regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    regions = args.regions or all_regions
    source_paths = [FIRMS, CWFIS, WEATHER_DAILY, STATIONS, LIGHTNING, EXPERIMENT_16_MODEL, ONSETS / "onset_cells.csv",
                    ONSETS / "onset_clusters.csv", *sorted(STATIC_DIR.glob("*.parquet"))]
    hashes = {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    context = Context(regions)
    labels = Labels(context, regions)
    drivers = Drivers(context, regions)
    static = StaticLayers()
    onsets = onset_categories()

    models = {}
    for name in LOADED:
        models[name] = catboost_model()
        models[name].load_model(str(EXPERIMENT_16_MODEL))
    if args.resume:
        for name in MODELS:
            models[name] = catboost_model()
            models[name].load_model(str(RESULTS / "models" / f"{name}.cbm"))
        print("Loaded saved fire-start models", flush=True)
    else:
        print("Building training sample (quiet cell-days, 2018-2021)", flush=True)
        sample, sample_stats = training_sample(regions, all_regions, context, labels, drivers, static)
        sample_stats.to_csv(RESULTS / "training_sample.csv", index=False)
        fit_checks = []
        y = sample[START].to_numpy()
        for name, feature_set in MODELS.items():
            X, quiet = design(sample, feature_set)
            assert quiet.all(), "a fire-start row outside Experiment 13's quiet stratum"
            models[name] = experiment_15.fit(name, X, y, FEATURE_SETS[feature_set] + ["quiet_stratum"], fit_checks)
            del X
        pd.DataFrame(fit_checks).to_csv(RESULTS / "fit_checks.csv", index=False)
        del sample
    for key in [k for k in labels.grids if k[1] in TRAIN_YEARS]:
        del labels.grids[key]
    labels.sources = None
    gc.collect()

    thresholds, parts = {}, []
    for region in regions:
        for year in (SELECTION_YEAR, EVALUATION_YEAR):
            path = RESULTS / "parts" / f"{region.lower().replace(' ', '_')}_{year}.pkl"
            if args.resume and path.exists():
                out = pickle.loads(path.read_bytes())
                print(f"{region} {year}: loaded finished part", flush=True)
            else:
                out = score_region_year(region, all_regions.index(region), year, context, labels, drivers, static, models,
                                        thresholds, onsets if year == EVALUATION_YEAR else None)
                out["thresholds"] = {k: v for k, v in thresholds.items() if k[0] == region}
                path.write_bytes(pickle.dumps(out))
            thresholds.update(out["thresholds"])
            parts.append(out)
            gc.collect()

    metrics = pd.DataFrame([r for p in parts for r in p["rows"]])
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    coverage = pd.DataFrame([p["coverage"] for p in parts])
    coverage.to_csv(RESULTS / "coverage.csv", index=False)
    daily = {k: v for p in parts if p["coverage"]["year"] == EVALUATION_YEAR for k, v in p["daily"].items()}
    names = list(SCORED)

    evaluation = metrics.loc[metrics.year.eq(EVALUATION_YEAR)]
    summary = evaluation.groupby(["definition", "model"])[["f1", "precision", "recall", "average_precision"]].median()
    summary = summary.add_prefix("median_").reset_index()
    for column, f in (("bc_wide_f1", f1_from_counts), ("bc_wide_precision", lambda t: t[1] / max(t[0], 1)),
                      ("bc_wide_recall", lambda t: t[3] / max(t[2], 1))):
        summary[column] = [float(f(np.sum([daily[r, d, m] for r in regions], axis=0).sum(0)))
                           for d, m in zip(summary.definition, summary.model)]
    summary.to_csv(RESULTS / "summary_2023.csv", index=False)

    watch = pd.DataFrame([r for p in parts for r in p["watch"]])
    watch.to_csv(RESULTS / "watch_list.csv", index=False)
    pooled = watch.loc[watch.year.eq(EVALUATION_YEAR)].groupby(["model", "list_size"])[
        ["fire_starts", "listed", "caught_next_day", "caught_next_day_margin", "caught_within_7d", "listed_start_within_7d"]].sum(min_count=1)
    for column in ("caught_next_day", "caught_next_day_margin", "caught_within_7d"):
        pooled[f"share_{column}"] = pooled[column] / pooled.fire_starts
    pooled["hit_rate_next_day"] = pooled.caught_next_day / pooled.listed
    pooled["hit_rate_within_7d"] = pooled.listed_start_within_7d / pooled.listed
    pooled.reset_index().to_csv(RESULTS / "watch_list_bc_2023.csv", index=False)

    groups = pd.DataFrame([r for p in parts for r in p["groups"]])
    groups.to_csv(RESULTS / "onset_groups_2023.csv", index=False)
    boot = bootstrap(daily, regions, names, "catboost_spread_noday")
    boot.to_csv(RESULTS / "bootstrap_2023.csv", index=False)

    assert hashes == {str(p.relative_to(EXPERIMENTS.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS), "selection_year": SELECTION_YEAR,
        "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "fire_start": f"label positive on day t, no detection of the label's source in the cell or its ring-1 neighbours on t-{QUIET_DAYS} .. t-1, and in Experiment 13's quiet stratum",
        "label": "S-NPP (FIRMS) or NOAA-20 (CWFIS) from 2021, S-NPP before; one-day gaps filled",
        "training_rows": "quiet cell-days: every fire start plus Experiment 21's 1% draw (same seeds)",
        "models": {k: {"feature_set": v, "inputs": FEATURE_SETS[v]} for k, v in MODELS.items()},
        "references": {"catboost_spread_noday": str(EXPERIMENT_16_MODEL.relative_to(EXPERIMENTS)),
                       "nearby_activity": "(MAX_RING + 1 - dist_nearest_3d) * 1e4 + active_cells_disk5_3d",
                       "lightning": "ltg_3x3_3d * 1e4 + ltg_9x9_3d", "random": "expected counts of a random daily list"},
        "thresholds": "maximize 2022 F1 per Fire Centre, definition and model, over quiet cell-days",
        "watch_lists": f"top {list(WATCH_SIZES)} quiet cells per Fire Centre per day; caught within 7 days = listed on t-6 .. t",
        "bootstrap": f"{BOOTSTRAP} resamples of whole ISO weeks of 2023, against catboost_spread_noday",
        "runtime_seconds": round(time.time() - started)}, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s", flush=True)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(), flush=True)
    print(pooled.reset_index().query("list_size == 50").round(4).to_string(), flush=True)
    print(boot.query("scope == 'BC-wide'").round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
