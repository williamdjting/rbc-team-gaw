"""Build a standalone visual HTML report from saved Experiment 18 outputs.

    python build_report.py                 # results/ -> RESULTS.html
    python build_report.py <results dir>   # another results folder (e.g. a smoke test), same output
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
FIGURES = HERE / "figures"
spec = importlib.util.spec_from_file_location("experiment_03_report", HERE.parent / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE, INK = report_03.RULE, report_03.INK
# Reference categorical slots 1-5 in fixed order; gray is reserved for rules and the random list.
SERIES = {"rule_yesterday": ("Fire yesterday (rule)", RULE, "D"),
          "random": ("Random quiet cells", RULE, "D"),
          "catboost_08": ("CatBoost, Experiment 13 features", "#2a78d6", "o"),
          "catboost_static": ("CatBoost, + static layers", "#eb6834", "^"),
          "catboost_quiet": ("CatBoost trained on quiet cells (+ history, static)", "#1baf7a", "s"),
          "nearby_activity": ("Ranking: recent fire nearby (30 days)", "#eda100", "v"),
          "burned_before": ("Ranking: burned in an earlier season", "#e87ba4", "P")}
FORECAST = ("rule_yesterday", "catboost_08", "catboost_static")
WATCH = ("catboost_08", "catboost_static", "catboost_quiet", "nearby_activity", "burned_before")
DEFINITION_NAMES = {"res7_margin1": "one-cell margin", "res7_exact": "exact cell"}
GROUP_NAMES = {"new BCWS wildfire": "New BCWS wildfire", "older BCWS wildfire, not recently detected": "Older wildfire flaring up",
               "no BCWS record": "No BCWS fire on record", "other BCWS record": "Other BCWS record"}
FEATURE_GROUPS = {"static": "static layer", "history": "history (Experiment 17)", "08": "Experiment 13"}
LIST_SIZE = 50


def f1_chart(evaluation, order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    for ax, definition in zip(axes, ("res7_margin1", "res7_exact")):
        part = evaluation.loc[evaluation.definition.eq(definition)]
        for name, offset in zip(FORECAST, (-.22, 0, .22)):
            label, colour, marker = SERIES[name]
            rows = part.loc[part.model.eq(name)].set_index("region")
            ax.scatter(rows.loc[order, "f1"], positions + offset, s=52, marker=marker, color=colour,
                       edgecolor="white", linewidth=1.2, zorder=3)
        if definition == "res7_margin1":
            ax.axvspan(.7, .8, color="#e1e0d9", zorder=0)
        ax.set(title=f"2023 F1, {DEFINITION_NAMES[definition]} (2022 threshold)", xlabel="F1")
        style(ax)
    axes[0].set(yticks=positions, yticklabels=order)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker=SERIES[n][2], markersize=8, markerfacecolor=SERIES[n][1],
                      markeredgecolor="white", label=SERIES[n][0]) for n in FORECAST]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#e1e0d9", label="F1 goal 0.70–0.80"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(FIGURES / "report_f1.png", dpi=150)
    plt.close(fig)


def pooled_watch(watch, year):
    """BC-wide shares: counts summed over Fire Centres."""
    part = watch.loc[watch.year.eq(year)].copy()
    for column in ("caught_next_day", "caught_within_7d"):
        part[column] = part[column] * part.quiet_fires
    for column in ("hit_rate_next_day", "hit_rate_within_7d"):
        part[column] = part[column] * part.listed_cell_days
    total = part.groupby(["model", "list_size"])[["quiet_fires", "listed_cell_days", "caught_next_day", "caught_within_7d",
                                                   "hit_rate_next_day", "hit_rate_within_7d"]].sum(min_count=1)
    for column in ("caught_next_day", "caught_within_7d"):
        total[column] /= total.quiet_fires
    for column in ("hit_rate_next_day", "hit_rate_within_7d"):
        total[column] /= total.listed_cell_days
    return total.reset_index()


def watch_chart(pooled):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), constrained_layout=True)
    panels = (("caught_next_day", "On the list the day before the fire"), ("caught_within_7d", "On the list on any of the 7 days up to the fire"))
    for ax, (column, title) in zip(axes, panels):
        for name in ("random", *WATCH):
            rows = pooled.loc[pooled.model.eq(name)].sort_values("list_size")
            if rows[column].isna().all():
                continue
            label, colour, marker = SERIES[name]
            ax.plot(rows.list_size, rows[column], color=colour, marker=marker, linewidth=2, markersize=8,
                    linestyle=":" if name == "random" else "-", markeredgecolor="white", label=label)
        ax.set(title=title, xlabel="Quiet cells on the list per Fire Centre per day", ylabel="Share of 2023 quiet fires",
               xscale="log", xticks=sorted(pooled.list_size.unique()))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylim(bottom=0)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_watch.png", dpi=150)
    plt.close(fig)


def importance_table(name, manifest, top=12):
    frame = pd.read_csv(RESULTS / f"feature_importance__{name}.csv").head(top)
    static = set(manifest["static_features"])
    history = set(json.loads((HERE.parent / "17_exp_history_nonwildfire_bc/results/manifest.json").read_text())["history_features"])

    def group(feature):
        if feature in static:
            return FEATURE_GROUPS["static"]
        if feature in history or feature.startswith("prior_"):
            return FEATURE_GROUPS["history"]
        return "detection or Experiment 13"
    return html_table(frame.assign(group=frame.feature.map(group)).rename(
        columns={"feature": "Input", "importance": "Importance (%)", "group": "Group"}), {"Importance (%)": "{:.1f}"})


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    evaluation_year = manifest["evaluation_year"]
    forecast = pd.read_csv(RESULTS / "forecast_metrics.csv")
    summary = pd.read_csv(RESULTS / "forecast_summary_2023.csv")
    boot = pd.read_csv(RESULTS / "forecast_bootstrap_2023.csv")
    watch = pd.read_csv(RESULTS / "watch_list.csv")
    watch_ap = pd.read_csv(RESULTS / "watch_list_average_precision.csv")
    onsets = pd.read_csv(RESULTS / "watch_list_onsets_2023.csv")
    flag = pd.read_csv(RESULTS / "wildfire_flag_metrics.csv")
    flag_onsets = pd.read_csv(RESULTS / "wildfire_flag_onsets.csv")
    fit_checks = pd.read_csv(RESULTS / "fit_checks.csv")
    regions = manifest["regions"]

    evaluation = forecast.loc[forecast.year.eq(evaluation_year)]
    pick = lambda definition, model: evaluation.loc[evaluation.definition.eq(definition) & evaluation.model.eq(model)].set_index("region")
    order = pick("res7_margin1", "catboost_static").f1.sort_values(ascending=False).index.tolist()
    f1_chart(evaluation, order)
    pooled = pooled_watch(watch, evaluation_year)
    watch_chart(pooled)

    # A. forecast
    rows = []
    for name in FORECAST:
        row = {"Model": SERIES[name][0]}
        for definition, label in DEFINITION_NAMES.items():
            s = summary.loc[summary.definition.eq(definition) & summary.model.eq(name)].iloc[0]
            row[f"F1, {label} (median)"] = s.f1
            row[f"F1, {label} (BC-wide)"] = s.bc_wide_f1
            row[f"Centres at 0.70+ ({label})"] = f"{int((pick(definition, name).f1 >= .7).sum())} of {len(regions)}"
        ap = pick("res7_exact", name).get("average_precision")
        row["AP, exact cell (median)"] = ap.median() if ap is not None and ap.notna().any() else np.nan
        rows.append(row)
    forecast_table = html_table(pd.DataFrame(rows), {c: "{:.3f}" for c in rows[0] if c.startswith(("F1", "AP"))})
    comparison_names = {("catboost_08", "rule_yesterday"): "Experiment 13 model − rule",
                        ("catboost_static", "rule_yesterday"): "+ static layers − rule",
                        ("catboost_static", "catboost_08"): "+ static layers − Experiment 13 model"}
    b = boot.loc[boot.scope.eq("BC-wide")].copy()
    b["Comparison"] = [comparison_names[(m, r)] for m, r in zip(b.model, b.baseline)]
    b["Definition"] = b.definition.map(DEFINITION_NAMES)
    b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
    boot_table = html_table(b[["Definition", "Comparison", "difference", "95% interval", "share_above"]].rename(
        columns={"difference": "F1 difference", "share_above": "Share of resamples above 0"}),
        {"F1 difference": "{:+.3f}", "Share of resamples above 0": "{:.0%}"})
    per_centre_boot = boot.loc[boot.scope.ne("BC-wide") & boot.definition.eq("res7_margin1")].copy()
    per_centre_boot["Comparison"] = [comparison_names[(m, r)] for m, r in zip(per_centre_boot.model, per_centre_boot.baseline)]
    per_centre_boot["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(per_centre_boot.ci_low, per_centre_boot.ci_high)]
    per_centre_boot_table = html_table(per_centre_boot[["scope", "Comparison", "difference", "95% interval"]].rename(
        columns={"scope": "Fire Centre", "difference": "F1 difference"}), {"F1 difference": "{:+.3f}"})
    columns = {f"{SERIES[m][0]} ({DEFINITION_NAMES[d]})": pick(d, m).loc[order, "f1"].to_numpy()
               for d in DEFINITION_NAMES for m in FORECAST}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, **columns}), {c: "{:.3f}" for c in columns})
    static_gain = pick("res7_margin1", "catboost_static").f1 - pick("res7_margin1", "catboost_08").f1
    rule_gain = pick("res7_margin1", "catboost_static").f1 - pick("res7_margin1", "rule_yesterday").f1
    bc = b.set_index(["definition", "model", "baseline"])
    bc_static_rule = bc.loc[("res7_margin1", "catboost_static", "rule_yesterday")]
    bc_static_08 = bc.loc[("res7_margin1", "catboost_static", "catboost_08")]

    # B. watch list
    at = pooled.loc[pooled.list_size.eq(LIST_SIZE)].set_index("model")
    ap = watch_ap.loc[watch_ap.year.eq(evaluation_year)].groupby("model").average_precision_quiet.median()
    base_rate = watch_ap.loc[watch_ap.year.eq(evaluation_year)].groupby("region").base_rate.first().median()
    watch_rows = []
    for name in ("random", *WATCH):
        r = at.loc[name]
        watch_rows.append({"Ranking": SERIES[name][0], "Caught the day before": r.caught_next_day,
                           "Caught within 7 days": r.caught_within_7d, "Listed cells burning next day": r.hit_rate_next_day,
                           "Listed cells burning within 7 days": r.hit_rate_within_7d,
                           "AP among quiet cells (median)": ap.get(name, base_rate)})
    watch_table = html_table(pd.DataFrame(watch_rows), {"Caught the day before": "{:.1%}", "Caught within 7 days": "{:.1%}",
                                                        "Listed cells burning next day": "{:.2%}",
                                                        "Listed cells burning within 7 days": "{:.2%}",
                                                        "AP among quiet cells (median)": "{:.4f}"})
    watch_table = watch_table.replace("<td>nan%</td>", "<td>–</td>")
    best_watch = at.loc[list(WATCH)].caught_within_7d.idxmax()
    quiet_fires = int(at.loc["catboost_08"].quiet_fires)
    quiet_listed = int(at.loc["catboost_08"].listed_cell_days)
    o = onsets.loc[onsets.list_size.eq(LIST_SIZE)].groupby(["onset_group", "model"])[["quiet_onset_cells", "caught_within_7d"]].sum()
    o["share"] = o.caught_within_7d / o.quiet_onset_cells
    groups = [g for g in GROUP_NAMES if g in o.index.get_level_values(0)]
    onset_table = html_table(pd.DataFrame(
        [{"Onset group (Experiment 16)": GROUP_NAMES[g], "Quiet onset cells": int(o.loc[(g, WATCH[0]), "quiet_onset_cells"]),
          **{SERIES[m][0]: o.loc[(g, m), "share"] for m in WATCH}} for g in groups]),
        {"Quiet onset cells": "{:,}", **{SERIES[m][0]: "{:.1%}" for m in WATCH}})

    # C. wildfire flag
    test = flag.loc[~flag.role.eq("train")].copy()
    test["Method"] = test.method.map({"wildfire_flag": "Wildfire flag (CatBoost)",
                                      "industry_or_farm_rule": "Rule: oil/gas site nearby or half the cell farmland"})
    flag_table = html_table(test[["year", "Method", "detected_cell_days", "non_wildfire_share", "precision", "recall", "f1",
                                  "roc_auc", "average_precision"]].rename(columns={
        "year": "Year", "detected_cell_days": "Detected cell-days", "non_wildfire_share": "Not wildfire-linked",
        "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC AUC", "average_precision": "AP"}),
        {"Year": "{}", "Detected cell-days": "{:,}", "Not wildfire-linked": "{:.1%}", "Precision": "{:.3f}", "Recall": "{:.3f}",
         "F1": "{:.3f}", "ROC AUC": "{:.3f}", "AP": "{:.3f}"}).replace("<td>nan</td>", "<td>–</td>")
    fo = flag_onsets.loc[flag_onsets.bcws_category.isin(GROUP_NAMES) & flag_onsets.onset_cells.ge(20)].copy()
    fo["Onset group (Experiment 16)"] = fo.bcws_category.map(GROUP_NAMES)
    flag_onset_table = html_table(fo[["year", "Onset group (Experiment 16)", "onset_cells", "mean_p_non_wildfire", "flagged_share",
                                      "label_non_wildfire_share"]].rename(columns={
        "year": "Year", "onset_cells": "Onset cells", "mean_p_non_wildfire": "Mean chance not wildfire",
        "flagged_share": "Flagged not wildfire", "label_non_wildfire_share": "Labelled not wildfire (perimeter link)"}),
        {"Year": "{}", "Onset cells": "{:,}", "Mean chance not wildfire": "{:.2f}", "Flagged not wildfire": "{:.0%}",
         "Labelled not wildfire (perimeter link)": "{:.0%}"})
    f = test.set_index(["year", "method"])
    sel, ev = manifest["selection_year"], evaluation_year
    no_record = flag_onsets.set_index(["year", "bcws_category"])
    reproduce = fit_checks.loc[fit_checks.model.eq("reproduces_experiment_12")]
    reproduce_note = ("Experiment 13's model and the rule reproduce Experiment 17's F1 exactly in every Fire Centre and year."
                      if len(reproduce) else "Reproduction check against Experiment 17 was not run (subset of Fire Centres).")

    images = {"f1": embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre"),
              "watch": embedded_image(FIGURES / "report_watch.png", "Share of quiet fires caught by watch-list size")}
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Static layers and watch list</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px;text-wrap:balance}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:18px 0 8px}}
p{{margin:8px 0;max-width:920px}}.purpose{{font-size:21px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0;max-width:980px}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0}}
.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.value{{font-size:40px;font-weight:750;color:var(--teal)}}
figure{{margin:16px 0 0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}img{{display:block;width:100%;height:auto}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:650}}td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 18 · Static layers, quiet-cell watch list, wildfire flag</div>
<h1>Do land, fuel and industry layers help, and can quiet cells get their own product?</h1>
<p class="purpose"><strong>Purpose:</strong> Experiments 16–17 showed that fires in quiet cells never reach the forecast's alert threshold, and that cells burning with no BC Wildfire Service record sit on flat, low farmland and oil and gas sites. This experiment adds static layers to the forecast, ranks quiet cells for a separate watch list, and builds a flag that separates wildfires from other burning.</p>
<p class="muted">{len(regions)} Fire Centres · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Static layers ({len(manifest["static_features"])} inputs):</strong> elevation and slope; fuel-type shares; Agricultural Land Reserve share; oil and gas facilities and wells in the cell and its neighbours; cutblock area harvested in the 3 previous years. Facilities, wells and cutblocks count only if dated before the season.</li>
<li><strong>A. Next-day forecast:</strong> CatBoost with Experiment 13's features, with and without the static layers, against "fire in this cell yesterday". Thresholds chosen on 2022, applied to 2023. Intervals resample whole weeks of 2023.</li>
<li><strong>B. Quiet-cell watch list:</strong> a quiet cell has no detection in it or its neighbours in the past 7 days. Each day, each Fire Centre lists its N highest-ranked quiet cells. Rankings: both forecast models, a CatBoost trained only on quiet cells (adding Experiment 17's history), and two simple rankings.</li>
<li><strong>C. Wildfire flag:</strong> for each detected cell-day, the chance it is not linked to a BCWS wildfire (Experiment 17's link to final perimeters and ignition points). Inputs are the static layers, the detection itself (intensity, night share, non-vegetation type, fire-complex size, recent activity), earlier seasons and location; no BCWS inputs.</li></ul></section>

<section><h2>A. Next-day forecast</h2>
<div class="cards"><div class="card"><h3>Static layers vs Experiment 13</h3><div class="value">{bc_static_08.difference:+.3f}</div><p>BC-wide F1 change, one-cell margin (95% interval {bc_static_08.ci_low:+.3f} to {bc_static_08.ci_high:+.3f}). Higher in {int((static_gain > 0).sum())} of {len(regions)} centres.</p></div>
<div class="card"><h3>Static-layer model vs rule</h3><div class="value">{bc_static_rule.difference:+.3f}</div><p>BC-wide F1, one-cell margin (95% interval {bc_static_rule.ci_low:+.3f} to {bc_static_rule.ci_high:+.3f}). Higher in {int((rule_gain > 0).sum())} of {len(regions)} centres.</p></div>
<div class="card"><h3>BC-wide F1, one-cell margin</h3><div class="value">{summary.set_index(["definition", "model"]).loc[("res7_margin1", "catboost_static"), "bc_wide_f1"]:.3f}</div><p>CatBoost with static layers; the rule scores {summary.set_index(["definition", "model"]).loc[("res7_margin1", "rule_yesterday"), "bc_wide_f1"]:.3f}.</p></div></div>
{forecast_table}
<figure>{images["f1"]}<figcaption>2023 F1 by Fire Centre. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Week-block bootstrap, BC-wide</h3>{boot_table}
<details><summary>Show intervals for each Fire Centre (one-cell margin)</summary>{per_centre_boot_table}</details>
<details><summary>Show F1 for every centre</summary>{per_centre}</details>
<details><summary>Show the most important inputs (CatBoost with static layers)</summary>{importance_table("catboost_static", manifest)}</details></section>

<section><h2>B. Quiet-cell watch list</h2>
<p>{quiet_fires:,} fires in 2023 started in quiet cells (no detection in the cell or its neighbours for 7 days). The forecast never flags them at its F1 threshold. A watch list instead names a fixed number of quiet cells per Fire Centre each day.</p>
<figure>{images["watch"]}<figcaption>BC-wide share of 2023 quiet fires whose cell was on the list. The random line is the expected share for a list drawn at random from that day's quiet cells.</figcaption></figure>
<h3>Top {LIST_SIZE} quiet cells per Fire Centre per day ({quiet_listed:,} listed cell-days in 2023)</h3>{watch_table}
<p class="muted">AP among quiet cells ranks every quiet cell-day by next-day detection; the base rate (median over centres) is {base_rate:.4f}, which is also the AP of a random ranking.</p>
<h3>Caught within 7 days, by Experiment 16 onset group (top {LIST_SIZE})</h3>{onset_table}
<details><summary>Show the most important inputs (CatBoost trained on quiet cells)</summary>{importance_table("catboost_quiet", manifest)}</details></section>

<section><h2>C. Wildfire flag</h2>
<p>Threshold chosen to maximize {sel} F1 for "not wildfire-linked", then applied to {ev}.</p>
{flag_table}
<h3>Check against Experiment 16's onset groups</h3>
<p>Experiment 16's groups come from BCWS ignition points and dates, not from final perimeters, so they are a separate check on the flag.</p>
{flag_onset_table}
<details><summary>Show the most important inputs (wildfire flag)</summary>{importance_table("wildfire_flag", manifest)}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>Forecast:</strong> static layers changed BC-wide F1 (one-cell margin) by {bc_static_08.difference:+.3f} over Experiment 13's model (95% interval {bc_static_08.ci_low:+.3f} to {bc_static_08.ci_high:+.3f}); against the rule, {bc_static_rule.difference:+.3f} ({bc_static_rule.ci_low:+.3f} to {bc_static_rule.ci_high:+.3f}). CONCLUSION_FORECAST</li>
<li><strong>Watch list:</strong> with {LIST_SIZE} quiet cells per Fire Centre per day, the best ranking ({SERIES[best_watch][0]}) had {at.loc[best_watch, "caught_within_7d"]:.0%} of 2023 quiet fires on its list within the 7 days before they started, against {at.loc["random", "caught_next_day"]:.1%} for a random list on the day before. CONCLUSION_WATCH</li>
<li><strong>Wildfire flag:</strong> in {sel}, F1 {f.loc[(sel, "wildfire_flag"), "f1"]:.2f} against {f.loc[(sel, "industry_or_farm_rule"), "f1"]:.2f} for the simple rule; in {ev}, {f.loc[(ev, "wildfire_flag"), "f1"]:.2f} against {f.loc[(ev, "industry_or_farm_rule"), "f1"]:.2f}. CONCLUSION_FLAG</li>
<li>Next: CONCLUSION_NEXT</li></ul></div>
<ul class="muted"><li>{reproduce_note}</li>
<li>Terrain, fuels and the Agricultural Land Reserve are current layers, applied to every year; they change slowly, but fuels can change after a large fire.</li>
<li>The wildfire link depends on BCWS records and final perimeters. Large perimeters (2023) take in nearby non-wildfire burning, so labels are least reliable in big fire years.</li>
<li>Untuned CatBoost, one seed. 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 18 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
