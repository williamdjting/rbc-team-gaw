"""Build a standalone visual HTML report from saved Experiment 21 outputs.

    python build_report.py                 # results/ -> RESULTS.html
    python build_report.py <results dir>   # another results folder (e.g. a smoke test)
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
RULE = report_03.RULE
# Gray tones for the two rules; reference categorical slots 1-5 in fixed order for models.
SERIES = {"rule_snpp_yesterday": ("Rule: S-NPP detection yesterday", "#b5b3aa", "d"),
          "rule_pair_yesterday": ("Rule: S-NPP or NOAA-20 detection yesterday", RULE, "D"),
          "catboost_stopgap": ("CatBoost, Experiment 20 (fire history)", "#2a78d6", "o"),
          "catboost_n20": ("+ NOAA-20 history", "#eb6834", "^"),
          "catboost_drivers": ("+ weather + lightning", "#1baf7a", "s"),
          "catboost_spread": ("+ upwind spread", "#eda100", "v"),
          "catboost_spread_noday": ("+ upwind spread, no day_of_year", "#e87ba4", "P")}
MODELS = [m for m in SERIES if not m.startswith("rule")]
DEFINITION_NAMES = {"res7_exact": "exact cell", "res7_margin1": "one-cell margin"}
METHOD_NAMES = {"f1_2022": "2022-tuned threshold", "rule_volume": "rule's daily volume"}
GROUP_OF = [("wx_", "weather"), ("ltg_", "lightning"), ("upwind_", "spread"), ("n20_", "NOAA-20"), ("pair_", "NOAA-20"),
            ("bcws_", "BCWS incidents")]


def group(feature):
    return next((name for prefix, name in GROUP_OF if feature.startswith(prefix)), "fire history")


def f1_chart(evaluation, order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    offsets = np.linspace(-.33, .33, len(SERIES))
    for ax, definition in zip(axes, DEFINITION_NAMES):
        part = evaluation.loc[evaluation.definition.eq(definition) & evaluation.method.eq("f1_2022")]
        for (name, (label, colour, marker)), offset in zip(SERIES.items(), offsets):
            rows = part.loc[part.model.eq(name)].set_index("region")
            ax.scatter(rows.loc[order, "f1"], positions + offset, s=46, marker=marker, color=colour,
                       edgecolor="white", linewidth=1.1, zorder=3)
        if definition == "res7_margin1":
            ax.axvspan(.7, .8, color="#e1e0d9", zorder=0)
        ax.set(title=f"2023 F1, {DEFINITION_NAMES[definition]}", xlabel="F1 (2022-tuned threshold)")
        style(ax)
    axes[0].set(yticks=positions, yticklabels=order)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=c, markeredgecolor="white", label=n)
               for n, c, m in SERIES.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(FIGURES / "report_f1.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    summary = pd.read_csv(RESULTS / "summary_2023.csv").set_index(["definition", "method", "model"])
    boot = pd.read_csv(RESULTS / "bootstrap_2023.csv")
    splits = pd.read_csv(RESULTS / "new_vs_continuing_2023.csv")
    sample = pd.read_csv(RESULTS / "training_sample.csv")
    regions = manifest["regions"]
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]

    def pick(definition, model, method="f1_2022", frame=evaluation):
        return frame.loc[frame.definition.eq(definition) & frame.model.eq(model) & frame.method.eq(method)].set_index("region")

    order = pick("res7_margin1", "rule_pair_yesterday").f1.sort_values(ascending=False).index.tolist()
    f1_chart(evaluation, order)
    s = lambda d, m, k="f1_2022", col="bc_wide_f1": summary.loc[(d, k, m), col]

    rows = []
    for name in SERIES:
        for method in METHOD_NAMES:
            if name.startswith("rule") and method == "rule_volume":
                continue
            row = {"Forecast": SERIES[name][0], "Threshold": "fixed (0/1)" if name.startswith("rule") else METHOD_NAMES[method]}
            for definition, label in DEFINITION_NAMES.items():
                row[f"F1, {label}"] = s(definition, name, method)
            row["Precision (exact)"] = s("res7_exact", name, method, "bc_wide_precision")
            row["Recall (exact)"] = s("res7_exact", name, method, "bc_wide_recall")
            row["Centres at 0.70+ (margin)"] = f"{int((pick('res7_margin1', name, method).f1 >= .7).sum())} of {len(regions)}"
            rows.append(row)
    numeric = {c: "{:.3f}" for c in rows[0] if c.startswith(("F1", "Precision", "Recall"))}
    summary_table = html_table(pd.DataFrame(rows), numeric)

    b = boot.loc[boot.scope.eq("BC-wide") & ~boot.model.eq("rule_snpp_yesterday")].copy()
    b["Forecast"] = b.model.map(lambda m: SERIES[m][0])
    b["Threshold"] = b.method.map(METHOD_NAMES)
    b["Definition"] = b.definition.map(DEFINITION_NAMES)
    b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
    boot_table = html_table(b[["Definition", "Forecast", "Threshold", "difference", "95% interval", "share_above"]].rename(
        columns={"difference": "F1 minus two-satellite rule", "share_above": "Share of resamples above 0"}),
        {"F1 minus two-satellite rule": "{:+.3f}", "Share of resamples above 0": "{:.0%}"})
    bb = b.set_index(["definition", "model", "method"])
    region_boot = boot.loc[boot.scope.ne("BC-wide") & boot.method.eq("f1_2022")].set_index(["scope", "definition", "model"])

    totals = splits.groupby(["model", "method"]).sum(numeric_only=True)
    split_rows = []
    for name in SERIES:
        for method in METHOD_NAMES:
            if (name, method) not in totals.index:
                continue
            t = totals.loc[(name, method)]
            split_rows.append({"Forecast": SERIES[name][0], "Threshold": "fixed (0/1)" if name.startswith("rule") else METHOD_NAMES[method],
                               "New fire cells caught": int(t.new_caught), "Continuing cells caught": int(t.continuing_caught),
                               "Stopping cells flagged (false alarms)": int(t.stopped_flagged),
                               "Other false alarms": int(t.fresh_false_alarms)})
    t0 = totals.loc[("rule_pair_yesterday", "f1_2022")]
    split_table = html_table(pd.DataFrame(split_rows), {c: "{:,}" for c in split_rows[0] if c not in ("Forecast", "Threshold")})

    def importance(name, top=15):
        frame = pd.read_csv(RESULTS / f"feature_importance__{name}.csv")
        groups = frame.assign(group=frame.feature.map(group)).groupby("group").importance.sum().sort_values(ascending=False)
        return frame.head(top), groups

    top_spread, groups_spread = importance("catboost_spread_noday")
    importance_table = html_table(top_spread.assign(Group=top_spread.feature.map(group)).rename(
        columns={"feature": "Input", "importance": "Importance (%)"})[["Input", "Group", "Importance (%)"]], {"Importance (%)": "{:.1f}"})
    group_table = html_table(groups_spread.reset_index().rename(columns={"group": "Input group", "importance": "Importance (%)"}),
                             {"Importance (%)": "{:.1f}"})
    columns = {f"{SERIES[m][0]} ({DEFINITION_NAMES[d]})": pick(d, m).loc[order, "f1"].to_numpy()
               for d in DEFINITION_NAMES for m in SERIES}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, **columns}), {c: "{:.3f}" for c in columns})
    selection = metrics.loc[metrics.year.eq(manifest["selection_year"]) & metrics.method.eq("f1_2022")].groupby(
        ["definition", "model"]).f1.median().unstack(0)
    selection_table = html_table(selection.reindex(list(SERIES)).rename(index=lambda m: SERIES[m][0]).reset_index().rename(
        columns={"model": "Forecast", "res7_exact": "Median F1, exact (2022)", "res7_margin1": "Median F1, margin (2022)"}),
        {"Median F1, exact (2022)": "{:.3f}", "Median F1, margin (2022)": "{:.3f}"})
    sample_table = html_table(sample.groupby("year").agg(rows=("kept_rows", "sum"), positives=("positives", "sum"),
                                                         missing=("weather_missing_share", "mean")).reset_index().rename(
        columns={"year": "Training year", "rows": "Rows kept", "positives": "Training-label positives", "missing": "Rows with no station within 100 km"}),
        {"Training year": "{}", "Rows kept": "{:,}", "Training-label positives": "{:,}", "Rows with no station within 100 km": "{:.1%}"})

    best = {d: max(MODELS, key=lambda m: s(d, m)) for d in DEFINITION_NAMES}
    card = lambda d: (f'<div class="card"><h3>Best model, {DEFINITION_NAMES[d]}</h3><div class="value">'
                      f'{bb.loc[(d, best[d], "f1_2022"), "difference"]:+.3f}</div><p>{SERIES[best[d]][0]} minus the two-satellite rule, '
                      f'BC-wide (95% interval {bb.loc[(d, best[d], "f1_2022"), "ci_low"]:+.3f} to {bb.loc[(d, best[d], "f1_2022"), "ci_high"]:+.3f}).</p></div>')
    above = lambda d, m: int((region_boot.xs((d, m), level=["definition", "model"]).ci_low > 0).sum())
    beats = lambda d, m: int((pick(d, m).f1 > pick(d, "rule_pair_yesterday").f1).sum())
    tn = lambda m, k="f1_2022": totals.loc[(m, k)]
    noday = "catboost_spread_noday"
    image = embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre")
    flagged = evaluation.loc[evaluation.method.eq("f1_2022")].groupby(["definition", "model"]).flagged.sum()
    t1 = tn(noday)
    sel = selection
    ci = lambda d, m, k="f1_2022": f'{bb.loc[(d, m, k), "difference"]:+.3f} (95% interval {bb.loc[(d, m, k), "ci_low"]:+.3f} to {bb.loc[(d, m, k), "ci_high"]:+.3f})'
    conclusions = f'''<ul>
<li><strong>For the first time, a model beats the two-satellite rule with an interval above zero, but only with the one-cell margin.</strong> The model with weather, lightning and spread inputs and no <code>day_of_year</code> reaches BC-wide F1 {s("res7_margin1", noday):.3f} against {s("res7_margin1", "rule_pair_yesterday"):.3f}: {ci("res7_margin1", noday)}. It is ahead in {beats("res7_margin1", noday)} of {len(regions)} centres ({above("res7_margin1", noday)} with intervals above zero), and {int((pick("res7_margin1", noday).f1 >= .7).sum())} of {len(regions)} centres reach 0.70.</li>
<li><strong>On the exact cell the rule still wins:</strong> {s("res7_exact", noday):.3f} against {s("res7_exact", "rule_pair_yesterday"):.3f}, {ci("res7_exact", noday)}. The model is within 0.01 of the rule in four centres (intervals include zero) and leads in Southeast, but trails by 0.023 in Prince George, which holds 72% of 2023's fire cells.</li>
<li><strong>Dropping <code>day_of_year</code> mattered more than any new input.</strong> With it, the same inputs score {s("res7_exact", "catboost_spread"):.3f} / {s("res7_margin1", "catboost_spread"):.3f}; without it, {s("res7_exact", noday):.3f} / {s("res7_margin1", noday):.3f}. Weather and lightning added to a model that keeps <code>day_of_year</code> do not help ({s("res7_exact", "catboost_drivers"):.3f} / {s("res7_margin1", "catboost_drivers"):.3f}). With the calendar available, the model learns when 2018 and 2021 peaked; without it, it has to use the weather, which is now its largest input group ({groups_spread.get("weather", 0):.0f}% of importance; noon temperature alone {top_spread.importance.iloc[0]:.0f}%). Lightning ({groups_spread.get("lightning", 0):.1f}%) and the upwind inputs ({groups_spread.get("spread", 0):.1f}%) contribute little.</li>
<li><strong>The win comes from how many cells are flagged, not from better choices among the rule's cells.</strong> At the rule's daily volume, the same model scores {s("res7_exact", noday, "rule_volume"):.3f} / {s("res7_margin1", noday, "rule_volume"):.3f}, a tie with the margin and behind on the exact cell; the NOAA-20 model without weather scores about the same ({s("res7_exact", "catboost_n20", "rule_volume"):.3f} / {s("res7_margin1", "catboost_n20", "rule_volume"):.3f}). With its 2022 threshold and the margin, the model flags {int(flagged[("res7_margin1", noday)]):,} cells against the rule's {int(flagged[("res7_margin1", "rule_pair_yesterday")]):,}: extra flags around active fires, which the margin counts as correct when fire appears next door. Its scores also carry 2022 thresholds over to 2023 better than the calendar models.</li>
<li><strong>On the exact cell, its errors shift but don't shrink enough.</strong> It drops {int(t0.stopped_flagged - t1.stopped_flagged):,} of the rule's {int(t0.stopped):,} false alarms on fires that stop and catches {int(t1.new_caught):,} of {int(t1.new_positives):,} new fire cells, but loses {int(t1.continuing_positives - t1.continuing_caught):,} still-burning cells and adds {int(t1.fresh_false_alarms):,} other false alarms.</li>
<li><strong>Selection check:</strong> this model was picked as the best of four on 2023, which flatters it. It was also the best model in 2022 ({sel.loc[noday, "res7_exact"]:.3f} exact, {sel.loc[noday, "res7_margin1"]:.3f} margin, medians), so the choice would have been the same without looking at 2023. A clean confirmation still needs 2024.</li>
<li><strong>Next:</strong> keep <code>day_of_year</code> out. Rerun with FIRMS NOAA-20 for 2018–2020 when it arrives. Try gridded weather (ERA5-Land): 15–24% of Coastal and Northwest cell-days have no station within 100 km, and a station's wind is a weak guide to where a fire will spread. Weather forecasts for the target day are the operational version of the missing information.</li></ul>'''

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Weather, lightning and spread inputs</title>
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
<header><div class="eyebrow">Experiment 21 · Weather, lightning and spread inputs</div>
<h1>Can drivers the rule cannot see beat persistence?</h1>
<p class="purpose"><strong>Purpose:</strong> on the two-satellite label, "fire yesterday" misses fire spreading into new cells and new starts, and keeps flagging fires that stop. This experiment gives the model weather, lightning and the wind direction relative to neighbouring fires, and checks whether it now beats the two-satellite rule.</p>
<p class="muted">{len(regions)} Fire Centres · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Label and training:</strong> as Experiment 20: S-NPP or NOAA-20 detection in the cell, gap-filled; 2018–2020 train on gap-filled S-NPP until FIRMS NOAA-20 arrives. NOAA-20 inputs are blank before 2021.</li>
<li><strong>Weather:</strong> BC Wildfire Service station values at noon local standard time on the day before the target (20:00 UTC, known before that UTC day ends): temperature, humidity, wind, 24 h and 3-day rain, Fire Weather Index codes. Each cell takes its nearest station with a reading, among its three nearest within 100 km.</li>
<li><strong>Lightning:</strong> CanCPLD cloud-to-ground flashes (0.1°) on that UTC day and the two before: the cell, 3×3 and 9×9 blocks, and days since lightning nearby.</li>
<li><strong>Spread:</strong> burning neighbours one and two cells away, each weighted by the cosine of the angle between its direction and the direction the wind blows from (positive = fire upwind), and that sum times wind speed.</li>
<li><strong>Thresholds:</strong> tuned to maximise 2022 F1 per Fire Centre, or with no tuning: each day, flag as many cells as the two-satellite rule does (the model only chooses which ones).</li></ul>
{sample_table}</section>

<section><h2>Results (2023)</h2>
<div class="cards"><div class="card"><h3>Two-satellite rule, BC-wide</h3><div class="value">{s("res7_exact", "rule_pair_yesterday"):.3f}</div><p>Exact cell; {s("res7_margin1", "rule_pair_yesterday"):.3f} with the one-cell margin.</p></div>
{card("res7_exact")}{card("res7_margin1")}</div>
{summary_table}
<p class="muted">BC-wide F1 pools all six Fire Centres. Precision and recall are BC-wide, exact cell.</p>
<figure>{image}<figcaption>2023 F1 by Fire Centre with 2022-tuned thresholds. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Week-block bootstrap against the two-satellite rule, BC-wide</h3>{boot_table}
<h3>Which errors change (exact cell, BC-wide, 2023)</h3>
<p>The two-satellite rule catches all {int(t0.continuing_positives):,} cells still burning, none of the {int(t0.new_positives):,} new fire cells, and flags all {int(t0.stopped):,} cells where the fire stopped.</p>
{split_table}
<h3>What the model uses (catboost_spread_noday)</h3>{group_table}
<details><summary>Show the 15 most important inputs</summary>{importance_table}</details>
<details><summary>Show F1 for every centre</summary>{per_centre}</details>
<details><summary>Show 2022 (threshold year) medians</summary>{selection_table}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2>{conclusions}</div>
<ul class="muted"><li>Training still uses the stopgap label: 2018–2020 lack NOAA-20 in both inputs and labels.</li>
<li>Station weather is a point reading up to 100 km away, at noon of the day before the target; the target day's own weather (rain, wind shifts) is unknown to the model. Weather forecasts would be the operational substitute.</li>
<li>2023 has been used to compare many experiments; only 2024 will give an untouched number. Untuned CatBoost, one seed. Neighbourhoods stop at each Fire Centre's grid edge.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 21 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
