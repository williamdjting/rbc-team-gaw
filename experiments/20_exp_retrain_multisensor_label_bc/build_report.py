"""Build a standalone visual HTML report from saved Experiment 20 outputs.

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
RULE, INK = report_03.RULE, report_03.INK
# Gray tones for the two rules; reference categorical slots 1-4 in fixed order for models.
SERIES = {"rule_snpp_yesterday": ("Rule: S-NPP detection yesterday", "#b5b3aa", "d"),
          "rule_pair_yesterday": ("Rule: S-NPP or NOAA-20 detection yesterday", RULE, "D"),
          "catboost_08": ("CatBoost, Experiment 13 (old label)", "#2a78d6", "o"),
          "catboost_stopgap": ("CatBoost, 2018–2021 stopgap label", "#eb6834", "^"),
          "catboost_2021": ("CatBoost, 2021 only", "#1baf7a", "s"),
          "catboost_2021_n20": ("CatBoost, 2021 only, + NOAA-20 inputs", "#eda100", "v")}
DEFINITION_NAMES = {"res7_exact": "exact cell", "res7_margin1": "one-cell margin"}


def f1_chart(evaluation, order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    offsets = np.linspace(-.3, .3, len(SERIES))
    for ax, definition in zip(axes, DEFINITION_NAMES):
        part = evaluation.loc[evaluation.definition.eq(definition)]
        for (name, (label, colour, marker)), offset in zip(SERIES.items(), offsets):
            rows = part.loc[part.model.eq(name)].set_index("region")
            ax.scatter(rows.loc[order, "f1"], positions + offset, s=48, marker=marker, color=colour,
                       edgecolor="white", linewidth=1.2, zorder=3)
        if definition == "res7_margin1":
            ax.axvspan(.7, .8, color="#e1e0d9", zorder=0)
        ax.set(title=f"2023 F1 on the new label, {DEFINITION_NAMES[definition]}", xlabel="F1 (2022 threshold)")
        style(ax)
    axes[0].set(yticks=positions, yticklabels=order)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=c, markeredgecolor="white", label=n)
               for n, c, m in SERIES.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_f1.png", dpi=150)
    plt.close(fig)


def pick_year(metrics, year, definition, model):
    return metrics.loc[metrics.year.eq(year) & metrics.definition.eq(definition) & metrics.model.eq(model)].set_index("region")


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    summary = pd.read_csv(RESULTS / "summary_2023.csv").set_index(["definition", "model"])
    boot = pd.read_csv(RESULTS / "bootstrap_2023.csv")
    sample = pd.read_csv(RESULTS / "training_sample.csv")
    regions = manifest["regions"]
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]
    pick = lambda definition, model: evaluation.loc[evaluation.definition.eq(definition) & evaluation.model.eq(model)].set_index("region")
    order = pick("res7_margin1", "rule_pair_yesterday").f1.sort_values(ascending=False).index.tolist()
    f1_chart(evaluation, order)

    rows = []
    for name in SERIES:
        row = {"Forecast": SERIES[name][0]}
        for definition, label in DEFINITION_NAMES.items():
            row[f"F1, {label} (BC-wide)"] = summary.loc[(definition, name), "bc_wide_f1"]
            row[f"F1, {label} (median)"] = summary.loc[(definition, name), "f1"]
            row[f"Centres at 0.70+ ({label})"] = f"{int((pick(definition, name).f1 >= .7).sum())} of {len(regions)}"
        rows.append(row)
    summary_table = html_table(pd.DataFrame(rows), {c: "{:.3f}" for c in rows[0] if c.startswith("F1")})
    b = boot.loc[boot.scope.eq("BC-wide")].copy()
    b["Forecast"] = b.model.map(lambda m: SERIES[m][0])
    b["Definition"] = b.definition.map(DEFINITION_NAMES)
    b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
    boot_table = html_table(b[["Definition", "Forecast", "difference", "95% interval", "share_above"]].rename(
        columns={"difference": "F1 minus two-satellite rule", "share_above": "Share of resamples above 0"}),
        {"F1 minus two-satellite rule": "{:+.3f}", "Share of resamples above 0": "{:.0%}"})
    columns = {f"{SERIES[m][0]} ({DEFINITION_NAMES[d]})": pick(d, m).loc[order, "f1"].to_numpy()
               for d in DEFINITION_NAMES for m in SERIES}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, **columns}), {c: "{:.3f}" for c in columns})
    selection = metrics.loc[metrics.year.eq(manifest["selection_year"])].groupby(["definition", "model"]).f1.median().unstack(0)
    selection_table = html_table(selection.reindex(list(SERIES)).rename(index=lambda m: SERIES[m][0]).reset_index().rename(
        columns={"model": "Forecast", "res7_exact": "Median F1, exact (2022)", "res7_margin1": "Median F1, margin (2022)"}),
        {"Median F1, exact (2022)": "{:.3f}", "Median F1, margin (2022)": "{:.3f}"})
    label_counts = sample.groupby("year")[["old_label_positives", "new_label_positives", "quiet_new_positives"]].sum().reset_index()
    label_table = html_table(label_counts.rename(columns={"year": "Training year", "old_label_positives": "Old-label positives",
                                                          "new_label_positives": "Training-label positives",
                                                          "quiet_new_positives": "…in quiet cells"}),
                             {"Training year": "{}", "Old-label positives": "{:,}", "Training-label positives": "{:,}", "…in quiet cells": "{:,}"})
    importance = pd.read_csv(RESULTS / "feature_importance__catboost_2021_n20.csv").head(12)
    importance_table = html_table(importance.rename(columns={"feature": "Input", "importance": "Importance (%)"}), {"Importance (%)": "{:.1f}"})
    s = lambda d, m: summary.loc[(d, m), "bc_wide_f1"]
    models = [m for m in SERIES if not m.startswith("rule")]
    best = {d: max(models, key=lambda m: s(d, m)) for d in DEFINITION_NAMES}
    bb = b.set_index(["definition", "model"])
    image = embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre on the new label")

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Retraining on multi-satellite label</title>
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
<header><div class="eyebrow">Experiment 20 · Retraining on the multi-satellite label</div>
<h1>With a cleaner label, does the model beat the rule?</h1>
<p class="purpose"><strong>Purpose:</strong> Experiment 19 built a more consistent label from both VIIRS satellites (S-NPP and NOAA-20) with one-day gap filling. This experiment retrains the forecast on it, using a stopgap for training years without NOAA-20.</p>
<p class="muted">{len(regions)} Fire Centres · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Label:</strong> an S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell, plus a cell positive the day before and the day after. Every forecast is scored on this label.</li>
<li><strong>Stopgap training label:</strong> 2021 has the full label; 2018–2020 use gap-filled S-NPP only (NOAA-20 is absent in 2018 and partial in 2019–2020). FIRMS NOAA-20 for 2018–2020 will replace this.</li>
<li><strong>Inputs:</strong> Experiment 13's features from S-NPP detections. One model adds NOAA-20 and two-satellite inputs (detections in the cell and its neighbours over 1, 3 and 7 days). No input uses gap filling.</li>
<li><strong>Rules:</strong> the old rule (S-NPP detection yesterday), and the new bar, an S-NPP or NOAA-20 detection yesterday.</li></ul>
{label_table}</section>

<section><h2>Results (2023)</h2>
<div class="cards"><div class="card"><h3>Two-satellite rule, BC-wide</h3><div class="value">{s("res7_exact", "rule_pair_yesterday"):.3f}</div><p>Exact cell; {s("res7_margin1", "rule_pair_yesterday"):.3f} with the one-cell margin.</p></div>
<div class="card"><h3>Best model, exact cell</h3><div class="value">{bb.loc[("res7_exact", best["res7_exact"]), "difference"]:+.3f}</div><p>{SERIES[best["res7_exact"]][0]} minus the two-satellite rule (95% interval {bb.loc[("res7_exact", best["res7_exact"]), "ci_low"]:+.3f} to {bb.loc[("res7_exact", best["res7_exact"]), "ci_high"]:+.3f}).</p></div>
<div class="card"><h3>Best model, one-cell margin</h3><div class="value">{bb.loc[("res7_margin1", best["res7_margin1"]), "difference"]:+.3f}</div><p>{SERIES[best["res7_margin1"]][0]} minus the two-satellite rule (95% interval {bb.loc[("res7_margin1", best["res7_margin1"]), "ci_low"]:+.3f} to {bb.loc[("res7_margin1", best["res7_margin1"]), "ci_high"]:+.3f}).</p></div></div>
{summary_table}
<figure>{image}<figcaption>2023 F1 by Fire Centre on the new label. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Week-block bootstrap against the two-satellite rule, BC-wide</h3>{boot_table}
<details><summary>Show F1 for every centre</summary>{per_centre}</details>
<details><summary>Show 2022 (threshold year) medians</summary>{selection_table}</details>
<details><summary>Show the most important inputs (2021 model with NOAA-20 inputs)</summary>{importance_table}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>The two-satellite rule is the best forecast of the new label.</strong> BC-wide F1 {s("res7_exact", "rule_pair_yesterday"):.3f} on the exact cell and {s("res7_margin1", "rule_pair_yesterday"):.3f} with the one-cell margin; {int((pick("res7_margin1", "rule_pair_yesterday").f1 >= .7).sum())} of {len(regions)} centres reach 0.70 with the margin. It beats the old S-NPP rule by {-bb.loc[("res7_exact", "rule_snpp_yesterday"), "difference"]:.3f} (exact) and every model; all bootstrap intervals against it lie below 0.</li>
<li><strong>Retraining on the stopgap label did not help:</strong> {s("res7_exact", "catboost_stopgap"):.3f} / {s("res7_margin1", "catboost_stopgap"):.3f} BC-wide, the same as Experiment 13's model on the old label ({s("res7_exact", "catboost_08"):.3f} / {s("res7_margin1", "catboost_08"):.3f}). Three of its four training years have no NOAA-20, so it mostly learns the old label, from S-NPP inputs only.</li>
<li><strong>NOAA-20 inputs help, but one training season is not enough.</strong> Among the 2021-only models, NOAA-20 inputs add {s("res7_exact", "catboost_2021_n20") - s("res7_exact", "catboost_2021"):+.3f} (exact) and {s("res7_margin1", "catboost_2021_n20") - s("res7_margin1", "catboost_2021"):+.3f} (margin). In 2022, where its thresholds were tuned, that model matches or beats the two-satellite rule with the margin in {int((pick_year(metrics, manifest["selection_year"], "res7_margin1", "catboost_2021_n20").f1 >= pick_year(metrics, manifest["selection_year"], "res7_margin1", "rule_pair_yesterday").f1).sum())} of {len(regions)} centres, but it falls behind in 2023: what it learned from one season does not carry over.</li>
<li><strong>Why the rule wins:</strong> the cleaner label makes "burning yesterday, burning today" more reliable, and the two-satellite rule uses that directly. The models see NOAA-20 in at most one training season, so they cannot learn how much to trust it.</li>
<li><strong>Next:</strong> rerun when FIRMS NOAA-20 for 2018–2020 arrives, so every training year has the full label and two-satellite inputs. Fire history alone has not beaten persistence in any experiment, so the next inputs should be drivers the rule cannot see: BCWS station weather (now downloaded) and lightning.</li></ul></div>
<ul class="muted"><li>The training label is a stopgap: 2018–2020 lack NOAA-20, so models trained on those years learn a less consistent label than the one they are scored on.</li>
<li>Models trained on 2021 alone see one fire season, so their thresholds and scores may not transfer to other seasons.</li>
<li>Neighbourhoods stop at each Fire Centre's grid edge. Untuned CatBoost, one seed. 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 20 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
