"""Build a standalone visual HTML report from saved Experiment 14 outputs."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
spec = importlib.util.spec_from_file_location("experiment_03_report", HERE.parent / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE, INK = report_03.RULE, report_03.INK
DEFINITIONS = {"res7_exact": ("Res-7, exact cell", "Current target: a detection in this ~5 km² cell tomorrow"),
               "res7_margin1": ("Res-7, one-cell margin", "A flag is correct if fire is detected in the cell or a neighbour; a detection is caught if it or a neighbour was flagged (~2–3 km)"),
               "res6_exact": ("Res-6, exact cell", "Cells of ~36 km² (about 7 res-7 cells); a res-6 cell's score is its highest res-7 score"),
               "res6_margin1": ("Res-6, one-cell margin", "Res-6 cells with the same one-cell margin")}
# Reference categorical slots 1-4 in fixed order; marker shape repeats the identity.
SERIES = {("gradient_boosting", "strict"): ("Boosting, trained on exact cell", "#2a78d6", "o"),
          ("catboost", "strict"): ("CatBoost, trained on exact cell", "#eb6834", "^"),
          ("gradient_boosting", "cell_or_neighbour"): ("Boosting, trained on cell or neighbour", "#1baf7a", "s"),
          ("catboost", "cell_or_neighbour"): ("CatBoost, trained on cell or neighbour", "#eda100", "D")}
GOAL = (0.70, 0.80)


def f1_panels(evaluation, order):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.4), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    for ax, (definition, (title, _)) in zip(axes, DEFINITIONS.items()):
        part = evaluation.loc[evaluation.definition.eq(definition)]
        ax.axvspan(*GOAL, color="#e1e0d9", zorder=0)
        rule = part.loc[part.model.eq("rule_local_1d")].set_index("region")
        ax.scatter(rule.loc[order, "f1"], positions, marker="D", s=58, color=RULE, edgecolor="white", linewidth=1.5, zorder=3)
        for (key, (name, colour, marker)), offset in zip(SERIES.items(), (-.27, -.09, .09, .27)):
            rows = part.loc[part.model.eq(key[0]) & part.training_target.eq(key[1])].set_index("region")
            ax.scatter(rows.loc[order, "f1"], positions + offset, s=40, marker=marker, color=colour,
                       edgecolor="white", linewidth=1.2, zorder=4)
        ax.set(title=title, xlim=(0.4, 0.9), xlabel="2023 F1 (2022 threshold)")
        style(ax)
    axes[0].set(yticks=positions, yticklabels=order)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=RULE, markeredgecolor="white",
                      label="Previous-day rule")]
    handles += [Line2D([], [], linestyle="none", marker=m, markersize=7, markerfacecolor=c, markeredgecolor="white", label=n)
                for n, c, m in SERIES.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#e1e0d9", label="F1 goal 0.70–0.80"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_f1_by_definition.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]
    models = evaluation.loc[evaluation.model.ne("rule_local_1d")]
    best_model = {d: models.loc[models.definition.eq(d)].groupby(["model", "training_target"]).f1.median().idxmax()
                  for d in DEFINITIONS}
    pick = lambda d, key: evaluation.loc[evaluation.definition.eq(d) & evaluation.model.eq(key[0])
                                         & evaluation.training_target.eq(key[1])].set_index("region")
    rule = {d: pick(d, ("rule_local_1d", "fixed")) for d in DEFINITIONS}
    best = {d: pick(d, best_model[d]) for d in DEFINITIONS}
    order = best["res7_margin1"].f1.sort_values(ascending=False).index.tolist()
    f1_panels(evaluation, order)

    overview = html_table(pd.DataFrame([{
        "Definition": DEFINITIONS[d][0], "What counts as correct": DEFINITIONS[d][1],
        "Rule F1 (median)": rule[d].f1.median(),
        "Best model": SERIES[best_model[d]][0],
        "Best model F1 (median)": best[d].f1.median(),
        "Range across centres": f"{best[d].f1.min():.3f}–{best[d].f1.max():.3f}",
        "Centres at 0.70 or more": f"{int((best[d].f1 >= GOAL[0]).sum())} of 6",
        "Best threshold in hindsight (median)": best[d].best_f1_any_2023_threshold.median()} for d in DEFINITIONS]),
        {"Rule F1 (median)": "{:.3f}", "Best model F1 (median)": "{:.3f}", "Best threshold in hindsight (median)": "{:.3f}"})
    detail = html_table(pd.DataFrame({"Fire Centre": order, **{
        f"{DEFINITIONS[d][0]}: {label}": frame.loc[order, column].to_numpy()
        for d in DEFINITIONS for label, frame, column in (("rule", rule[d], "f1"), ("model", best[d], "f1"),
                                                           ("precision", best[d], "precision"), ("recall", best[d], "recall"))}}),
        {c: ("{:.1%}" if c.endswith(("precision", "recall")) else "{:.3f}")
         for c in [f"{DEFINITIONS[d][0]}: {label}" for d in DEFINITIONS for label in ("rule", "model", "precision", "recall")]})
    full = evaluation.assign(name=evaluation.model.str.replace("_", " ") + " · " + evaluation.training_target.str.replace("_", " "),
                             definition_name=evaluation.definition.map(lambda d: DEFINITIONS[d][0]))
    full_table = html_table(full.sort_values(["region", "definition", "model", "training_target"])[
        ["region", "definition_name", "name", "threshold", "precision", "recall", "f1", "best_f1_any_2023_threshold", "flagged", "positives"]].rename(columns={
            "region": "Fire Centre", "definition_name": "Definition", "name": "Model · trained on", "threshold": "Threshold (2022)",
            "precision": "Precision", "recall": "Recall", "f1": "F1", "best_f1_any_2023_threshold": "Best F1 in hindsight",
            "flagged": "Cells flagged", "positives": "Positive cells"}),
        {"Threshold (2022)": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}",
         "Best F1 in hindsight": "{:.3f}", "Cells flagged": "{:,}", "Positive cells": "{:,}"})
    margin_gain = best["res7_margin1"].f1 - rule["res7_margin1"].f1
    strict_gain = best["res7_exact"].f1 - rule["res7_exact"].f1
    image = embedded_image(FIGURES / "report_f1_by_definition.png", "2023 F1 by Fire Centre under four target definitions")
    reach = [DEFINITIONS[d][0] for d in DEFINITIONS if best[d].f1.median() >= GOAL[0]]

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Target definitions and F1</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 12px}}
p{{margin:8px 0}}.purpose{{font-size:21px;max-width:900px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0}}figure{{margin:0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:8px 4px}}
.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.value{{font-size:40px;font-weight:750;color:var(--teal)}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted);font-weight:650}}
td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
.wrap td{{white-space:normal}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 14 · Target definitions</div>
<h1>Which target definition can reach F1 0.70–0.80?</h1>
<p class="purpose"><strong>Purpose:</strong> Experiment 13 showed that, with a detection in the exact ~5 km² cell as the target, no model beats the previous-day rule, and the rule sits near 0.55. Most errors are one-cell shifts. This experiment scores the same models under four definitions of a correct forecast.</p>
<p class="muted">Six Fire Centres · Experiment 13 features · train 2018–2021 · thresholds 2022 per centre, definition and model · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Results</h2>
<div class="cards"><div class="card"><h3>Best model, res-7 with one-cell margin</h3><div class="value">{best["res7_margin1"].f1.median():.3f}</div><p>Median 2023 F1; {int((best["res7_margin1"].f1 >= GOAL[0]).sum())} of 6 centres at 0.70 or more.</p></div>
<div class="card"><h3>Model minus rule, one-cell margin</h3><div class="value">{margin_gain.median():+.3f}</div><p>Median F1 change; model higher in {int((margin_gain > 0).sum())} of 6 centres. On the exact cell it is {strict_gain.median():+.3f}.</p></div>
<div class="card"><h3>Best model, res-6 with one-cell margin</h3><div class="value">{best["res6_margin1"].f1.median():.3f}</div><p>Median 2023 F1 with ~36 km² cells.</p></div></div>
<div class="table wrap">{overview[len('<div class="table">'):-len('</div>')]}</div>
<figure>{image}<figcaption>Each panel is one definition. The shaded band is the 0.70–0.80 goal. Gray diamonds are the previous-day rule scored the same way.</figcaption></figure>
<details><summary>Show rule and best-model F1, precision and recall for every centre</summary>{detail}</details>
<details><summary>Show every model under every definition</summary>{full_table}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>The definition sets the ceiling.</strong> Median best-model F1 moves from {best["res7_exact"].f1.median():.3f} on the exact cell to {best["res7_margin1"].f1.median():.3f} with a one-cell margin, with the same models and data. Definitions with a median of at least 0.70: {", ".join(reach) or "none"}.</li>
<li><strong>The one-cell margin keeps 5 km² cells.</strong> It accepts a forecast that is off by one cell (about 2–3 km), which is how fire spread and the satellite's pixel position move detections. This is the usual neighbourhood check in forecast verification.</li>
<li>Exact-cell F1 should stay in every report next to the chosen definition, so results remain comparable with Experiments 02–07 and 13.</li>
<li>Next: NOAA-20 detections reduce missed days under every definition; ERA5-Land day-t rain and wind target the fires that stop.</li></ul></div>
<ul class="muted"><li>Targets are <strong>recorded detections</strong>, not ignitions or confirmed absence. Coverage remains unknown.</li>
<li>Margin scoring: a flagged cell is correct if a detection (any Fire Centre) falls in it or a neighbour; a detected cell is caught if it or an in-centre neighbour is flagged. Res-6 scores are the highest res-7 score among each cell's children.</li>
<li>Thresholds are chosen on 2022 per centre, definition and model. Untuned models and a single seed; differences under about 0.01 F1 are within noise. 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 14 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    output = HERE / "RESULTS.html"
    output.write_text(page, encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
