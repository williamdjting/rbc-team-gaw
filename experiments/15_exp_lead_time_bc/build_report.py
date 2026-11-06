"""Build a standalone visual HTML report from saved Experiment 15 outputs."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
spec = importlib.util.spec_from_file_location("experiment_03_report", HERE.parent / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE = report_03.RULE
DEFINITIONS = {"res7_margin1": "Res-7, one-cell margin", "res7_exact": "Res-7, exact cell"}
# Reference categorical slots 1-2 plus the gray reserved for rules.
LINES = {"model_f1": ("CatBoost, 2022 threshold", "#2a78d6", "o", "-"),
         "model_best_f1_any_2023_threshold": ("CatBoost, best threshold in hindsight", "#2a78d6", "o", ":"),
         "best_rule_f1": ("Best persistence rule (chosen on 2022)", RULE, "D", "-")}


def lead_chart(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, (definition, title) in zip(axes, DEFINITIONS.items()):
        median = comparison.loc[comparison.definition.eq(definition)].groupby("lead_days")[list(LINES)].median()
        for column, (name, colour, marker, line) in LINES.items():
            ax.plot(median.index, median[column], color=colour, marker=marker, linestyle=line, linewidth=2,
                    markersize=7, markeredgecolor="white")
        for lead, value in median.model_f1.items():
            ax.annotate(f"{value:.2f}", (lead, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=title, xlabel="Lead time (days after the last day of history)", xticks=list(median.index), ylim=(0, .85))
        style(ax)
    axes[0].set_ylabel("2023 F1, median of six Fire Centres")
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, linewidth=2, markeredgecolor="white", label=n)
               for n, c, m, ls in LINES.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_lead_time.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    comparison = pd.read_csv(RESULTS / "comparison_2023.csv")
    comparison["hindsight_minus_rule"] = comparison.model_best_f1_any_2023_threshold - comparison.best_rule_f1
    lead_chart(comparison)
    tables = {}
    for definition in DEFINITIONS:
        part = comparison.loc[comparison.definition.eq(definition)]
        median = part.groupby("lead_days").agg(
            model=("model_f1", "median"), hindsight=("model_best_f1_any_2023_threshold", "median"),
            rule=("best_rule_f1", "median"), yesterday=("yesterday_rule_f1", "median"),
            gain=("model_minus_best_rule", "median"), wins=("model_minus_best_rule", lambda s: f"{int((s > 0).sum())} of 6"))
        tables[definition] = html_table(median.reset_index().rename(columns={
            "lead_days": "Lead (days)", "model": "CatBoost F1", "hindsight": "CatBoost F1, best threshold in hindsight",
            "rule": "Best rule F1", "yesterday": "Fire-yesterday rule F1", "gain": "CatBoost minus best rule",
            "wins": "Centres where CatBoost is higher"}),
            {"Lead (days)": "{}", "CatBoost F1": "{:.3f}", "CatBoost F1, best threshold in hindsight": "{:.3f}",
             "Best rule F1": "{:.3f}", "Fire-yesterday rule F1": "{:.3f}", "CatBoost minus best rule": "{:+.3f}"})
    per_region = comparison.loc[comparison.definition.eq("res7_margin1")].pivot(index="region", columns="lead_days",
                                                                                  values="model_minus_best_rule")
    per_region_table = html_table(per_region.reset_index().rename(columns=lambda c: f"{c} days" if isinstance(c, int) else "Fire Centre"),
                                  {f"{lead} days": "{:+.3f}" for lead in per_region.columns})
    margin = comparison.loc[comparison.definition.eq("res7_margin1")].groupby("lead_days").median(numeric_only=True)
    image = embedded_image(FIGURES / "report_lead_time.png", "2023 F1 against lead time")

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Forecast lead time</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:16px 0 8px}}
p{{margin:8px 0;max-width:900px}}.purpose{{font-size:21px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
figure{{margin:0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}img{{display:block;width:100%;height:auto}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:650}}td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 15 · Forecast lead time</div>
<h1>Does the model pull away from persistence at longer leads?</h1>
<p class="purpose"><strong>Purpose:</strong> at a 1-day lead the previous-day rule is hard to beat. Its F1 falls quickly with lead time, and most fires at longer leads appear in cells that were not burning on day 0. This experiment trains one CatBoost model per lead (1, 2, 3, 5, 7 days) on Experiment 13's features and compares it with the best of six persistence rules.</p>
<p class="muted">Six Fire Centres · train 2018–2021 · rules and thresholds chosen on 2022 per centre, lead and definition · evaluate 2023 · 2024 excluded</p></header>
<section><h2>Results</h2>
<figure>{image}<figcaption>Median 2023 F1 across the six Fire Centres. The dotted line uses the best 2023 threshold in hindsight, which cannot be used in practice and shows the model's ceiling.</figcaption></figure>
<h3>{DEFINITIONS["res7_margin1"]}</h3>{tables["res7_margin1"]}
<h3>{DEFINITIONS["res7_exact"]}</h3>{tables["res7_exact"]}
<details><summary>Show CatBoost minus best rule for every centre (one-cell margin)</summary>{per_region_table}</details></section>
<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>The model's lead over persistence stays small at every lead.</strong> With the one-cell margin it is a median {margin.model_minus_best_rule.loc[2]:+.3f} at 2 days and {margin.model_minus_best_rule.loc[3]:+.3f} at 3 days. Even with the best threshold in hindsight it is {margin.hindsight_minus_rule.loc[5]:+.3f} at 5 days and {margin.hindsight_minus_rule.loc[7]:+.3f} at 7 days.</li>
<li><strong>Fire-history inputs carry little beyond persistence at any lead.</strong> The fires the rules miss at longer leads start in quiet cells, and nothing in the current inputs says where. Predicting them needs drivers the model does not yet have: weather for the forecast days, fuels and terrain, and lightning.</li>
<li><strong>At 5 and 7 days, 2022 thresholds are unreliable.</strong> 2022 had almost no fire at long leads in several centres, so some 2023 results drop well below what the model can do (for example Cariboo at 5 days). Pooled-year thresholds are needed before long leads are compared again.</li></ul></div>
<ul class="muted"><li>Targets are recorded detections on day e + L. Margin scoring follows Experiment 14.</li>
<li>Untuned CatBoost, one seed, one model per lead. 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 15 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
