"""Build a standalone visual HTML report from saved Experiment 33 outputs.

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
EXPERIMENTS = HERE.parent
FIGURES = HERE / "figures"
spec = importlib.util.spec_from_file_location("experiment_03_report", EXPERIMENTS / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE = report_03.RULE
# Gray tones for the two rules; reference categorical slots 1-5 in fixed order for models.
SERIES = {"rule_snpp_yesterday": ("Rule: S-NPP detection yesterday", "#b5b3aa", "d"),
          "rule_pair_yesterday": ("Rule: S-NPP or NOAA-20 detection yesterday", RULE, "D"),
          "catboost_spread_noday": ("CatBoost (Experiment 21)", "#2a78d6", "o"),
          "mlp": ("MLP (Experiment 22)", "#eb6834", "s"),
          "gru": ("MLP + GRU (Experiment 22)", "#1baf7a", "v"),
          "mlp_reg": ("MLP, regularised (Experiment 23)", "#eda100", "^"),
          "gru_reg": ("MLP + GRU, regularised (Experiment 23)", "#8a5cc2", "P")}
DEFINITION_NAMES = {"res7_exact": "exact cell", "res7_margin1": "one-cell margin"}
METHOD_NAMES = {"f1_2022": "2022-tuned threshold", "rule_volume": "rule's daily volume"}
DEVELOPMENT = {"rule_snpp_yesterday": "21_exp_weather_lightning_spread_bc", "rule_pair_yesterday": "21_exp_weather_lightning_spread_bc",
               "catboost_spread_noday": "21_exp_weather_lightning_spread_bc", "mlp": "22_exp_neural_models_bc",
               "gru": "22_exp_neural_models_bc"}  # Experiment 23 never finished scoring 2023


def f1_chart(test, order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    offsets = np.linspace(-.36, .36, len(SERIES))
    for ax, definition in zip(axes, DEFINITION_NAMES):
        part = test.loc[test.definition.eq(definition) & test.method.eq("f1_2022")]
        for (name, (label, colour, marker)), offset in zip(SERIES.items(), offsets):
            rows = part.loc[part.model.eq(name)].set_index("region")
            ax.scatter(rows.loc[order, "f1"], positions + offset, s=42, marker=marker, color=colour,
                       edgecolor="white", linewidth=1.0, zorder=3)
        if definition == "res7_margin1":
            ax.axvspan(.7, .8, color="#e1e0d9", zorder=0)
        ax.set(title=f"2024 F1, {DEFINITION_NAMES[definition]}", xlabel="F1 (2022-tuned threshold)")
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
    test_year = manifest["test_year"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    summary = pd.read_csv(RESULTS / f"summary_{test_year}.csv").set_index(["definition", "method", "model"])
    boot = pd.read_csv(RESULTS / f"bootstrap_{test_year}.csv")
    splits = pd.read_csv(RESULTS / f"new_vs_continuing_{test_year}.csv")
    coverage = pd.read_csv(RESULTS / "label_coverage.csv")
    regions = manifest["regions"]
    test = metrics.loc[metrics.year.eq(test_year)]
    development = {m: pd.read_csv(EXPERIMENTS / folder / "results/summary_2023.csv").set_index(["definition", "method", "model"])
                   for m, folder in DEVELOPMENT.items()}

    def pick(definition, model, method="f1_2022"):
        return test.loc[test.definition.eq(definition) & test.model.eq(model) & test.method.eq(method)].set_index("region")

    order = pick("res7_margin1", "rule_pair_yesterday").f1.sort_values(ascending=False).index.tolist()
    f1_chart(test, order)
    s = lambda d, m, k="f1_2022", col="bc_wide_f1": summary.loc[(d, k, m), col]
    dev = lambda d, m, k="f1_2022": development[m].loc[(d, k, m), "bc_wide_f1"] if m in development else np.nan

    rows = []
    for name in SERIES:
        for method in METHOD_NAMES:
            if name.startswith("rule") and method == "rule_volume":
                continue
            row = {"Forecast": SERIES[name][0], "Threshold": "fixed (0/1)" if name.startswith("rule") else METHOD_NAMES[method]}
            for definition, label in DEFINITION_NAMES.items():
                row[f"2024 F1, {label}"] = s(definition, name, method)
                row[f"2023 F1, {label}"] = "" if np.isnan(dev(definition, name, method)) else f"{dev(definition, name, method):.3f}"
            row["2024 precision (exact)"] = s("res7_exact", name, method, "bc_wide_precision")
            row["2024 recall (exact)"] = s("res7_exact", name, method, "bc_wide_recall")
            row["Centres at 0.70+ (margin)"] = f"{int((pick('res7_margin1', name, method).f1 >= .7).sum())} of {len(regions)}"
            rows.append(row)
    numeric = {c: "{:.3f}" for c in rows[0] if c.startswith(("2024",))}
    summary_table = html_table(pd.DataFrame(rows), numeric)

    def boot_table(baseline, label):
        b = boot.loc[boot.scope.eq("BC-wide") & boot.baseline.eq(baseline) & ~boot.model.isin(["rule_snpp_yesterday"])].copy()
        b["Forecast"] = b.model.map(lambda m: SERIES[m][0])
        b["Threshold"] = b.method.map(METHOD_NAMES)
        b["Definition"] = b.definition.map(DEFINITION_NAMES)
        b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
        column = f"F1 minus {label}"
        return html_table(b[["Definition", "Forecast", "Threshold", "difference", "95% interval", "share_above"]].rename(
            columns={"difference": column, "share_above": "Share of resamples above 0"}),
            {column: "{:+.3f}", "Share of resamples above 0": "{:.0%}"})

    bb = boot.loc[boot.scope.eq("BC-wide")].set_index(["baseline", "definition", "model", "method"])
    totals = splits.groupby(["model", "method"]).sum(numeric_only=True)
    split_rows = [{"Forecast": SERIES[name][0], "Threshold": "fixed (0/1)" if name.startswith("rule") else METHOD_NAMES[method],
                   "New fire cells caught": int(t.new_caught), "Continuing cells caught": int(t.continuing_caught),
                   "Stopping cells flagged (false alarms)": int(t.stopped_flagged), "Other false alarms": int(t.fresh_false_alarms)}
                  for name in SERIES for method in METHOD_NAMES if (name, method) in totals.index
                  for t in [totals.loc[(name, method)]]]
    t0 = totals.loc[("rule_pair_yesterday", "f1_2022")]
    split_table = html_table(pd.DataFrame(split_rows), {c: "{:,}" for c in split_rows[0] if c not in ("Forecast", "Threshold")})
    columns = {f"{SERIES[m][0]} ({DEFINITION_NAMES[d]})": pick(d, m).loc[order, "f1"].to_numpy() for d in DEFINITION_NAMES for m in SERIES}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, **columns}), {c: "{:.3f}" for c in columns})
    cov = coverage.pivot(index="region", columns="year", values="label_cell_days").loc[order]
    coverage_table = html_table(pd.DataFrame({"Fire Centre": order, "Fire cell-days, 2022": cov[2022].to_numpy(),
                                              f"Fire cell-days, {test_year}": cov[test_year].to_numpy()}),
                                {"Fire cell-days, 2022": "{:,}", f"Fire cell-days, {test_year}": "{:,}"})

    rule, ref = "rule_pair_yesterday", "catboost_spread_noday"
    learned = [m for m in SERIES if not m.startswith("rule")]
    best = {d: max(learned, key=lambda m: s(d, m)) for d in DEFINITION_NAMES}
    ci = lambda base, d, m, k="f1_2022": (f'{bb.loc[(base, d, m, k), "difference"]:+.3f} (95% interval '
                                          f'{bb.loc[(base, d, m, k), "ci_low"]:+.3f} to {bb.loc[(base, d, m, k), "ci_high"]:+.3f})')
    card = lambda d: (f'<div class="card"><h3>Best learned model, {DEFINITION_NAMES[d]}</h3><div class="value">{s(d, best[d]):.3f}</div>'
                      f'<p>{SERIES[best[d]][0]}. Minus the two-satellite rule: {ci(rule, d, best[d])}.</p></div>')
    tn = lambda m, k="f1_2022": totals.loc[(m, k)]
    image = embedded_image(FIGURES / "report_f1.png", f"{test_year} F1 by Fire Centre")
    conclusions = ""

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Final test on {test_year}</title>
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
<header><div class="eyebrow">Experiment 33 · Final test on {test_year}</div>
<h1>How do the forecasts do on a season none of them has seen?</h1>
<p class="purpose"><strong>Purpose:</strong> 2023 was used to compare many models, so its scores are optimistic. {test_year} was never loaded until now. The saved models of Experiments 20–23 are scored on it once, with thresholds chosen on 2022, nothing retrained or tuned.</p>
<p class="muted">{len(regions)} Fire Centres · trained on 2018–2021 · thresholds 2022 · tested on {test_year} · models and metrics fixed in README.md before scoring</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Label:</strong> S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell on the target day, one-day gaps filled, as in 2023. NOAA-21, new in {test_year}, is not used.</li>
<li><strong>Inputs for {test_year}:</strong> downloaded and built into separate files with the same scripts as 2018–2023. Rebuilding 2023's feature tables with the {test_year} script reproduces Experiment 02's exactly.</li>
<li><strong>Checks:</strong> on 2022, every forecast reproduces Experiments 21 and 22 on every Fire Centre, so the thresholds are the ones used before.</li>
<li><strong>A quieter season:</strong> {test_year} had far fewer fire cell-days than 2023, but similar to 2022 in most centres.</li></ul>
{coverage_table}</section>

<section><h2>Results ({test_year})</h2>
<div class="cards"><div class="card"><h3>Two-satellite rule, BC-wide</h3><div class="value">{s("res7_margin1", rule):.3f}</div><p>One-cell margin; {s("res7_exact", rule):.3f} on the exact cell. In 2023: {dev("res7_margin1", rule):.3f} / {dev("res7_exact", rule):.3f}.</p></div>
{card("res7_exact")}{card("res7_margin1")}</div>
{summary_table}
<p class="muted">BC-wide F1 pools all six Fire Centres. 2023 columns are the development scores from Experiments 21 and 22; Experiment 23's models were never fully scored on 2023.</p>
<figure>{image}<figcaption>{test_year} F1 by Fire Centre with 2022-tuned thresholds. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Week-block bootstrap against the two-satellite rule, BC-wide</h3>{boot_table(rule, "two-satellite rule")}
<h3>Week-block bootstrap against Experiment 21's CatBoost, BC-wide</h3>{boot_table(ref, "CatBoost")}
<h3>Which errors change (exact cell, BC-wide, {test_year})</h3>
<p>The two-satellite rule catches all {int(t0.continuing_positives):,} cells still burning, none of the {int(t0.new_positives):,} new fire cells, and flags all {int(t0.stopped):,} cells where the fire stopped.</p>
{split_table}
<details><summary>Show F1 for every centre</summary>{per_centre}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2>{conclusions}</div>
<ul class="muted"><li>One season: week-block intervals cover day-to-day variation within {test_year}, not how different another season might be.</li>
<li>Models were trained with the stopgap label (2018–2020 without NOAA-20).</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 33 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
