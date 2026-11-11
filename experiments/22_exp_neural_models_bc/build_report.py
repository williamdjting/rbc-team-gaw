"""Build a standalone visual HTML report from saved Experiment 22 outputs.

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
from matplotlib.ticker import MaxNLocator
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
          "catboost_spread_noday": ("CatBoost, Experiment 21 (300 trees)", "#2a78d6", "o"),
          "mlp": ("MLP", "#eb6834", "s"),
          "gru": ("MLP + GRU over 14 days", "#1baf7a", "v")}
MODELS = [m for m in SERIES if not m.startswith("rule")]
DEFINITION_NAMES = {"res7_exact": "exact cell", "res7_margin1": "one-cell margin"}
METHOD_NAMES = {"f1_2022": "2022-tuned threshold", "rule_volume": "rule's daily volume"}
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




def curves_chart(curves):
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for name in ("mlp", "gru"):
        part = curves.loc[curves.model.eq(name)]
        label, colour, marker = SERIES[name]
        ax.plot(part.epoch, part.valid_log_loss, color=colour, marker=marker, markersize=6, linewidth=2, label=f"{label}, held-out weeks")
        ax.plot(part.epoch, part.train_log_loss, color=colour, linewidth=1.3, linestyle=":", label=f"{label}, training")
    ax.set(xlabel="Epoch", ylabel="Log loss", title="Neural training curves")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.savefig(FIGURES / "report_curves.png", dpi=150)
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
    fits = pd.read_csv(RESULTS / "fit_checks.csv").set_index("model")
    curves = pd.read_csv(RESULTS / "training_curves.csv")
    regions = manifest["regions"]
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]

    def pick(definition, model, method="f1_2022", frame=evaluation):
        return frame.loc[frame.definition.eq(definition) & frame.model.eq(model) & frame.method.eq(method)].set_index("region")

    order = pick("res7_margin1", "rule_pair_yesterday").f1.sort_values(ascending=False).index.tolist()
    f1_chart(evaluation, order)
    curves_chart(curves)
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
            row["Median AP (exact)"] = f'{summary.loc[("res7_exact", "f1_2022", name), "average_precision"]:.3f}' if not name.startswith("rule") and method == "f1_2022" else ""
            row["Centres at 0.70+ (margin)"] = f"{int((pick('res7_margin1', name, method).f1 >= .7).sum())} of {len(regions)}"
            rows.append(row)
    numeric = {c: "{:.3f}" for c in rows[0] if c.startswith(("F1", "Precision", "Recall"))}
    summary_table = html_table(pd.DataFrame(rows), numeric)

    def boot_table(baseline, label):
        b = boot.loc[boot.scope.eq("BC-wide") & boot.baseline.eq(baseline) & ~boot.model.eq("rule_snpp_yesterday")].copy()
        b["Forecast"] = b.model.map(lambda m: SERIES[m][0])
        b["Threshold"] = b.method.map(METHOD_NAMES)
        b["Definition"] = b.definition.map(DEFINITION_NAMES)
        b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
        return html_table(b[["Definition", "Forecast", "Threshold", "difference", "95% interval", "share_above"]].rename(
            columns={"difference": f"F1 minus {label}", "share_above": "Share of resamples above 0"}),
            {f"F1 minus {label}": "{:+.3f}", "Share of resamples above 0": "{:.0%}"})

    bb = boot.loc[boot.scope.eq("BC-wide")].set_index(["baseline", "definition", "model", "method"])
    region_boot = boot.loc[boot.scope.ne("BC-wide") & boot.method.eq("f1_2022")].set_index(["baseline", "scope", "definition", "model"])

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

    columns = {f"{SERIES[m][0]} ({DEFINITION_NAMES[d]})": pick(d, m).loc[order, "f1"].to_numpy()
               for d in DEFINITION_NAMES for m in SERIES}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, **columns}), {c: "{:.3f}" for c in columns})
    selection = metrics.loc[metrics.year.eq(manifest["selection_year"]) & metrics.method.eq("f1_2022")].groupby(
        ["definition", "model"]).f1.median().unstack(0)
    selection_table = html_table(selection.reindex(list(SERIES)).rename(index=lambda m: SERIES[m][0]).reset_index().rename(
        columns={"model": "Forecast", "res7_exact": "Median F1, exact (2022)", "res7_margin1": "Median F1, margin (2022)"}),
        {"Median F1, exact (2022)": "{:.3f}", "Median F1, margin (2022)": "{:.3f}"})
    fit_rows = pd.DataFrame([{"Model": SERIES[m][0], "Training rows": int(fits.loc[m, "rows"]),
                              "Inputs": int(fits.loc[m, "inputs"]),
                              "Parameters": "" if pd.isna(fits.loc[m].get("parameters")) else f'{int(fits.loc[m, "parameters"]):,}',
                              "Best epoch": "" if pd.isna(fits.loc[m].get("best_epoch")) else int(fits.loc[m, "best_epoch"]),
                              "Held-out log loss": "" if pd.isna(fits.loc[m].get("valid_log_loss")) else f'{fits.loc[m, "valid_log_loss"]:.4f}',
                              "Fit time (s)": fits.loc[m, "fit_seconds"]} for m in fits.index])
    fit_table = html_table(fit_rows, {"Training rows": "{:,}", "Fit time (s)": "{:.0f}"})
    sample_table = html_table(sample.groupby("year").agg(rows=("kept_rows", "sum"), positives=("positives", "sum")).reset_index().rename(
        columns={"year": "Training year", "rows": "Rows kept", "positives": "Training-label positives"}),
        {"Training year": "{}", "Rows kept": "{:,}", "Training-label positives": "{:,}"})

    neural = ("mlp", "gru")
    best = {d: max(neural, key=lambda m: s(d, m)) for d in DEFINITION_NAMES}
    ref = "catboost_spread_noday"
    ci = lambda base, d, m, k="f1_2022": (f'{bb.loc[(base, d, m, k), "difference"]:+.3f} (95% interval '
                                          f'{bb.loc[(base, d, m, k), "ci_low"]:+.3f} to {bb.loc[(base, d, m, k), "ci_high"]:+.3f})')
    card = lambda d: (f'<div class="card"><h3>Best new model, {DEFINITION_NAMES[d]}</h3><div class="value">'
                      f'{bb.loc[(ref, d, best[d], "f1_2022"), "difference"]:+.3f}</div><p>{SERIES[best[d]][0]} minus Experiment 21\'s CatBoost, '
                      f'BC-wide (95% interval {bb.loc[(ref, d, best[d], "f1_2022"), "ci_low"]:+.3f} to {bb.loc[(ref, d, best[d], "f1_2022"), "ci_high"]:+.3f}).</p></div>')
    beats = lambda d, m, base=ref: int((pick(d, m).f1 > pick(d, base).f1).sum())
    tn = lambda m, k="f1_2022": totals.loc[(m, k)]
    images = (embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre"),
              embedded_image(FIGURES / "report_curves.png", "Neural training curves"))
    ap = lambda m: summary.loc[("res7_exact", "f1_2022", m), "average_precision"]
    rule = "rule_pair_yesterday"
    conclusions = f'''
<p><strong>Partly. Both networks rank cells better than CatBoost and win on the exact cell. With the one-cell margin, CatBoost keeps a small lead.</strong></p>
<ul><li><strong>Exact cell:</strong> the MLP scores {s("res7_exact", "mlp"):.3f} BC-wide. CatBoost scores {s("res7_exact", ref):.3f}, so the MLP is {ci(ref, "res7_exact", "mlp")} ahead of it. The MLP+GRU scores {s("res7_exact", "gru"):.3f}, {ci(ref, "res7_exact", "gru")} ahead of CatBoost. Both gains hold at the rule's daily volume too (MLP {ci(ref, "res7_exact", "mlp", "rule_volume")}). The MLP beats CatBoost in {beats("res7_exact", "mlp")} of {len(regions)} centres. Median average precision, which needs no threshold, rises from {ap(ref):.3f} to {ap("mlp"):.3f} (MLP) and {ap("gru"):.3f} (MLP+GRU).</li>
<li><strong>But the exact cell is still a tie with the rule:</strong> the MLP minus the two-satellite rule ({s("res7_exact", rule):.3f}) is {ci(rule, "res7_exact", "mlp")}. At the rule's volume it is {bb.loc[(rule, "res7_exact", "mlp", "rule_volume"), "difference"]:+.3f}. This is the first learned model to reach the rule on the exact cell, though not to beat it.</li>
<li><strong>One-cell margin:</strong> CatBoost stays ahead. MLP minus CatBoost is {ci(ref, "res7_margin1", "mlp")}; MLP+GRU minus CatBoost is {ci(ref, "res7_margin1", "gru")}. Both networks still beat the rule (MLP {ci(rule, "res7_margin1", "mlp")}). At the rule's volume the networks edge CatBoost (MLP {ci(ref, "res7_margin1", "mlp", "rule_volume")}). The margin difference therefore comes from where each model's 2022 threshold lands, not from better ranking: CatBoost's threshold buys recall ({s("res7_margin1", ref, col="bc_wide_recall"):.3f} vs {s("res7_margin1", "mlp", col="bc_wide_recall"):.3f}), which the margin rewards.</li>
<li><strong>Where the exact-cell gain comes from:</strong> the networks keep more cells that are still burning. The MLP keeps {int(tn("mlp").continuing_caught):,} and CatBoost {int(tn(ref).continuing_caught):,}, out of {int(t0.continuing_positives):,}. They catch about as many new fire cells ({int(tn("mlp").new_caught):,} vs {int(tn(ref).new_caught):,} of {int(t0.new_positives):,}). The networks do not find new fires; they are better at not dropping fires that continue.</li>
<li><strong>The 14-day GRU adds nothing over the MLP.</strong> It catches more new fire cells ({int(tn("gru").new_caught):,}) but pays for them in false alarms ({int(tn("gru").fresh_false_alarms):,} vs {int(tn("mlp").fresh_false_alarms):,}). Its F1 and AP equal or trail the MLP's, and it takes about four times longer to fit. The hand-built rolling features already summarise the recent history.</li>
<li><strong>Both networks peak after one epoch.</strong> Held-out loss rises afterwards while training loss keeps falling, so the networks overfit the training seasons almost at once. Lower learning rates, stronger regularisation or averaging several seeds are the obvious next steps. The current numbers are a lower bound on what a network can do here.</li></ul>
<p>The deep learning model is a modest step: it closes the exact-cell gap to the rule without opening a new one, and all models stay within about 0.02 F1 of each other. With the margin every model is above the 0.70–0.80 goal ({s("res7_margin1", "gru"):.3f} to {s("res7_margin1", ref):.3f}); the exact cell stays near {s("res7_exact", "mlp"):.2f}. Neither result moves the information ceiling seen since Experiment 15. FIRMS NOAA-20 for 2018–2020 and gridded day-ahead weather remain the levers most likely to.</p>'''

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Deep learning vs CatBoost</title>
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
figure.narrow img{{max-width:760px}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:650}}td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 22 · Deep learning vs CatBoost</div>
<h1>Does a deep learning model learn more than CatBoost?</h1>
<p class="purpose"><strong>Purpose:</strong> every model so far has been gradient-boosted trees on hand-built features. This experiment trains an MLP on the same inputs, and an MLP plus a GRU that reads the last 14 days of each cell's fire, weather, lightning and spread history day by day, to see whether either learns something CatBoost misses.</p>
<p class="muted">{len(regions)} Fire Centres · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Same everything except the model:</strong> Experiment 21's label, training draw, inputs (weather, lightning and upwind spread, no <code>day_of_year</code>), thresholds and scoring. Experiment 21's CatBoost is loaded, not retrained, as the reference to beat; it reproduces its numbers exactly.</li>
<li><strong>MLP:</strong> three hidden layers (256, 256, 128). Heavy-tailed inputs get a signed log; all inputs are standardized; blanks (NOAA-20 before 2021, no station within 100 km) become 0 with a missing flag.</li>
<li><strong>MLP + GRU:</strong> the same MLP, plus a 64-unit GRU over the 14 days ending on the day before the target. Each day has {len(manifest["sequence_channels"])} channels per cell: S-NPP detections in the cell and rings 1–2, FRP, night detections, NOAA-20 and two-satellite detections nearby, station weather, lightning, upwind fire and an in-season flag.</li>
<li><strong>Training:</strong> binary cross-entropy with AdamW, the same sampled rows and quiet-row correction as CatBoost. One 7-day block in six of each training season is held out to pick the epoch; the neural models therefore train on about 5/6 of the rows CatBoost uses.</li></ul>
{sample_table}{fit_table}
<figure class="narrow">{images[1]}<figcaption>Training and held-out log loss per epoch; the saved model is the epoch with the lowest held-out loss.</figcaption></figure></section>

<section><h2>Results (2023)</h2>
<div class="cards"><div class="card"><h3>Experiment 21 CatBoost, BC-wide</h3><div class="value">{s("res7_margin1", ref):.3f}</div><p>One-cell margin; {s("res7_exact", ref):.3f} on the exact cell. Two-satellite rule: {s("res7_margin1", "rule_pair_yesterday"):.3f} / {s("res7_exact", "rule_pair_yesterday"):.3f}.</p></div>
{card("res7_exact")}{card("res7_margin1")}</div>
{summary_table}
<p class="muted">BC-wide F1 pools all six Fire Centres. Precision and recall are BC-wide, exact cell. Average precision (AP) is the median over centres and needs no threshold.</p>
<figure>{images[0]}<figcaption>2023 F1 by Fire Centre with 2022-tuned thresholds. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Week-block bootstrap against Experiment 21's CatBoost, BC-wide</h3>{boot_table(ref, "Experiment 21 CatBoost")}
<h3>Week-block bootstrap against the two-satellite rule, BC-wide</h3>{boot_table("rule_pair_yesterday", "two-satellite rule")}
<h3>Which errors change (exact cell, BC-wide, 2023)</h3>
<p>The two-satellite rule catches all {int(t0.continuing_positives):,} cells still burning, none of the {int(t0.new_positives):,} new fire cells, and flags all {int(t0.stopped):,} cells where the fire stopped.</p>
{split_table}
<details><summary>Show F1 for every centre</summary>{per_centre}</details>
<details><summary>Show 2022 (threshold year) medians</summary>{selection_table}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2>{conclusions}</div>
<ul class="muted"><li>Training still uses the stopgap label: 2018–2020 lack NOAA-20 in both inputs and labels.</li>
<li>One seed and one architecture per neural model, lightly tuned (no search over width, depth or learning rate); trained on CPU.</li>
<li>2023 has been used to compare many experiments; only 2024 will give an untouched number. Neighbourhoods stop at each Fire Centre's grid edge.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 22 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
