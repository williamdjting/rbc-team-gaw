"""Build a standalone visual HTML report from saved Experiment 23 outputs.

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
EXPERIMENT_17 = HERE.parent / "22_exp_neural_models_bc/results"
FIGURES = HERE / "figures"
spec = importlib.util.spec_from_file_location("experiment_03_report", HERE.parent / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE = report_03.RULE
# Gray tones for the two rules; reference categorical slots 1-4 in fixed order for models.
SERIES = {"rule_snpp_yesterday": ("Rule: S-NPP detection yesterday", "#b5b3aa", "d"),
          "rule_pair_yesterday": ("Rule: S-NPP or NOAA-20 detection yesterday", RULE, "D"),
          "catboost_spread_noday": ("CatBoost, Experiment 21", "#2a78d6", "o"),
          "mlp": ("MLP, Experiment 22", "#eb6834", "s"),
          "mlp_reg": ("MLP, regularised", "#1baf7a", "^"),
          "gru_reg": ("MLP + GRU, regularised", "#eda100", "v")}
TRAINED = ("mlp_reg", "gru_reg")
DEFINITION_NAMES = {"res7_exact": "exact cell", "res7_margin1": "one-cell margin"}
METHOD_NAMES = {"f1_2022": "2022-tuned threshold", "rule_volume": "rule's daily volume"}
BASELINE_NAMES = {"catboost_spread_noday": "Experiment 21 CatBoost", "mlp": "Experiment 22 MLP",
                  "rule_pair_yesterday": "two-satellite rule"}


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
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_f1.png", dpi=150)
    plt.close(fig)


def curves_chart(curves, old_curves):
    """Held-out loss per epoch: Experiment 22's settings (faint) against the regularised ones."""
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for frame, name, label, colour, marker, alpha in (
            (old_curves, "mlp", "MLP, Experiment 22", SERIES["mlp"][1], "s", .55),
            (old_curves, "gru", "MLP + GRU, Experiment 22", SERIES["gru_reg"][1], "v", .55),
            (curves, "mlp_reg", SERIES["mlp_reg"][0], SERIES["mlp_reg"][1], "^", 1),
            (curves, "gru_reg", SERIES["gru_reg"][0], SERIES["gru_reg"][1], "v", 1)):
        part = frame.loc[frame.model.eq(name)]
        ax.plot(part.epoch, part.valid_log_loss, color=colour, marker=marker, markersize=5, alpha=alpha,
                linewidth=2 if alpha == 1 else 1.3, linestyle="-" if alpha == 1 else "--", label=label)
    ax.set(xlabel="Epoch", ylabel="Held-out log loss", title="Held-out loss per epoch")
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
    fits = pd.read_csv(RESULTS / "fit_checks.csv").set_index("model")
    old_fits = pd.read_csv(EXPERIMENT_17 / "fit_checks.csv").set_index("model")
    curves = pd.read_csv(RESULTS / "training_curves.csv")
    old_curves = pd.read_csv(EXPERIMENT_17 / "training_curves.csv")
    old_summary = pd.read_csv(EXPERIMENT_17 / "summary_2023.csv").set_index(["definition", "method", "model"])
    regions = manifest["regions"]
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]

    def pick(definition, model, method="f1_2022", frame=evaluation):
        return frame.loc[frame.definition.eq(definition) & frame.model.eq(model) & frame.method.eq(method)].set_index("region")

    order = pick("res7_margin1", "rule_pair_yesterday").f1.sort_values(ascending=False).index.tolist()
    f1_chart(evaluation, order)
    curves_chart(curves, old_curves)
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
    gru_17 = {d: old_summary.loc[(d, "f1_2022", "gru"), "bc_wide_f1"] for d in DEFINITION_NAMES}

    def boot_table(baseline):
        b = boot.loc[boot.scope.eq("BC-wide") & boot.baseline.eq(baseline)].copy()
        b["Forecast"] = b.model.map(lambda m: SERIES[m][0])
        b["Threshold"] = b.method.map(METHOD_NAMES)
        b["Definition"] = b.definition.map(DEFINITION_NAMES)
        b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
        label = f"F1 minus {BASELINE_NAMES[baseline]}"
        return html_table(b[["Definition", "Forecast", "Threshold", "difference", "95% interval", "share_above"]].rename(
            columns={"difference": label, "share_above": "Share of resamples above 0"}),
            {label: "{:+.3f}", "Share of resamples above 0": "{:.0%}"})

    bb = boot.loc[boot.scope.eq("BC-wide")].set_index(["baseline", "definition", "model", "method"])

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

    neural, old = manifest["neural"], manifest["experiment_17_neural"]
    fit_rows = pd.DataFrame([{"Model": label, "Settings": settings, "Best epoch": int(f.loc[m, "best_epoch"]),
                              "Epochs run": int(f.loc[m, "epochs_run"]) if "epochs_run" in f else int(c.loc[c.model.eq(m), "epoch"].max()),
                              "Held-out log loss": f.loc[m, "valid_log_loss"], "Fit time (s)": f.loc[m, "fit_seconds"]}
                             for label, settings, f, c, m in (
                                 ("MLP", "Experiment 22", old_fits, old_curves, "mlp"),
                                 ("MLP", "regularised", fits, curves, "mlp_reg"),
                                 ("MLP + GRU", "Experiment 22", old_fits, old_curves, "gru"),
                                 ("MLP + GRU", "regularised", fits, curves, "gru_reg"))])
    fit_table = html_table(fit_rows, {"Held-out log loss": "{:.4f}", "Fit time (s)": "{:.0f}"})
    settings_table = html_table(pd.DataFrame([
        {"Setting": "Learning rate", "Experiment 22": f'{old["lr"]:g}', "Regularised": f'{neural["lr"]:g}'},
        {"Setting": "Dropout", "Experiment 22": f'{old["dropout"]:g} after the first two layers',
         "Regularised": f'{neural["dropout"]:g} after every layer and on the GRU summary'},
        {"Setting": "Weight decay (AdamW)", "Experiment 22": f'{old["weight_decay"]:g}, all parameters',
         "Regularised": f'{neural["weight_decay"]:g}, weights only'},
        {"Setting": "Epochs (max, patience)", "Experiment 22": f'{old["max_epochs"]}, {old["patience"]}',
         "Regularised": f'{neural["max_epochs"]}, {neural["patience"]}'}]), {})

    ref, base, rule = "catboost_spread_noday", "mlp", "rule_pair_yesterday"
    ci = lambda b, d, m, k="f1_2022": (f'{bb.loc[(b, d, m, k), "difference"]:+.3f} (95% interval '
                                       f'{bb.loc[(b, d, m, k), "ci_low"]:+.3f} to {bb.loc[(b, d, m, k), "ci_high"]:+.3f})')
    best = {d: max(TRAINED, key=lambda m: s(d, m)) for d in DEFINITION_NAMES}
    card = lambda d: (f'<div class="card"><h3>Best regularised model, {DEFINITION_NAMES[d]}</h3><div class="value">'
                      f'{s(d, best[d]):.3f}</div><p>{SERIES[best[d]][0]}. Against Experiment 22\'s MLP: {ci(base, d, best[d])}. '
                      f'Against Experiment 21\'s CatBoost: {ci(ref, d, best[d])}.</p></div>')
    ap = lambda m: summary.loc[("res7_exact", "f1_2022", m), "average_precision"]
    tn = lambda m, k="f1_2022": totals.loc[(m, k)]
    images = (embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre"),
              embedded_image(FIGURES / "report_curves.png", "Held-out loss per epoch"))
    conclusions = ""

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Regularised networks</title>
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
<header><div class="eyebrow">Experiment 23 · Regularised networks</div>
<h1>Do the networks improve with a lower learning rate and strong regularisation?</h1>
<p class="purpose"><strong>Purpose:</strong> in Experiment 22 the MLP and the MLP + GRU did best after a single epoch, then overfit the training seasons. This experiment retrains both with a ten times lower learning rate, more dropout and strong weight decay, to see whether slower, more constrained training gets more out of the same inputs.</p>
<p class="muted">{len(regions)} Fire Centres · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Only the training settings change.</strong> Label, training draw, inputs, input scaling, layer sizes, thresholds and scoring are Experiment 22's. Experiment 21's CatBoost and Experiment 22's MLP are loaded, not retrained, and reproduce their earlier numbers exactly.</li>
<li>One seed per model, as in Experiment 22. The epoch is still chosen on held-out weeks of the training seasons (one 7-day block in six), never on 2022 or 2023.</li></ul>
{settings_table}{fit_table}
<figure class="narrow">{images[1]}<figcaption>Held-out log loss per epoch. Dashed: Experiment 22's settings. Solid: regularised. The saved model is the epoch with the lowest held-out loss.</figcaption></figure></section>

<section><h2>Results (2023)</h2>
<div class="cards"><div class="card"><h3>References, BC-wide</h3><div class="value">{s("res7_exact", base):.3f}</div><p>Experiment 22 MLP, exact cell ({s("res7_margin1", base):.3f} with the margin). Experiment 21 CatBoost: {s("res7_exact", ref):.3f} / {s("res7_margin1", ref):.3f}. Two-satellite rule: {s("res7_exact", rule):.3f} / {s("res7_margin1", rule):.3f}.</p></div>
{card("res7_exact")}{card("res7_margin1")}</div>
{summary_table}
<p class="muted">BC-wide F1 pools all six Fire Centres. Precision and recall are BC-wide, exact cell. Average precision (AP) is the median over centres and needs no threshold. Experiment 22's MLP + GRU (not rescored here) had BC-wide F1 {gru_17["res7_exact"]:.3f} exact and {gru_17["res7_margin1"]:.3f} with the margin.</p>
<figure>{images[0]}<figcaption>2023 F1 by Fire Centre with 2022-tuned thresholds. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Week-block bootstrap against Experiment 22's MLP, BC-wide</h3>{boot_table(base)}
<h3>Week-block bootstrap against Experiment 21's CatBoost, BC-wide</h3>{boot_table(ref)}
<h3>Week-block bootstrap against the two-satellite rule, BC-wide</h3>{boot_table(rule)}
<h3>Which errors change (exact cell, BC-wide, 2023)</h3>
<p>The two-satellite rule catches all {int(t0.continuing_positives):,} cells still burning, none of the {int(t0.new_positives):,} new fire cells, and flags all {int(t0.stopped):,} cells where the fire stopped.</p>
{split_table}
<details><summary>Show F1 for every centre</summary>{per_centre}</details>
<details><summary>Show 2022 (threshold year) medians</summary>{selection_table}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2>{conclusions}</div>
<ul class="muted"><li>One setting of learning rate, dropout and weight decay, not a search; one seed per model.</li>
<li>Training still uses the stopgap label: 2018–2020 lack NOAA-20 in both inputs and labels.</li>
<li>2023 has been used to compare many experiments; only 2024 will give an untouched number.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 23 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
