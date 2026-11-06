"""Build a standalone visual HTML report from saved Experiment 17 outputs."""
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
# Reference categorical slots 1-3 in fixed order; gray is reserved for rules.
SERIES = {"rule_yesterday": ("Fire yesterday (rule)", RULE, "D"),
          "rule_yesterday_bcws": ("Fire yesterday and a known BCWS fire within 10 km (rule)", "#b5b3aa", "d"),
          "catboost_08": ("CatBoost, Experiment 13 features", "#2a78d6", "o"),
          "catboost_history": ("CatBoost, + history features", "#eb6834", "^"),
          "catboost_history_wildfire": ("CatBoost, + history, trained on wildfire target", "#1baf7a", "s")}
TARGET_NAMES = {"recorded": "Recorded detections (current target)", "wildfire": "Wildfire-linked detections"}
DEFINITION_NAMES = {"res7_margin1": "one-cell margin", "res7_exact": "exact cell"}
GROUP_NAMES = {"new BCWS wildfire": "New BCWS wildfire", "older BCWS wildfire, not recently detected": "Older wildfire flaring up",
               "no BCWS record": "No BCWS fire on record", "other BCWS record": "Other BCWS record",
               "not an Experiment 16 onset": "Other quiet positives"}


def f1_chart(evaluation, order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    for ax, target in zip(axes, TARGET_NAMES):
        part = evaluation.loc[evaluation.target.eq(target) & evaluation.definition.eq("res7_margin1")]
        offsets = np.linspace(-.3, .3, len(SERIES))
        for (name, (label, colour, marker)), offset in zip(SERIES.items(), offsets):
            rows = part.loc[part.model.eq(name)].set_index("region")
            ax.scatter(rows.loc[order, "f1"], positions + offset, s=48, marker=marker, color=colour,
                       edgecolor="white", linewidth=1.2, zorder=3)
        ax.axvspan(.7, .8, color="#e1e0d9", zorder=0)
        ax.set(title=f"{TARGET_NAMES[target]}, one-cell margin", xlabel="2023 F1 (2022 threshold)")
        style(ax)
    axes[0].set(yticks=positions, yticklabels=order)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=c, markeredgecolor="white", label=n)
               for n, c, m in SERIES.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#e1e0d9", label="F1 goal 0.70–0.80"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_f1.png", dpi=150)
    plt.close(fig)


def catch_chart(catches):
    totals = catches.groupby(["onset_group", "model"])[["quiet_fires", "caught"]].sum().reset_index()
    totals["share"] = totals.caught / totals.quiet_fires
    groups = [g for g in GROUP_NAMES if g in set(totals.onset_group)]
    models = [m for m in SERIES if not m.startswith("rule")]
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    width = .8 / len(models)
    for i, model in enumerate(models):
        rows = totals.loc[totals.model.eq(model)].set_index("onset_group").reindex(groups)
        x = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * width
        ax.bar(x, rows.share, width=width * .92, color=SERIES[model][1], label=SERIES[model][0])
        for xi, value in zip(x, rows.share):
            ax.annotate(f"{value:.0%}", (xi, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    counts = totals.drop_duplicates("onset_group").set_index("onset_group").quiet_fires.reindex(groups)
    ax.set(xticks=np.arange(len(groups)), xticklabels=[f"{GROUP_NAMES[g]}\n({counts[g]:,} fires)" for g in groups],
           ylabel="Share caught (exact cell)", title="2023 quiet fires caught, by Experiment 16 onset group (rules catch none)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    fig.savefig(FIGURES / "report_quiet_catch.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    catches = pd.read_csv(RESULTS / "quiet_catch_2023.csv")
    sample = pd.read_csv(RESULTS / "training_sample.csv")
    importance = pd.read_csv(RESULTS / "feature_importance__catboost_history.csv")
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]
    pick = lambda target, definition, model: evaluation.loc[evaluation.target.eq(target) & evaluation.definition.eq(definition)
                                                            & evaluation.model.eq(model)].set_index("region")
    order = pick("recorded", "res7_margin1", "catboost_history").f1.sort_values(ascending=False).index.tolist()
    f1_chart(evaluation, order)
    catch_chart(catches)

    def summary(target):
        rows = []
        for model in SERIES:
            if target == "recorded" and model == "catboost_history_wildfire":
                continue
            row = {"Model": SERIES[model][0]}
            for definition, label in DEFINITION_NAMES.items():
                part = pick(target, definition, model)
                row[f"F1, {label} (median)"] = part.f1.median()
                row[f"Centres at 0.70+ ({label})"] = f"{int((part.f1 >= .7).sum())} of 6"
            ap = pick(target, "res7_exact", model).get("average_precision")
            row["AP, exact cell (median)"] = ap.median() if ap is not None else np.nan
            rows.append(row)
        return html_table(pd.DataFrame(rows), {c: "{:.3f}" for c in rows[0] if c.startswith(("F1", "AP"))})

    columns = {f"{TARGET_NAMES[t].split(' (')[0]}: {SERIES[m][0]}": pick(t, "res7_margin1", m).loc[order, "f1"].to_numpy()
               for t in TARGET_NAMES for m in ("rule_yesterday", "catboost_08", "catboost_history", "catboost_history_wildfire")
               if not (t == "recorded" and m == "catboost_history_wildfire")}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, **columns}), {c: "{:.3f}" for c in columns})
    labels = evaluation.loc[evaluation.definition.eq("res7_exact") & evaluation.model.eq("rule_yesterday")].pivot(
        index="region", columns="target", values="positives")
    train_share = sample.wildfire_positives.sum() / sample.recorded_positives.sum()
    label_table = html_table(pd.DataFrame({
        "Fire Centre": order, "2023 recorded positives": labels.loc[order, "recorded"].to_numpy(),
        "2023 wildfire-linked": labels.loc[order, "wildfire"].to_numpy(),
        "Share wildfire-linked": (labels.wildfire / labels.recorded).loc[order].to_numpy()}),
        {"2023 recorded positives": "{:,}", "2023 wildfire-linked": "{:,}", "Share wildfire-linked": "{:.1%}"})
    history = set(manifest["history_features"])
    top = importance.sort_values("importance", ascending=False).head(12)
    importance_table = html_table(top.assign(group=top.feature.map(lambda f: "history" if f in history else "Experiment 13"))
                                  .rename(columns={"feature": "Input", "importance": "Importance (%)", "group": "Group"}),
                                  {"Importance (%)": "{:.1f}"})
    totals = catches.groupby(["onset_group", "model"])[["quiet_fires", "caught"]].sum()
    caught = lambda model, group: totals.loc[(group, model), "caught"] / totals.loc[(group, model), "quiet_fires"]
    older = "older BCWS wildfire, not recently detected"
    shares = (totals.caught / totals.quiet_fires).unstack("model")
    max_quiet = float(shares.loc[[g for g in shares.index if g in GROUP_NAMES and totals.loc[g].quiet_fires.iloc[0] >= 100],
                                 [m for m in shares.columns if not m.startswith("rule")]].max().max())
    gain_recorded = pick("recorded", "res7_margin1", "catboost_history").f1 - pick("recorded", "res7_margin1", "rule_yesterday").f1
    gain_wild = pick("wildfire", "res7_margin1", "catboost_history_wildfire").f1 - pick("wildfire", "res7_margin1", "rule_yesterday").f1
    history_gain = pick("recorded", "res7_margin1", "catboost_history").f1 - pick("recorded", "res7_margin1", "catboost_08").f1
    images = {"f1": embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre for both targets"),
              "catch": embedded_image(FIGURES / "report_quiet_catch.png", "Quiet fires caught by onset group")}

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | History features and non-wildfire flag</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:18px 0 8px}}
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
.wrap td{{white-space:normal;text-align:left}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 17 · History features and non-wildfire flag</div>
<h1>Does longer memory, and separating wildfires from other burning, help the model?</h1>
<p class="purpose"><strong>Purpose:</strong> Experiment 16 showed that most "quiet" fires are either burning with no BC Wildfire Service fire on record (recurring, small, often at night) or older wildfires flaring up beyond the 7-day, one-cell history window. This experiment adds 30-day, season and prior-year history, and a wildfire-linked target.</p>
<p class="muted">Six Fire Centres · next-day forecasts · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>History features ({len(manifest["history_features"])}):</strong> detections in the cell and its neighbours over 30 days; active cells within 5 rings over 30 days; distance to the nearest detection in 30 days; days with activity since May 1; recent days with and without a known BCWS fire nearby; and, from earlier seasons, how many years the cell and its neighbours burned and how many of those detection days were or were not wildfire-linked.</li>
<li><strong>Wildfire-linked target:</strong> a detection counts if it falls inside, or within 2 km of, the final perimeter of a BCWS fire active within 3 days, or within 10 km of an active BCWS fire's ignition point. Final perimeters are known only after the season, so they define labels and prior-year features only; current-season features use the ignition-point link with incidents already reported.</li>
<li><strong>Models:</strong> CatBoost with Experiment 13's features (the previous best), with the history features added, and the latter retrained on the wildfire target. All use Experiment 03's sampling and correction.</li></ul>
<p>{train_share:.0%} of 2018–2021 training positives are wildfire-linked. In 2023:</p>
{label_table}</section>

<section><h2>Results</h2>
<div class="cards"><div class="card"><h3>History features, recorded target</h3><div class="value">{history_gain.median():+.3f}</div><p>Median F1 change over Experiment 13 features (one-cell margin); higher in {int((history_gain > 0).sum())} of 6 centres.</p></div>
<div class="card"><h3>Model vs rule, recorded target</h3><div class="value">{gain_recorded.median():+.3f}</div><p>CatBoost with history minus "fire yesterday" (one-cell margin); model higher in {int((gain_recorded > 0).sum())} of 6.</p></div>
<div class="card"><h3>Model vs rule, wildfire target</h3><div class="value">{gain_wild.median():+.3f}</div><p>CatBoost trained on the wildfire target minus "fire yesterday"; model higher in {int((gain_wild > 0).sum())} of 6.</p></div></div>
<h3>{TARGET_NAMES["recorded"]}</h3>{summary("recorded")}
<h3>{TARGET_NAMES["wildfire"]}</h3>{summary("wildfire")}
<figure>{images["f1"]}<figcaption>2023 F1 with the one-cell margin. The shaded band is the 0.70–0.80 goal.</figcaption></figure>
<h3>Quiet fires: nothing in the cell or its neighbours for 7 days</h3>
<p>Rules never flag these cells. Share caught by each model at its 2022 threshold, by Experiment 16 group:</p>
<figure>{images["catch"]}<figcaption>At F1-optimal thresholds, almost no quiet fires are flagged by any model; groups with fewer than 100 fires are shown for completeness.</figcaption></figure>
<details><summary>Show the most important inputs (CatBoost with history, recorded target)</summary>{importance_table}</details>
<details><summary>Show F1 for every centre (one-cell margin)</summary>{per_centre}</details></section>

<section><h2>Data that would separate these groups further</h2>
<div class="table wrap"><table><thead><tr><th>Group</th><th>Data</th><th>What it adds</th><th>Access</th></tr></thead><tbody>
<tr><td>Not wildfire (agricultural, slash piles, industrial)</td><td>AAFC Annual Crop Inventory (30 m); land cover (ESA WorldCover 10 m or NRCan)</td><td>Cropland and pasture, where fall stubble and land-clearing burns happen</td><td>Free, open download</td></tr>
<tr><td></td><td>BC consolidated cutblocks and RESULTS openings (DataBC)</td><td>Recent harvest, where slash piles are burned in autumn</td><td>Free, open (same service as the BCWS data)</td></tr>
<tr><td></td><td>BC Energy Regulator facilities and wells; VIIRS Nightfire (Payne Institute)</td><td>Gas flares and plants; Nightfire also estimates source temperature, which separates flares from vegetation</td><td>Free; Nightfire needs registration</td></tr>
<tr><td></td><td>BCWS open-burning registrations (Category 2 and 3)</td><td>Where and when permitted burns took place</td><td>Request from BCWS; not openly published</td></tr>
<tr><td>Older wildfires flaring up</td><td>ERA5-Land wind, humidity, temperature and rain</td><td>Hot, dry, windy days restart smouldering fires</td><td>Free; needs a Copernicus key (script ready)</td></tr>
<tr><td></td><td>NOAA-20 VIIRS detections</td><td>Fewer days where a burning fire goes unseen</td><td>Requested from FIRMS</td></tr>
<tr><td>New wildfires</td><td>Lightning: GOES-West Geostationary Lightning Mapper (AWS), or ERA5 lightning density; CLDN on request</td><td>71% of new fires here are lightning-caused</td><td>GLM and ERA5 free; CLDN licensed</td></tr>
<tr><td></td><td>Fuel types (CFS FBP grid), elevation and slope (Copernicus GLO-30), roads and settlements (BC Digital Road Atlas)</td><td>Where fuels can carry fire and where people start fires</td><td>Free, open</td></tr>
<tr><td>Label checking</td><td>National Burned Area Composite (CWFIS); Sentinel-2 or Landsat burn scars</td><td>Confirms whether a detection burned vegetation</td><td>Free, open</td></tr>
</tbody></table></div></section>

<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>History features did not help.</strong> Median F1 changed by {history_gain.median():+.3f} over Experiment 13 features on the recorded target (one-cell margin), and exact-cell AP fell slightly. The best configuration is still Experiment 13's CatBoost.</li>
<li><strong>Quiet fires stay below the alert threshold.</strong> At thresholds chosen for F1, the models flag at most {max_quiet:.1%} of quiet fires in any group. History features raise quiet cells' scores (quiet positives average far more nearby 30-day activity than quiet negatives), but a quiet cell's chance of fire tomorrow stays well under the threshold that maximizes F1, so they are never flagged.</li>
<li><strong>Wildfire-only target:</strong> {train_share:.0%} of training positives are wildfire-linked, and in 2023 nearly all are, so the cleaner target changes little in 2023 ({gain_wild.median():+.3f} median F1 against the rule). It matters in quiet years and in the Peace region, where non-wildfire burning dominates.</li>
<li><strong>Filtering yesterday's detections by known BCWS fires hurts</strong> (median F1 well below the plain rule): many real fire detections are not yet within 10 km of a reported ignition point.</li>
<li>Next: land-use, cutblock, industrial-site, fuel and terrain layers (being built in <code>datasets/static_layers/</code>) give the model direct evidence about non-wildfire burning and ignition risk; quiet cells may also need their own model and operating point instead of one F1 threshold for all cells.</li></ul></div>
<ul class="muted"><li>The wildfire link depends on BCWS records; small wildfires never reported to BCWS are counted as non-wildfire.</li>
<li>30-day windows are truncated at May 1, since detections cover May–September only. Prior-year features start in 2018, so 2018 rows have none.</li>
<li>Untuned CatBoost, one seed. 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 17 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
