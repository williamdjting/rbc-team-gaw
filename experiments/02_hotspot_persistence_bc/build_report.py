"""Create an offline HTML report after all six regional runs have completed."""
import base64
from html import escape
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_class_balance import add_class_balance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def image(path, description):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{escape(description)}" loading="lazy">'


def main():
    results = HERE / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    assert manifest["all_six_complete"], "Finish all regions before generating the report"
    selected = pd.read_csv(results / "selected_rules_2023.csv").sort_values("region")
    metrics = pd.read_csv(results / "all_region_metrics.csv")
    checks = pd.read_csv(results / "all_region_checks.csv")
    audit = pd.read_csv(results / "boundary_membership_audit.csv")
    assert len(selected) == 6 and selected.region.nunique() == 6
    assert len(metrics) == 6*6*7 and len(checks) == 6*6
    assert selected.rule.eq("local_1d").all()
    macro = selected[["precision", "recall", "f1"]].mean()
    for year in range(2018, 2024):
        assert checks.loc[checks.year.eq(year), "detections"].sum() == audit.loc[audit.year.eq(year), "regional_membership_count"].item()
    for region in selected.region:
        validation = metrics.loc[metrics.region.eq(region) & metrics.year.eq(2022) & metrics.rule.ne("always_no_detection")]
        winner = validation.sort_values(["f1", "rule"], ascending=[False, True]).iloc[0]
        assert winner.rule == selected.loc[selected.region.eq(region), "rule"].item()
    figures = HERE / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    y = np.arange(6)
    for offset, column, label, colour in ((-.18, "precision", "Precision", "#bc571c"),
                                          (.18, "recall", "Recall", "#237e85")):
        axes[0].barh(y+offset, selected[column]*100, height=.34, label=label, color=colour)
        for pos, value in zip(y+offset, selected[column]):
            axes[0].text(value*100+1, pos, f"{value:.1%}", va="center", fontsize=9)
    axes[0].set(yticks=y, yticklabels=selected.region, xlim=(0, 75), xlabel="Percent", title="2023 precision and recall")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper right")
    axes[1].barh(y, selected.f1, height=.55, color="#385b66")
    for pos, value in zip(y, selected.f1):
        axes[1].text(value+.01, pos, f"{value:.3f}", va="center")
    axes[1].set(yticks=y, yticklabels=selected.region, xlim=(0, .75), xlabel="F1", title="Selected rule: previous-day activity in the same cell")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", linestyle=":", alpha=.35)
    fig.savefig(figures / "report_regional_metrics.png", dpi=160)
    plt.close(fig)

    spatial = []
    for row in selected.itertuples():
        folder = HERE / "regions" / row.region.lower().replace(" ", "_") / "figures"
        plots = {(year, kind): image(folder / f"{kind}_{year}.png", f"{row.region}, {year}: {kind} recorded hotspot observations")
                 for year in (2018, 2023) for kind in ("monthly", "weekly")}
        spatial.append(f'''<details class="region" {'open' if row.region == 'Kamloops' else ''}>
<summary>{escape(row.region)} <span>2023 F1 {row.f1:.3f}</span></summary>
<div class="pair"><figure>{plots[(2018, 'weekly')]}<figcaption>2018 · August periods</figcaption></figure>
<figure>{plots[(2023, 'weekly')]}<figcaption>2023 · August periods</figcaption></figure></div>
<details class="monthly"><summary>Monthly context: June–September</summary><div class="pair">
<figure>{plots[(2018, 'monthly')]}<figcaption>2018 monthly observations</figcaption></figure>
<figure>{plots[(2023, 'monthly')]}<figcaption>2023 monthly observations</figcaption></figure></div></details></details>''')
    rows = "".join(f'<tr><th scope="row">{escape(r.region)}</th><td>{r.precision:.1%}</td><td>{r.recall:.1%}</td><td>{r.f1:.3f}</td><td>{int(r.positive_cell_days):,}</td></tr>' for r in selected.itertuples())
    positives_2022 = metrics.loc[metrics.year.eq(2022) & metrics.rule.eq("local_1d")].set_index("region").positive_cell_days
    low, high = selected.loc[selected.f1.idxmin()], selected.loc[selected.f1.idxmax()]
    comparison = image(figures / "report_regional_metrics.png", "Six-region precision, recall and F1 for the 2022-selected persistence rule, evaluated on 2023")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>BC hotspot persistence | Experiment 02</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f5;color:#20343a;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:40px 26px}}header{{padding-bottom:28px;border-bottom:2px solid #20343a}}
.eyebrow{{color:#b4521d;text-transform:uppercase;font-size:12px;letter-spacing:2px;font-weight:750}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 14px}}h3{{font-size:18px;margin:0 0 8px}}
.purpose{{font-size:21px;max-width:1000px}}p{{margin:8px 0}}section{{padding:30px 0;border-bottom:1px solid #d9e2df}}
ul{{padding-left:22px}}li{{margin:5px 0}}.muted,figcaption{{font-size:14px;color:#52676b}}
.pair,.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.cards{{grid-template-columns:repeat(3,1fr);margin:20px 0}}
.card,figure,.step{{background:white;border:1px solid #dae3df;border-radius:12px;padding:18px}}figure{{margin:0;padding:10px}}img{{width:100%;height:auto;display:block}}
.number{{font-size:38px;font-weight:750;color:#237e85}}figcaption{{padding:8px}}details{{margin:14px 0}}summary{{cursor:pointer;padding:14px 18px;background:#e8efeb;border-radius:9px;font-weight:700}}
summary span{{float:right;font-size:14px;font-weight:500}}details .pair{{margin-top:14px}}.monthly{{margin:12px 18px 20px}}.monthly summary{{background:white;border:1px solid #dae3df}}
.timeline{{display:grid;grid-template-columns:1fr 40px 1fr;gap:12px;align-items:center}}.arrow{{text-align:center;font-size:30px}}
.rule,.conclusion{{background:#e8efeb;border-radius:12px;padding:22px;margin:18px 0}}.rule{{border-left:4px solid #237e85;font-size:21px}}
table{{border-collapse:collapse;width:100%;font-size:15px;margin-top:18px}}th,td{{padding:10px 14px;border-bottom:1px solid #d9e2df;text-align:right}}th:first-child{{text-align:left}}thead{{background:#e8efeb}}
footer{{font-size:13px;color:#52676b;padding:20px 0}}.scroll{{overflow-x:auto}}@media(max-width:800px){{main{{padding:24px 14px}}.pair,.cards{{grid-template-columns:1fr}}summary span{{float:none;display:block}}}}
@media print{{body{{background:white}}main{{padding:0}}figure,.card{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 02 · Six BC Fire Centres · Completed</div>
<h1>Does hotspot persistence extend beyond Kamloops?</h1>
<p class="purpose"><strong>Purpose:</strong> repeat the same experiment across BC’s six Fire Centres to test whether recent recorded hotspot activity predicts next-day detections.</p>
<p class="muted">Cariboo · Coastal · Kamloops · Northwest · Prince George · Southeast<br>Regional Fire Centre boundaries, not municipal city limits. H3 resolution 7; 2024 excluded.</p></header>

<section><h2>01 / Spatial context: 2018 and 2023</h2>
<ul><li>Inspect monthly maps, then August periods, to see where recorded hotspot clusters recur or change.</li>
<li>Recurring locations suggest persistence. The temporal experiment below measures it; maps do not establish ignition or fire identity.</li></ul>
{''.join(spatial)}
<p class="muted">Open a region to inspect its plots. August 22–31 spans 10 days; other periods span 7. Colour scales are shared within each figure, but differ across figures. Compare locations across figures, not colour darkness.</p></section>

<section><h2>02 / Same methodology; different regional data</h2>
<div class="timeline"><div class="step"><div class="eyebrow">Features</div><h3>Previous 1, 3 or 7 days</h3>Recorded detection counts in the cell, or the cell plus immediate in-region neighbours.</div>
<div class="arrow">→</div><div class="step"><div class="eyebrow">Target</div><h3>Any recorded detection next day?</h3><strong>1:</strong> at least one observation<br><strong>0:</strong> none recorded in the archive</div></div>
<div class="rule"><strong>Fixed rule:</strong> a positive past-window count → predict a detection on the target day.</div>
<ul><li><strong>Example:</strong> predict August 15 using August 14, August 12–14, or August 8–14. Target-day observations never enter features.</li>
<li><strong>Same setup:</strong> fixed resolution-7 grids; full cell-date tables; May 8–September 30 targets; seven-day warmup; unknown coverage retained.</li>
<li><strong>Evaluation:</strong> 2018–2021 diagnostics → select each region’s highest-F1 rule on 2022 → evaluate on 2023.</li>
<li>No fitted classifier or GNN yet. The rule predicts recorded activity, including ongoing burning.</li></ul></section>

<section><h2>03 / Results: previous-day activity wins in all six regions</h2>
<h3>2023 average evaluation metrics</h3>
<div class="cards"><div class="card"><h3>Average precision</h3><div class="number">{macro.precision:.1%}</div></div>
<div class="card"><h3>Average recall</h3><div class="number">{macro.recall:.1%}</div></div>
<div class="card"><h3>Average F1</h3><div class="number">{macro.f1:.3f}</div></div></div>
<p class="muted">Macro averages: arithmetic means of the six regional metrics, with equal weight per Fire Centre. Average F1 is the mean of regional F1 scores. These are not pooled BC-wide cell-day metrics.</p>
<div class="cards"><div class="card"><h3>Selected rule</h3><div class="number">6 / 6</div><p>Regions selected yesterday’s activity in the same cell using 2022 F1.</p></div>
<div class="card"><h3>2023 precision range</h3><div class="number">{selected.precision.min():.1%}–{selected.precision.max():.1%}</div><p>Share of flagged cell-days with a recorded detection.</p></div>
<div class="card"><h3>2023 recall range</h3><div class="number">{selected.recall.min():.1%}–{selected.recall.max():.1%}</div><p>Share of recorded-positive cell-days correctly flagged.</p></div></div>
<figure>{comparison}<figcaption>Each regional rule was selected on 2022 before evaluation on 2023. Precision and recall are similar because the rule carries yesterday’s positive cells forward one day.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>Fire Centre</th><th>Precision</th><th>Recall</th><th>F1</th><th>2023 positive cell-days</th></tr></thead><tbody>{rows}</tbody><tfoot><tr><th scope="row">Equal-region average</th><td>{macro.precision:.1%}</td><td>{macro.recall:.1%}</td><td>{macro.f1:.3f}</td><td>Not pooled</td></tr></tfoot></table></div>
<ul><li><strong>Range:</strong> F1 {low.f1:.3f} in {escape(low.region)} to {high.f1:.3f} in {escape(high.region)}; persistence strength varies by region.</li>
<li><strong>Unequal evidence:</strong> Cariboo’s 2022 selection period had only {int(positives_2022['Cariboo']):,} positive cell-days. Its selected rule is based on a small sample.</li>
<li>Metrics are per region; overlapping border cells are not pooled into a unique BC-wide score.</li></ul></section>

<section><div class="conclusion"><h2>04 / Conclusion</h2><ul>
<li><strong>Day-to-day persistence is useful across all six regions, not only Kamloops.</strong></li>
<li>The simple rule correctly flags roughly half of next-day recorded-positive cell-days, while missing many others and producing false positives.</li>
<li>This is the benchmark for learned cell-only, neighbour-feature and graph models under the same chronological evaluation.</li></ul></div>
<ul class="muted"><li>These are recorded hotspot observations, not new ignitions, unique fires, severity or verified fire absence. Observation coverage remains unknown.</li>
<li>2023 is development evaluation; 2024 remains excluded. Previous-day records are assumed available; operational reporting delay is not modeled.</li>
<li>Checks passed: grid coverage, H3 assignment, past-only features, source-count reconciliation and exact reproduction of the Kamloops metrics and sample features.</li>
<li>Cached boundary audit: {int(audit.unassigned_to_cached_centres.sum()):,} detections outside all centre polygons; {int(audit.multiple_centre_matches.sum()):,} match multiple centres. Exceptions are saved; regional filters remain unchanged.</li></ul></section>
<footer>Built from the completed experiment outputs. All 25 figures are embedded for offline viewing. Detailed metrics remain in results/all_region_metrics.csv.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(add_class_balance(html, HERE), encoding="utf-8")
    (results / "report_manifest.json").write_text(json.dumps({
        "report": "RESULTS.html", "all_regions_complete": True, "embedded_figures": 25,
        "metrics_source": "results/selected_rules_2023.csv", "source_count_reconciliation_passed": True,
        "rule_selection_verified": True, "macro_2023": macro.to_dict(),
        "averaging": "Equal-weight arithmetic mean across six Fire Centres"}, indent=2), encoding="utf-8")
    print(f"Created {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
