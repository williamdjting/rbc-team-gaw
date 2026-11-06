"""Build a standalone visual HTML report from saved Experiment 19 outputs."""
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
LABELS = {"snpp_firms": "S-NPP, FIRMS (current label)",
          "snpp_cwfis": "S-NPP, CWFIS copy",
          "viirs_pair": "S-NPP + NOAA-20",
          "cwfis_viirs": "S-NPP + NOAA-20, CWFIS",
          "viirs_modis": "S-NPP + NOAA-20 + MODIS",
          "snpp_firms_filled": "S-NPP, gap-filled",
          "viirs_pair_filled": "S-NPP + NOAA-20, gap-filled",
          "cwfis_viirs_filled": "S-NPP + NOAA-20, CWFIS, gap-filled",
          "viirs_modis_filled": "S-NPP + NOAA-20 + MODIS, gap-filled"}
# Reference categorical slots 1-3 for the three years with complete NOAA-20 coverage.
YEARS = {2021: ("#2a78d6", "o"), 2022: ("#eb6834", "^"), 2023: ("#1baf7a", "s")}
FULL_YEARS = list(YEARS)


def rule_chart(bc):
    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)
    positions = np.arange(len(order))
    for ax, (column, title) in zip(axes, (("rule_f1_exact", "Exact cell"), ("rule_f1_margin", "One-cell margin"))):
        for (year, (colour, marker)), offset in zip(YEARS.items(), (-.22, 0, .22)):
            rows = bc.loc[bc.year.eq(year)].set_index("label").loc[order]
            ax.scatter(rows[column], positions + offset, s=52, marker=marker, color=colour, edgecolor="white",
                       linewidth=1.2, zorder=3)
        current = bc.loc[bc.label.eq("snpp_firms") & bc.year.isin(FULL_YEARS), column]
        ax.axvspan(current.min(), current.max(), color="#e1e0d9", zorder=0)
        ax.set(title=f"Previous-day rule F1, {title.lower()}", xlabel="F1 (BC-wide)")
        style(ax)
    axes[0].set(yticks=positions, yticklabels=[LABELS[k] for k in order])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=c, markeredgecolor="white", label=str(y))
               for y, (c, m) in YEARS.items()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#e1e0d9", label="Current label, 2021–2023 range"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(FIGURES / "report_rule_f1.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    table = pd.read_csv(RESULTS / "label_metrics.csv")
    alignment = pd.read_csv(RESULTS / "cwfis_firms_alignment.csv")
    bc = table.loc[table.region.eq("BC-wide")]
    rule_chart(bc)
    get = lambda year, label, column: float(bc.loc[bc.year.eq(year) & bc.label.eq(label), column].iloc[0])

    def year_table(year):
        rows = bc.loc[bc.year.eq(year)].set_index("label").loc[list(LABELS)].reset_index()
        rows["label"] = rows.label.map(LABELS)
        frame = rows[["label", "positives", "positives_vs_firms", "persistence", "gap_share", "quiet_share",
                      "rule_f1_exact", "rule_f1_margin", "inside_perimeter_share"]].rename(columns={
            "label": "Label", "positives": "Positive cell-days", "positives_vs_firms": "× current label",
            "persistence": "P(fire today | fire yesterday)", "gap_share": "Missed-day share", "quiet_share": "Quiet share",
            "rule_f1_exact": "Rule F1, exact", "rule_f1_margin": "Rule F1, margin",
            "inside_perimeter_share": "Inside BCWS perimeter"})
        return html_table(frame, {"Positive cell-days": "{:,}", "× current label": "{:.2f}",
                                  "P(fire today | fire yesterday)": "{:.3f}", "Missed-day share": "{:.1%}",
                                  "Quiet share": "{:.1%}", "Rule F1, exact": "{:.3f}", "Rule F1, margin": "{:.3f}",
                                  "Inside BCWS perimeter": "{:.1%}"})

    per_region = table.loc[table.year.eq(2023) & table.label.isin(["snpp_firms", "viirs_pair", "viirs_pair_filled",
                                                                   "cwfis_viirs_filled"]) & table.region.ne("BC-wide")]
    region_table = per_region.pivot(index="region", columns="label", values="rule_f1_margin")[
        ["snpp_firms", "viirs_pair", "viirs_pair_filled", "cwfis_viirs_filled"]].rename(columns=LABELS).reset_index().rename(
        columns={"region": "Fire Centre"})
    region_html = html_table(region_table, {c: "{:.3f}" for c in region_table.columns if c != "Fire Centre"})
    align = alignment.loc[alignment.lag_days.eq(0)].assign(
        **{"S-NPP cell-days, CWFIS ÷ FIRMS": lambda f: [get(y, "snpp_cwfis", "positives_vs_firms") for y in f.year]})
    align_html = html_table(align[["year", "jaccard", "S-NPP cell-days, CWFIS ÷ FIRMS"]].rename(
        columns={"year": "Year", "jaccard": "Overlap (shared ÷ either)"}),
        {"Year": "{}", "Overlap (shared ÷ either)": "{:.3f}", "S-NPP cell-days, CWFIS ÷ FIRMS": "{:.2f}"})
    lag = alignment.pivot(index="year", columns="lag_days", values="jaccard")
    quiet_years = [y for y in (2019, 2020, 2022)]
    cwfis_inside = ", ".join(f"{y}: {get(y, 'snpp_cwfis', 'inside_perimeter_share'):.0%} vs {get(y, 'snpp_firms', 'inside_perimeter_share'):.0%}"
                             for y in quiet_years)
    image = embedded_image(FIGURES / "report_rule_f1.png", "Previous-day rule F1 by label")

    c, p, f = "snpp_firms", "viirs_pair", "viirs_pair_filled"
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Multi-satellite labels</title>
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
<header><div class="eyebrow">Experiment 19 · Multi-satellite labels</div>
<h1>Can more satellites make the next-day fire label more consistent?</h1>
<p class="purpose"><strong>Purpose:</strong> the current label is one satellite's detection (VIIRS on S-NPP). A burning cell often goes undetected for a day because of cloud, smoke or the overpass, which caps every forecast. This experiment builds labels from more satellites and from filling one-day gaps, and measures how consistent each one is.</p>
<p class="muted">Six Fire Centres · target days May 8 – September 30 · NOAA-20 complete 2021–2023 · 2024 excluded</p></header>

<section><h2>Data</h2>
<ul><li><strong>CWFIS hotspot archive</strong> (Natural Resources Canada, open): VIIRS on S-NPP and NOAA-20, MODIS on Terra and Aqua, and AVHRR, merged for North America. Kept: BC, May–September 2018–2023, assigned to H3 res-7 cells. NOAA-20 is complete from 2021, partial in 2019 (from July) and 2020, and absent in 2018. AVHRR (2018–2020 only) and VIIRS M-band are left out so the sensor mix stays stable.</li>
<li><strong>Gap filling:</strong> a cell-day also counts as positive if the cell is positive the day before and the day after. This uses the next day, so it is for labels only. The rule and the quiet test always use the unfilled label, which is all a forecast can see.</li>
<li><strong>Measures:</strong> P(fire today | fire yesterday); the missed-day share (a cell positive within 3 days before and after, but not that day); the share of positives in quiet cells (nothing in the cell or its neighbours for 7 days); the previous-day rule's F1; and the share of positives inside, or within 2 km of, a final BCWS fire perimeter of that year.</li>
<li><strong>Alignment check:</strong> CWFIS's own copy of S-NPP matches FIRMS best with no date shift (overlap {lag[0].min():.2f}–{lag[0].max():.2f} at lag 0 vs at most {max(lag[-1].max(), lag[1].max()):.2f} at ±1 day), so dates line up.</li></ul></section>

<section><h2>Results</h2>
<div class="cards"><div class="card"><h3>Missed-day share, 2023</h3><div class="value">{get(2023, c, "gap_share"):.0%} → {get(2023, f, "gap_share"):.0%}</div><p>Current label → S-NPP + NOAA-20, gap-filled.</p></div>
<div class="card"><h3>Rule F1, exact cell, 2023</h3><div class="value">{get(2023, c, "rule_f1_exact"):.3f} → {get(2023, f, "rule_f1_exact"):.3f}</div><p>NOAA-20 alone: {get(2023, p, "rule_f1_exact"):.3f}. In 2022: {get(2022, c, "rule_f1_exact"):.3f} → {get(2022, f, "rule_f1_exact"):.3f}.</p></div>
<div class="card"><h3>Rule F1, one-cell margin, 2023</h3><div class="value">{get(2023, c, "rule_f1_margin"):.3f} → {get(2023, f, "rule_f1_margin"):.3f}</div><p>NOAA-20 alone: {get(2023, p, "rule_f1_margin"):.3f}. In 2022: {get(2022, c, "rule_f1_margin"):.3f} → {get(2022, f, "rule_f1_margin"):.3f}.</p></div></div>
<figure>{image}<figcaption>BC-wide previous-day rule F1 for each candidate label, in the three years with complete NOAA-20 coverage. The shaded band is the current label's range.</figcaption></figure>
<h3>2023 (evaluation year)</h3>{year_table(2023)}
<h3>2022 (threshold year)</h3>{year_table(2022)}
<details><summary>Show 2021</summary>{year_table(2021)}</details>
<details><summary>Show rule F1 by Fire Centre, 2023, one-cell margin</summary>{region_html}</details>
<details><summary>Show CWFIS vs FIRMS S-NPP agreement</summary>{align_html}
<p class="muted">In quiet years CWFIS keeps fewer S-NPP detections than FIRMS, and more of them fall inside BCWS perimeters ({cwfis_inside}). CWFIS appears to drop some industrial and other non-wildfire hotspots.</p></details></section>

<section><div class="conclusion"><h2>Conclusion</h2><ul>
<li><strong>Adding NOAA-20 makes the label clearly more consistent.</strong> In 2023 it adds {get(2023, p, "positives_vs_firms") - 1:.0%} more positive cell-days, nearly all inside BCWS perimeters ({get(2023, p, "inside_perimeter_share"):.1%}), and raises P(fire today | fire yesterday) from {get(2023, c, "persistence"):.3f} to {get(2023, p, "persistence"):.3f}.</li>
<li><strong>Gap filling adds the most.</strong> With both, the missed-day share falls from {get(2023, c, "gap_share"):.0%} to {get(2023, f, "gap_share"):.0%}. The rule's F1 on the exact cell rises from {get(2023, c, "rule_f1_exact"):.3f} to {get(2023, f, "rule_f1_exact"):.3f} in 2023, and from {get(2022, c, "rule_f1_exact"):.3f} to {get(2022, f, "rule_f1_exact"):.3f} in 2022.</li>
<li><strong>MODIS does not help.</strong> Its 1 km pixels spread detections over neighbouring cells: it adds positives but not consistency, and slightly fewer fall inside perimeters.</li>
<li><strong>CWFIS's copy cleans quiet years.</strong> Using CWFIS for both VIIRS satellites gives the cleanest label in 2022 ({get(2022, "cwfis_viirs_filled", "inside_perimeter_share"):.1%} inside perimeters vs {get(2022, f, "inside_perimeter_share"):.1%}) at the cost of {1 - get(2022, "cwfis_viirs_filled", "firms_snpp_recovered"):.0%} of FIRMS S-NPP positives.</li>
<li><strong>Correction:</strong> an earlier estimate of the rule on gap-filled labels (0.815, one-cell margin, 2023) let the rule see the filled day before, which uses the target day. Scored correctly, it is {get(2023, "snpp_firms_filled", "rule_f1_margin"):.3f}.</li>
<li><strong>Next:</strong> retrain the forecast on the S-NPP + NOAA-20, gap-filled label and test whether the model now beats the rule. Training years need NOAA-20 too: CWFIS has it only from 2021 (partial 2019–2020), so the FIRMS NOAA-20 request (2018–2020) completes the training label.</li></ul></div>
<ul class="muted"><li>A higher rule F1 means a more consistent, more predictable label. It does not by itself mean a better forecast of the current label; both are measured against their own label.</li>
<li>"Inside BCWS perimeter" uses final perimeters of the same year with a 2 km margin and no date window.</li>
<li>Neighbourhoods stop at each Fire Centre's grid edge. 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 19 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
