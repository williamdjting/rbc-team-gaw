"""How many fires in cells that were not burning on day 0 do the models and rules catch?

Rescores 2023 with the saved per-lead CatBoost models at their 2022 thresholds (exact-cell
definition) and the best persistence rule chosen on 2022. Each 2023 positive (detection on
day e + L) is put in one group by the cell's state on day e:
- burning on day 0: a detection in the cell on day e;
- nearby activity: none in the cell on day e, but some in the cell or its in-centre
  neighbours during the 7 days ending on day e;
- quiet: nothing in the cell or its neighbours for those 7 days.
Reports the share of each group caught (the cell itself flagged), and how many of the
model's flags fall on cells that were not burning on day 0 and how many of those are right.
Writes results/quiet_cell_catch_2023.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment_10", HERE / "run_experiment.py")
experiment_10 = sys.modules["experiment_10"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_10)
from experiment_10 import (LEADS, RESULTS, RULES, EVALUATION_YEAR, SELECTION_YEAR, Context, REFERENCE,  # noqa: E402
                           catboost_model, lead_targets, load, predict, target_grid)


def main():
    regions = sorted(pd.read_csv(REFERENCE / "results/selected_rules_2023.csv").region)
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    exact = metrics.loc[metrics.definition.eq("res7_exact")]
    selection = exact.loc[exact.year.eq(SELECTION_YEAR) & exact.kind.eq("rule")]
    best_rule = selection.loc[selection.groupby(["region", "lead_days"]).f1.idxmax()].set_index(["region", "lead_days"]).name
    threshold = exact.loc[exact.year.eq(SELECTION_YEAR) & exact.kind.eq("catboost")].set_index(["region", "lead_days"]).threshold
    models = {}
    for lead in LEADS:
        models[lead] = catboost_model()
        models[lead].load_model(str(RESULTS / "models" / f"catboost__lead{lead}.cbm"))
    context = Context(regions)
    rows = []
    for region in regions:
        cells = context.grids[region]["cells"]
        frame = context.features(region, EVALUATION_YEAR, load(region, EVALUATION_YEAR, with_cells=True))
        targets = lead_targets(frame, cells, target_grid(frame, cells))
        burning = frame.local_count_1d.to_numpy() > 0
        nearby = ~burning & ((frame.local_count_7d.to_numpy() + frame.neighbour_count_7d.to_numpy()) > 0)
        quiet = ~burning & ~nearby
        for lead in LEADS:
            valid = targets[lead] >= 0
            positive = valid & (targets[lead] == 1)
            flags = {"catboost": predict(models[lead], frame, "all") >= threshold.loc[(region, lead)],
                     "best_rule": frame[RULES[best_rule.loc[(region, lead)]]].sum(axis=1).to_numpy() > 0,
                     "fire_yesterday_rule": burning}
            for name, flag in flags.items():
                flag = flag & valid
                row = {"region": region, "lead_days": lead, "flagger": name,
                       "rule_used": best_rule.loc[(region, lead)] if name == "best_rule" else "",
                       "flags": int(flag.sum()), "flags_on_cells_not_burning": int((flag & ~burning).sum()),
                       "correct_flags_on_cells_not_burning": int((flag & ~burning & positive).sum())}
                for group, mask in (("burning", burning), ("nearby", nearby), ("quiet", quiet)):
                    row[f"positives_{group}"] = int((positive & mask).sum())
                    row[f"caught_{group}"] = int((positive & mask & flag).sum())
                rows.append(row)
        print(f"{region}: done", flush=True)
        del frame
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "quiet_cell_catch_2023.csv", index=False)
    totals = table.groupby(["lead_days", "flagger"]).sum(numeric_only=True)
    for group in ("burning", "nearby", "quiet"):
        totals[f"recall_{group}"] = totals[f"caught_{group}"] / totals[f"positives_{group}"]
    totals["precision_on_cells_not_burning"] = totals.correct_flags_on_cells_not_burning / totals.flags_on_cells_not_burning.clip(lower=1)
    pd.set_option("display.width", 250)
    print(totals[["positives_burning", "positives_nearby", "positives_quiet", "recall_burning", "recall_nearby",
                  "recall_quiet", "flags", "flags_on_cells_not_burning", "correct_flags_on_cells_not_burning",
                  "precision_on_cells_not_burning"]].round(3).to_string())


if __name__ == "__main__":
    main()
