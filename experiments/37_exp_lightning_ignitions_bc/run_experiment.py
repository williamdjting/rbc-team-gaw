"""Experiment 36 with the label restricted to lightning-caused (natural) wildfires of 10 ha or more.

Person-caused, unknown-cause and smaller fires are dropped from the label only: a cell-day is
positive if a BCWS wildfire with FIRE_CAUSE "Lightning" and a recorded size (CURRENT_SIZE, the
final size for past seasons) of at least 10 ha starts in it on any of days t .. t+6. Size is used
only to choose which fires the label counts; it is never an input. The inputs are
unchanged, so Experiment 13's BCWS inputs still see every known active fire, whatever its cause.
Everything else (areas, models, references, training draw, thresholds, scoring, bootstrap) is
Experiment 36's, run through its own main() with its label function replaced. 2024 is not loaded.

    python run_experiment.py                      # full run
    python run_experiment.py --regions Southeast  # smoke test
    python run_experiment.py --resume             # load saved models, skip finished region-years
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
spec = importlib.util.spec_from_file_location("experiment_22", EXPERIMENTS / "36_exp_bcws_ignitions_bc/run_experiment.py")
experiment_22 = sys.modules["experiment_22"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_22)

CAUSE = "Lightning"
MIN_HA = 10.0
_ignition_grid = experiment_22.ignition_grid


def lightning_grid(incidents, cells, year):
    return _ignition_grid(incidents.loc[incidents.FIRE_CAUSE.eq(CAUSE) & incidents.CURRENT_SIZE.ge(MIN_HA)], cells, year)


def main():
    experiment_22.ignition_grid = lightning_grid
    experiment_22.HERE = HERE  # results/ default to this folder
    experiment_22.main()
    results = Path(sys.argv[sys.argv.index("--results") + 1]) if "--results" in sys.argv else HERE / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    manifest["label"] = manifest["label"].replace("BCWS incident of type Fire", f"BCWS incident of type Fire with FIRE_CAUSE {CAUSE} "
                                                  f"and CURRENT_SIZE >= {MIN_HA:g} ha")
    manifest["label_cause"], manifest["label_min_ha"] = CAUSE, MIN_HA
    manifest["inputs_note"] = "inputs unchanged: BCWS inputs still see active fires of every cause"
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
