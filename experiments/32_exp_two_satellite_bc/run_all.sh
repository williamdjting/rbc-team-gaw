#!/usr/bin/env bash
# Exp 32: rerun Exp 30 (ERA5), Exp 31 (spread features) and the stacked / side-by-side models on the two-satellite table.
# Run from the repo root after build_pair_table.py. Outputs: experiments/32_exp_two_satellite_bc/exp30, exp31.
set -e
PY=.venv/forecast/Scripts/python.exe
export GAW_TABLE=pair
$PY -I experiments/30_exp_era5_features_bc/run_experiment.py . all
$PY -I experiments/31_exp_spread_features_bc/run_experiment.py . all
$PY -I experiments/31_exp_spread_features_bc/stack.py .
echo ALL DONE
