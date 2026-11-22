#!/usr/bin/env bash
# Label v1 vs v2: same pipeline for both (run from repo root after build_label_v2.py).
set -e
PY=.venv/forecast/Scripts/python.exe
$PY -I experiments/32_exp_two_satellite_bc/label_v2/run_label.py . v1
$PY -I experiments/32_exp_two_satellite_bc/label_v2/run_label.py . v2
echo BOTH DONE
