# Experiment 04a: Graph Neural Network with CEMS Fire Weather for BC

This experiment is the exact CEMS-enhanced counterpart to **Experiment 04 (`04_exp_gnn_bc`)**. It uses your joined `datasets/firms_with_cems_18_24.csv` dataset (plus `datasets/dataset_CEMS_18_24.csv` for quiet/0-detection days) to train the same 2-layer GraphSAGE-style GNN and no-message-passing MLP control on the `234,554`-node H3 graph across all six BC Fire Centres, adding the four shortlisted Copernicus CEMS Fire Weather Index features (`fwinx`, `ffmcode`, `drtcode`, `infsinx`) to each node's input vector.

## Run

From the repository root (`rbc-team-gaw/`):

```bash
/Users/arshdeepkaur/wildfire_detection/.venv/bin/python experiments/04a_exp_gnn_cems_bc/run_experiment.py
/Users/arshdeepkaur/wildfire_detection/.venv/bin/python experiments/04a_exp_gnn_cems_bc/build_report.py
```

## Experimental Design

- **Graph Structure:** Identical to Experiment 04 (`234,554` H3 resolution-7 nodes, directed edges joining in-centre H3 neighbours).
- **Node Inputs (12 features):**
  - The 8 node features from Experiment 04 (`log_local_count_1d`, `log_local_count_3d`, `log_local_count_7d`, `log_days_since_previous_detection`, `history_left_censored`, `quiet_stratum`, `day_of_year_scaled`, `in_region_neighbours_scaled`)
  - Plus the 4 shortlisted CEMS weather features from `history_end_date` ($t - 1$):
    - `log_fwinx`: $\ln(1 + \text{FWI})$
    - `ffmcode_scaled`: $(\text{FFMC} - 70) / 25$
    - `log_drtcode`: $\ln(1 + \text{DC}) / 6$
    - `log_infsinx`: $\ln(1 + \text{ISI})$
- **Models:**
  - `gnn`: 2 GraphSAGE-style mean-aggregation layers (2-hop receptive field), residual connections, `LayerNorm`, hidden size `32`.
  - `mlp`: Identical network with message passing turned off, isolating the effect of graph message passing over fire + weather states.
- **Training & Evaluation Split:**
  - **Train:** `2018–2021` (15 epochs, Adam `lr=3e-3`, fresh 1% draw of quiet negatives per daily snapshot).
  - **Selection:** `2022` chooses the best epoch (by mean per-centre AP) and best-F1 threshold.
  - **Evaluation:** `2023` (compared directly against Experiment 02 rules, Experiment 04 FIRMS-only GNN, and Experiment 05 Gradient Boosting + CEMS).
  - **Holdout:** `2024` is never loaded.

