# Experiment 04: graph neural network for BC

Step 3 of the planned comparison in `hotspot-forecasting-design.md`: a standard graph
neural network on the same rows, target, split and evaluation as Experiment 03.

Run from this folder after Experiments 02 and 03 have completed:

```powershell
python run_experiment.py
python build_report.py
```

Dependencies: torch (CPU or CUDA), h3 4.5.0, pandas, numpy, scikit-learn,
pyarrow, matplotlib. CUDA is used automatically when available. Shared loading,
metric and threshold functions are imported from
`../03_exp_learned_baselines_bc/run_experiment.py`; inputs are read from
Experiment 02. Neither earlier experiment is modified.

- **Graph:** one graph per target day. Every Fire Centre grid cell is a node
  (234,554 in total). Edges join H3 neighbours inside the same centre, matching
  the in-region neighbours of Experiments 02-03, and the six centres are
  disconnected components. Node degrees are checked against Experiment 02.
- **Node inputs:** only the cell's own features: log past 1/3/7-day counts,
  log days since previous detection, history-censored flag, quiet-stratum flag,
  day of year and in-centre neighbour count. No neighbour counts are given; any
  neighbour information must come from message passing.
- **GNN:** two GraphSAGE-style layers with mean aggregation (2-hop receptive
  field), residual connections and LayerNorm, hidden size 32. **MLP control:**
  the identical network with message passing removed, so the difference between
  the two isolates the graph's contribution.
- **Training:** 2018-2021, one daily snapshot of all six centres per step, Adam
  (lr 3e-3), 15 epochs, seed 42. The loss uses all positives and active nodes plus
  a fresh 1% draw of quiet negatives each step. Quiet-node log-odds are shifted by
  log(0.01) at prediction, exactly as in Experiment 03.
- **Selection:** the epoch with the best mean per-centre 2022 average precision
  is kept (03 used 2022 only for thresholds). Thresholds maximize 2022 F1 per
  centre and network. 2023 is the development evaluation; 2024 is never loaded.

Checks: graph degrees match Experiment 02, every node-day filled exactly once,
past-only history, no 2024 rows, Experiment 02's previous-day rule reproduced on
every evaluated region-year, and source hashes unchanged.

`results/` contains metrics by year, a 2023 comparison combining Experiment 02
rules, Experiment 03 models and these networks, precision-recall curves, per-epoch
training history, model weights, per-centre 2023 predictions and a manifest.
`RESULTS.html` is the standalone report.
