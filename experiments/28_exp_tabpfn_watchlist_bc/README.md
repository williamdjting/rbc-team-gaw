# Experiment 28: TabPFN as a quiet-cell watch-list ranking

Experiment 07 used TabPFN only on active cells. This experiment uses it to rank every quiet
cell for Experiment 27's watch list, with Experiment 25's inputs (S-NPP history + CanCPLD
lightning), and compares it with the Experiment 25/26 MLPs, Experiment 13's CatBoost and
Experiment 18's simple rankings.

```powershell
pip install tabpfn==9.2.0     # uses the ungated TabPFN v2 weights
python run_experiment.py      # about 75 minutes on CPU; resumes from results/scores/
```

Needs Experiment 25's full-grid tables (`data/snpp_lightning_h3r7/`), Experiment 26 scores,
Experiment 13 predictions, and Experiment 27's code. Results: [RESULTS.md](RESULTS.md).
