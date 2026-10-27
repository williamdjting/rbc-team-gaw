# TabPFN instructions

Use these instructions to run TabPFN against the CatBoost and learned baselines. Create this folder first so the experiment numbering stays consistent:

```text
experiments/07_tabpfn/
```

The script is `experiments/07_tabpfn/run_experiment.py`. It imports helpers from Experiment 05 and loads Experiment 06's fitted CatBoost models, so **run Experiments 05 and 06 first**.

## Background

TabPFN (Hollmann et al., `william/week2/TabPFN.pdf`) is a pretrained transformer for tabular data. It is not trained on our data. Training rows are passed in as a *context* at prediction time, and the model predicts from that context in a single forward pass. This has two consequences here:

- **Context size is capped.** On CPU, TabPFN-3 and later allow about 5,000 rows, and older versions about 1,000. GPU and Apple Silicon allow more. Experiment 06's undersampled training set (1.51M / 1.76M rows) is about 300× over the CPU cap.
- **Prediction cost grows with test rows.** Every test row is scored against the full context. Scoring all ~68M rows from 2022 and 2023 is not feasible on a laptop or on the free cloud tier.

## Setup

Use a fresh virtual environment with Python 3.11–3.13, which has the best PyTorch wheel support.

```bash
cd rbc-team-gaw
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.13" tabpfn pandas numpy pyarrow scikit-learn catboost
```

On Apple Silicon, check that the GPU is available:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

This should print `True`. The first `TabPFNClassifier.fit` call downloads the model weights from Hugging Face. If the download fails with a licence or authentication error, accept the model licence on the Hugging Face model page, then run `pip install huggingface_hub` and `hf auth login` (on older versions, `huggingface-cli login`), and retry. Follow the TabPFN README for the version you installed.

No GPU, or local runs too slow? The free hosted inference is available through `pip install tabpfn-client`, but the script targets the local package. Check the client's row and usage limits before relying on it. The FIRMS data is public, but confirm the team is comfortable sending it to a third-party API.

## Split, target and features

Unchanged from the CatBoost instructions. Train on **2018–2021**, select thresholds on **2022**, evaluate on **2023**, and never load 2024. The target is `target_recorded_detection` (recorded hotspot activity, not ignitions). There are two configurations, `cell_only` and `cell_plus_neighbours`, with the same features as Experiments 03–06. `quiet_stratum` is dropped for TabPFN because it is always 0 on the rows TabPFN scores.

## Design: hybrid by activity stratum

Quiet/active follows the CatBoost instructions. Quiet means no recorded activity in the past seven days, in the cell for `cell_only`, and in the cell and its in-region neighbours for `cell_plus_neighbours`.

| Rows | Share of rows | Scored by |
| --- | --- | --- |
| Active | ~0.1–0.3% | TabPFN |
| Quiet | ~99.7–99.9% | Experiment 06 CatBoost, with the 0.01 quiet-row odds correction |

1. Collect every 2018–2021 active row per configuration. Check that the four-category counts sum to all training rows, and that the active counts equal Experiment 06's class-balance table, which keeps every active row.
2. Draw a uniform random context from the active rows: 10,000 rows per draw, 3 draws, seeded by `(42, configuration, draw)`. Do not rebalance classes. Active rows are already about 26% (`cell_only`) and 11% (`cell_plus_neighbours`) positive, so the scores stay on the active population's scale and need no odds correction.
3. Fit one `TabPFNClassifier(n_estimators=4, ignore_pretraining_limits=True)` per draw, and average the positive-class probabilities across draws.
4. Score every 2022 and 2023 row. Active rows get the TabPFN average, and quiet rows get Experiment 06's corrected CatBoost score.
5. For each configuration and Fire Centre, pick the threshold that maximizes 2022 F1 on the combined scores, then apply it unchanged to 2023.

The script also reports **active-stratum AP** for TabPFN and for Experiment 06 CatBoost on the same active rows. This isolates TabPFN's contribution, since the quiet rows are identical in both models.

## Running

Start with a smoke test, which uses a 2,000-row context, 1 draw and 1 estimator:

```bash
python experiments/07_tabpfn/run_experiment.py --quick
```

Each region-year line prints TabPFN throughput (rows/s). Use it to estimate the full run time: about 3 draws × (all active rows in 2022 and 2023, both configurations) ÷ rate. Then run the full version:

```bash
python experiments/07_tabpfn/run_experiment.py
```

Optional flags: `--context-rows N`, `--contexts K`, and `--device mps|cuda|cpu`. If MPS runs out of memory, lower `--context-rows`, or lower `predict_batch` in `SETTINGS`.

## Outputs

Saved in `experiments/07_tabpfn/results/`:

```text
tabpfn_metrics_2023.csv            per centre and configuration: AP, precision, recall, F1, threshold, TP, FP, FN, TN,
                                   plus active_ap_tabpfn and active_ap_catboost_06
tabpfn_metrics_by_year.csv         2022 (threshold selection) and 2023
tabpfn_predictions_2023.parquet    every 2023 row with quiet flags and hybrid scores
tabpfn_context_balance.csv         rows and positives in each context draw
tabpfn_fit_checks.csv
tabpfn_comparison_2023.csv         beside Experiments 03, 04, 05 and 06 on the same 2023 rows
tabpfn_manifest.json               versions, device, seed, settings, counts, thresholds, throughput, source hashes
```

## Interpretation

- Compare `active_ap_tabpfn` with `active_ap_catboost_06` first. This is the clean TabPFN vs CatBoost comparison.
- Overall AP and F1 also depend on how the TabPFN active scores rank against the CatBoost quiet scores. Both are probabilities on the original population scale, but they come from different models, so state this when comparing with Experiments 03–06.
- TabPFN sees at most 30,000 of about 133k (`cell_only`) or 386k (`cell_plus_neighbours`) active training rows. A gap with CatBoost may reflect the context size rather than the model. Try `--context-rows 20000` or more draws if memory and time allow.
- Positives are recorded hotspot activity, not confirmed ignitions.
