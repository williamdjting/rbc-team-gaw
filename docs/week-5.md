# Week 5

**Date:** November 12, 2026

## Milestone

Train simple classifiers and compare prediction windows using time-based validation within the first three years.

## Deliverable

Model and prediction-window comparison.

## Presentation

[Week 5 detailed presentation](../experiments/Week5_GAW_detailed.html)

An 8-slide deck, mostly tables and charts, comparing the CanadaFireSat paper
(arXiv 2506.08690) with our work: the paper at a glance, data, target, how its test set is built (a map of which cells get scored),
our model under the paper's evaluation strategy, and two slides on why our sampled score is high (fire history, feature importance), and a final slide on new fires under both evaluations. 

Slide 5 scores one model (Experiment 21's CatBoost, trained 2018-2021) on the
8-day target in 2023, three ways: every cell (F1 0.468), sampled 1 fire : 2 non-fire
(0.853), and sampled with FWI-matched non-fire (0.726). Numbers come from
Experiment 38's results (`38_exp_canadafiresat_protocol_bc/results/`).

Use arrow keys or the slide selector; print mode displays all slides.

## Error analysis

[Week 5 error analysis](../experiments/Week5_error_analysis.html): Experiment 21's CatBoost (best model for now), next-day target, 2023, every cell: confusion matrices vs the persistence rule, continuing vs new fires caught, false alarms (cells that stopped burning vs no fire nearby), and F1 by Fire Centre (exact and one-cell margin). An Exp 25 MLP version with precision–recall, threshold, histogram, calibration and ROC curves, a one-day map and F1 by month is in [Week5_error_analysis_exp25_mlp.html](../experiments/Week5_error_analysis_exp25_mlp.html). Scripts: `experiments/week5_deck_scripts/`.
