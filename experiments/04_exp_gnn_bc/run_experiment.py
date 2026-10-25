"""Graph neural network on daily Fire Centre snapshots (planned step 3).

Same rows, target, split, sampling and evaluation as Experiment 03. Each target
day is one graph: every Fire Centre grid cell is a node, edges join H3
neighbours inside the same centre (as in Experiments 02-03), and the six centres
are disconnected components. Nodes receive only their own cell's features, so
any neighbour information must come from message passing. The same network
without message passing (an MLP) isolates the graph's contribution.
2024 is never loaded.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENT_02 = HERE.parent / "02_hotspot_persistence_bc"
EXPERIMENT_03 = HERE.parent / "03_exp_learned_baselines_bc"
spec = importlib.util.spec_from_file_location("experiment_03", EXPERIMENT_03 / "run_experiment.py")
experiment_03 = sys.modules["experiment_03"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_03)
from experiment_03 import (load, feature_path, slug, threshold_metrics, best_f1_threshold, downsampled_curve,  # noqa: E402
                           TARGET, QUIET_SAMPLE_RATE, SEED, TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR)

RESULTS = HERE / "results"
DAYS = 146  # May 8-September 30 targets, as in Experiments 01-03
INPUTS = ["log_local_count_1d", "log_local_count_3d", "log_local_count_7d", "log_days_since_previous_detection",
          "history_left_censored", "quiet_stratum", "day_of_year_scaled", "in_region_neighbours_scaled"]
HIDDEN, LAYERS, LEARNING_RATE, EPOCHS = 32, 2, 3e-3, 15
NETWORKS = {"gnn": True, "mlp": False}  # name -> message passing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_graph(regions):
    """Concatenate per-centre grids; edges only between in-centre H3 neighbours."""
    cells, offsets, rows, columns = [], {}, [], []
    for region in regions:
        grid = pd.read_parquet(EXPERIMENT_02 / "regions" / slug(region) / "data/h3_grid.parquet", columns=["h3_cell"])
        local = grid.h3_cell.tolist()
        position = {cell: index for index, cell in enumerate(local)}
        offsets[region] = (len(cells), local)
        local_rows = []
        for cell, index in position.items():
            for neighbour in sorted(h3.grid_disk(cell, 1)):
                if neighbour != cell and neighbour in position:
                    local_rows.append(index)
                    columns.append(len(cells) + position[neighbour])
        reference = pd.read_csv(EXPERIMENT_02 / "regions" / slug(region) / "results/neighbour_counts.csv")
        degrees = np.bincount(local_rows, minlength=len(local))
        assert reference.h3_cell.tolist() == local and np.array_equal(reference.in_region_neighbours, degrees), region
        rows += [len(cells) + index for index in local_rows]
        cells.extend(local)
    rows, columns = np.array(rows), np.array(columns)
    degree = np.bincount(rows, minlength=len(cells)).astype(np.float32)
    values = 1 / degree[rows]  # mean over neighbours
    adjacency = torch.sparse_coo_tensor(np.vstack([rows, columns]), values, (len(cells), len(cells))).coalesce()
    return adjacency.to_sparse_csr(), offsets, len(cells), len(rows)


def node_features(frame):
    log = lambda column: np.log1p(frame[column].to_numpy(dtype=np.float32))
    return np.column_stack([
        log("local_count_1d"), log("local_count_3d"), log("local_count_7d"),
        np.log1p(frame.days_since_previous_detection.fillna(0).to_numpy(dtype=np.float32)),
        frame.history_left_censored.to_numpy(dtype=np.float32),
        frame.local_count_7d.eq(0).to_numpy(dtype=np.float32),  # same quiet stratum as 03's cell_only
        (frame.day_of_year.to_numpy(dtype=np.float32) - 200) / 45,
        frame.in_region_neighbours.to_numpy(dtype=np.float32) / 6])


def load_indexed(region, year, offsets, nodes):
    """One centre's feature rows with their (day, node) position in the daily snapshots."""
    start, local = offsets[region]
    frame = load(region, year, with_cells=True)
    day = (frame.target_date - pd.Timestamp(f"{year}-05-08")).dt.days.to_numpy()
    node = start + pd.Categorical(frame.h3_cell, categories=local).codes
    assert len(frame) == DAYS * len(local) and day.min() == 0 and day.max() == DAYS - 1
    assert (node >= start).all() and not pd.Series(day * nodes + node).duplicated().any()
    return frame, day, node


def snapshots(regions, offsets, nodes, year):
    """Inputs (days, nodes, features) and targets (days, nodes) for one year; one centre in memory at a time."""
    X = np.zeros((DAYS, nodes, len(INPUTS)), dtype=np.float16)
    y = np.full((DAYS, nodes), -1, dtype=np.int8)
    for region in regions:
        frame, day, node = load_indexed(region, year, offsets, nodes)
        X[day, node] = node_features(frame)
        y[day, node] = frame[TARGET].to_numpy()
        del frame
    assert (y >= 0).all(), "Every node-day must be filled exactly once"
    return X, y


class Network(nn.Module):
    def __init__(self, message_passing):
        super().__init__()
        self.message_passing = message_passing
        self.encode = nn.Sequential(nn.Linear(len(INPUTS), HIDDEN), nn.ReLU())
        self.own = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(LAYERS))
        self.neighbours = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(LAYERS)) if message_passing else None
        self.norms = nn.ModuleList(nn.LayerNorm(HIDDEN) for _ in range(LAYERS))
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x, adjacency):
        h = self.encode(x)
        for layer in range(LAYERS):
            update = self.own[layer](h)
            if self.message_passing:  # GraphSAGE-style mean aggregation over in-centre neighbours
                update = update + self.neighbours[layer](torch.sparse.mm(adjacency, h))
            h = self.norms[layer](h + torch.relu(update))
        return self.head(h).squeeze(-1)


def predict(network, X, adjacency):
    """Probabilities with the quiet-negative sampling undone (quiet_stratum is an input)."""
    network.eval()
    shift = np.log(QUIET_SAMPLE_RATE)
    out = np.empty(X.shape[:2], dtype=np.float32)
    with torch.no_grad():
        for day in range(len(X)):
            x = torch.from_numpy(X[day]).to(DEVICE, torch.float32)
            logits = network(x, adjacency) + shift * x[:, INPUTS.index("quiet_stratum")]
            out[day] = torch.sigmoid(logits).cpu().numpy()
    return out


def mean_region_ap(score, y, offsets):
    aps = {region: experiment_03.average_precision_score(y[:, start:start+len(local)].ravel(),
                                                         score[:, start:start+len(local)].ravel())
           for region, (start, local) in offsets.items()}
    return float(np.mean(list(aps.values()))), aps


def train(name, message_passing, train_X, train_y, select_X, select_y, adjacency, offsets):
    torch.manual_seed(SEED)
    network = Network(message_passing).to(DEVICE)
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    quiet_column = INPUTS.index("quiet_stratum")
    history, best = [], (-1.0, None, None)
    for epoch in range(1, EPOCHS + 1):
        network.train()
        tic, losses = time.time(), []
        for index in rng.permutation(len(train_X)):
            x = torch.from_numpy(train_X[index]).to(DEVICE, torch.float32)
            y = torch.from_numpy(train_y[index]).to(DEVICE, torch.float32)
            # Fresh 1% draw of quiet negatives each step; positives and active nodes always count.
            drawn = torch.from_numpy(rng.random(len(y)) < QUIET_SAMPLE_RATE).to(DEVICE)
            keep = (x[:, quiet_column] == 0) | (y == 1) | drawn
            loss = loss_fn(network(x, adjacency)[keep], y[keep])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        selection_ap, region_aps = mean_region_ap(predict(network, select_X, adjacency), select_y, offsets)
        history.append({"network": name, "epoch": epoch, "training_loss": float(np.mean(losses)),
                        "selection_mean_region_ap": selection_ap, "seconds": round(time.time() - tic, 1),
                        **{f"selection_ap_{slug(r)}": ap for r, ap in region_aps.items()}})
        print(f"{name} epoch {epoch:2d}: loss {np.mean(losses):.4f}, 2022 mean AP {selection_ap:.3f} "
              f"({time.time()-tic:.0f}s)", flush=True)
        if selection_ap > best[0]:
            best = (selection_ap, epoch, copy.deepcopy(network.state_dict()))
    network.load_state_dict(best[2])
    return network, best[1], history


def main():
    started = time.time()
    for folder in (RESULTS / "models", RESULTS / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    regions = sorted(pd.read_csv(EXPERIMENT_02 / "results/selected_rules_2023.csv").region)
    rules = pd.read_csv(EXPERIMENT_02 / "results/all_region_metrics.csv")
    learned = pd.read_csv(EXPERIMENT_03 / "results/metrics_by_year.csv")
    years = [*TRAIN_YEARS, SELECTION_YEAR, EVALUATION_YEAR]
    source_paths = ([feature_path(r, y) for r in regions for y in years]
                    + [EXPERIMENT_02 / "regions" / slug(r) / "data/h3_grid.parquet" for r in regions]
                    + [EXPERIMENT_02 / "results/all_region_metrics.csv", EXPERIMENT_03 / "results/metrics_by_year.csv",
                       EXPERIMENT_03 / "run_experiment.py"])
    hashes = {str(p.relative_to(HERE.parent)): digest(p) for p in source_paths}

    adjacency, offsets, nodes, edges = build_graph(regions)
    adjacency = adjacency.to(DEVICE)
    print(f"Graph: {nodes:,} nodes, {edges:,} directed edges across {len(regions)} centres; device {DEVICE}", flush=True)

    # Filled year by year to avoid holding two copies of ~2 GB of float16 inputs.
    train_X = np.empty((len(TRAIN_YEARS) * DAYS, nodes, len(INPUTS)), dtype=np.float16)
    train_y = np.empty((len(TRAIN_YEARS) * DAYS, nodes), dtype=np.int8)
    for index, year in enumerate(TRAIN_YEARS):
        block = slice(index * DAYS, (index + 1) * DAYS)
        train_X[block], train_y[block] = snapshots(regions, offsets, nodes, year)
    select_X, select_y = snapshots(regions, offsets, nodes, SELECTION_YEAR)
    print(f"Training snapshots: {len(train_X)} days x {nodes:,} nodes; positives {int((train_y == 1).sum()):,}", flush=True)

    fitted, histories, best_epochs = {}, [], {}
    for name, message_passing in NETWORKS.items():
        network, best_epochs[name], history = train(name, message_passing, train_X, train_y, select_X, select_y,
                                                    adjacency, offsets)
        fitted[name] = network
        histories += history
        torch.save(network.state_dict(), RESULTS / "models" / f"{name}.pt")
        print(f"{name}: best epoch {best_epochs[name]} by 2022 mean per-centre AP", flush=True)
    pd.DataFrame(histories).to_csv(RESULTS / "training_history.csv", index=False)
    del train_X, train_y

    rows, curves, thresholds = [], [], {}
    for year in (SELECTION_YEAR, EVALUATION_YEAR):
        X = select_X if year == SELECTION_YEAR else snapshots(regions, offsets, nodes, year)[0]
        scores = {name: predict(network, X, adjacency) for name, network in fitted.items()}
        del X
        if year == SELECTION_YEAR:
            del select_X, select_y
        for region in regions:
            frame, day, node = load_indexed(region, year, offsets, nodes)
            y = frame[TARGET].to_numpy()
            reference = rules.loc[rules.region.eq(region) & rules.year.eq(year) & rules.rule.eq("local_1d")].iloc[0]
            check = threshold_metrics(y, frame.local_count_1d.to_numpy(dtype=np.float32), 1)
            assert all(check[k] == reference[k] for k in ("rows", "positive_cell_days", "tp", "fp", "fn")), region
            predictions = frame[["target_date", "h3_cell", TARGET]].copy() if year == EVALUATION_YEAR else None
            for name in fitted:
                score = scores[name][day, node]
                if year == SELECTION_YEAR:
                    thresholds[(region, name)] = best_f1_threshold(y, score)
                rows.append({"region": region, "year": year,
                             "role": "threshold_selection" if year == SELECTION_YEAR else "development_evaluation",
                             "model": name, "feature_set": "cell_only_inputs", "best_epoch": best_epochs[name],
                             **threshold_metrics(y, score, thresholds[(region, name)])})
                if year == EVALUATION_YEAR:
                    predictions[name] = score
                    recall, precision = downsampled_curve(y, score)
                    curves.append(pd.DataFrame({"region": region, "model": name, "recall": recall, "precision": precision}))
            if predictions is not None:
                predictions.to_parquet(RESULTS / "predictions" / f"{slug(region)}_{EVALUATION_YEAR}.parquet", index=False)
            latest = [r for r in rows if r["region"] == region and r["year"] == year]
            print(f"{region} {year}: " + "; ".join(f"{r['model']} AP={r['average_precision']:.3f}" for r in latest), flush=True)
            del frame, predictions
        del scores

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RESULTS / "pr_curves_2023.csv", index=False)

    # One 2023 table: Experiment 02 rules, Experiment 03 learned models, this experiment's networks.
    rule_2023 = rules.loc[rules.year.eq(EVALUATION_YEAR) & rules.rule.ne("always_no_detection")]
    selected = pd.read_csv(EXPERIMENT_02 / "results/selected_rules_2023.csv").set_index("region")
    combined = pd.concat([
        selected.reset_index().assign(source="02_rules", model="selected_rule", feature_set=selected.rule.to_numpy()),
        rule_2023.loc[rule_2023.groupby("region").average_precision.idxmax()].assign(
            source="02_rules", model="best_rule_hindsight", feature_set=lambda f: f.rule),
        learned.loc[learned.year.eq(EVALUATION_YEAR)].assign(source="03_learned_baselines"),
        metrics.loc[metrics.year.eq(EVALUATION_YEAR)].assign(source="04_gnn")], ignore_index=True)
    combined[["source", "region", "model", "feature_set", "average_precision", "precision", "recall", "f1", "tp", "fp", "fn"]] \
        .sort_values(["region", "source", "model", "feature_set"]).to_csv(RESULTS / "comparison_2023.csv", index=False)

    assert hashes == {str(p.relative_to(HERE.parent)): digest(p) for p in source_paths}
    (RESULTS / "manifest.json").write_text(json.dumps({
        "source_hashes": hashes, "regions": regions, "training_years": list(TRAIN_YEARS),
        "selection_year": SELECTION_YEAR, "evaluation_year": EVALUATION_YEAR, "holdout_loaded": False,
        "graph": {"nodes": nodes, "directed_edges": edges, "rule": "one node per Fire Centre grid cell; edges between "
                  "H3 neighbours inside the same centre; centres disconnected; degrees match Experiment 02"},
        "node_inputs": INPUTS, "neighbour_inputs": "none; neighbour information only through message passing",
        "networks": {"gnn": f"{LAYERS} GraphSAGE-style mean-aggregation layers (2-hop), residual + LayerNorm, hidden {HIDDEN}",
                     "mlp": "identical network with message passing removed"},
        "training": {"optimiser": "Adam", "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "seed": SEED,
                     "batch": "one daily snapshot (all six centres) per step",
                     "sampling": "loss on all positives and active nodes plus a fresh 1% draw of quiet negatives per step; "
                                 "log(rate) added to quiet-node log-odds at prediction (exact: quiet_stratum is an input)",
                     "epoch_selection": "best mean per-centre 2022 average precision", "best_epochs": best_epochs},
        "threshold_rule": "per Fire Centre and network, maximize 2022 F1; applied unchanged to 2023",
        "checks": "graph degrees match Experiment 02, every node-day filled once, past-only history, no 2024 rows, "
                  "Experiment 02 previous-day rule reproduced on every evaluated region-year, source hashes unchanged",
        "device": str(DEVICE), "runtime_seconds": round(time.time() - started),
        "versions": {"torch": torch.__version__, "h3": h3.__version__, "pandas": pd.__version__, "numpy": np.__version__}},
        indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time()-started:.0f}s. Results: {RESULTS}", flush=True)
    print(combined.pivot_table(index="region", columns=["source", "model"], values="average_precision", aggfunc="max")
          .round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
