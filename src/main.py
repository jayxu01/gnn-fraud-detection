"""End-to-end pipeline: build graph -> train (MLP baseline + 3 GNNs) -> evaluate -> communities.

Usage (from the project root):
    .venv\\Scripts\\python -m src.main
"""
import time

import numpy as np
import pandas as pd
import torch

from . import config
from .dataset import load_elliptic
from .models import build_model
from .train import train_model, set_seed
from .evaluate import evaluate_model, plot_curves, plot_recall_at_k, plot_model_comparison, report_by_degree
from .community import analyze_community, save_community_stats, plot_communities

MODEL_NAMES = ["MLP", "GCN", "GraphSAGE", "GAT"]


def _aggregate(per_seed, keys):
    """Mean and std across seeds for each metric."""
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_seed]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"] = round(float(np.std(vals)), 4)
    return agg


def run_models(data):
    """Train and evaluate all models; returns (metrics DataFrame, by-degree DataFrame)."""
    in_dim = data.x.shape[1]
    rows, degree_rows = [], []

    for name in MODEL_NAMES:
        per_seed = []
        y_true = y_score = None
        last_model = None
        t0 = time.time()
        print(f"\n===== model {name} =====")

        for seed in config.SEEDS:
            set_seed(seed)
            model = build_model(name, in_dim, config.HIDDEN_DIM,
                                config.NUM_LAYERS, config.DROPOUT, config.GAT_HEADS)
            train_model(model, data, seed=seed)
            metrics, y_true, y_score = evaluate_model(model, data)
            per_seed.append(metrics)
            last_model = model

        # save the best weights of the last seed (reused by the EDA t-SNE)
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(last_model.state_dict(), config.MODEL_DIR / f"{name}.pt")

        keys = [k for k in per_seed[0] if k != "confusion"]
        agg = _aggregate(per_seed, keys)
        agg["model"] = name
        agg["confusion"] = per_seed[-1]["confusion"]
        agg["seconds"] = round(time.time() - t0, 1)
        rows.append(agg)

        fig_dir = config.FIGURE_DIR / name.lower()
        plot_curves(y_true, y_score, fig_dir)
        plot_recall_at_k(y_true, y_score, fig_dir)

        # graph-structure value (skip MLP, it has no graph)
        if name != "MLP":
            for grp, v in report_by_degree(last_model, data).items():
                degree_rows.append({"model": name, "group": grp, **v})

    return pd.DataFrame(rows), pd.DataFrame(degree_rows)


def main():
    t_start = time.time()
    data = load_elliptic()

    df, degree_df = run_models(data)

    config.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.METRIC_DIR / "metrics.csv", index=False)
    degree_df.to_csv(config.METRIC_DIR / "by_degree.csv", index=False)
    plot_model_comparison(df, config.FIGURE_DIR)

    cols = ["model", "auc_mean", "pr_auc_mean", "recall@100_mean", "f1_mean"]
    print("\n===== model metrics (mean +- std, test) =====")
    print(df[cols].to_string(index=False))
    print(f"\nmetrics saved to {config.METRIC_DIR / 'metrics.csv'}")

    if not degree_df.empty:
        print("\n===== graph-structure value (test isolated vs non-isolated, PR-AUC) =====")
        print(degree_df.to_string(index=False))

    print("\n===== community detection (Louvain) =====")
    cdf, global_rate = analyze_community(data)
    save_community_stats(cdf, config.METRIC_DIR / "community_stats.csv")
    plot_communities(cdf, global_rate, config.FIGURE_DIR)
    print(f"global labeled illicit rate (baseline): {global_rate:.4f}")
    print("top-10 high-density communities:")
    print(cdf.head(10).to_string(index=False))

    print(f"\ndone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
