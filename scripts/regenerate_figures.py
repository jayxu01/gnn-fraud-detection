"""Regenerate all figures from the saved model weights (no retraining required).

Usage:
    .venv\\Scripts\\python scripts/regenerate_figures.py
"""
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.dataset import load_elliptic
from src.models import build_model
from src.evaluate import evaluate_model, plot_curves, plot_recall_at_k, plot_model_comparison
from src.community import analyze_community, plot_communities


def main():
    data = load_elliptic()
    in_dim = data.x.shape[1]

    # per-model ROC / PR / Recall@K curves, from the saved last-seed best weights
    for name in ["MLP", "GCN", "GraphSAGE", "GAT"]:
        ckpt = config.MODEL_DIR / f"{name}.pt"
        if not ckpt.exists():
            print(f"  skip {name}: no checkpoint at {ckpt}")
            continue
        model = build_model(name, in_dim, config.HIDDEN_DIM,
                            config.NUM_LAYERS, config.DROPOUT, config.GAT_HEADS)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        _, y_true, y_score = evaluate_model(model, data)
        fig_dir = config.FIGURE_DIR / name.lower()
        plot_curves(y_true, y_score, fig_dir)
        plot_recall_at_k(y_true, y_score, fig_dir)
        print(f"  {name}: curves regenerated")

    # model comparison chart
    metrics = pd.read_csv(config.METRIC_DIR / "metrics.csv")
    plot_model_comparison(metrics, config.FIGURE_DIR)
    print("  model comparison chart regenerated")

    # community plot
    cdf, global_rate = analyze_community(data)
    plot_communities(cdf, global_rate, config.FIGURE_DIR)
    print("  community plot regenerated")

    print(f"figures written to {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
