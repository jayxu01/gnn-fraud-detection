"""Evaluation: AUC / PR-AUC / Recall@K / reference F1, plus ROC/PR curves.

Primary (threshold-free) metrics: AUC, PR-AUC. Business metrics: Recall@K, Precision@K.
The threshold-based F1/Recall/Precision pick their threshold on val and are reported on
test only as a reference.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

from . import config


def recall_at_k(y_true, y_score, k):
    """Recall@K / Precision@K.

    Pool = the labeled nodes in y_true (not all ~200k nodes). Rank by score
    descending and take the top-K: Recall@K = hits / total positives,
    Precision@K = hits / K.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(k, len(y_true))
    order = np.argsort(-y_score, kind="stable")
    hits = int(y_true[order[:k]].sum())
    total_pos = int(y_true.sum())
    recall = hits / total_pos if total_pos > 0 else 0.0
    precision = hits / k if k > 0 else 0.0
    return float(recall), float(precision)


def best_f1_threshold(y_true, y_score):
    """Threshold maximizing F1; fall back to 0.5 when a single class is present."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]) if best < len(thresholds) else 0.5


def evaluate_model(model, data, device=None):
    """Evaluate on test_mask, returning (metrics, test_true, test_score).

    The F1/Recall/Precision threshold is picked on val (not test) as a reference.
    """
    device = device or config.DEVICE
    model = model.to(device)
    model.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        logits = model(x, edge_index).squeeze(1).cpu().numpy()

    y = data.y.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    val_true = y[val_mask]
    val_score = logits[val_mask]
    test_true = y[test_mask]
    test_score = logits[test_mask]

    # threshold chosen on val, applied to test
    thr = best_f1_threshold(val_true, val_score)

    metrics = {
        "auc": float(roc_auc_score(test_true, test_score)),
        "pr_auc": float(average_precision_score(test_true, test_score)),
    }
    for k in config.TOP_K:
        r, p = recall_at_k(test_true, test_score, k)
        metrics[f"recall@{k}"] = r
        metrics[f"precision@{k}"] = p

    test_pred = (test_score >= thr).astype(int)
    metrics["f1"] = float(f1_score(test_true, test_pred))
    metrics["recall"] = float(recall_score(test_true, test_pred))
    metrics["precision"] = float(precision_score(test_true, test_pred))
    metrics["threshold"] = thr
    metrics["confusion"] = confusion_matrix(test_true, test_pred).tolist()

    return metrics, test_true, test_score


def plot_curves(y_true, y_score, fig_dir):
    """ROC and PR curves, saved to fig_dir/roc.png and pr.png."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend()
    plt.savefig(fig_dir / "roc.png", dpi=150, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.savefig(fig_dir / "pr.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_recall_at_k(y_true, y_score, fig_dir, ks=None):
    """Recall@K as a function of K."""
    ks = ks or [1, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
    ks = [k for k in ks if k <= len(y_true)]
    recalls = [recall_at_k(y_true, y_score, k)[0] for k in ks]
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(ks, recalls, marker="o")
    plt.xlabel("K (top-K alerts)")
    plt.ylabel("Recall@K")
    plt.title("Recall@K")
    plt.xscale("log")
    plt.savefig(fig_dir / "recall_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(metrics_df, fig_dir):
    """Grouped bar chart comparing the models on the key metrics."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = metrics_df.copy()
    panels = [("auc_mean", "ROC-AUC"), ("pr_auc_mean", "PR-AUC"),
              ("recall@300_mean", "Recall@300")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (col, title) in zip(axes, panels):
        ax.bar(df["model"], df[col], color="#5b8ff9")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        for i, v in enumerate(df[col]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def report_by_degree(model, data, device=None):
    """PR-AUC on test split by isolated vs non-isolated nodes (graph-structure value)."""
    device = device or config.DEVICE
    model = model.to(device)
    model.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        score = model(x, edge_index).squeeze(1).cpu().numpy()

    y = data.y.cpu().numpy()
    deg = torch.bincount(edge_index[0].cpu(), minlength=data.num_nodes).cpu().numpy()
    test = data.test_mask.cpu().numpy()

    result = {}
    for name, m in [("isolated (deg=0)", test & (deg == 0)),
                    ("non-isolated (deg>0)", test & (deg > 0))]:
        tt, ss = y[m], score[m]
        n = int(len(tt))
        n_pos = int(tt.sum())
        if n > 0 and 0 < n_pos < n:
            result[name] = {"n": n, "n_illicit": n_pos,
                            "pr_auc": float(average_precision_score(tt, ss))}
    return result
