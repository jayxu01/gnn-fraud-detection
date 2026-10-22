"""Training: loss on train_mask only, early stopping on val_mask, class imbalance."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def focal_loss(logits, targets, gamma: float = 2.0, alpha=None):
    """Focal loss: down-weights easy samples to focus on hard ones (imbalance).

    logits: [N, 1] or [N]; targets: [N] in {0, 1}.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    loss = (1 - pt) ** gamma * bce
    if alpha is not None:
        at = alpha[targets.long()]
        loss = at * loss
    return loss.mean()


def _safe_pr_auc(y_true, y_score):
    """PR-AUC; returns NaN when a single class makes it undefined."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return average_precision_score(y_true, y_score)


def _safe_auc(y_true, y_score):
    """ROC-AUC with the same single-class guard."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return roc_auc_score(y_true, y_score)


def train_model(model, data, seed=None, device=None):
    """Train one model. Returns the best state dict (by val PR-AUC early stopping);
    the model is reloaded with the best weights before returning."""
    if seed is not None:
        set_seed(seed)
    device = device or config.DEVICE
    model = model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)

    # Class weight: positive (illicit) is the minority, weight it by licit/illicit ratio.
    y_train = y[train_mask]
    n_pos = (y_train == 1).sum().float()
    n_neg = (y_train == 0).sum().float()
    pos_weight = (n_neg / n_pos).clamp(min=1.0) if n_pos > 0 else torch.tensor(1.0)
    pos_weight = pos_weight.to(device)

    if config.LOSS == "focal":
        alpha = torch.tensor([1.0, pos_weight.item()], device=device)

        def criterion(logits, targets):
            return focal_loss(logits, targets, gamma=config.FOCAL_GAMMA, alpha=alpha)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_metric = -float("inf")
    best_state = None
    patience_left = config.PATIENCE

    for epoch in range(1, config.EPOCHS + 1):
        # train step
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask].squeeze(1), y[train_mask].float())
        loss.backward()
        optimizer.step()

        # evaluate (log train/val loss and val PR-AUC each epoch)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            train_loss = F.binary_cross_entropy_with_logits(
                out[train_mask].squeeze(1), y[train_mask].float()).item()
            val_loss = F.binary_cross_entropy_with_logits(
                out[val_mask].squeeze(1), y[val_mask].float()).item()
            val_score = out[val_mask].squeeze(1).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()

        # early-stopping metric: PR-AUC, falling back to ROC-AUC, then -val_loss
        metric = _safe_pr_auc(val_true, val_score)
        metric_name = "val_pr_auc"
        if np.isnan(metric):
            metric = _safe_auc(val_true, val_score)
            metric_name = "val_auc"
        if np.isnan(metric):
            metric = -val_loss
            metric_name = "val_neg_loss"

        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | {metric_name} {metric:.4f}")

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch} ({metric_name}={best_metric:.4f})")
                break

    model.load_state_dict(best_state)
    return best_state
