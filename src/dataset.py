"""Data loading and graph construction.

Reads the three Elliptic CSVs, validates edge endpoints, maps labels, standardizes
features, and applies a leakage-free temporal split. Returns a PyG Data object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from . import config

# Raw class values -> numeric labels.
# Official mapping: 1 = illicit, 2 = licit, unknown = unlabeled.
# Illicit is the positive class (1), licit is negative (0), unknown is -1 (masked out).
LABEL_MAP = {"unknown": -1, "1": 1, "2": 0}

# Published label counts, used to hard-check the label direction (a flipped 1/2 copy
# would swap these two numbers).
_KNOWN_ILLICIT = 4545
_KNOWN_LICIT = 42019


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Column names of the features (everything except txId and time_step)."""
    return [c for c in df.columns if c not in ("txId", "time_step")]


def _norm_class(v) -> str:
    """Normalize a class value to "1" / "2" / "unknown", tolerating 1.0 / "1" / NaN."""
    s = str(v).strip().lower()
    if s in ("1", "1.0"):
        return "1"
    if s in ("2", "2.0"):
        return "2"
    return "unknown"


def build_data(features: pd.DataFrame, classes: pd.DataFrame, edgelist: pd.DataFrame) -> Data:
    """Build a PyG Data from three DataFrames. Pure function, unit-tested.

    Args:
        features: columns txId, time_step, and the feature columns
        classes:  columns txId, class (unknown / 1 / 2)
        edgelist: columns txId1, txId2 (directed payment flows)

    Returns: Data(x, edge_index, y, time_step, train_mask, val_mask, test_mask)
    """
    # 1. node features and time steps
    tx_ids = features["txId"].astype(np.int64).values
    node_idx = {int(tx): i for i, tx in enumerate(tx_ids)}
    num_nodes = len(tx_ids)

    feat_cols = _feature_columns(features)
    x = features[feat_cols].values.astype(np.float32)          # [N, F]
    time_step = features["time_step"].astype(np.int64).values  # [N]

    # 2. label mapping (classes.csv covers every node)
    class_of = dict(zip(classes["txId"].astype(np.int64).values,
                        classes["class"].values))
    y = np.array([LABEL_MAP[_norm_class(class_of.get(int(tx), "unknown"))]
                  for tx in tx_ids], dtype=np.int64)

    # 3. edge list -> edge_index
    src_ids = edgelist["txId1"].astype(np.int64).values
    dst_ids = edgelist["txId2"].astype(np.int64).values

    # validate that every edge endpoint is a known node
    known = set(node_idx.keys())
    unknown_src = set(src_ids.tolist()) - known
    unknown_dst = set(dst_ids.tolist()) - known
    if unknown_src or unknown_dst:
        raise ValueError(
            f"Edges reference unknown nodes: {len(unknown_src)} sources, "
            f"{len(unknown_dst)} targets"
        )

    src = np.array([node_idx[int(t)] for t in src_ids], dtype=np.int64)
    dst = np.array([node_idx[int(t)] for t in dst_ids], dtype=np.int64)

    # drop self-loops
    keep = src != dst
    src, dst = src[keep], dst[keep]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)  # [2, E]

    # convert to undirected (add reverse edges + dedupe) for bidirectional message passing
    if config.UNDIRECTED:
        edge_index = to_undirected(edge_index)

    # 4. leakage-free temporal split, labeled nodes only
    y_tensor = torch.tensor(y, dtype=torch.long)
    ts = torch.tensor(time_step, dtype=torch.long)
    labeled = y_tensor != -1

    train_mask = labeled & (ts <= config.TRAIN_TS_END)
    val_mask = labeled & (ts > config.TRAIN_TS_END) & (ts <= config.VAL_TS_END)
    test_mask = labeled & (ts > config.VAL_TS_END)

    # 5. z-score standardization fitted on train nodes only, applied to the whole graph
    # This keeps test statistics out of the training pipeline; val/test/unlabeled nodes
    # are scaled with the same train-fitted mean/std.
    train_x = x[train_mask.numpy()]
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x = torch.tensor((x - mean) / std, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y_tensor, time_step=ts,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_nodes = num_nodes
    return data


def _check_data_files() -> None:
    """Ensure the three CSVs exist, with an actionable message otherwise."""
    missing = [p for p in (config.FEATURES_CSV, config.EDGELIST_CSV, config.CLASSES_CSV)
               if not p.exists()]
    if missing:
        names = "\n  ".join(str(m) for m in missing)
        raise FileNotFoundError(
            f"Missing data files:\n  {names}\n"
            "Download the Elliptic dataset (ellipticco/elliptic-data-set) from Kaggle "
            "into data/raw/ — see data/raw/README.md."
        )


def _print_summary(data: Data) -> None:
    """Print node/edge/label/split counts for sanity checking."""
    def cnt(mask, val):
        return int(((data.y == val) & mask).sum())

    print("=" * 60)
    print("Data summary")
    print(f"  nodes: {data.num_nodes:,}    edges (undirected): {data.edge_index.shape[1]:,}")
    print(f"  feature dim: {data.x.shape[1]}")
    for name, m in [("train", data.train_mask), ("val", data.val_mask),
                    ("test", data.test_mask)]:
        n = int(m.sum())
        print(f"  {name:>5}: labeled {n:>7,} | illicit {cnt(m, 1):>6,} | licit {cnt(m, 0):>7,}")
    print("=" * 60)


def _read_features(path):
    """Read features.csv, handling both header and header-less copies.

    With header: txId,time_step,0,1,...,N
    Without header: the first line is data (txId,time_step,feature values...)
    """
    with open(path) as f:
        first = f.readline().strip()
    if "txId" in first.lower():
        return pd.read_csv(path)
    df = pd.read_csv(path, header=None)
    n_feat = df.shape[1] - 2
    df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(n_feat)]
    return df


def load_elliptic() -> Data:
    """Load the real CSVs and build the graph, with a hard label-direction check."""
    _check_data_files()

    features = _read_features(config.FEATURES_CSV)
    classes = pd.read_csv(config.CLASSES_CSV)
    edgelist = pd.read_csv(config.EDGELIST_CSV)

    data = build_data(features, classes, edgelist)

    # Hard check: illicit/licit counts must match the published numbers, otherwise the
    # 1/2 direction has been flipped.
    y = data.y
    n_illicit = int((y == 1).sum())
    n_licit = int((y == 0).sum())
    if n_illicit != _KNOWN_ILLICIT or n_licit != _KNOWN_LICIT:
        raise ValueError(
            f"Label counts do not match the published values: illicit={n_illicit} "
            f"(expected {_KNOWN_ILLICIT}), licit={n_licit} (expected {_KNOWN_LICIT}). "
            "Check whether the 1/2 direction in classes.csv is flipped."
        )

    _print_summary(data)
    return data
