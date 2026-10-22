"""Unit tests for dataset: label mapping, temporal split, graph construction.

Runs on synthetic data, no real Elliptic CSVs required.
"""
import numpy as np
import pandas as pd
import pytest

from src.dataset import build_data, LABEL_MAP


def _make_dfs():
    """Build a small synthetic dataset covering train/val/test and a self-loop."""
    n = 10
    features = pd.DataFrame({
        "txId": list(range(n)),
        "time_step": [1, 5, 20, 29, 30, 34, 35, 40, 49, 30],
        "f0": np.arange(n, dtype=float),
        "f1": np.arange(n, dtype=float) * 2.0,
    })
    classes = pd.DataFrame({
        "txId": list(range(n)),
        "class": ["1", "2", "unknown", "1", "2", "unknown", "2", "1", "2", "1"],
    })
    edgelist = pd.DataFrame({
        "txId1": [0, 1, 2, 3, 3, 4],
        "txId2": [1, 2, 3, 4, 3, 5],  # 3->3 is a self-loop
    })
    return features, classes, edgelist


def test_label_map():
    assert LABEL_MAP == {"unknown": -1, "1": 1, "2": 0}


def test_shapes():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    assert data.x.shape == (10, 2)
    assert data.y.shape == (10,)
    # 6 directed edges minus 1 self-loop = 5, undirected *2 = 10
    assert data.edge_index.shape == (2, 10)


def test_label_mapping_values():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    assert data.y.tolist() == [1, 0, -1, 1, 0, -1, 0, 1, 0, 1]


def test_train_mask():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    assert data.train_mask.nonzero().flatten().tolist() == [0, 1, 3]


def test_val_mask():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    assert data.val_mask.nonzero().flatten().tolist() == [4, 9]


def test_test_mask():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    assert data.test_mask.nonzero().flatten().tolist() == [6, 7, 8]


def test_no_self_loop():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    src, dst = data.edge_index
    assert (src != dst).all().item()


def test_standardization():
    f, c, e = _make_dfs()
    data = build_data(f, c, e)
    x = data.x.numpy()
    # z-score is fit on train only: train nodes should have mean 0 and std 1
    train_x = x[data.train_mask.numpy()]
    assert np.allclose(train_x.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(train_x.std(axis=0), 1.0, atol=1e-5)
    # no NaN anywhere
    assert not np.isnan(x).any()


def test_unknown_endpoint_raises():
    f, c, _ = _make_dfs()
    bad = pd.DataFrame({"txId1": [0, 999], "txId2": [1, 2]})
    with pytest.raises(ValueError):
        build_data(f, c, bad)
