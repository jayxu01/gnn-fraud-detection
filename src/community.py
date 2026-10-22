"""Louvain community detection and high-density (suspected gang) community stats.

A community's illicit density = illicit share among its labeled nodes (not all nodes);
enrichment is relative to the global labeled illicit rate (~9.76%). Communities smaller
than the threshold are ignored to reduce small-sample noise.
"""
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def build_nx_graph(edge_index, num_nodes):
    """Convert a PyG edge_index into an undirected networkx graph."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    src, dst = edge_index.cpu().tolist()
    G.add_edges_from(zip(src, dst))
    return G


def analyze_community(data, min_size=None):
    """Run Louvain; returns (community stats DataFrame, global labeled illicit rate)."""
    min_size = min_size or config.MIN_COMMUNITY_SIZE
    G = build_nx_graph(data.edge_index, data.num_nodes)
    print("  running Louvain (~1-2 min for ~200k nodes)...")
    partition = community_louvain.best_partition(G, random_state=42)

    y = data.y.cpu().numpy()
    labeled = y != -1

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    global_rate = float((y == 1).sum() / labeled.sum()) if labeled.sum() else 0.0

    rows = []
    for cid, nodes in communities.items():
        nodes = np.asarray(nodes)
        size = len(nodes)
        if size < min_size:
            continue
        lab = labeled[nodes]
        n_labeled = int(lab.sum())
        n_illicit = int((y[nodes] == 1).sum())
        density = n_illicit / n_labeled if n_labeled > 0 else 0.0
        enrichment = density / global_rate if global_rate > 0 else 0.0
        rows.append({
            "community": cid,
            "size": size,
            "n_labeled": n_labeled,
            "n_illicit": n_illicit,
            "illicit_density": round(density, 4),
            "enrichment_vs_global": round(enrichment, 2),
        })

    df = pd.DataFrame(rows).sort_values("illicit_density", ascending=False)
    return df, global_rate


def save_community_stats(df, path):
    """Save community stats to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_communities(df, global_rate, fig_dir):
    """Left: illicit-density histogram (with global baseline); right: enrichment vs size."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["illicit_density"], bins=40, color="#5b8ff9", edgecolor="white")
    axes[0].axvline(global_rate, color="#e8684a", ls="--", lw=2,
                    label=f"global baseline {global_rate:.3f}")
    axes[0].set_xlabel("illicit density (labeled nodes)")
    axes[0].set_ylabel("number of communities")
    axes[0].set_title("Community illicit density")
    axes[0].legend()

    top = df.head(20)
    rest = df.iloc[20:]
    axes[1].scatter(rest["size"], rest["enrichment_vs_global"], s=10,
                    color="#5b8ff9", alpha=0.5, label="other communities")
    axes[1].scatter(top["size"], top["enrichment_vs_global"], s=30,
                    color="#e8684a", label="top-20 density")
    axes[1].axhline(1.0, color="grey", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("community size (log)")
    axes[1].set_ylabel("enrichment vs global baseline")
    axes[1].set_title("Community enrichment")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "community.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
