"""Optional figures: the sentence-similarity graph and a word graph."""

from pathlib import Path

import matplotlib

# A non-interactive backend writes PNGs without opening (and blocking on) a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from summarizer.pagerank import build_similarity_graph  # noqa: E402
from summarizer.sentence_graph import END, START  # noqa: E402


def plot_similarity_graph(similarity: np.ndarray, scores: np.ndarray, path: Path) -> None:
    """Draw sentences as nodes sized by PageRank, with edge thickness showing similarity."""
    graph = build_similarity_graph(similarity)
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges])
    widths = 0.3 + 4 * weights / weights.max() if len(weights) and weights.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(8, 8))
    layout = nx.circular_layout(graph)
    nx.draw_networkx_edges(graph, layout, width=widths, alpha=0.35, edge_color="#4a6fa5", ax=ax)
    nx.draw_networkx_nodes(
        graph, layout, node_size=4000 * scores / scores.max(), node_color="#f4a259", ax=ax
    )
    nx.draw_networkx_labels(graph, layout, labels={i: f"S{i}" for i in graph}, ax=ax)
    ax.set_title("Sentence similarity graph (node size = PageRank)")
    ax.axis("off")
    _save(fig, path)


def plot_word_graph(graph: nx.DiGraph, path: Path, highlight: list | None = None) -> None:
    """Draw a bigram word graph left to right; ``highlight`` marks a chosen path."""
    # Two-line labels keep neighbouring bigram boxes from overlapping.
    labels = {node: node if node in (START, END) else "\n".join(node) for node in graph}
    highlighted = set(highlight or [])
    colors = [
        "#f4a259" if node in (START, END) else "#8cb369" if node in highlighted else "#bcd4e6"
        for node in graph
    ]

    fig, ax = plt.subplots(figsize=(18, 6))
    layout = _layered_layout(graph) if nx.is_directed_acyclic_graph(graph) else nx.spring_layout(graph, seed=0)
    nx.draw_networkx(
        graph, layout, labels=labels, node_color=colors, node_shape="s",
        node_size=2200, font_size=8, arrowsize=14, edge_color="#777777", ax=ax,
    )
    ax.set_title("Word graph (nodes are bigrams; green = generated path)")
    ax.axis("off")
    _save(fig, path)


def _layered_layout(graph: nx.DiGraph) -> dict:
    """Place nodes in columns by their longest distance from START (requires a DAG)."""
    depth = {START: 0}
    for node in nx.topological_sort(graph):
        for successor in graph.successors(node):
            depth[successor] = max(depth.get(successor, 0), depth.get(node, 0) + 1)
    # Lay out a copy so the caller's graph is not given a "layer" attribute.
    layered = graph.copy()
    nx.set_node_attributes(layered, depth, "layer")
    return nx.multipartite_layout(layered, subset_key="layer")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
