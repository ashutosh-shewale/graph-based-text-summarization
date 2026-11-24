"""Sentence ranking with PageRank on a sentence-similarity graph (Day-1 Task 3)."""

import networkx as nx
import numpy as np


def build_similarity_graph(similarity: np.ndarray) -> nx.Graph:
    """Complete undirected graph: one node per sentence, edges weighted by cosine similarity.

    Self-loops are left out on purpose: every sentence has similarity 1 with
    itself, which would make each node "vote" for itself and flatten the ranking.
    """
    graph = nx.Graph()
    num_sentences = similarity.shape[0]
    graph.add_nodes_from(range(num_sentences))
    for u in range(num_sentences):
        for v in range(u + 1, num_sentences):
            graph.add_edge(u, v, weight=float(similarity[u, v]))
    return graph


def pagerank_scores(similarity: np.ndarray, damping: float = 0.85) -> np.ndarray:
    """PageRank score of every sentence, indexed by sentence position."""
    graph = build_similarity_graph(similarity)
    scores = nx.pagerank(graph, alpha=damping, weight="weight")
    return np.array([scores[node] for node in range(similarity.shape[0])])


def top_n_indices(scores: np.ndarray, n: int) -> list[int]:
    """Indices of the ``n`` highest-scoring sentences, returned in document order."""
    # A stable sort on -score keeps the earlier sentence first when scores tie.
    best = np.argsort(-scores, kind="stable")[:n]
    return sorted(int(index) for index in best)
