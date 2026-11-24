import numpy as np

from summarizer.pagerank import build_similarity_graph, pagerank_scores, top_n_indices

# Sentence 1 is strongly similar to everyone else, so it should be the most central.
SIMILARITY = np.array([
    [1.0, 0.8, 0.1, 0.0],
    [0.8, 1.0, 0.7, 0.6],
    [0.1, 0.7, 1.0, 0.1],
    [0.0, 0.6, 0.1, 1.0],
])


def test_graph_is_complete_without_self_loops():
    graph = build_similarity_graph(SIMILARITY)
    assert graph.number_of_edges() == 4 * 3 // 2
    assert all(u != v for u, v in graph.edges)
    assert graph[0][1]["weight"] == 0.8


def test_most_connected_sentence_ranks_first():
    scores = pagerank_scores(SIMILARITY)
    assert int(np.argmax(scores)) == 1
    assert np.isclose(scores.sum(), 1.0)


def test_top_n_is_returned_in_document_order():
    scores = np.array([0.1, 0.4, 0.2, 0.3])
    assert top_n_indices(scores, 2) == [1, 3]
