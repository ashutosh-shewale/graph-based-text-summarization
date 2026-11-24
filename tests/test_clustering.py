import random

import numpy as np
import pytest

from summarizer.clustering import (
    closest_to_centroid,
    kmeans_cosine,
    kmeans_sklearn,
    recompute_centroids,
)

# Two clear topics: the first three vectors use dimensions 0-1, the rest use 2-3.
VECTORS = np.array([
    [3.0, 1.0, 0.0, 0.0],
    [2.0, 2.0, 0.0, 0.0],
    [1.0, 3.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 1.0],
    [0.0, 0.0, 1.0, 3.0],
])


def _groups(labels) -> set[frozenset[int]]:
    return {frozenset(np.flatnonzero(labels == c).tolist()) for c in set(labels.tolist())}


@pytest.mark.parametrize("seed", range(5))
def test_scratch_kmeans_separates_topics(seed):
    clustering = kmeans_cosine(VECTORS, k=2, rng=random.Random(seed))
    assert _groups(clustering.labels) == {frozenset({0, 1, 2}), frozenset({3, 4})}


def test_sklearn_kmeans_agrees():
    clustering = kmeans_sklearn(VECTORS, k=2, seed=0)
    assert _groups(clustering.labels) == {frozenset({0, 1, 2}), frozenset({3, 4})}


def test_empty_cluster_keeps_previous_centroid():
    previous = np.array([[1.0, 0, 0, 0], [9.0, 9, 9, 9]])
    labels = np.zeros(len(VECTORS), dtype=int)  # nobody in cluster 1
    centroids = recompute_centroids(VECTORS, labels, previous)
    np.testing.assert_array_equal(centroids[1], previous[1])
    assert not np.isnan(centroids).any()


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        kmeans_cosine(VECTORS, k=6, rng=random.Random(0))


def test_closest_to_centroid():
    centroid = np.array([1.0, 1.0, 0.0, 0.0])
    assert closest_to_centroid(VECTORS, [0, 1, 2], centroid, random.Random(0)) == 1
