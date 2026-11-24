"""K-means sentence clustering with cosine similarity (Day-2 Task 1).

The from-scratch version follows the assignment step by step. A scikit-learn
version is included as a cross-check.
"""

import random
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

KMEANS_METHODS = ("scratch", "sklearn")


@dataclass(frozen=True)
class Clustering:
    """Cluster label per sentence plus the final centroids."""

    labels: np.ndarray
    centroids: np.ndarray

    def members(self, cluster: int) -> list[int]:
        """Sentence indices assigned to ``cluster``."""
        return [int(i) for i in np.flatnonzero(self.labels == cluster)]


def assign_to_nearest(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Label each vector with the centroid it has the highest cosine similarity to."""
    return np.argmax(cosine_similarity(vectors, centroids), axis=1)


def recompute_centroids(vectors: np.ndarray, labels: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Mean of each cluster's members.

    A cluster that lost all its members keeps its previous centroid, since
    the mean of an empty set is NaN and would break later steps.
    """
    centroids = previous.copy()
    for cluster in range(len(previous)):
        members = vectors[labels == cluster]
        if len(members):
            centroids[cluster] = members.mean(axis=0)
    return centroids


def kmeans_cosine(
    vectors: np.ndarray,
    k: int,
    rng: random.Random,
    max_iterations: int = 100,
) -> Clustering:
    """Cluster sentence vectors into ``k`` groups (from scratch).

    Initial centroids are ``k`` sentences drawn with ``random.sample``.
    Iteration stops when the assignment no longer changes. ``max_iterations``
    is a safety cap in case it never settles.
    """
    if not 1 <= k <= len(vectors):
        raise ValueError(f"k must be between 1 and {len(vectors)}, got {k}")

    centroids = vectors[rng.sample(range(len(vectors)), k)].astype(float)
    labels = assign_to_nearest(vectors, centroids)
    for _ in range(max_iterations):
        centroids = recompute_centroids(vectors, labels, centroids)
        new_labels = assign_to_nearest(vectors, centroids)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return Clustering(labels=labels, centroids=centroids)


def kmeans_sklearn(vectors: np.ndarray, k: int, seed: int) -> Clustering:
    """scikit-learn K-means on L2-normalised vectors.

    On unit vectors, Euclidean distance ranks points the same way as cosine
    similarity, so this is a fair baseline for the from-scratch version.
    """
    unit_vectors = normalize(vectors)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(unit_vectors)
    return Clustering(labels=model.labels_, centroids=model.cluster_centers_)


def closest_to_centroid(
    vectors: np.ndarray,
    members: list[int],
    centroid: np.ndarray,
    rng: random.Random,
) -> int:
    """The member sentence most similar to the centroid; ties are broken at random."""
    similarities = cosine_similarity(vectors[members], centroid[np.newaxis, :]).ravel()
    best = np.flatnonzero(np.isclose(similarities, similarities.max()))
    return members[rng.choice(list(best))]
