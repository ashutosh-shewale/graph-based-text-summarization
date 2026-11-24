"""End-to-end summarization pipelines built from the individual stages."""

import random
from dataclasses import dataclass, field

import numpy as np

from summarizer.clustering import Clustering, closest_to_centroid, kmeans_cosine, kmeans_sklearn
from summarizer.mmr import mmr_select
from summarizer.pagerank import pagerank_scores, top_n_indices
from summarizer.preprocessing import Sentence, preprocess_document
from summarizer.sentence_graph import find_partner, fuse_sentences
from summarizer.tfidf import TfidfMatrix, build_tfidf, similarity_matrix


@dataclass(frozen=True)
class Document:
    """A pre-processed document with its TF-IDF matrix, similarities and PageRank scores."""

    sentences: list[Sentence]
    tfidf: TfidfMatrix
    similarity: np.ndarray
    scores: np.ndarray = field(repr=False)

    @classmethod
    def from_text(cls, text: str, split_mode: str = "line") -> "Document":
        """Run preprocessing, TF-IDF and PageRank once so every summarizer can reuse them."""
        sentences = preprocess_document(text, split_mode)
        if not sentences:
            raise ValueError("The input document contains no sentences.")
        tfidf = build_tfidf([s.processed_tokens for s in sentences])
        similarity = similarity_matrix(tfidf)
        return cls(sentences, tfidf, similarity, pagerank_scores(similarity))


@dataclass(frozen=True)
class ClusterSentence:
    """One summary sentence produced from a cluster."""

    order: int          # index of S1 in the document; decides output order (Task 3)
    text: str
    anchor: int         # S1
    partner: int | None  # S2, or None when no sentence shared enough bigrams


def summarize_pagerank(document: Document, n: int) -> list[Sentence]:
    """Top-``n`` sentences by PageRank, in document order."""
    return [document.sentences[i] for i in top_n_indices(document.scores, n)]


def summarize_mmr(document: Document, n: int, lambda_: float = 0.7) -> list[Sentence]:
    """``n`` sentences picked by MMR (PageRank importance minus redundancy), in document order."""
    picked = mmr_select(document.scores, document.similarity, n, lambda_)
    return [document.sentences[i] for i in sorted(picked)]


def cluster_document(document: Document, k: int, seed: int, method: str = "scratch") -> Clustering:
    """Cluster sentence TF-IDF vectors with the chosen K-means implementation."""
    vectors = document.tfidf.sentence_vectors
    if method == "scratch":
        return kmeans_cosine(vectors, k, random.Random(seed))
    if method == "sklearn":
        return kmeans_sklearn(vectors, k, seed)
    raise ValueError(f"Unknown K-means method {method!r}")


def summarize_clusters(
    document: Document,
    k: int,
    seed: int = 42,
    method: str = "scratch",
) -> list[ClusterSentence]:
    """One sentence per cluster: S1 fused with S2 through the word graph when possible.

    Clusters are ordered by the position of their S1 in the original document.
    """
    k = min(k, len(document.sentences))
    rng = random.Random(seed)
    clustering = cluster_document(document, k, seed, method)

    summary = []
    for cluster in range(k):
        members = clustering.members(cluster)
        if members:
            summary.append(_summarize_cluster(document, clustering, cluster, members, rng))
    return sorted(summary, key=lambda item: item.order)


def _summarize_cluster(
    document: Document,
    clustering: Clustering,
    cluster: int,
    members: list[int],
    rng: random.Random,
) -> ClusterSentence:
    """Pick S1 and S2 for one cluster and build its summary sentence."""
    vectors = document.tfidf.sentence_vectors
    anchor = closest_to_centroid(vectors, members, clustering.centroids[cluster], rng)
    anchor_tokens = document.sentences[anchor].cleaned_tokens
    candidates = {i: document.sentences[i].cleaned_tokens for i in members if i != anchor}
    partner = find_partner(anchor_tokens, candidates)

    if partner is None:
        # Assignment rule: without an S2, keep S1 unchanged.
        text = document.sentences[anchor].original
    else:
        words = fuse_sentences(anchor_tokens, candidates[partner], rng)
        text = _as_sentence(words)
    return ClusterSentence(order=anchor, text=text, anchor=anchor, partner=partner)


def _as_sentence(words: list[str]) -> str:
    """Turn fused lowercase tokens into a readable sentence."""
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."
