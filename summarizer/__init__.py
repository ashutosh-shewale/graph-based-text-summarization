"""Extractive and graph-based text summarization.

Pipeline: preprocessing -> TF-IDF -> sentence similarity graph -> PageRank,
optionally followed by MMR re-ranking, or by k-means clustering plus
word-graph sentence fusion.
"""

from summarizer.pipeline import (
    summarize_clusters,
    summarize_mmr,
    summarize_pagerank,
)

__all__ = ["summarize_pagerank", "summarize_mmr", "summarize_clusters"]
