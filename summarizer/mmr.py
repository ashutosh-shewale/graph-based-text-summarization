"""Maximal Marginal Relevance re-ranking (Day-2, redundancy removal).

    MMR = argmax_{D_i in R \\ S} [ lambda * Imp(D_i) - (1 - lambda) * max_{D_j in S} Sim(D_i, D_j) ]

Imp is the PageRank score and Sim is TF-IDF cosine similarity.
"""

import numpy as np


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max scale scores to [0, 1].

    PageRank scores sum to 1, so they are tiny next to cosine similarities in
    [0, 1]. Without rescaling, the redundancy penalty would dominate and lambda
    would lose its meaning.
    """
    spread = scores.max() - scores.min()
    if spread == 0:
        return np.ones_like(scores)
    return (scores - scores.min()) / spread


def mmr_select(
    importance: np.ndarray,
    similarity: np.ndarray,
    n: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Greedily pick ``n`` sentences that balance importance and novelty.

    Returns indices in selection order (most valuable first). ``lambda_`` = 1
    reduces to pure PageRank; ``lambda_`` = 0 maximises diversity only.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    relevance = normalize_scores(importance)
    remaining = list(range(len(importance)))
    selected: list[int] = []

    while remaining and len(selected) < n:
        def marginal_relevance(candidate: int) -> float:
            # Redundancy is the similarity to the *closest* already-picked
            # sentence; nothing is picked yet on the first round, so it is 0.
            redundancy = max((similarity[candidate, s] for s in selected), default=0.0)
            return lambda_ * relevance[candidate] - (1 - lambda_) * redundancy

        best = max(remaining, key=marginal_relevance)
        selected.append(best)
        remaining.remove(best)

    return selected
