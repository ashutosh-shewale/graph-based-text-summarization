"""TF-IDF sentence representation (Day-1 Task 2), implemented from scratch.

The assignment's formula is used instead of sklearn's ``TfidfVectorizer``,
which smooths IDF and L2-normalises rows by default:

    TF(w, s)   = count of w in sentence s
    IDF(w)     = log(N / number of sentences containing w)
    TFIDF(w,s) = TF(w, s) * IDF(w)
"""

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class TfidfMatrix:
    """Word-by-sentence TF-IDF matrix: ``values[i, j]`` = TF-IDF of word i in sentence j."""

    vocabulary: list[str]
    values: np.ndarray

    @property
    def sentence_vectors(self) -> np.ndarray:
        """One row per sentence (the transpose), the shape most algorithms expect."""
        return self.values.T


def term_frequencies(token_lists: Sequence[Sequence[str]], vocabulary: list[str]) -> np.ndarray:
    """Raw counts, shaped (words, sentences)."""
    word_row = {word: row for row, word in enumerate(vocabulary)}
    counts = np.zeros((len(vocabulary), len(token_lists)))
    for column, tokens in enumerate(token_lists):
        for word, count in Counter(tokens).items():
            counts[word_row[word], column] = count
    return counts


def inverse_document_frequency(counts: np.ndarray) -> np.ndarray:
    """IDF per word; a word found in every sentence gets 0 (it cannot discriminate)."""
    num_sentences = counts.shape[1]
    document_frequency = np.count_nonzero(counts, axis=1)
    return np.log(num_sentences / document_frequency)


def build_tfidf(token_lists: Sequence[Sequence[str]]) -> TfidfMatrix:
    """Build the TF-IDF matrix for a document given each sentence's tokens."""
    vocabulary = sorted({word for tokens in token_lists for word in tokens})
    counts = term_frequencies(token_lists, vocabulary)
    idf = inverse_document_frequency(counts)
    # Broadcasting multiplies each word's row of counts by that word's IDF.
    return TfidfMatrix(vocabulary=vocabulary, values=counts * idf[:, np.newaxis])


def similarity_matrix(tfidf: TfidfMatrix) -> np.ndarray:
    """Pairwise cosine similarity between sentences (0 for all-zero vectors)."""
    return cosine_similarity(tfidf.sentence_vectors)
