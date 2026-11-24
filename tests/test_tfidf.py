import math

import numpy as np

from summarizer.tfidf import build_tfidf, similarity_matrix


def test_matches_assignment_formula():
    tfidf = build_tfidf([["cat", "cat", "dog"], ["dog", "fish"], ["dog"]])

    assert tfidf.vocabulary == ["cat", "dog", "fish"]
    assert tfidf.values.shape == (3, 3)  # rows = words, columns = sentences
    expected = np.array([
        [2 * math.log(3 / 1), 0, 0],  # cat: TF 2 in s0, in 1 of 3 sentences
        [0, 0, 0],                    # dog is in every sentence -> IDF 0
        [0, math.log(3 / 1), 0],      # fish
    ])
    np.testing.assert_allclose(tfidf.values, expected)


def test_sentence_vectors_are_columns():
    tfidf = build_tfidf([["a"], ["b"]])
    np.testing.assert_array_equal(tfidf.sentence_vectors, tfidf.values.T)


def test_similarity_handles_empty_sentences():
    similarity = similarity_matrix(build_tfidf([["a", "b"], ["a", "c"], []]))
    assert similarity.shape == (3, 3)
    assert similarity[2].tolist() == [0.0, 0.0, 0.0]
    assert 0 < similarity[0, 1] < 1
