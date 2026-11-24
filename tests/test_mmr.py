import numpy as np
import pytest

from summarizer.mmr import mmr_select, normalize_scores

# Sentences 0 and 1 are near-duplicates and both important; sentence 2 is different.
IMPORTANCE = np.array([0.40, 0.38, 0.22])
SIMILARITY = np.array([
    [1.0, 0.95, 0.05],
    [0.95, 1.0, 0.05],
    [0.05, 0.05, 1.0],
])


def test_near_duplicate_is_replaced_by_diverse_sentence():
    assert mmr_select(IMPORTANCE, SIMILARITY, n=2, lambda_=0.5) == [0, 2]


def test_lambda_one_is_plain_ranking():
    assert mmr_select(IMPORTANCE, SIMILARITY, n=2, lambda_=1.0) == [0, 1]


def test_n_larger_than_document():
    assert sorted(mmr_select(IMPORTANCE, SIMILARITY, n=10)) == [0, 1, 2]


def test_invalid_lambda_rejected():
    with pytest.raises(ValueError):
        mmr_select(IMPORTANCE, SIMILARITY, n=1, lambda_=1.5)


def test_normalize_scores_constant_input():
    assert normalize_scores(np.array([0.2, 0.2])).tolist() == [1.0, 1.0]
