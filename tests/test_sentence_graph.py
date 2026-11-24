import random

import networkx as nx
import pytest

from summarizer.sentence_graph import (
    END,
    START,
    build_word_graph,
    common_bigrams,
    find_partner,
    fuse_sentences,
    path_to_words,
)

# The example from the assignment.
S1 = "dharara tower built in 1832 collapses in kathmandu during earthquake".split()
S2 = "historic dharara tower collapses in kathmandu after 7.9 earthquake".split()
EXPECTED_FUSION = "dharara tower built in 1832 collapses in kathmandu after 7.9 earthquake"


def _all_generated_sentences() -> set[str]:
    graph = build_word_graph([S1, S2])
    return {
        " ".join(path_to_words(path[1:-1]))
        for path in nx.all_simple_paths(graph, START, END)
    }


def test_shared_bigrams():
    assert common_bigrams(S1, S2) == {
        ("dharara", "tower"),
        ("collapses", "in"),
        ("in", "kathmandu"),
    }


def test_partner_needs_three_shared_bigrams():
    unrelated = "the weather was calm".split()
    assert find_partner(S1, {5: unrelated, 7: S2}) == 7
    assert find_partner(S1, {5: unrelated}) is None


def test_shared_bigrams_become_one_node():
    graph = build_word_graph([S1, S2])
    bigram_nodes = set(graph) - {START, END}
    assert len(bigram_nodes) == 9 + 8 - 3


def test_assignment_example_sentence_is_reachable():
    generated = _all_generated_sentences()
    assert EXPECTED_FUSION in generated
    assert " ".join(S1) in generated and " ".join(S2) in generated


@pytest.mark.parametrize("seed", range(10))
def test_random_fusion_is_a_valid_graph_path(seed):
    fused = " ".join(fuse_sentences(S1, S2, random.Random(seed)))
    assert fused in _all_generated_sentences()


@pytest.mark.parametrize("seed", range(10))
def test_fusion_rejects_shortcut_fragments(seed):
    # "heavy rain" ends sentence b, so START -> (heavy, rain) -> END is a valid
    # but useless two-word path.
    a = "heavy rain caused the river to flood the town on tuesday night".split()
    b = "the river flooded the town on tuesday night after hours of heavy rain".split()
    fused = fuse_sentences(a, b, random.Random(seed))
    assert len(fused) >= min(len(a), len(b))
