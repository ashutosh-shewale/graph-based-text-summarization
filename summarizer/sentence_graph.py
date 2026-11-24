"""Word-graph sentence fusion (Day-2 Task 2).

Two related sentences are merged into a directed graph whose nodes are word
bigrams. Bigrams the sentences share become the same node, so a walk from
START to END can switch from one sentence to the other at those shared points
and produce a new sentence.
"""

import random
from typing import Optional, Sequence

import networkx as nx

START = "<START>"
END = "<END>"
MIN_COMMON_BIGRAMS = 3

Bigram = tuple[str, str]


def bigrams(tokens: Sequence[str]) -> list[Bigram]:
    """Consecutive word pairs, in sentence order."""
    return list(zip(tokens, tokens[1:]))


def common_bigrams(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> set[Bigram]:
    """Bigrams that occur in both sentences."""
    return set(bigrams(tokens_a)) & set(bigrams(tokens_b))


def find_partner(
    anchor_tokens: Sequence[str],
    candidates: dict[int, Sequence[str]],
    min_common: int = MIN_COMMON_BIGRAMS,
) -> Optional[int]:
    """Pick S2: the candidate sharing the most bigrams with S1 (at least ``min_common``).

    Returns ``None`` when no candidate qualifies. If several tie, the one with
    the lowest key (earliest in the document) wins.
    """
    best_index, best_overlap = None, min_common - 1
    for index, tokens in sorted(candidates.items()):
        overlap = len(common_bigrams(anchor_tokens, tokens))
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
    return best_index


def build_word_graph(token_sequences: Sequence[Sequence[str]]) -> nx.DiGraph:
    """Directed bigram graph START -> b_1 -> ... -> b_m -> END for each sentence.

    The assignment's pseudo-code only links the last bigram to END. Here the
    last bigram is also linked from its predecessor, so the chain is never broken.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from([START, END])
    for tokens in token_sequences:
        chain = bigrams(tokens)
        if chain:
            nx.add_path(graph, [START, *chain, END])
    return graph


def random_path(graph: nx.DiGraph, rng: random.Random) -> Optional[list[Bigram]]:
    """Random START -> END path that never revisits a node, without the dummy nodes.

    Uses depth-first search with shuffled successors: each run follows a random
    route, and backtracks from dead ends, so a path is found whenever one exists.
    Visiting each node at most once also stops the walk looping on repeated bigrams.
    """

    def shuffled_successors(node) -> list:
        successors = list(graph.successors(node))
        rng.shuffle(successors)
        return successors

    path, visited = [START], {START}
    pending = [shuffled_successors(START)]
    while path:
        if not pending[-1]:
            path.pop()
            pending.pop()
            continue
        nxt = pending[-1].pop()
        if nxt == END:
            return path[1:]
        if nxt not in visited:
            visited.add(nxt)
            path.append(nxt)
            pending.append(shuffled_successors(nxt))
    return None


def path_to_words(path: Sequence[Bigram]) -> list[str]:
    """Collapse a bigram path back into words (each bigram adds its second word)."""
    return [path[0][0]] + [second for _, second in path]


def fuse_sentences(
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    rng: random.Random,
    max_attempts: int = 25,
) -> list[str]:
    """Build the word graph of two sentences and read off a random START -> END path.

    A shared bigram near the end of one sentence can create a shortcut to END
    (e.g. "heavy rain caused ..." + "... hours of heavy rain" can yield just
    "heavy rain"). Such fragments are rejected: paths are re-sampled until the
    result is at least as long as the shorter input, falling back to S1.
    """
    graph = build_word_graph([tokens_a, tokens_b])
    min_words = min(len(tokens_a), len(tokens_b))
    for _ in range(max_attempts):
        path = random_path(graph, rng)
        if path and len(path) + 1 >= min_words:
            return path_to_words(path)
    return list(tokens_a)
