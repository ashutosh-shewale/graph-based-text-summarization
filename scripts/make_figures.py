"""Regenerate the figures used in the README (docs/images/).

Run from the repository root:  python scripts/make_figures.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from summarizer.nltk_setup import ensure_nltk_data  # noqa: E402
from summarizer.pipeline import Document  # noqa: E402
from summarizer.sentence_graph import build_word_graph, random_path  # noqa: E402
from summarizer.visualize import plot_similarity_graph, plot_word_graph  # noqa: E402

IMAGES = ROOT / "docs" / "images"

# The assignment's word-graph example.
S1 = "dharara tower built in 1832 collapses in kathmandu during earthquake".split()
S2 = "historic dharara tower collapses in kathmandu after 7.9 earthquake".split()


def main() -> None:
    ensure_nltk_data()

    text = (ROOT / "data" / "inaugural_speech.txt").read_text(encoding="utf-8")
    document = Document.from_text(text, split_mode="sentence")
    plot_similarity_graph(document.similarity, document.scores, IMAGES / "similarity_graph.png")

    graph = build_word_graph([S1, S2])
    # Seed chosen so the highlighted path switches sentences (a genuinely new sentence).
    path = random_path(graph, random.Random(3))
    plot_word_graph(graph, IMAGES / "word_graph.png", highlight=path)
    print(f"Figures written to {IMAGES}")


if __name__ == "__main__":
    main()
