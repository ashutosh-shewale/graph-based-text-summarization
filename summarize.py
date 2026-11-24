"""Command-line entry point.

Example:
    python summarize.py data/inaugural_speech.txt --method all -n 4 --seed 42
"""

import argparse
from pathlib import Path

from summarizer.clustering import KMEANS_METHODS
from summarizer.nltk_setup import ensure_nltk_data
from summarizer.pipeline import (
    Document,
    summarize_clusters,
    summarize_mmr,
    summarize_pagerank,
)
from summarizer.preprocessing import SPLIT_MODES

OUTPUT_FILES = {
    "pagerank": "Summary_PR.txt",
    "mmr": "Summary_MMR.txt",
    "cluster": "Summary_SentenceGraph.txt",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Graph-based extractive text summarization.")
    parser.add_argument("input", type=Path, help="text file to summarize")
    parser.add_argument("--method", choices=[*OUTPUT_FILES, "all"], default="all")
    parser.add_argument("-n", "--num-sentences", type=int, default=3,
                        help="summary length (also the number of clusters K)")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.7,
                        help="MMR trade-off: 1 = importance only, 0 = diversity only")
    parser.add_argument("--split", choices=SPLIT_MODES, default="line",
                        help="'line': one sentence per line (assignment); 'sentence': NLTK Punkt")
    parser.add_argument("--kmeans", choices=KMEANS_METHODS, default="scratch")
    parser.add_argument("--seed", type=int, default=42, help="seed for all random choices")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--plot", action="store_true", help="also save graph figures to --out")
    return parser.parse_args()


def build_summaries(document: Document, args: argparse.Namespace) -> dict[str, list[str]]:
    """Run the requested summarizers and return their sentences keyed by method."""
    methods = OUTPUT_FILES if args.method == "all" else [args.method]
    n = args.num_sentences
    runners = {
        "pagerank": lambda: [s.original for s in summarize_pagerank(document, n)],
        "mmr": lambda: [s.original for s in summarize_mmr(document, n, args.lambda_)],
        "cluster": lambda: [c.text for c in summarize_clusters(document, n, args.seed, args.kmeans)],
    }
    return {method: runners[method]() for method in methods}


def write_summary(path: Path, sentences: list[str]) -> None:
    """Write one summary sentence per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def save_figures(document: Document, out_dir: Path) -> None:
    """Save the sentence-similarity graph (imported lazily: matplotlib is slow to load)."""
    from summarizer.visualize import plot_similarity_graph

    plot_similarity_graph(document.similarity, document.scores, out_dir / "similarity_graph.png")


def main() -> None:
    args = parse_args()
    ensure_nltk_data()
    document = Document.from_text(args.input.read_text(encoding="utf-8"), args.split)
    print(f"{args.input}: {len(document.sentences)} sentences, "
          f"{len(document.tfidf.vocabulary)} distinct words\n")

    for method, sentences in build_summaries(document, args).items():
        path = args.out / OUTPUT_FILES[method]
        write_summary(path, sentences)
        print(f"== {method} -> {path}")
        print("\n".join(sentences), end="\n\n")

    if args.plot:
        save_figures(document, args.out)


if __name__ == "__main__":
    main()
