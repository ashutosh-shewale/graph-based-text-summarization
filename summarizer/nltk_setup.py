"""Download the NLTK resources the project needs, only if they are missing."""

import nltk

# (lookup path, download id). Both old and new names of the tokenizer and
# tagger are listed because NLTK renamed them in 3.8.2+.
_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def ensure_nltk_data() -> None:
    """Download any missing NLTK corpora/models (no-op when already present)."""
    for path, package in _RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
