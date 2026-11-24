"""Sentence segmentation and text normalisation (Day-1 Task 1).

Each sentence is kept in three forms:
- ``original``: the raw text, used when printing the final summary;
- ``cleaned_tokens``: lowercase, punctuation-free words *with* stopwords,
  used by the word graph so that fused sentences stay readable;
- ``processed_tokens``: stopwords removed and lemmatized, used for TF-IDF.
"""

import string
from dataclasses import dataclass
from functools import lru_cache

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# string.punctuation covers ASCII only; the sample texts also contain curly
# quotes and dashes, which would otherwise survive as separate "words".
_PUNCTUATION = string.punctuation + "‘’“”–—…"
_PUNCTUATION_TABLE = str.maketrans("", "", _PUNCTUATION)

SPLIT_MODES = ("line", "sentence")


@dataclass(frozen=True)
class Sentence:
    """One sentence of the input document in its raw and normalised forms."""

    index: int
    original: str
    cleaned_tokens: tuple[str, ...]
    processed_tokens: tuple[str, ...]

    @property
    def processed_text(self) -> str:
        """Pre-processed sentence as a single string (the Day-1 T1 output)."""
        return " ".join(self.processed_tokens)


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def split_sentences(document: str, mode: str = "line") -> list[str]:
    """Split a document into non-empty sentences.

    ``line`` follows the assignment (one sentence per newline);
    ``sentence`` uses NLTK's Punkt model for prose that wraps across lines.
    """
    if mode == "line":
        candidates = document.splitlines()
    elif mode == "sentence":
        candidates = sent_tokenize(" ".join(document.split()))
    else:
        raise ValueError(f"Unknown split mode {mode!r}; expected one of {SPLIT_MODES}")
    return [text.strip() for text in candidates if text.strip()]


def clean_tokens(text: str) -> list[str]:
    """Remove punctuation, lowercase and tokenize."""
    return word_tokenize(text.translate(_PUNCTUATION_TABLE).lower())


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank tag to the WordNet POS the lemmatizer expects.

    Without a POS the lemmatizer treats every word as a noun, so verbs such
    as "walking" or "met" would not reduce to "walk" / "meet".
    """
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Drop stopwords and lemmatize the remaining tokens.

    POS tagging runs on the full token list (before stopword removal) so the
    tagger still sees the grammatical context of each word.
    """
    stop_words = _stop_words()
    lemmatizer = _lemmatizer()
    return [
        lemmatizer.lemmatize(word, _wordnet_pos(tag))
        for word, tag in pos_tag(tokens)
        if word not in stop_words
    ]


def preprocess_document(document: str, split_mode: str = "line") -> list[Sentence]:
    """Segment and normalise a document, preserving sentence order."""
    sentences = []
    for index, text in enumerate(split_sentences(document, split_mode)):
        cleaned = clean_tokens(text)
        sentences.append(
            Sentence(
                index=index,
                original=text,
                cleaned_tokens=tuple(cleaned),
                processed_tokens=tuple(normalize_tokens(cleaned)),
            )
        )
    return sentences
