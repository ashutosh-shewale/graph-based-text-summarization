"""Shared test setup: make sure NLTK data is available before any test runs."""

import sys
from pathlib import Path

# Let tests import the package without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarizer.nltk_setup import ensure_nltk_data  # noqa: E402

ensure_nltk_data()
