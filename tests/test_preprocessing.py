import pytest

from summarizer.preprocessing import clean_tokens, preprocess_document, split_sentences

TOY_DOCUMENT = "I was walking down the street.\nI met my friends Rayan, George and Seni.\n"


def test_assignment_toy_example():
    first, second = preprocess_document(TOY_DOCUMENT)
    # The assignment example keeps "i"/"down"/"my", but they are NLTK stopwords,
    # so they are removed. The verb lemmas match the example: walking -> walk, met -> meet.
    assert first.processed_text == "walk street"
    assert second.processed_text == "meet friend rayan george seni"


def test_original_text_and_order_preserved():
    sentences = preprocess_document(TOY_DOCUMENT)
    assert [s.index for s in sentences] == [0, 1]
    assert sentences[1].original == "I met my friends Rayan, George and Seni."


def test_clean_tokens_strip_unicode_punctuation_and_keep_stopwords():
    assert clean_tokens("Our nation’s capital — today!") == ["our", "nations", "capital", "today"]


def test_blank_lines_are_skipped():
    assert split_sentences("one\n\n  \ntwo\n") == ["one", "two"]


def test_sentence_mode_joins_wrapped_lines():
    text = "This sentence wraps\nacross two lines. Second one."
    assert split_sentences(text, "sentence") == [
        "This sentence wraps across two lines.",
        "Second one.",
    ]


def test_unknown_split_mode_rejected():
    with pytest.raises(ValueError):
        split_sentences("x", "paragraph")
