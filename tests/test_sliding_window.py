import pytest
from sliding_window import sliding_window_sentiment_analysis

def test_empty_reviews_list_returns_empty_lists():
    afinn = {"good": 3, "bad": -3}
    emot = {":)": 2}
    out = sliding_window_sentiment_analysis([], k=3, afinn=afinn, emoticons=emot)
    assert out == []

def test_k_zero_or_negative_yields_empty_windows():
    reviews = ["Good movie :)"]
    afinn = {"good": 3}
    emot = {":)": 2}
    assert sliding_window_sentiment_analysis(reviews, 0, afinn, emot) == [[]]
    assert sliding_window_sentiment_analysis(reviews, -1, afinn, emot) == [[]]

def test_review_shorter_than_k_gives_empty_windows():
    reviews = ["only two"]
    afinn = {"only": 0, "two": 0}
    emot = {}
    assert sliding_window_sentiment_analysis(reviews, 3, afinn, emot) == [[]]

def test_basic_window_counts_and_starts():
    reviews = ["a b c d e"]
    afinn = {}
    emot = {}
    out = sliding_window_sentiment_analysis(reviews, 2, afinn, emot)
    # Expect starts: 0,1,2,3 for k=2 over 5 tokens
    starts = [s for s, _ in out[0]]
    assert starts == [0, 1, 2, 3]

def test_scoring_integration_with_words_and_emoticons():
    reviews = ["good :)", "bad :-("]
    afinn = {"good": 3, "bad": -3}
    emot = {":)": 2, ":-(": -2}
    out = sliding_window_sentiment_analysis(reviews, 1, afinn, emot)
    # First review windows (k=1): ["good"], [":)"]
    first_scores = [v for _, v in out[0]]
    assert 3 in first_scores and 2 in first_scores
    # Second review windows (k=1): ["bad"], [":-("]
    second_scores = [v for _, v in out[1]]
    assert -3 in second_scores and -2 in second_scores

def test_consistent_tokenization_punctuation_kept():
    reviews = ["Amazing!!"]
    afinn = {"amazing": 4}
    emot = {}
    out = sliding_window_sentiment_analysis(reviews, 1, afinn, emot)
    # Tokens are ["Amazing", "!", "!"], so three windows
    assert len(out[0]) == 3
    # At least one window should carry positive score from "Amazing"
    assert any(score > 0 for _, score in out[0])
