import pytest
from sentiment_scoring import calculate_window_sentiment

def test_scoring_basic_words():
    afinn = {"good": 3, "bad": -3, "great": 3}
    emot = {}
    assert calculate_window_sentiment(["good"], afinn, emot) > 0
    assert calculate_window_sentiment(["bad"], afinn, emot) < 0
    assert calculate_window_sentiment(["great", "movie"], afinn, emot) >= 3

def test_scoring_emoticons():
    afinn = {}
    emot = {":)": 2, ":-(": -2}
    assert calculate_window_sentiment([":)"], afinn, emot) == 2
    assert calculate_window_sentiment([":-("], afinn, emot) == -2

def test_scoring_case_and_punct():
    afinn = {"amazing": 4}
    emot = {}
    assert calculate_window_sentiment(["Amazing", "!"], afinn, emot) >= 4

def test_scoring_negation_and_intensity():
    afinn = {"good": 3}
    emot = {}
    pos = calculate_window_sentiment(["very", "good"], afinn, emot)
    neg = calculate_window_sentiment(["not", "good"], afinn, emot)
    assert pos >= 3
    assert neg <= -2
