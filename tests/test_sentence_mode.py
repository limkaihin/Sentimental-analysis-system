import pytest
from lib.sliding_window import sliding_window_sentiment_analysis

def test_sentence_windows_basic_independent():
    # Minimal inline lexicons (no external files)
    afinn = {"good": 3, "bad": -3, "ending": 0, "movie": 0, "acting": 0, "very": 0}
    emot = {}

    # Three short sentences; windows of k=2 sentences
    text = "Good movie. Bad acting. Very good ending!"
    out = sliding_window_sentiment_analysis(
        [text], k=2, afinn=afinn, emoticons=emot, unit="sentence"
    )

    # Starts should be [0, 1] for two windows over three sentences
    starts = [s for s, _ in out[0]]
    assert starts == [0, 1]

    # Scores: [good + bad] then [bad + very+good]
    scores = [v for _, v in out[0]]
    assert len(scores) == 2
    assert scores[0] < scores[1]  # second window should be more positive

def test_sentence_windows_short_text_independent():
    afinn = {"good": 3}
    emot = {}
    # Only one sentence -> with k=2 should return empty window list
    text = "Good!"
    out = sliding_window_sentiment_analysis(
        [text], k=2, afinn=afinn, emoticons=emot, unit="sentence"
    )
    assert out == [[]]

def test_sentence_vs_token_indices_independent():
    afinn = {"amazing": 4}
    emot = {}
    text = "Amazing!! Wow."
    # sentence mode k=1 -> one window per sentence
    out_sent = sliding_window_sentiment_analysis(
        [text], k=1, afinn=afinn, emoticons=emot, unit="sentence"
    )
    assert len(out_sent[0]) == 2  # two sentences

    # token mode k=1 -> one window per token (many more)
    out_tok = sliding_window_sentiment_analysis(
        [text], k=1, afinn=afinn, emoticons=emot, unit="token"
    )
    assert len(out_tok[0]) >= len(out_sent[0])  # token windows >= sentence windows
