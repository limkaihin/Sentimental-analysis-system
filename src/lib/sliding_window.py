from __future__ import annotations
from typing import List, Tuple, Dict, Literal
from lib.preprocessing import tokenize, split_sentences
from lib.sentiment_scoring import calculate_window_sentiment

Unit = Literal["token", "sentence"]

def sliding_window_sentiment_analysis(
    reviews: List[str],
    k: int,
    afinn: Dict[str, int],
    emoticons: Dict[str, int],
    unit: Unit = "token",  # default preserves previous token-based behavior
) -> List[List[Tuple[int, int]]]:
    """
    Compute sentiment windows for each review.

    - unit='token':
        Tokenize the review, slide a length-k token window, score each window.
        start_index refers to token index.

    - unit='sentence':
        Split review into tokenized sentences; build windows of k consecutive
        sentences; score each window by summing the sentence scores.
        start_index refers to sentence index.

    Returns
    -------
    A list (per review) of (start_index, window_score) tuples.
    """
    results: List[List[Tuple[int, int]]] = []
    if k <= 0:
        return [[] for _ in reviews]

    for review in reviews:
        if unit == "token":
            tokens = tokenize(review)
            n = len(tokens)
            if n < k:
                results.append([])
                continue

            cur: List[Tuple[int, int]] = []
            for i in range(n - k + 1):
                window = tokens[i : i + k]
                score = calculate_window_sentiment(window, afinn, emoticons)
                cur.append((i, score))
            results.append(cur)

        else:  # unit == "sentence"
            sents: List[List[str]] = split_sentences(review)  # each item is a tokenized sentence
            s = len(sents)
            if s < k:
                results.append([])
                continue

            cur: List[Tuple[int, int]] = []
            for i in range(s - k + 1):
                window_sents = sents[i : i + k]
                # Sum sentiment per sentence in the window
                score = 0
                for sent in window_sents:
                    score += calculate_window_sentiment(sent, afinn, emoticons)
                cur.append((i, score))
            results.append(cur)

    return results
