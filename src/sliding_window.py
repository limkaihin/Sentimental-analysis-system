from __future__ import annotations
from typing import List, Tuple, Dict
from preprocessing import tokenize
from sentiment_scoring import calculate_window_sentiment

def sliding_window_sentiment_analysis(
    reviews: List[str],
    k: int,
    afinn: Dict[str, int],
    emoticons: Dict[str, int],
) -> List[List[Tuple[int, int]]]:
    """
    For each review string:
    - Tokenize text (keeps punctuation/emoticons).
    - Slide a length-k window across tokens.
    - Score each window using calculate_window_sentiment.
    Returns: list (per review) of (start_index, window_score).
    """
    results: List[List[Tuple[int, int]]] = []
    for review in reviews:
        tokens = tokenize(review)
        n = len(tokens)
        if k <= 0 or n < k:
            results.append([])
            continue

        cur: List[Tuple[int, int]] = []
        for i in range(n - k + 1):
            window = tokens[i : i + k]
            score = calculate_window_sentiment(window, afinn, emoticons)
            cur.append((i, score))
        results.append(cur)
    return results
