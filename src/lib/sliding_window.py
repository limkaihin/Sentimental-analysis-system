from __future__ import annotations
from typing import List, Tuple, Dict, Literal
from src.lib.preprocessing import tokenize, split_sentences
from src.lib.sentiment_scoring import calculate_window_sentiment

Unit = Literal["token", "sentence"]
Window = Tuple[int, int, int]  # (start, end, score)

def sliding_window_sentiment_analysis(
    reviews: List[str],
    k: int,
    afinn: Dict[str, int],
    emoticons: Dict[str, int],
    unit: Unit = "token",
    debug: bool = False,
) -> List[List[Window]]:
    results: List[List[Window]] = []
    if k <= 0:
        return [[] for _ in reviews]

    for review in reviews:
        if unit == "token":
            tokens = tokenize(review)
            n = len(tokens)
            if n < k:
                results.append([])
                continue
            cur: List[Window] = []
            for i in range(n - k + 1):
                window = tokens[i:i+k]
                s = calculate_window_sentiment(window, afinn, emoticons, debug=(debug and i == 0))
                cur.append((i, i + k - 1, s))
            results.append(cur)
        else:
            # sentence mode: split into tokenized sentences first
            sent_tokens: List[List[str]] = split_sentences(review)
            s = len(sent_tokens)
            if s < k:
                results.append([])
                continue
            cur: List[Window] = []
            for i in range(s - k + 1):
                window_sents = sent_tokens[i:i+k]
                total = 0
                for sent in window_sents:
                    total += calculate_window_sentiment(sent, afinn, emoticons, debug=False)
                cur.append((i, i + k - 1, total))
            results.append(cur)

    return results
