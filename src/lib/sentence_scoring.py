# src/lib/sentence_scoring.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from src.lib.preprocessing import split_sentences
from src.lib.sentiment_scoring import calculate_window_sentiment

def sentences_and_scores(
    text: str,
    afinn: Dict[str, int],
    emoticons: Dict[str, int],
) -> Tuple[List[List[str]], List[int]]:
    sents = split_sentences(text)
    scores = [calculate_window_sentiment(tokens, afinn, emoticons, debug=False) for tokens in sents]
    return sents, scores

def most_positive_negative_sentence(scores: List[int]) -> Tuple[Optional[int], Optional[int]]:
    if not scores:
        return None, None
    max_idx = max(range(len(scores)), key=lambda i: scores[i])
    min_idx = min(range(len(scores)), key=lambda i: scores[i])
    return max_idx, min_idx
