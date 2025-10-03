# sentiment_scoring.py
# Purpose: Score tokens and sentences using AFINN + emoticon dictionaries.

from __future__ import annotations
from typing import Iterable
from .preprocessing import split_sentences

def calculate_window_sentiment(tokens: Iterable[str], afinn: dict[str, int], 
                               emot: dict[str, int], debug: bool = False) -> int:
    """
    Sum sentiment scores for tokens using emoticon dict first, then AFINN.
    """
    score = 0
    for tok in tokens:
        key = tok.lower()
        if key in emot:
            score += emot[key]
        elif key in afinn:
            score += afinn[key]
    return score

def sentence_scores(text: str, afinn: dict[str, int], emot: dict[str, int]) -> list[int]:
    """
    Return one integer score per sentence.
    Sentences are tokenized via split_sentences, which applies segmentation if enabled.
    """
    sents = split_sentences(text)
    return [calculate_window_sentiment(t, afinn, emot, debug=False) for t in sents]
