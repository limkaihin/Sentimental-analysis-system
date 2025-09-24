# src/lib/varlen_segments.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from src.lib.preprocessing import split_sentences
from src.lib.sentiment_scoring import calculate_window_sentiment

Segment = Optional[Tuple[int, int, int]]  # (start_sentence_idx, end_sentence_idx, sum)

def sentence_scores(text: str, afinn: Dict[str, int], emoticons: Dict[str, int]) -> List[int]:
    sents = split_sentences(text)
    return [calculate_window_sentiment(tokens, afinn, emoticons, debug=False) for tokens in sents]

def _max_subarray(scores: List[int]) -> Segment:
    """
    Maximum-sum contiguous subarray with tie-breaks:
    - Restart on ties to prefer later starts.
    - Prefer shorter span when sums tie.
    """
    if not scores:
        return None
    best = cur = scores[0]
    start = end = s = 0
    for i in range(1, len(scores)):
        # Restart on tie to avoid dragging negatives into a positive span
        if cur + scores[i] <= scores[i]:
            cur = scores[i]
            s = i
        else:
            cur += scores[i]
        # Update on improvement; prefer shorter span on equal sums
        if (cur > best) or (cur == best and (i - s) < (end - start)):
            best = cur
            start = s
            end = i
    return (start, end, best)

def _min_subarray(scores: List[int]) -> Segment:
    """Minimum-sum contiguous subarray (standard implementation)."""
    if not scores:
        return None
    best = cur = scores[0]
    start = end = s = 0
    for i in range(1, len(scores)):
        if cur + scores[i] > scores[i]:
            cur = scores[i]
            s = i
        else:
            cur += scores[i]
        if cur < best:
            best = cur
            start = s
            end = i
    return (start, end, best)

def best_varlen_segments(text: str, afinn: Dict[str, int], emoticons: Dict[str, int]) -> Tuple[List[int], Segment, Segment]:
    scores = sentence_scores(text, afinn, emoticons)
    return scores, _max_subarray(scores), _min_subarray(scores)

def segment_sentences(text: str, seg: Segment) -> List[str]:
    if seg is None:
        return []
    start, end, _ = seg
    sent_tokens = split_sentences(text)  # List[List[str]]
    return [" ".join(toks) for toks in sent_tokens[start:end + 1]]
