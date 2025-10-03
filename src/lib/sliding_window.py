# sliding_window.py
# Purpose: Fixed-size sliding windows over sentence scores; find best +/- windows.

from __future__ import annotations

def sliding_window_sentiment_over_sentences(scores: list[int], k: int) -> list[tuple[int, int, int]]:
    """
    Return [(start_idx, end_idx, sum)] for every k-sentence window.
    """
    if k <= 0 or not scores or len(scores) < k:
        return []
    
    out: list[tuple[int, int, int]] = []
    cur = sum(scores[:k])
    out.append((0, k - 1, cur))
    
    for i in range(k, len(scores)):
        cur += scores[i] - scores[i - k]
        out.append((i - k + 1, i, cur))
    
    return out

def extrema_segments(windows: list[tuple[int, int, int]]):
    """
    Return (best_positive_window, best_negative_window) or (None, None).
    """
    if not windows:
        return None, None
    
    best_pos = max(windows, key=lambda t: t[2])
    best_neg = min(windows, key=lambda t: t[2])
    
    return (best_pos if best_pos[2] > 0 else None,
            best_neg if best_neg[2] < 0 else None)
