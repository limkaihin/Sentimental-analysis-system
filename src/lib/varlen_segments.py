# varlen_segments.py
# Purpose: Arbitrary-length best positive/negative segments via Kadane's algorithm.

from __future__ import annotations

def _max_subarray(scores: list[int]):
    """Kadane's algorithm for maximum sum subarray."""
    if not scores:
        return None
    best = cur = scores[0]
    start = end = s = 0
    
    for i in range(1, len(scores)):
        if cur + scores[i] < scores[i]:
            cur = scores[i]
            s = i
        else:
            cur += scores[i]
        
        if cur > best:
            best = cur
            start = s
            end = i
    
    return (start, end, best)

def _min_subarray(scores: list[int]):
    """Kadane's algorithm for minimum sum subarray."""
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

def best_varlen_segments_from_scores(scores: list[int]):
    """
    Return (best_positive_segment, best_negative_segment) as tuples (start, end, sum)
    or (None, None) if none exist.
    """
    if not scores:
        return None, None
    
    pos = _max_subarray(scores)
    neg = _min_subarray(scores)
    
    pos = pos if pos and pos[2] > 0 else None
    neg = neg if neg and neg[2] < 0 else None
    
    return pos, neg
