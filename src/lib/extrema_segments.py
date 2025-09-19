from __future__ import annotations
from typing import List, Optional, Tuple

# Segment is (start_index, end_index_inclusive, sum_of_window)
Segment = Optional[Tuple[int, int, int]]

def extrema_segments(windows: List[Tuple[int, int]], k: int) -> Tuple[Segment, Segment]:
    """
    Given [(start, score), ...] and window size k, return:
    - Most positive segment (max sum).
    - Most negative segment (min sum).
    Each as (start, end_inclusive, sum). Returns (None, None) if no windows.
    """
    if not windows or k <= 0:
        return None, None

    best_pos = max(windows, key=lambda x: x[1])
    best_neg = min(windows, key=lambda x: x[1])

    s_pos, v_pos = best_pos
    s_neg, v_neg = best_neg

    return (s_pos, s_pos + k - 1, v_pos), (s_neg, s_neg + k - 1, v_neg)
