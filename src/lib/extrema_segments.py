from __future__ import annotations
from typing import List, Tuple, Optional

Window = Tuple[int, int, int]      # (start, end, score)
Segment = Optional[Tuple[int, int, int]]

def extrema_segments(windows: List[Tuple[int, ...]], k: int) -> Tuple[Segment, Segment]:
    """
    Return the most positive and most negative segments from a window list.
    Accepts:
      - Triples (start, end, score) [preferred]
      - Pairs (start, score) [legacy]; end is inferred as start + k - 1
    """
    if not windows:
        return None, None

    # Normalize to triples if pairs were provided
    if isinstance(windows[0], tuple) and len(windows[0]) == 2:
        windows = [(s, s + k - 1, v) for (s, v) in windows]  # type: ignore[list-item]

    best_pos: Segment = None
    best_neg: Segment = None

    for start, end, s in windows:  # type: ignore[misc]
        # Best positive
        if best_pos is None:
            best_pos = (start, end, s)
        else:
            bs, be, bv = best_pos
            if (s > bv) or (s == bv and (end - start > be - bs)) or (s == bv and end - start == be - bs and start < bs):
                best_pos = (start, end, s)

        # Best negative
        if best_neg is None:
            best_neg = (start, end, s)
        else:
            bs, be, bv = best_neg
            if (s < bv) or (s == bv and (end - start > be - bs)) or (s == bv and end - start == be - bs and start < bs):
                best_neg = (start, end, s)

    return best_pos, best_neg
