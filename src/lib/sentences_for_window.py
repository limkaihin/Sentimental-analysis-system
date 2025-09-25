"""
Helper to extract exact sentences covered by a sliding window segment.
Drop this file into your src/lib/ directory (or anywhere importable).
"""

from typing import List
from src.lib.preprocessing import split_sentences

def sentences_for_window(text: str, start: int, end: int) -> List[str]:
    """
    Given the original text and a (start, end) window of sentence indices,
    return the exact sentence strings for that window.

    Args:
        text: full review/document string
        start: starting sentence index (inclusive)
        end: ending sentence index (inclusive)

    Returns:
        List of sentences (joined tokens) from start..end
    """
    sent_tokens = split_sentences(text)  # List[List[str]]
    return [" ".join(toks) for toks in sent_tokens[start:end + 1]]
