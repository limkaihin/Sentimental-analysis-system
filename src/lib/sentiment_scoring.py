from __future__ import annotations
from typing import List, Dict
import unicodedata

# Simple lists of negators and intensifiers to improve lexicon behavior.
NEGATORS = {"not", "no", "never", "without", "hardly", "scarcely", "n't"}
INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "too": 1.2,
    "slightly": 0.8, "somewhat": 0.9, "barely": 0.7, "quite": 1.2
}

def _norm(tok: str) -> str:
    """
    Lowercase and normalize a token for word-lexicon lookup.
    Emoticon lookup uses the original raw token to preserve symbols.
    """
    return unicodedata.normalize("NFKC", tok).strip().lower()

def calculate_window_sentiment(window: List[str], afinn: Dict[str, int], emoticons: Dict[str, int]) -> int:
    """
    Compute sentiment for a token window using:
    - AFINN word scores (case-insensitive, normalized).
    - Emoticon scores (exact symbol match, raw token).
    - Negation: flips sentiment if a negator appears among the previous 3 tokens.
    - Intensifiers: multiplicative scaling based on recent intensifier tokens.
    Returns an integer so downstream summations and comparisons are simple.
    """
    tokens = [_norm(t) for t in window if t.strip()]
    score = 0.0
    for i, tok in enumerate(tokens):
        raw = window[i]
        # Emoticons/emoji: check raw form first.
        if raw in emoticons:
            score += emoticons[raw]
            continue

        base = afinn.get(tok, 0)
        if base == 0:
            continue

        # Look back a few tokens for negators and intensifiers.
        scope_start = max(0, i - 3)
        negated = any(t in NEGATORS for t in tokens[scope_start:i])
        intensity = 1.0
        for t in tokens[scope_start:i]:
            intensity *= INTENSIFIERS.get(t, 1.0)

        val = base * intensity
        if negated:
            val = -val
        score += val

    return int(round(score))
