from __future__ import annotations
from typing import Dict, List
import unicodedata
import string

# Negation and intensity heuristics
NEGATORS = {"not", "no", "never", "without", "hardly", "scarcely", "n't"}
INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "too": 1.2,
    "slightly": 0.8, "somewhat": 0.9, "barely": 0.7, "quite": 1.2, "pretty": 1.2
}

# Extra Unicode punctuation to trim from word edges
EXTRA_PUNCT = "“”‘’–—…·•«»‹›‒‐‑‱※"

def _norm_word(tok: str) -> str:
    """
    Normalize a token for AFINN lookup:
    - Unicode NFKC normalization
    - lowercase
    - strip surrounding ASCII and common Unicode punctuation
    """
    t = unicodedata.normalize("NFKC", tok).lower()
    return t.strip(string.punctuation + EXTRA_PUNCT)

def calculate_window_sentiment(window: List[str],
                               afinn: Dict[str, int],
                               emoticons: Dict[str, int]) -> int:
    """
    Compute sentiment for a token window.

    Rules:
    - Emoticons/emoji are matched on the raw token (exact symbol match).
    - Words are normalized via _norm_word() before AFINN lookup.
    - Negation flips the sign of the current word if a negator occurs within
      the previous 3 normalized tokens.
    - Intensifiers multiply the magnitude based on any intensifier tokens
      within the same previous 3-token scope.
    - The final score is rounded to an integer for stability.
    """
    if not window:
        return 0

    # Precompute normalized tokens for scope checks
    norm_tokens = [_norm_word(t) for t in window]

    score = 0.0
    for i, raw in enumerate(window):
        # Emoticons/emoji: check raw token first
        emo = emoticons.get(raw)
        if emo:
            score += emo
            continue

        tok = norm_tokens[i]
        if not tok:
            continue

        base = afinn.get(tok, 0)
        if base == 0:
            continue

        # Look back up to 3 previous tokens for context
        scope_start = max(0, i - 3)
        scope = norm_tokens[scope_start:i]

        # Negation
        negated = any(t in NEGATORS for t in scope)

        # Intensifiers (multiplicative)
        intensity = 1.0
        for t in scope:
            intensity *= INTENSIFIERS.get(t, 1.0)

        val = base * intensity
        if negated:
            val = -val
        score += val

    return int(round(score))
