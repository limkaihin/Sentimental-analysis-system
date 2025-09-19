from __future__ import annotations
from typing import Dict, List
import unicodedata
import string

NEGATORS = {"not", "no", "never", "without", "hardly", "scarcely", "n't"}
INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "too": 1.2,
    "slightly": 0.8, "somewhat": 0.9, "barely": 0.7, "quite": 1.2, "pretty": 1.2
}

EXTRA_PUNCT = "“”‘’–—…·•«»‹›‒‐‑‱※"

def _norm_word(tok: str) -> str:
    t = unicodedata.normalize("NFKC", tok).lower()
    return t.strip(string.punctuation + EXTRA_PUNCT)

def calculate_window_sentiment(window: List[str],
                               afinn: Dict[str, int],
                               emoticons: Dict[str, int],
                               debug: bool = False) -> int:
    """
    Returns an integer sentiment score for the given token window.
    Supports:
      - AFINN word lookup on normalized tokens
      - emoticon/emoji lookup on raw tokens
      - simple negation within a 3-token scope
      - multiplicative intensifiers within the same scope
    """
    if not window:
        return 0

    norm_tokens = [_norm_word(t) for t in window]
    score = 0.0
    dbg = []

    for i, raw in enumerate(window):
        emo_val = emoticons.get(raw)
        if emo_val:
            score += emo_val
            if debug:
                dbg.append((raw, raw, 0, emo_val, False, 1.0))
            continue

        tok = norm_tokens[i]
        if not tok:
            continue

        base = afinn.get(tok, 0)
        if base == 0:
            continue

        scope_start = max(0, i - 3)
        scope = norm_tokens[scope_start:i]
        negated = any(t in NEGATORS for t in scope)

        intensity = 1.0
        for t in scope:
            intensity *= INTENSIFIERS.get(t, 1.0)

        val = base * intensity
        if negated:
            val = -val
        score += val

        if debug:
            dbg.append((raw, tok, base, 0, negated, intensity))

    if debug and dbg:
        print("Matches (raw, norm, AFINN, EMO, negated, intensity):")
        for m in dbg[:20]:
            print("  ", m)

    return int(round(score))
