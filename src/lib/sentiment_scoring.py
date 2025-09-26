from __future__ import annotations
from typing import Dict, List, Tuple
import unicodedata
import string

# Heuristics [negation/intensity]
NEGATORS = {"not", "no", "never", "without", "hardly", "scarcely", "n't"}  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
INTENSIFIERS: Dict[str, float] = {  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "too": 1.2,
    "slightly": 0.8, "somewhat": 0.9, "barely": 0.7, "quite": 1.2, "pretty": 1.2,
}
EXTRA_PUNCT = "“”‘’–—…·•«»‹›‒‐‑‱※"  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]

def _norm_word(tok: str) -> str:
    t = unicodedata.normalize("NFKC", tok).lower()
    return t.strip(string.punctuation + EXTRA_PUNCT)  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]

def calculate_window_sentiment(
    window: List[str],
    afinn: Dict[str, int],
    emoticons: Dict[str, int],
    debug: bool = False,
    return_contrib: bool = False,
):
    """
    Score a token window with:
      1) Raw-token emoticon lookup (exact match, requires tokenizer to preserve). [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
      2) Normalized AFINN lookup for words. [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
      3) 3-token scope negation flip and multiplicative intensifiers. [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
    Returns int score, or (score, contrib_dict) when return_contrib=True. [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
    """
    if not window:
        return (0, {}) if return_contrib else 0  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]

    norm_tokens = [_norm_word(t) for t in window]  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
    score = 0.0
    contrib: Dict[str, float] = {}
    dbg: List[Tuple[str, str, int, int, bool, float]] = []

    for i, raw in enumerate(window):
        # 1) Emoticon raw lookup
        emo_val = emoticons.get(raw)
        if emo_val:
            score += emo_val
            if return_contrib:
                contrib[raw] = contrib.get(raw, 0.0) + emo_val
            if debug:
                dbg.append((raw, raw, 0, emo_val, False, 1.0))
            continue  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]

        # 2) AFINN normalized word
        tok = norm_tokens[i]
        if not tok:
            if debug:
                dbg.append((raw, tok, 0, 0, False, 1.0))
            continue
        base = afinn.get(tok, 0)
        if base == 0:
            if debug:
                dbg.append((raw, tok, 0, 0, False, 1.0))
            continue  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]

        # 3) Local scope modifiers
        scope = norm_tokens[max(0, i - 3) : i + 1]
        negated = any(t in NEGATORS for t in scope)
        intensity = 1.0
        for t in scope:
            intensity *= INTENSIFIERS.get(t, 1.0)

        val = base * intensity
        if negated:
            val = -val
        score += val

        if return_contrib:
            contrib[tok] = contrib.get(tok, 0.0) + val
        if debug:
            dbg.append((raw, tok, base, 0, negated, intensity))

    if debug and dbg:
        print("Matches (raw, norm, AFINN, EMO, negated, intensity):")
        for m in dbg[:20]:
            print(" ", m)  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]

    out = int(round(score))
    return (out, contrib) if return_contrib else out  # [attached_file:7e17d3a9-c17a-4c8f-b10a-39954f4aabd4]
