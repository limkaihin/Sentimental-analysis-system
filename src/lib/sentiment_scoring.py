from pathlib import Path
from typing import Dict, List, Tuple, Union
import unicodedata
import string

# Heuristics [negation/intensity]
NEGATORS = {"not", "no", "never", "without", "hardly", "scarcely", "n't"}
INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "too": 1.2,
    "slightly": 0.8, "somewhat": 0.9, "barely": 0.7, "quite": 1.2, "pretty": 1.2,
}
EXTRA_PUNCT = "“”‘’–—…·•«»‹›‒‐‑‱※"

def _norm_word(tok: str) -> str:
    t = unicodedata.normalize("NFKC", tok).lower()
    return t.strip(string.punctuation + EXTRA_PUNCT)

def load_tab_lexicon(path: Union[str, Path]) -> Dict[str, int]:
    """
    Load token-to-integer sentiment lexicon from file with tab or whitespace separators.
    """
    d: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    continue
            token, score_str = parts
            try:
                d[token] = int(score_str)
            except ValueError:
                continue
    return d

def calculate_window_sentiment(
    window: List[str],
    afinn: Dict[str, int],
    emoticons: Dict[str, int],
    debug: bool = False,
    return_contrib: bool = False,
) -> int | Tuple[int, Dict[str, float]]:
    """
    Score a token window with:
      1) Raw-token emoticon lookup (exact match, requires tokenizer to preserve).
      2) Normalized AFINN lookup for words.
      3) 3-token scope negation flip and multiplicative intensifiers.
    Returns int score, or (score, contrib_dict) when return_contrib=True.
    """
    if not window:
        return (0, {}) if return_contrib else 0

    norm_tokens = [_norm_word(t) for t in window]
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
            continue

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
            continue

        # 3) Local scope modifiers
        scope_start = max(0, i - 3)
        scope = norm_tokens[scope_start:i + 1]
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
            print(" ", m)

    out = int(round(score))
    return (out, contrib) if return_contrib else out

if __name__ == "__main__":
    afinn_path = Path("data/lexicon/AFINN-en-165.txt")
    emoticon_path = Path("data/lexicon/AFINN-emoticon-8.txt")

    afinn = load_tab_lexicon(afinn_path)
    emoticons = load_tab_lexicon(emoticon_path)

    tokens = [":)", "I", "am", "very", "happy", "but", "not", "sad", ":("]

    score = calculate_window_sentiment(tokens, afinn, emoticons, debug=True)
    print(f"Total sentiment score: {score}")
