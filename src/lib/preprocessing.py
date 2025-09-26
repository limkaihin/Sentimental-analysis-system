from __future__ import annotations
from typing import Iterable, List, Union
from pathlib import Path
import re
import unicodedata
import gzip

# 1) Emoticon extractor (match before generic tokens)
_EMOTICON_PATTERNS = [
    # Smiles/frowns (with/without nose), capture repeats
    r":-?\)+", r":-?\(+", r";-?\)+", r";-?\(+",
    # 8-face variants (smile/frown + big grin)
    r"8-?\)+", r"8-?\(+", r"8-?D",
    # Tongue/grin/concern/slash/pipe (single or nose)
    r":-?D", r":D", r":-?P", r":P", r":-?p", r":p",
    r":-?S", r":S", r":-?/", r":/", r":-?\|", r":\|",
    # Crying
    r":'\(", r":'\)",
    # Star-kiss (present in AFINN)
    r":-?\*", r":\*",
    # Square/curly bracket faces
    r":-?[\]\[\}\{]",
    # Literal ':-?' in AFINN
    r":-\?",
    # :o) face
    r":o\)",
    # Equals-sign faces and slashes
    r"=\(", r"=\)", r"=\/", r"=\\", r"=\^/",
    # Literal ://
    r"://",
    # Arms up
    r"\\o/",
    # XD family
    r"X-?D", r"XD", r"x-?D", r"xD",
    # Hugs/Kisses (explicit forms as per AFINN)
    r"XO", r"XOXO", r"XOXOXO", r"xo", r"xoxo", r"xoxoxo", r"xoxoxoxo",
    # Hearts (graded by repeats)
    r"<3+",
    # Unicode heart
    r"♥",
]
_EMO_RE = re.compile("|".join(_EMOTICON_PATTERNS))

# 2) Generic token: words/numbers or single non-space symbol
_GENERIC_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

def normalize_text(text: str) -> str:
    """
    Unicode/whitespace normalization for stable tokenization. [attached_file:8461b178-9149-448f-a4ee-5846bce7560a]
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\f\v]+", " ", t)
    return t.strip()

def tokenize(text: str) -> List[str]:
    """
    Emoticon-aware tokenizer: preserves ':)', ':-(', ':D', etc., as single tokens, then tokenizes the rest. [attached_file:8461b178-9149-448f-a4ee-5846bce7560a]
    """
    text = normalize_text(text)
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _EMO_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        m2 = _GENERIC_RE.match(text, i)
        if m2:
            tok = m2.group(0)
            if not tok.isspace():
                tokens.append(tok)
            i = m2.end()
            continue
        i += 1
    return tokens

def read_text_files(root: Union[str, Path], pattern: str = "*") -> List[str]:
    """
    Recursively read UTF-8 .txt and .gz files, skipping others. [attached_file:8461b178-9149-448f-a4ee-5846bce7560a]
    """
    root = Path(root)
    docs: List[str] = []
    for p in sorted(root.rglob(pattern)):
        try:
            if p.suffix == ".gz":
                with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                    docs.append(f.read())
            elif p.suffix == ".txt":
                docs.append(p.read_text(encoding="utf-8", errors="ignore"))
            else:
                continue
        except Exception as e:
            print(f"Warning: Failed to read {p}: {e}")
            continue
    return docs

SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[list[str]]:
    """
    Split text into sentences and tokenize each into emoticon-preserving tokens. [attached_file:8461b178-9149-448f-a4ee-5846bce7560a]
    """
    text = normalize_text(text)
    if not text:
        return []
    raw_sents = SENT_BOUNDARY_RE.split(text)
    return [tokenize(s) for s in raw_sents if s.strip()]

def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs by blank lines. [attached_file:8461b178-9149-448f-a4ee-5846bce7560a]
    """
    text = normalize_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def sliding_windows(tokens: List[str], k: int) -> Iterable[List[str]]:
    """
    Yield consecutive windows of length k from a token list. [attached_file:8461b178-9149-448f-a4ee-5846bce7560a]
    """
    if k <= 0 or len(tokens) < k:
        return []
    return (tokens[i : i + k] for i in range(len(tokens) - k + 1))


# === Fuzzy word segmentation fallback =========================================
# A small built-in vocabulary to help split common phrases even if not in AFINN.
_COMMON_FALLBACK_VOCAB = {
    # pronouns / function words
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "a", "an", "the", "and", "or", "but", "if", "think", "then", "than", "that", "this", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "did", "done", "does",
    "of", "to", "in", "on", "for", "with", "as", "by", "at", "from", "about", "into", "over",
    "not", "no", "very", "really", "too", "so", "just", "only", "more", "most", "less", "least",
    # common review words
    "good", "great", "amazing", "awesome", "excellent", "nice", "love", "like", "fun", "cool",
    "bad", "terrible", "awful", "poor", "boring", "hate", "dislike", "meh", "ok", "okay",
    "sad", "happy", "angry", "mad",
    "movie", "film", "plot", "acting", "actor", "actors", "actress", "story", "sound", "music",
    "trash", "garbage", "mess", "gem", "must", "watch", "rewatch", "slow", "fast",
    "worst", "best", "better", "worse",
}

# Config flags for segmentation
_ENABLE_WORD_SEGMENTATION = False
_ENABLE_FUZZY_SEGMENTATION = True
_WORD_SEG_VOCAB: set = set()

def set_word_segmentation(enabled: bool, vocab: Iterable[str] | None = None, fuzzy: bool = True):
    """
    Enable/disable dictionary-based segmentation. If fuzzy=True, use a dynamic-programming
    fallback that favors known words but can still segment unknown text.
    """
    global _ENABLE_WORD_SEGMENTATION, _ENABLE_FUZZY_SEGMENTATION, _WORD_SEG_VOCAB
    _ENABLE_WORD_SEGMENTATION = bool(enabled)
    _ENABLE_FUZZY_SEGMENTATION = bool(fuzzy)
    if vocab is not None:
        _WORD_SEG_VOCAB = {str(w).strip().lower() for w in vocab if str(w).strip()}

def get_word_segmentation_state():
    return _ENABLE_WORD_SEGMENTATION, _WORD_SEG_VOCAB, _ENABLE_FUZZY_SEGMENTATION

def word_break_one_fuzzy(text: str, vocab: set[str] | None = None, max_word_len: int = 20) -> list[str]:
    """
    DP-based word break that prefers known words but will still split unknown text.
    Cost model:
      - known word (in vocab or fallback): small cost (-5 bonus -> cost -5)
      - unknown word: positive cost proportional to length (len)
    We minimize total cost, which yields longer sequences of known words; unknown chunks are
    split into shorter pieces rather than left as one long token.
    """
    if not text:
        return []
    t = text.lower()
    V = set(vocab or set())
    V |= _COMMON_FALLBACK_VOCAB

    n = len(t)
    INF = 10**9
    # dp[i] = (cost, prev_index)
    dp = [(INF, -1)] * (n + 1)
    dp[0] = (0, -1)

    for i in range(n):
        base_cost, _ = dp[i]
        if base_cost >= INF:
            continue
        for j in range(i + 1, min(n, i + max_word_len) + 1):
            w = t[i:j]
            if w in V:
                cost = base_cost - 5  # reward known words
            else:
                # penalize unknowns; shorter unknown chunks preferred
                cost = base_cost + len(w)
            if cost < dp[j][0]:
                dp[j] = (cost, i)

    # reconstruct
    if dp[n][0] >= INF:
        return [text]
    parts = []
    cur = n
    while cur > 0:
        prev = dp[cur][1]
        parts.append(t[prev:cur])
        cur = prev
    parts.reverse()
    return parts

def _segment_token_if_needed(tok: str) -> list[str]:
    """
    If segmentation is enabled, attempt strict dictionary split first.
    If that doesn't split, and fuzzy is enabled, try word_break_one_fuzzy.
    """
    if not _ENABLE_WORD_SEGMENTATION:
        return [tok]
    t = tok.lower()
    if not t.isalpha() or len(t) <= 3 or t in _WORD_SEG_VOCAB:
        return [tok]
    # 1) Strict dictionary-based split (if available)
    try:
        from src.lib.word_segmentation import word_break_one
        seg = word_break_one(t, _WORD_SEG_VOCAB)
        if isinstance(seg, (list, tuple)) and len(seg) > 1:
            return list(seg)
        if isinstance(seg, str) and " " in seg.strip():
            parts = [p for p in seg.strip().split() if p]
            if len(parts) > 1:
                return parts
    except Exception:
        pass
    # 2) Fuzzy fallback
    if _ENABLE_FUZZY_SEGMENTATION:
        parts = word_break_one_fuzzy(t, _WORD_SEG_VOCAB)
        if len(parts) > 1:
            return parts
    return [tok]


def tokenize_smart(text: str) -> list[str]:
    base = tokenize(text)
    out: list[str] = []
    for tok in base:
        out.extend(_segment_token_if_needed(tok))
    return out

def split_sentences(text: str) -> List[List[str]]:
    t = normalize_text(text)
    if not t:
        return []
    pieces = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
    if len(pieces) == 1:
        pieces = [s.strip() for s in re.split(r'(?<=[.!?])', t) if s.strip()]
    return [tokenize_smart(p) for p in pieces]
