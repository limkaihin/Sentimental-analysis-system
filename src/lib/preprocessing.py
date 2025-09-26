from __future__ import annotations
from typing import Iterable, List, Union
from pathlib import Path
import re
import unicodedata
import gzip

# 1) Emoticon extractor (match before generic tokens)
_EMOTICON_PATTERNS = [
    r":-?\)", r":-?\(", r";-?\)", r";-?\(", r":-?D", r":D", r":-?P", r":P",
    r":-?S", r":S", r":-?/", r":/", r":'\(", r":'\)", r":-?\|", r":\|",
    r":-?O", r":O"
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


# === Smart tokenization & robust sentence splitting (glued-word support) ===
# Global config toggled from the Streamlit app
_ENABLE_WORD_SEGMENTATION = False
_WORD_SEG_VOCAB: set = set()

def set_word_segmentation(enabled: bool, vocab: Iterable[str] | None = None):
    """
    Enable/disable dictionary-based segmentation for glued words (e.g., 'iamsad' -> 'i am sad').
    If vocab is provided, it replaces the current vocabulary.
    """
    global _ENABLE_WORD_SEGMENTATION, _WORD_SEG_VOCAB
    _ENABLE_WORD_SEGMENTATION = bool(enabled)
    if vocab is not None:
        _WORD_SEG_VOCAB = {str(w).strip().lower() for w in vocab if str(w).strip()}

def get_word_segmentation_state():
    return _ENABLE_WORD_SEGMENTATION, _WORD_SEG_VOCAB

def _segment_token_if_needed(tok: str) -> list[str]:
    """
    If segmentation is enabled and the token looks like a glued word,
    try to segment it using the dictionary-based word breaker.
    """
    if not _ENABLE_WORD_SEGMENTATION:
        return [tok]
    t = tok.lower()
    if not t.isalpha() or len(t) <= 3 or t in _WORD_SEG_VOCAB:
        return [tok]
    try:
        # Lazy import to avoid circular cost
        from src.lib.word_segmentation import word_break_one
        seg = word_break_one(t, _WORD_SEG_VOCAB)
        # Accept segmentation only if it yields >1 tokens
        if isinstance(seg, (list, tuple)) and len(seg) > 1:
            return list(seg)
        # Some implementations return a string; split by spaces if present
        if isinstance(seg, str) and " " in seg.strip():
            parts = [p for p in seg.strip().split() if p]
            if len(parts) > 1:
                return parts
    except Exception:
        pass
    return [tok]

def tokenize_smart(text: str) -> list[str]:
    """
    Wrap existing tokenize() and post-process tokens with optional dictionary segmentation.
    """
    base = tokenize(text)  # uses existing pipeline
    out: list[str] = []
    for tok in base:
        out.extend(_segment_token_if_needed(tok))
    return out

def split_sentences(text: str) -> List[List[str]]:
    """
    Split input into sentences even when there is no space after punctuation.
    Then tokenize each sentence with tokenize_smart(), so glued words are split if enabled.
    """
    t = normalize_text(text)
    if not t:
        return []
    # First pass: standard case with whitespace after punctuation
    pieces = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
    # Fallback: handle text with *no* whitespace between sentences
    if len(pieces) == 1:
        pieces = [s.strip() for s in re.split(r'(?<=[.!?])', t) if s.strip()]
    return [tokenize_smart(p) for p in pieces]
