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
