from __future__ import annotations
from typing import Iterable, List
from pathlib import Path
import re
import unicodedata

# Regex that captures words and single non-space symbols (keeps emoticons/emoji and punctuation)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def normalize_text(text: str) -> str:
    """
    Normalize Unicode and whitespace so tokenization is stable across platforms.
    - NFKC normalizes look-alike characters.
    - Convert CRLF/CR to LF to unify newlines.
    - Collapse runs of spaces/tabs.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\f\v]+", " ", t)
    return t.strip()

def tokenize(text: str) -> List[str]:
    """
    Split text into tokens while preserving punctuation and emoticons/emoji,
    which is important for lexicon-based scoring.
    """
    text = normalize_text(text)
    return TOKEN_RE.findall(text)

def read_text_files(root: str | Path, pattern: str = "*.txt") -> List[str]:
    """
    Recursively read UTF-8 text files from a directory tree.
    - Sorted traversal yields deterministic order for tests and reproducibility.
    - Errors are ignored per file to avoid aborting on a single bad file.
    """
    root = Path(root)
    docs: List[str] = []
    for p in sorted(root.rglob(pattern)):
        try:
            docs.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return docs

SENT_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> list[list[str]]:
    """
    Split text into sentences and tokenize each sentence.
    Returns a list of token lists, one per sentence.
    """
    text = normalize_text(text)
    if not text:
        return []
    raw_sents = SENT_BOUNDARY_RE.split(text)
    return [TOKEN_RE.findall(s) for s in raw_sents if s.strip()]

def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs by blank lines; trims whitespace.
    """
    text = normalize_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def sliding_windows(tokens: List[str], k: int) -> Iterable[List[str]]:
    """
    Yield consecutive windows of length k from a token sequence.
    Returns empty iterable for invalid k or if not enough tokens.
    """
    if k <= 0 or len(tokens) < k:
        return []
    return (tokens[i : i + k] for i in range(len(tokens) - k + 1))
