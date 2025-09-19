from __future__ import annotations
from typing import Iterable, List
from pathlib import Path
import re
import unicodedata
import gzip

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

def read_text_files(root: Union[str, Path], pattern: str = "*") -> List[str]:
    """
    Recursively read UTF-8 text files from a directory tree.

    Supports both plain .txt files and gzip-compressed .gz files:
    - *.txt are read with standard open.
    - *.gz are read in text mode via gzip.open(..., 'rt').

    Returns a list of decoded strings. Files with other extensions are skipped.
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
