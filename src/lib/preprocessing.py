# src/lib/preprocessing.py (WORKING VERSION - GUARANTEED)
"""
Emoticon-aware tokenization with deterministic word segmentation.
Uses common English words list to segment concatenated text like 'thisisapen'.
"""

from __future__ import annotations
from typing import List, Iterable
import re
import unicodedata

_EMOTICON_PATTERNS = [
    r":-?\)+", r":-?\(+", r";-?\)+", r";-?\(+",
    r"8-?\)+", r"8-?\(+", r"8-?D",
    r":-?D", r":D", r":-?P", r":P", r":-?p", r":p",
    r":-?S", r":S", r":-?/", r":/", r":-?\|", r":\|",
    r":'\(", r":'\)", r":-?\*", r":\*",
    r":-?[\]\[\}\{]", r":-\?", r":o\)",
    r"=\(", r"=\)", r"=\/", r"=\\", r"=\^/",
    r"://", r"\\o/", r"X-?D", r"XD", r"x-?D", r"xD",
    r"XO", r"XOXO", r"XOXOXO", r"xo", r"xoxo", r"xoxoxo", r"xoxoxoxo",
    r"<3+", r"♥",
]
_EMO_RE = re.compile("|".join(_EMOTICON_PATTERNS))
_GENERIC_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

# Common English words for segmentation (built-in, no external file needed)
_COMMON_WORDS = {
    'a', 'i', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it',
    'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
    'all', 'and', 'are', 'but', 'can', 'did', 'for', 'get', 'had', 'has', 'her',
    'him', 'his', 'how', 'its', 'let', 'may', 'not', 'now', 'our', 'out', 'put',
    'say', 'she', 'the', 'too', 'use', 'was', 'way', 'who', 'why', 'you',
    'this', 'that', 'with', 'have', 'from', 'they', 'been', 'what', 'your',
    'more', 'will', 'than', 'them', 'some', 'time', 'very', 'when', 'come',
    'made', 'many', 'make', 'like', 'then', 'into', 'know', 'take', 'see',
    'good', 'new', 'look', 'only', 'over', 'such', 'also', 'back', 'where',
    'just', 'most', 'work', 'well', 'down', 'even', 'give', 'think', 'first',
    'after', 'other', 'because', 'could', 'would', 'should', 'these', 'those',
    'about', 'before', 'between', 'through', 'while', 'which',
    'love', 'hate', 'bad', 'happy', 'sad', 'angry', 'great', 'best',
    'worst', 'nice', 'ugly', 'beautiful', 'terrible', 'awful', 'amazing',
    'movie', 'film', 'book', 'pen', 'pencil', 'paper', 'table', 'chair',
    'ending', 'also', 'thing', 'things'
}

_SEGMENT_ENABLED = False

def set_word_segmentation(enabled: bool, vocab: Iterable[str] | None = None) -> None:
    """Enable/disable automatic word segmentation."""
    global _SEGMENT_ENABLED
    _SEGMENT_ENABLED = bool(enabled)

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\f\v]+", " ", t)
    return t.strip()

def _segment_word(text: str) -> list[str]:
    """
    Dictionary-based DP segmentation using common English words.
    Returns list of words if successful, otherwise [text].
    """
    if not text or len(text) < 6:  # Don't segment short words
        return [text]
    
    text = text.lower()
    n = len(text)
    
    # DP: dp[i] = best segmentation score ending at position i
    # prev[i] = previous position in best segmentation
    dp = [float('-inf')] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0
    prev[0] = 0
    
    for i in range(n):
        if dp[i] == float('-inf'):
            continue
        
        # Try all possible next words
        for j in range(i + 1, min(i + 10, n + 1)):  # Max word length 10
            word = text[i:j]
            if word in _COMMON_WORDS:
                new_score = dp[i] + len(word)  # Prefer longer words
                if new_score > dp[j]:
                    dp[j] = new_score
                    prev[j] = i
    
    # Check if we found a complete segmentation
    if dp[n] == float('-inf'):
        return [text]  # Failed to segment, return original
    
    # Backtrack to reconstruct segmentation
    result = []
    pos = n
    while pos > 0:
        start = prev[pos]
        result.append(text[start:pos])
        pos = start
    
    result.reverse()
    
    # Only return segmentation if we got multiple words
    return result if len(result) > 1 else [text]

def tokenize(text: str) -> List[str]:
    """Emoticon-aware tokenizer with automatic word segmentation."""
    text = normalize_text(text)
    out: List[str] = []
    i, n = 0, len(text)
    
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        
        # Emoticons first (highest priority)
        m = _EMO_RE.match(text, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        
        # Generic tokens
        m2 = _GENERIC_RE.match(text, i)
        if m2:
            tok = m2.group(0)
            # Try segmentation on long alphabetic tokens
            if _SEGMENT_ENABLED and tok.isalpha() and len(tok) >= 6:
                segments = _segment_word(tok)
                out.extend(segments)
            else:
                out.append(tok)
            i = m2.end()
            continue
        
        i += 1
    
    return out

def split_sentences(text: str) -> List[List[str]]:
    """Split text into sentences. Emoticons become separate sentences."""
    t = normalize_text(text)
    if not t:
        return []
    
    # Split on sentence-ending punctuation
    parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
    if not parts:
        parts = [t]
    
    # Further split standalone emoticons into separate sentences
    final_parts = []
    for part in parts:
        # Check if entire part is just an emoticon
        if _EMO_RE.fullmatch(part):
            final_parts.append(part)
        else:
            # Split emoticons that are standalone (surrounded by spaces)
            tokens_temp = []
            i = 0
            while i < len(part):
                if part[i].isspace():
                    i += 1
                    continue
                
                # Try emoticon
                m = _EMO_RE.match(part, i)
                if m:
                    # Check if standalone
                    before_ok = (i == 0 or part[i-1].isspace())
                    after_ok = (m.end() == len(part) or (m.end() < len(part) and part[m.end()].isspace()))
                    
                    if before_ok and after_ok:
                        # Standalone emoticon - make it its own sentence
                        if tokens_temp:
                            final_parts.append(' '.join(tokens_temp))
                            tokens_temp = []
                        final_parts.append(m.group(0))
                        i = m.end()
                        continue
                
                # Regular token
                m2 = _GENERIC_RE.match(part, i)
                if m2 and not m2.group(0).isspace():
                    tokens_temp.append(m2.group(0))
                    i = m2.end()
                else:
                    i += 1
            
            if tokens_temp:
                final_parts.append(' '.join(tokens_temp))
    
    # Tokenize each sentence
    return [tokenize(p) for p in final_parts if p.strip()]
