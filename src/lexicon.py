from __future__ import annotations
from pathlib import Path
from typing import Dict

def load_tab_lexicon(path: str | Path) -> Dict[str, int]:
    """
    Load a token-to-integer-score mapping from a file.
    - Supports both TAB and whitespace separation.
    - Skips malformed lines gracefully.
    Works for AFINN-en-165.txt and AFINN-emoticon-8.txt.
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
            token, score = parts
            try:
                d[token] = int(score)
            except ValueError:
                continue
    return d
