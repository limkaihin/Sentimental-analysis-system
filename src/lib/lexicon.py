# lexicon.py
# Purpose: Load tab-separated lexicons (AFINN, emoticons) into Python dicts.

from __future__ import annotations

def load_tab_lexicon(path: str) -> dict[str, int]:
    """
    Load a TSV lexicon file with format: token<TAB>score
    Returns dict mapping lowercased tokens to integer scores.
    """
    lex: dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                k, v = parts
                try:
                    lex[k.lower()] = int(v)
                except Exception:
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to load lexicon from {path}: {e}")
    return lex
