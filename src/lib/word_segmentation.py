# src/lib/word_segmentation.py
from __future__ import annotations
from typing import List, Optional, Set

def word_break_one(s: str, vocab: Set[str]) -> Optional[List[str]]:
    n = len(s)
    prev: List[Optional[int]] = [None] * (n + 1)
    prev[0] = 0
    for i in range(n):
        if prev[i] is None:
            continue
        for j in range(i + 1, n + 1):
            if prev[j] is None and s[i:j] in vocab:
                prev[j] = i
                if j == n:
                    break
    if prev[n] is None:
        return None
    out: List[str] = []
    cur = n
    while cur > 0:
        p = prev[cur]
        assert p is not None
        out.append(s[p:cur])
        cur = p
    return out[::-1]

def word_break_all(s: str, vocab: Set[str], max_solutions: Optional[int] = None) -> List[List[str]]:
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dfs(i: int) -> List[List[str]]:
        if i == len(s):
            return [[]]
        res: List[List[str]] = []
        for j in range(i + 1, len(s) + 1):
            w = s[i:j]
            if w in vocab:
                for tail in dfs(j):
                    res.append([w] + tail)
                    if max_solutions is not None and len(res) >= max_solutions:
                        return res
        return res
    return dfs(0)
