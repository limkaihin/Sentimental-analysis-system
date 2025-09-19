import pytest
from lib.preprocessing import normalize_text, tokenize, sliding_windows

def test_normalize_text_basic():
    raw = "Great movie!\r\nLoved it.\t\tSo good."
    norm = normalize_text(raw)
    assert "\r" not in norm
    assert "  " not in norm

def test_tokenize_keeps_emoticons_and_punct():
    text = "Wow!! :) This is A+"
    toks = tokenize(text)
    assert ":)" in toks
    assert "!" in toks
    assert "+" in toks

def test_sliding_windows_shapes():
    toks = ["a", "b", "c", "d"]
    k = 2
    wins = list(sliding_windows(toks, k))
    assert len(wins) == 3
    assert wins[0] == ["a", "b"]
    assert wins[-1] == ["c", "d"]

def test_sliding_windows_edge_cases():
    assert list(sliding_windows(["a"], 2)) == []
    assert list(sliding_windows([], 1)) == []
    assert list(sliding_windows(["a","b"], 0)) == []
