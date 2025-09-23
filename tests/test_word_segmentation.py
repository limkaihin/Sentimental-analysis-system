from src.lib.word_segmentation import word_break_one, word_break_all

def test_word_break_one_basic():
    vocab = {"this","is","a","pen"}
    assert word_break_one("thisisapen", vocab) == ["this","is","a","pen"]

def test_word_break_all_multiple():
    vocab = {"apple","pen","applepen","pine","pineapple"}
    outs = word_break_all("pineapplepenapple", vocab)
    assert any(o == ["pine", "apple", "pen", "apple"] for o in outs)
    assert any(o == ["pineapple", "pen", "apple"] for o in outs)
    assert any(o == ["pine", "applepen", "apple"] for o in outs)
