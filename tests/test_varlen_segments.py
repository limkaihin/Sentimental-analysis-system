from src.lib.varlen_segments import best_varlen_segments

def test_varlen_segments_basic():
    text = "Good. Bad. Very good."
    afinn = {"good":3, "bad":-3}
    emot = {}
    scores, pos_seg, neg_seg = best_varlen_segments(text, afinn, emot)
    assert len(scores) == 3
    assert pos_seg is not None and pos_seg[2] > 0
    assert neg_seg is not None and neg_seg[2] < 0
