from src.lib.extrema_segments import extrema_segments

def test_extrema_sign_enforced():
    k = 2
    windows = [(0,1,-3), (1,2,-1)]
    pos, neg = extrema_segments(windows, k)
    assert pos is None
    assert neg is not None and neg[2] == -1 or neg[2] == -3
