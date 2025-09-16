# extrema_segments.py
from sliding_window import sliding_window_sentiment_analysis

def extrema_segments(values: List[int], k: int, afinn: dict, emoticons: dict):
    """
    Identify the most positive and most negative sentiment segments in a list of sentiment values.
    
    Args:
        values: A list of sentiment scores for a review (from sliding window analysis).
        k: The window size for the sliding window.
        afinn: A dictionary containing AFINN lexicon for words and their sentiment scores.
        emoticons: A dictionary containing emoticons and their sentiment scores.

    Returns:
        A tuple containing two segments: (most_positive_segment, most_negative_segment),
        where each segment is represented as (start_idx, end_idx_inclusive, sentiment_sum).
    """
    windows = sliding_window_sentiment_analysis(values, k, afinn, emoticons)
    
    if not windows:
        return None, None

    # Find the most positive and most negative segments
    best_pos = max(windows, key=lambda x: x[1])
    best_neg = min(windows, key=lambda x: x[1])

    s_pos, v_pos = best_pos
    s_neg, v_neg = best_neg

    return (s_pos, s_pos + k - 1, v_pos), (s_neg, s_neg + k - 1, v_neg)
