# sentiment_scoring.py
from typing import List, Dict

def calculate_window_sentiment(window: List[str], afinn: Dict[str, int], emoticons: Dict[str, int]) -> int:
    """
    Calculate sentiment for a given window of words using AFINN lexicon and emoticons.
    
    Args:
        window: A list of words (str) in a sliding window.
        afinn: A dictionary containing AFINN lexicon for words and their sentiment scores.
        emoticons: A dictionary containing emoticons and their sentiment scores.

    Returns:
        An integer representing the sentiment score for the given window.
    """
    sentiment = 0
    for word in window:
        word_lower = word.lower()
        sentiment += afinn.get(word_lower, 0)  # Add word's sentiment score from AFINN
        sentiment += emoticons.get(word, 0)  # Add emoticon's sentiment score
    return sentiment
