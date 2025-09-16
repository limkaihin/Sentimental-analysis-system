# sliding_window.py
from typing import List, Tuple
from sentiment_scoring import calculate_window_sentiment

def sliding_window_sentiment_analysis(reviews: List[str], k: int, afinn: dict, emoticons: dict) -> List[List[Tuple[int, int]]]:
    """
    Analyze sentiment for each sliding window in the list of reviews.
    
    Args:
        reviews: List of reviews (strings) to analyze.
        k: The window size for sliding window.
        afinn: A dictionary containing AFINN lexicon for words and their sentiment scores.
        emoticons: A dictionary containing emoticons and their sentiment scores.

    Returns:
        A list of lists of tuples, where each tuple is (window_start_index, sentiment_score).
    """
    result = []
    for review in reviews:
        words = review.split()  # Tokenize review into words
        review_sentiments = []
        for i in range(len(words) - k + 1):
            window = words[i:i + k]  # Extract the window of words
            sentiment_score = calculate_window_sentiment(window, afinn, emoticons)
            review_sentiments.append((i, sentiment_score))  # Store window index and its sentiment
        result.append(review_sentiments)
    return result
