# main.py
from sliding_window import sliding_window_sentiment_analysis
from sentiment_scoring import calculate_window_sentiment
from extrema_segments import extrema_segments
from typing import List
import os

# Load AFINN lexicon
def load_afinn(filepath: str) -> dict:
    """
    Load AFINN lexicon from a file.
    Args:
        filepath: Path to the AFINN lexicon file.
    Returns:
        A dictionary where keys are words and values are sentiment scores.
    """
    afinn = {}
    with open(filepath, 'r') as file:
        for line in file:
            word, score = line.split()
            afinn[word] = int(score)
    return afinn

# Load emoticons lexicon
def load_emoticons(filepath: str) -> dict:
    """
    Load emoticons sentiment lexicon from a file.
    Args:
        filepath: Path to the emoticons lexicon file.
    Returns:
        A dictionary where keys are emoticons and values are sentiment scores.
    """
    emoticons = {}
    with open(filepath, 'r') as file:
        for line in file:
            emoticon, score = line.split()
            emoticons[emoticon] = int(score)
    return emoticons

# Load IMDB reviews from a directory
def load_imdb_reviews(directory: str) -> List[str]:
    """
    Load all reviews from a directory.
    Args:
        directory: Path to the directory containing review files.
    Returns:
        A list of reviews (strings).
    """
    reviews = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            reviews.append(file.read())
    return reviews

# Paths to lexicons and review directory
afinn_path = './data/AFINN-en-165.txt'
emoticons_path = './data/AFINN-emoticon-8.txt'
train_dir = './data/aclImdb/train/pos'  # For positive reviews

# Load the lexicons
afinn = load_afinn(afinn_path)
emoticons = load_emoticons(emoticons_path)

# Load reviews
reviews = load_imdb_reviews(train_dir)

# Set window size
k = 3  # Sliding window size

# Perform sliding window sentiment analysis
sentiment_results = sliding_window_sentiment_analysis(reviews, k, afinn, emoticons)

# Get the most positive and negative segments
for review_idx, review_sentiments in enumerate(sentiment_results):
    positive_segment, negative_segment = extrema_segments(review_sentiments, k, afinn, emoticons)
    
    print(f"Review {review_idx + 1}:")
    print("Most positive segment:", positive_segment)
    print("Most negative segment:", negative_segment)
    print()
