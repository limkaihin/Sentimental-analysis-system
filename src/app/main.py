# main.py
# Command-line interface for sentiment analysis

from __future__ import annotations
import argparse
from pathlib import Path

from src.lib.lexicon import load_tab_lexicon
from src.lib.preprocessing import normalize_text, split_sentences, set_word_segmentation
from src.lib.sentiment_scoring import sentence_scores
from src.lib.sliding_window import sliding_window_sentiment_over_sentences, extrema_segments
from src.lib.varlen_segments import best_varlen_segments_from_scores
from src.lib.visualisation import plot_sentence_scores, plot_sentiment_timeline, save_chart_as_html

def main():
    parser = argparse.ArgumentParser(description="AFINN sentiment analysis CLI")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="Text file to analyze")
    parser.add_argument("--k", type=int, default=3, help="Window size")
    parser.add_argument("--dict", type=str, default="data/lexicon/dictionary.txt",
                       help="Dictionary file for segmentation")
    args = parser.parse_args()
    
    # Load lexicons
    afinn = load_tab_lexicon("data/lexicon/AFINN-en-165.txt")
    emot = load_tab_lexicon("data/lexicon/AFINN-emoticon-8.txt")
    
    # Load dictionary for segmentation
    dict_path = Path(args.dict)
    vocab = set()
    if dict_path.exists():
        vocab = {w.strip().lower() for w in dict_path.read_text().splitlines() if w.strip()}
    set_word_segmentation(bool(vocab), vocab)
    
    # Get text
    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        print("Please provide --text or --file")
        return
    
    # Analyze
    cleaned = normalize_text(text)
    sents = split_sentences(cleaned)
    scores = sentence_scores(cleaned, afinn, emot)
    
    print("Sentences and scores:")
    for i, toks in enumerate(sents):
        print(f"[{i}] {' '.join(toks)} → {scores[i]}")
    
    # SAVES CHARTS FOR CLI
    bar_chart = plot_sentence_scores(scores, sents)
    timeline_chart = plot_sentiment_timeline(scores, sents)
    save_chart_as_html(bar_chart, "cli_sentence_scores.html")
    save_chart_as_html(timeline_chart, "cli_sentiment_timeline.html")
    print("\n[INFO] Charts saved as cli_sentence_scores.html and cli_sentiment_timeline.html")

    print(f"\nFixed windows (k={args.k}):")
    wins = sliding_window_sentiment_over_sentences(scores, args.k)
    pos_win, neg_win = extrema_segments(wins)
    if pos_win:
        a, b, s = pos_win
        print(f"Best +: [{a}-{b}] sum={s}")
    if neg_win:
        a, b, s = neg_win
        print(f"Best -: [{a}-{b}] sum={s}")

if __name__ == "__main__":
    main()
