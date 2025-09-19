from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

from lib.lexicon import load_tab_lexicon
from lib.preprocessing import read_text_files
from lib.sliding_window import sliding_window_sentiment_analysis
from lib.extrema_segments import extrema_segments
from lib.visualisation import plot_review_windows, annotate_extrema

def parse_args() -> argparse.Namespace:
    """
    CLI with sensible defaults pointing to the data/ folder:
    - data contains AFINN-en-165.txt, AFINN-emoticon-8.txt.
    - data/reviews contains sample review .txt files.
    All defaults can be overridden without changing code.
    """
    p = argparse.ArgumentParser(description="Sliding-window sentiment with AFINN and emoticons")
    p.add_argument("--data_dir", type=str, default="data", help="Base data dir containing lexicons and reviews")
    p.add_argument("--reviews_subdir", type=str, default="reviews", help="Subfolder under data_dir with .txt reviews")
    p.add_argument("--afinn_name", type=str, default="AFINN-en-165.txt", help="AFINN lexicon filename under data_dir")
    p.add_argument("--emoticons_name", type=str, default="AFINN-emoticon-8.txt", help="Emoticon lexicon filename under data_dir")
    p.add_argument("--window_size", type=int, default=3, help="Sliding window size k")
    p.add_argument("--plot_first_n", type=int, default=0, help="Plot first N reviews (0 disables)")
    p.add_argument("--save_dir", type=str, default="", help="Optional output directory for plots")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    base = Path(args.data_dir)

    # Resolve paths under data/ based on CLI args (no hardcoding in logic).
    reviews_dir = base / args.reviews_subdir
    afinn_path = base / args.afinn_name
    emoticons_path = base / args.emoticons_name

    # Load data and lexicons.
    reviews = read_text_files(reviews_dir)
    if not reviews:
        print(f"No .txt reviews found under: {reviews_dir}")
        return

    afinn = load_tab_lexicon(afinn_path)
    emoticons = load_tab_lexicon(emoticons_path)

    # Sliding-window analysis for all reviews.
    k = args.window_size
    per_review_windows: List[List[tuple[int, int]]] = sliding_window_sentiment_analysis(reviews, k, afinn, emoticons)

    # Print extrema and optionally plot.
    for i, windows in enumerate(per_review_windows, start=1):
        pos_seg, neg_seg = extrema_segments(windows, k)
        print(f"Review {i}:")
        print(f"  Most positive: {pos_seg}")
        print(f"  Most negative: {neg_seg}")

        if args.plot_first_n and i <= args.plot_first_n:
            ax = plot_review_windows(windows, k, title=f"Review {i}")
            annotate_extrema(ax, pos_seg, neg_seg)
            if args.save_dir:
                outdir = Path(args.save_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                ax.figure.savefig(outdir / f"review_{i}.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
