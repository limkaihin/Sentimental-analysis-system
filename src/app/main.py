from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import List

from src.lib.lexicon import load_tab_lexicon
from src.lib.preprocessing import read_text_files
from src.lib.sliding_window import sliding_window_sentiment_analysis
from src.lib.extrema_segments import extrema_segments
from src.lib.visualisation import plot_review_windows, annotate_extrema

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sliding-window sentiment with AFINN and emoticons")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--reviews_subdir", type=str, default="reviews")
    p.add_argument("--afinn_name", type=str, default="AFINN-en-165.txt")
    p.add_argument("--emoticons_name", type=str, default="AFINN-emoticon-8.txt")
    p.add_argument("--window_size", type=int, default=3)
    p.add_argument("--unit", type=str, default="token", choices=["token", "sentence"])
    p.add_argument("--plot_first_n", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--debug", action="store_true", help="Print first-window contributions")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    base = Path(args.data_dir)
    reviews_dir = base / args.reviews_subdir
    afinn_path = base / args.afinn_name
    emoticons_path = base / args.emoticons_name

    reviews = read_text_files(reviews_dir, pattern="*")
    if not reviews:
        print(f"No .txt or .gz reviews found under: {reviews_dir}")
        return

    afinn = load_tab_lexicon(afinn_path)
    emoticons = load_tab_lexicon(emoticons_path)
    print(f"AFINN entries: {len(afinn)}; Emoticons entries: {len(emoticons)}")

    k = args.window_size
    per_review_windows: List[List[tuple[int, int, int]]] = sliding_window_sentiment_analysis(
        reviews, k, afinn, emoticons, unit=args.unit, debug=args.debug
    )

    # Prepare output directory
    outdir = Path(args.save_dir) if args.save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for i, windows in enumerate(per_review_windows, start=1):
        print(f"Review {i} windows: {len(windows)}")
        print("First 20 windows (start,end,score):", windows[:20])

        # CSV dump
        if outdir:
            csv_path = outdir / f"review_{i}_windows.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["start", "end", "score"])
                w.writerows(windows)
            print(f"Wrote: {csv_path}")

        # Extrema
        pos_seg, neg_seg = extrema_segments(windows, k)
        print(f"Review {i}:")
        print(f"  Most positive: {pos_seg}")
        print(f"  Most negative: {neg_seg}")

        # Plot
        if args.plot_first_n and i <= args.plot_first_n and outdir:
            ax = plot_review_windows(windows, k, title=f"Review {i}")
            annotate_extrema(ax, pos_seg, neg_seg)
            img_path = outdir / f"review_{i}.png"
            ax.figure.savefig(img_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {img_path}")

if __name__ == "__main__":
    main()
