# src/app/main.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# ---- package imports (relative) ----
from ..lib.lexicon import load_tab_lexicon
from ..lib.preprocessing import normalize_text as clean_text, tokenize as tokenize_text, split_sentences
from ..lib.sentiment_scoring import calculate_window_sentiment as score_text
from ..lib.sliding_window import sliding_window_sentiment_analysis
from ..lib.extrema_segments import extrema_segments

# Optional plotting (loaded only if --plot is used)
try:
    from ..lib.visualisation import plot_review_windows, annotate_extrema  # type: ignore
except Exception:
    plot_review_windows = None
    annotate_extrema = None

# ---------- helpers ----------
ROOT = Path(__file__).resolve().parents[2]  # project root (folder that contains "src")

def resolve_path(p: Path | str) -> Path:
    """Resolve to absolute; if relative, treat as relative to project root."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def combine_lexicons(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out

def label_from_score(score: float) -> str:
    return "positive" if score > 0 else ("negative" if score < 0 else "neutral")

# ----- local helpers for arbitrary-length segments (Kadane) -----
Segment = Optional[Tuple[int, int, int]]  # (start, end, sum)

def _max_subarray(scores: List[int]) -> Segment:
    if not scores:
        return None
    best = cur = scores[0]
    start = end = s = 0
    for i in range(1, len(scores)):
        if cur + scores[i] < scores[i]:
            cur = scores[i]; s = i
        else:
            cur += scores[i]
        if cur > best:
            best = cur; start = s; end = i
    return (start, end, best)

def _min_subarray(scores: List[int]) -> Segment:
    if not scores:
        return None
    best = cur = scores[0]
    start = end = s = 0
    for i in range(1, len(scores)):
        if cur + scores[i] > scores[i]:
            cur = scores[i]; s = i
        else:
            cur += scores[i]
        if cur < best:
            best = cur; start = s; end = i
    return (start, end, best)

# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentiment Analysis System")

    # Paths (folders kept separate from filenames to avoid double-prefix)
    p.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    p.add_argument("--reviews_subdir", type=str, default="reviews", help="Reviews subfolder inside data_dir")
    p.add_argument("--lexicon_subdir", type=str, default="lexicon", help="Lexicon subfolder inside data_dir")

    # Lexicon filenames ONLY (not full paths)
    p.add_argument("--afinn_name", type=str, default="AFINN-en-165.txt")
    p.add_argument("--emoticons_name", type=str, default="AFINN-emoticon-8.txt")

    # Compatibility flags
    p.add_argument("--window_size", type=int, default=5, help="Size for sliding-window ops")
    p.add_argument("--unit", type=str, choices={"token", "sentence"}, default="token")
    p.add_argument("--plot_first_n", type=int, default=0, help="Plot first N items (dataset mode)")
    p.add_argument("--save_dir", type=str, default=None, help="If set, save plots/files here")
    p.add_argument("--debug", action="store_true")

    # Single-review inputs
    p.add_argument("--text", type=str, default=None, help="Analyse a single review passed on the command line.")
    p.add_argument("--input_file", type=str, default=None, help="Analyse a single review from a .txt file.")

    # Dataset controls
    p.add_argument("--limit", type=int, default=None, help="Process only first N reviews (dataset mode)")
    p.add_argument("--plot", action="store_true", help="Enable plotting for single-review or dataset mode")

    return p

# ---------- main ----------
def main() -> None:
    args = build_parser().parse_args()

    # Canonical paths
    data_dir = resolve_path(args.data_dir)
    reviews_dir = data_dir / args.reviews_subdir
    lexicon_dir = data_dir / args.lexicon_subdir

    afinn_path = resolve_path(lexicon_dir / args.afinn_name)
    emoticon_path = resolve_path(lexicon_dir / args.emoticons_name)

    # Load lexicons
    if not afinn_path.exists():
        raise FileNotFoundError(f"AFINN file not found: {afinn_path}")
    if not emoticon_path.exists():
        raise FileNotFoundError(f"Emoticon file not found: {emoticon_path}")

    afinn = load_tab_lexicon(afinn_path)
    emoticon = load_tab_lexicon(emoticon_path)
    _ = combine_lexicons(afinn, emoticon)  # kept for compatibility

    # ---------- SINGLE-REVIEW FAST PATH ----------
    review_text: str | None = None
    if args.text:
        review_text = args.text
    elif args.input_file:
        ip = resolve_path(args.input_file)
        if not ip.exists():
            raise FileNotFoundError(f"Input file not found: {ip}")
        review_text = read_text_file(ip)

    if review_text is not None:
        cleaned = clean_text(review_text)
        tokens = tokenize_text(cleaned)

        # helper: try a few variants to hit lexicon entries
        def lookup(tok: str) -> int:
            # 1) exact
            if tok in afinn: return afinn[tok]
            if tok in emoticon: return emoticon[tok]
            # 2) strip simple punct
            base = tok.strip(".,!?;:\"'()[]{}")
            if base in afinn: return afinn[base]
            if base in emoticon: return emoticon[base]
            # 3) crude lemmatization fallbacks
            for suf, n in (("ing", -3), ("ed", -2), ("es", -2), ("s", -1)):
                if len(base) > abs(n) + 1 and base.endswith(suf):
                    stem = base[:n]
                    if stem in afinn: return afinn[stem]
            return 0

        # 1) Simple lexicon sum (per token)
        simple_sum = sum(lookup(t) for t in tokens)

        # 2) Official score over the full window: PASS TOKENS (bug fix)
        official_score = score_text(tokens, afinn, emoticon, debug=args.debug)

        print("=== Sentiment Analysis (single review) ===")
        print("Original:", review_text.strip())
        print("Cleaned :", cleaned)
        print("Simple lexicon sum:", simple_sum, "(per-token lookup)")
        print("Official score     :", official_score, "(full-window over tokens)")

        if args.debug:
            print("\n[debug] token contributions (non-zero):")
            for t in tokens:
                sc = lookup(t)
                if sc != 0:
                    print(f"  {t:>12} -> {sc}")

        # 3) Per-sentence scores + extrema sentences
        sent_tokens = split_sentences(cleaned)
        sent_scores = [score_text(st, afinn, emoticon, debug=False) for st in sent_tokens]
        print("\n--- Sentence scores ---")
        if not sent_scores:
            print("No sentences found.")
        else:
            for i, (toks, sc) in enumerate(zip(sent_tokens, sent_scores)):
                print(f"[sent {i}] score={sc} :: {' '.join(toks)}")
            max_idx = max(range(len(sent_scores)), key=lambda i: sent_scores[i]) if sent_scores else None
            min_idx = min(range(len(sent_scores)), key=lambda i: sent_scores[i]) if sent_scores else None
            if max_idx is not None:
                print(f"Most positive sentence: idx={max_idx}, score={sent_scores[max_idx]}")
            if min_idx is not None:
                print(f"Most negative sentence: idx={min_idx}, score={sent_scores[min_idx]}")

        # 4) Fixed-size windows (token or sentence unit)
        if args.window_size > 0:
            sw = sliding_window_sentiment_analysis(
                reviews=[cleaned],
                k=args.window_size,
                afinn=afinn,
                emoticons=emoticon,
                unit=args.unit,
                debug=False,
            )[0]
            print(f"\n--- {args.window_size}-{args.unit} windows ---")
            if not sw:
                print("No windows available (text too short for k).")
            else:
                for (start, end, sc) in sw:
                    print(f"[win {start}-{end}] score={sc}")
                pos_seg, neg_seg = extrema_segments(sw, args.window_size)
                if pos_seg:
                    print(f"Most positive window : {pos_seg[0]}-{pos_seg[1]} score={pos_seg[2]}")
                if neg_seg:
                    print(f"Most negative window : {neg_seg[0]}-{neg_seg[1]} score={neg_seg[2]}")

        # 5) Arbitrary-length segments over sentences (Kadane)
        if sent_scores:
            pos_seg = _max_subarray(sent_scores)
            neg_seg = _min_subarray(sent_scores)
            print("\n--- Arbitrary-length segments (sentences) ---")
            print("Sentence scores:", sent_scores)
            if pos_seg:
                ps, pe, pv = pos_seg
                print(f"Best positive segment: {ps}-{pe} sum={pv}")
                for i in range(ps, pe + 1):
                    print(f"  [sent {i}] {' '.join(sent_tokens[i])}")
            if neg_seg:
                ns, ne, nv = neg_seg
                print(f"Best negative segment: {ns}-{ne} sum={nv}")
                for i in range(ns, ne + 1):
                    print(f"  [sent {i}] {' '.join(sent_tokens[i])}")

        # Optional plotting for single review
        if args.plot and plot_review_windows is not None:
            # Plot whichever unit was requested, then shade fixed-size extrema if available
            sw = sliding_window_sentiment_analysis(
                reviews=[cleaned],
                k=args.window_size,
                afinn=afinn,
                emoticons=emoticon,
                unit=args.unit,
                debug=False,
            )[0]
            ax = plot_review_windows(sw, k=args.window_size, title="Sliding-window sentiment")
            # Shade extrema for fixed-size windows
            if sw and annotate_extrema is not None:
                pos_seg, neg_seg = extrema_segments(sw, args.window_size)
                annotate_extrema(ax, pos_seg, neg_seg, k=args.window_size)
            # Save if requested
            if args.save_dir:
                out_dir = resolve_path(args.save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ax.figure.tight_layout()
                ax.figure.savefig(out_dir / "single_review_windows.png", bbox_inches="tight")
        raise SystemExit(0)

    # ---------- DATASET MODE (runs only if no --text/--input_file) ----------
    if not reviews_dir.exists():
        print(f"(Info) Reviews folder not found: {reviews_dir}")
        print("(Info) Use --text or --input_file to analyse a single review.")
        return

    processed = 0
    for txt in reviews_dir.rglob("*.txt"):
        text = read_text_file(txt)
        cleaned = clean_text(text)  # BUG FIX: was clean_text(review_text)

        tokens = tokenize_text(cleaned)
        score = score_text(tokens, afinn, emoticon, debug=False)
        label = label_from_score(score)

        if args.debug:
            print(f"[{txt}] score={score} label={label}")

        # Optional plotting in dataset mode (first N only)
        if args.plot and args.plot_first_n and processed < args.plot_first_n and plot_review_windows is not None:
            sw = sliding_window_sentiment_analysis(
                reviews=[cleaned],
                k=args.window_size,
                afinn=afinn,
                emoticons=emoticon,
                unit=args.unit,
                debug=False,
            )[0]
            ax = plot_review_windows(sw, k=args.window_size, title=txt.name)
            if sw and annotate_extrema is not None:
                pos_seg, neg_seg = extrema_segments(sw, args.window_size)
                annotate_extrema(ax, pos_seg, neg_seg, k=args.window_size)
            if args.save_dir:
                out_dir = resolve_path(args.save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ax.figure.tight_layout()
                ax.figure.savefig(out_dir / f"{txt.stem}_windows.png", bbox_inches="tight")

        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

if __name__ == "__main__":
    main()
