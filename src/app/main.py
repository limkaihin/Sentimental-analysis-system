# src/app/main.py
from __future__ import annotations

import argparse
from pathlib import Path

# ---- package imports (relative) ----
from ..lib.lexicon import load_tab_lexicon
from ..lib.preprocessing import normalize_text as clean_text, tokenize as tokenize_text
from ..lib.sentiment_scoring import calculate_window_sentiment as score_text
# from ..lib.sliding_window import sliding_window            # keep if you use it
# from ..lib.visualisation import plot_review_windows, ...   # imported lazily below if --plot

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

    # You had these in your script — keep them for compatibility
    p.add_argument("--window_size", type=int, default=5, help="Size for any sliding-window ops")
    p.add_argument("--unit", type=str, choices={"token", "sentence"}, default="token")
    p.add_argument("--plot_first_n", type=int, default=0, help="Plot first N items (dataset mode)")
    p.add_argument("--save_dir", type=str, default=None, help="If set, save plots/files here")
    p.add_argument("--debug", action="store_true")

    # NEW: single-review inputs
    p.add_argument("--text", type=str, default=None,
                   help="Analyse a single review passed directly on the command line.")
    p.add_argument("--input_file", type=str, default=None,
                   help="Analyse a single review read from a .txt file.")

    # Quality-of-life
    p.add_argument("--limit", type=int, default=None, help="When in dataset mode, process only first N reviews")
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
    combined = combine_lexicons(afinn, emoticon)  # adjust if your scorer expects separate dicts

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
            for cut in (("ing", -3), ("ed", -2), ("es", -2), ("s", -1)):
                suf, n = cut
                if len(base) > abs(n) + 1 and base.endswith(suf):
                    stem = base[:n]
                    if stem in afinn: return afinn[stem]
            return 0

        # compute a plain lexicon sum for transparency
        lexicon_sum = sum(lookup(t) for t in tokens)

        # also compute your project’s official score (windowed) so both are visible
        official_score = score_text(
            cleaned,
            afinn,
            emoticon,
        )

        # show per-token contributions (use --debug to see this)
        if getattr(args, "debug", False):
            print("\n[debug] token contributions:")
            for t in tokens:
                sc = lookup(t)
                if sc != 0:
                    print(f"  {t:>12}  -> {sc}")

        # final label from the lexicon_sum (simple, intuitive)
        label = "positive" if lexicon_sum > 0 else ("negative" if lexicon_sum < 0 else "neutral")

        print("=== Sentiment Analysis (single review) ===")
        print("Original:", review_text.strip())
        print("Cleaned :", cleaned)
        print("Score   :", lexicon_sum, "(simple lexicon sum)")
        print("Label   :", label)
        print("(project official score:", official_score, ")")

        raise SystemExit(0)


    # ---------- DATASET MODE (runs only if no --text/--input_file) ----------
    if not reviews_dir.exists():
        print(f"(Info) Reviews folder not found: {reviews_dir}")
        print("(Info) Use --text or --input_file to analyse a single review.")
        return

    processed = 0
    for txt in reviews_dir.rglob("*.txt"):
        text = read_text_file(txt)
        cleaned = clean_text(review_text)
        score = score_text(
            cleaned,
            afinn,                 # AFINN dict you already loaded
            emoticon,              # emoticons dict you already loaded
        )
        label = label_from_score(score)

        if args.debug:
            print(f"[{txt}] score={score} label={label}")

        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

        # Optional plotting in dataset mode (first N only)
        if args.plot and args.plot_first_n and processed <= args.plot_first_n:
            try:
                from ..lib.visualisation import plot_review_windows  # type: ignore
                out_dir = resolve_path(args.save_dir) if args.save_dir else None
                plot_review_windows([cleaned], save_dir=out_dir)
            except Exception as e:
                if args.debug:
                    print("Plotting error:", e)


if __name__ == "__main__":
    main()
