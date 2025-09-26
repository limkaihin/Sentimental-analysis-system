from __future__ import annotations

import streamlit as st
from pathlib import Path

# Core pipeline
from src.lib.lexicon import load_tab_lexicon
from src.lib.preprocessing import normalize_text, tokenize, tokenize_smart, set_word_segmentation
from src.lib.sentiment_scoring import calculate_window_sentiment
from src.lib.sentence_scoring import sentences_and_scores, most_positive_negative_sentence
from src.lib.sliding_window import sliding_window_sentiment_analysis
from src.lib.extrema_segments import extrema_segments
from src.lib.varlen_segments import best_varlen_segments, segment_sentences
from src.lib.word_segmentation import word_break_one, word_break_all

# Visuals
from src.lib.visualisation import plot_review_windows, annotate_extrema, plot_bar_counts

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sentiment & Segmentation", layout="centered")
st.title("Sentiment Analysis and Word Segmentation")

# -----------------------------------------------------------------------------
# Sidebar configuration (no hardcoding; all paths and options are user-settable)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    data_root = st.text_input("Data root", value="data")
    lex_root = st.text_input("Lexicon subfolder", value="lexicon")
    afinn_name = st.text_input("AFINN filename", value="AFINN-en-165.txt")
    emot_name = st.text_input("Emoticon filename", value="AFINN-emoticon-8.txt")

    # Resolve paths relative to data root
    afinn_path = Path(data_root) / lex_root / afinn_name
    emot_path = Path(data_root) / lex_root / emot_name

    # Load lexicons once per run
    afinn = load_tab_lexicon(str(afinn_path))
    emot = load_tab_lexicon(str(emot_path))

    st.caption(f"Loaded AFINN: {afinn_path}")
    st.caption(f"Loaded Emoticons: {emot_path}")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Analysis", "Word Segmentation"])

# -----------------------------------------------------------------------------
# Analysis tab
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Input")
    source = st.radio("Input source", ["Manual", "Upload .txt/.csv"], horizontal=True, key="input_source")
    uploaded_docs = []
    selected_text_from_upload = None
    if source == "Upload .txt/.csv":
        uploads = st.file_uploader("Upload one or more .txt or .csv files", type=["txt", "csv"], accept_multiple_files=True)
        if uploads:
            import csv
            for f in uploads:
                name = f.name
                if name.lower().endswith(".txt"):
                    try:
                        content = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        content = f.read().decode("latin-1", errors="ignore")
                    uploaded_docs.append((name, content))
                elif name.lower().endswith(".csv"):
                    try:
                        text = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                    	text = f.read().decode("latin-1", errors="ignore")
                    rows = list(csv.reader(text.splitlines()))
                    if rows:
                        headers = rows[0]
                        data_rows = rows[1:]
                        st.info(f"CSV detected: **{name}** with {len(data_rows)} rows and {len(headers)} columns.")
                        if headers:
                            col = st.selectbox(f"Pick text column for {name}", headers, key=f"col_{name}")
                            col_idx = headers.index(col)
                            mode = st.radio(f"How to use rows from {name}?", ["Pick one row", "Combine all rows"], key=f"mode_{name}", horizontal=True)
                            if mode == "Pick one row":
                                if data_rows:
                                    row_i = st.number_input(f"Row index (1..{len(data_rows)}) for {name}", min_value=1, max_value=len(data_rows), value=1, step=1, key=f"row_{name}")
                                    val = data_rows[row_i-1][col_idx] if col_idx < len(data_rows[row_i-1]) else ""
                                    uploaded_docs.append((f"{name} [row {row_i}::{col}]", val))
                            else:
                                vals = [r[col_idx] for r in data_rows if col_idx < len(r)]
                                uploaded_docs.append((f"{name} [all::{col}]", "\n\n".join(vals)))
                        else:
                            st.warning(f"No headers found in {name}; please ensure the first row contains column names.")
                    else:
                        st.warning(f"{name} appears to be empty.")
        if uploaded_docs:
            labels = [f"{i+1}. {nm}" for i,(nm,_) in enumerate(uploaded_docs)]
            pick = st.selectbox("Select an uploaded document to analyze", labels, index=0)
            idx = labels.index(pick)
            selected_text_from_upload = uploaded_docs[idx][1]
            st.caption("Preview (first 400 chars):")
            st.code(selected_text_from_upload[:400])

    text = st.text_area(
        "Enter text",
        "Good movie. Bad acting. Very good ending!",
        height=120,
        placeholder="Paste or type a review/document here...",
    )
    unit = st.selectbox("Sliding-window unit", ["sentence", "token"], index=0)
    seg_enabled = st.checkbox("Split glued words (e.g., iamsad → i am sad)", value=True)
    seg_fuzzy   = st.checkbox("Fuzzy split unknown words (helps with thismovieistrash)", value=True)
    common = {"i","am","is","are","was","were","be","being","been","very","not","no","good","bad","sad","happy","movie","film","acting","the","a","an","this","that","trash","great","awful","amazing","terrible"}
    vocab = set(afinn.keys()) | common
    set_word_segmentation(seg_enabled, vocab, fuzzy=seg_fuzzy)

    k = st.slider("Window size (k)", min_value=3, max_value=15, value=3, help="Fixed-size window length")

    if selected_text_from_upload:
        text = selected_text_from_upload
    if st.button("Run analysis", type="primary"):
        # Normalize once
        cleaned = normalize_text(text)

        # ---------------------------------------------------------------------
        # Official project score (full-window over tokens)
        # ---------------------------------------------------------------------
        tokens = tokenize_smart(cleaned)
        official_score = calculate_window_sentiment(tokens, afinn, emot, debug=False)
        st.subheader("Official project score")
        # Show score depending on sliding-window unit without changing other logic
        if unit == "sentence":
            # Reuse the same pipeline to compute a full-window sentence score
            _sents_fw, _sent_scores_fw = sentences_and_scores(cleaned, afinn, emot)
            _full_window_sentence_score = sum(_sent_scores_fw) if _sent_scores_fw else 0
            st.write(f"Full-window sentence score: {_full_window_sentence_score}")
        else:
            st.write(f"Full-window token score: {official_score}")

        # Overall score summary (added)
        try:
            # Overall review score as sum of sentence scores for easy interpretability
            # (keeps existing logic intact; purely presentational)
            overall_review_score = sum(sent_scores) if 'sent_scores' in locals() and sent_scores else official_score
        except Exception:
            overall_review_score = official_score

        st.metric(label="Overall Review Score", value=overall_review_score)


        # ---------------------------------------------------------------------
        # Per-sentence scores and most positive/negative single sentences
        # ---------------------------------------------------------------------
        sents, sent_scores = sentences_and_scores(cleaned, afinn, emot)
        st.subheader("Sentence scores")
        st.write(sent_scores)

        pos_idx, neg_idx = most_positive_negative_sentence(sent_scores) if sent_scores else (None, None)
        if pos_idx is not None and sent_scores[pos_idx] > 0:
            st.success(f"Most positive sentence [{pos_idx}]: {' '.join(sents[pos_idx])}")
        else:
            st.info("Most positive sentence: none (> 0 not found)")

        if neg_idx is not None and sent_scores[neg_idx] < 0:
            st.error(f"Most negative sentence [{neg_idx}]: {' '.join(sents[neg_idx])}")
        else:
            st.info("Most negative sentence: none (< 0 not found)")

        # ---------------------------------------------------------------------
        # Fixed-k windows
        # ---------------------------------------------------------------------
        st.subheader(f"Fixed-k windows (k={k}, unit={unit})")
        windows = sliding_window_sentiment_analysis([cleaned], k, afinn, emot, unit=unit)[0]
        st.write(windows)

        pos_win, neg_win = extrema_segments(windows, k)
        if pos_win and pos_win[2] > 0:
            st.success(f"Most positive window: {pos_win}")
        else:
            st.info("Most positive window: none (> 0 not found)")

        if neg_win and neg_win[2] < 0:
            st.error(f"Most negative window: {neg_win}")
        else:
            st.info("Most negative window: none (< 0 not found)")

        # Plot fixed-k line chart with shaded extrema
        if windows:
            ax = plot_review_windows(windows, k=k, title="Sliding-window sentiment")
            annotate_extrema(
                ax,
                pos_win if (pos_win and pos_win[2] > 0) else None,
                neg_win if (neg_win and neg_win[2] < 0) else None,
            )
            st.pyplot(ax.figure)

        # ---------------------------------------------------------------------
        # Variable-length (Kadane) spans over sentences
        # ---------------------------------------------------------------------
        st.subheader("Arbitrary-length sentence segments")
        scores2, pos_seg, neg_seg = best_varlen_segments(cleaned, afinn, emot)
        st.write("Sentence scores (again for reference):", scores2)

        # Positive segment list (sign-filtered)
        if pos_seg and pos_seg[2] > 0:
            ps, pe, pv = pos_seg
            st.success(f"Best positive segment: {pos_seg}")
            pos_items = segment_sentences(cleaned, pos_seg)
            st.write([s for i, s in enumerate(pos_items, start=ps) if scores2[i] > 0])
        else:
            st.info("Best positive segment: none (> 0 not found)")

        # Negative segment list (sign-filtered)
        if neg_seg and neg_seg[2] < 0:
            ns, ne, nv = neg_seg
            st.error(f"Best negative segment: {neg_seg}")
            neg_items = segment_sentences(cleaned, neg_seg)
            st.write([s for i, s in enumerate(neg_items, start=ns) if scores2[i] < 0])
        else:
            st.info("Best negative segment: none (< 0 not found)")

        # ---------------------------------------------------------------------
        # Histogram of sentence labels
        # ---------------------------------------------------------------------
        if sent_scores:
            counts = {
                "negative": sum(1 for s in sent_scores if s < 0),
                "neutral": sum(1 for s in sent_scores if s == 0),
                "positive": sum(1 for s in sent_scores if s > 0),
            }
            axh = plot_bar_counts(
                counts,
                title="Sentence sentiment distribution",
                xlabel="Sentiment class",
                ylabel="Count",
            )
            st.pyplot(axh.figure)

# -----------------------------------------------------------------------------
# Word segmentation tab
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Dictionary-based segmentation")
    raw = st.text_input("Spaceless string", "thisisapen")
    vocab_text = st.text_area("Dictionary (one word per line)", "this\nis\na\npen")
    all_solutions = st.checkbox("Show all segmentations", value=False)
    max_solutions = st.number_input("Max solutions", min_value=1, max_value=1000, value=20, step=1)

    if st.button("Segment", key="segment"):
        vocab = {w.strip().lower() for w in vocab_text.splitlines() if w.strip()}
        if all_solutions:
            st.write(word_break_all(raw.lower(), vocab, max_solutions=int(max_solutions)))
        else:
            st.write(word_break_one(raw.lower(), vocab))
