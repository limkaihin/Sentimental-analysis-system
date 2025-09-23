import streamlit as st
from pathlib import Path
from src.lib.lexicon import load_tab_lexicon
from src.lib.preprocessing import normalize_text, split_sentences
from src.lib.sentiment_scoring import calculate_window_sentiment
from src.lib.sliding_window import sliding_window_sentiment_analysis
from src.lib.extrema_segments import extrema_segments
from src.lib.sentence_scoring import sentences_and_scores, most_positive_negative_sentence
from src.lib.varlen_segments import best_varlen_segments, segment_sentences
from src.lib.word_segmentation import word_break_one, word_break_all
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment & Segmentation", layout="centered")
st.title("Sentiment Analysis and Word Segmentation")

with st.sidebar:
    afinn_path = st.text_input("AFINN path", "data/lexicon/AFINN-en-165.txt")
    emot_path = st.text_input("Emoticon path", "data/lexicon/AFINN-emoticon-8.txt")
    afinn = load_tab_lexicon(afinn_path)
    emot = load_tab_lexicon(emot_path)

tab1, tab2 = st.tabs(["Analysis", "Word Segmentation"])

with tab1:
    text = st.text_area("Enter text", "Good movie. Bad acting. Very good ending!")
    unit = st.selectbox("Unit", ["sentence","token"], index=0)
    k = st.slider("Window size (k)", 1, 10, 2)
    if st.button("Run analysis"):
        cleaned = normalize_text(text)
        sents, scores = sentences_and_scores(cleaned, afinn, emot)
        st.write("Sentence scores:", scores)
        max_i, min_i = most_positive_negative_sentence(scores) if scores else (None, None)
        if max_i is not None and scores[max_i] > 0: st.success(f"Most positive sentence [{max_i}]: {' '.join(sents[max_i])}")
        if min_i is not None and scores[min_i] < 0: st.error(f"Most negative sentence [{min_i}]: {' '.join(sents[min_i])}")

        # Fixed-k windows
        windows = sliding_window_sentiment_analysis([cleaned], k, afinn, emot, unit=unit)[0]
        st.write(f"{k}-{unit} windows:", windows)
        pos_win, neg_win = extrema_segments(windows, k)
        if pos_win and pos_win[2] > 0: st.success(f"Most positive window: {pos_win}")
        if neg_win and neg_win[2] < 0: st.error(f"Most negative window: {neg_win}")

        # Varlen
        scores2, pos_seg, neg_seg = best_varlen_segments(cleaned, afinn, emot)
        st.write("Arbitrary-length sentence scores:", scores2)
        if pos_seg and pos_seg[2] > 0:
            st.success(f"Best positive segment: {pos_seg}")
            st.write(segment_sentences(cleaned, pos_seg))
        if neg_seg and neg_seg[2] < 0:
            st.error(f"Best negative segment: {neg_seg}")
            st.write(segment_sentences(cleaned, neg_seg))

        # Line plot
        if windows:
            fig1, ax1 = plt.subplots(figsize=(8,3))
            xs = [w[0] for w in windows]
            ys = [w[-1] for w in windows]
            ax1.plot(xs, ys, marker="o")
            ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax1.set_title("Sliding-window sentiment")
            st.pyplot(fig1)

        # Histogram
        if scores:
            counts = {"negative": sum(1 for s in scores if s < 0), "neutral": sum(1 for s in scores if s == 0), "positive": sum(1 for s in scores if s > 0)}
            fig2, ax2 = plt.subplots(figsize=(5,3))
            ax2.bar(list(counts.keys()), list(counts.values()), color=["red","gray","green"])
            ax2.set_title("Sentence sentiment distribution")
            st.pyplot(fig2)

with tab2:
    raw = st.text_input("Spaceless string", "thisisapen")
    vocab_text = st.text_area("Dictionary (one word per line)", "this\nis\na\npen")
    all_solutions = st.checkbox("Show all segmentations", value=False)
    max_solutions = st.number_input("Max solutions", min_value=1, max_value=1000, value=20)
    if st.button("Segment"):
        vocab = {w.strip().lower() for w in vocab_text.splitlines() if w.strip()}
        if all_solutions:
            st.write(word_break_all(raw.lower(), vocab, max_solutions=int(max_solutions)))
        else:
            st.write(word_break_one(raw.lower(), vocab))
