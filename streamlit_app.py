from __future__ import annotations
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

from src.lib.lexicon import load_tab_lexicon
from src.lib.preprocessing import normalize_text, split_sentences, set_word_segmentation
from src.lib.sentiment_scoring import sentence_scores
from src.lib.sliding_window import sliding_window_sentiment_over_sentences, extrema_segments
from src.lib.varlen_segments import best_varlen_segments_from_scores
from src.lib.visualisation import plot_sentence_scores, plot_sentiment_timeline

st.set_page_config(page_title="AFINN Sentiment Analysis", layout="centered")
st.title("AFINN Sentiment Analysis")

LEX_ROOT = Path("data") / "lexicon"
AFINN_PATH = LEX_ROOT / "AFINN-en-165.txt"
EMOT_PATH = LEX_ROOT / "AFINN-emoticon-8.txt"

try:
    afinn = load_tab_lexicon(str(AFINN_PATH))
    emot  = load_tab_lexicon(str(EMOT_PATH))
except Exception as e:
    st.error(f"Failed to load lexicons: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    st.caption(f"✓ AFINN: {len(afinn)} entries")
    st.caption(f"✓ Emoticons: {len(emot)} entries")
    
    enable_seg = st.checkbox("Enable word segmentation", value=True,
                             help="Split long glued words (>10 chars) like 'thisisapencilbook' → 'this is a pencil book'")
    k = st.slider("Window size (sentences)", 2, 8, 3, 1)

if enable_seg:
    set_word_segmentation(True, afinn.keys())
    st.sidebar.success("✓ Auto-segmentation enabled")
else:
    set_word_segmentation(False, None)

st.subheader("Input")
default_text = "I love this movie! But the ending was bad. :( Also, thisisapen."
text = st.text_area("Enter text to analyze", default_text, height=160)

if st.button("Run Analysis", type="primary"):
    cleaned = normalize_text(text)
    sents = split_sentences(cleaned)
    scores = sentence_scores(cleaned, afinn, emot)
    
    st.subheader("Sentence Analysis")
    if not sents:
        st.warning("No sentences found.")
        st.stop()
    
    for i, toks in enumerate(sents):
        score = scores[i] if i < len(scores) else 0
        st.write(f"**[{i}]** {' '.join(toks)} → Score: **{score}**")
    
    if scores:
        pos_idx = int(np.argmax(scores))
        neg_idx = int(np.argmin(scores))
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Most Positive [{pos_idx}]**\n\nScore: {scores[pos_idx]}\n\n{' '.join(sents[pos_idx])}")
        with col2:
            st.error(f"**Most Negative [{neg_idx}]**\n\nScore: {scores[neg_idx]}\n\n{' '.join(sents[neg_idx])}")
    
    # Visualizations (ONLY ONCE - prevent repeats)
    if scores:
        st.subheader("Score Visualizations")
        st.altair_chart(plot_sentence_scores(scores, sents), use_container_width=True)
        st.altair_chart(plot_sentiment_timeline(scores, sents), use_container_width=True)
    
    # Rest of analysis...
    st.subheader(f"Fixed Window Analysis (k={k})")
    wins = sliding_window_sentiment_over_sentences(scores, k)
    pos_win, neg_win = extrema_segments(wins)
    
    col1, col2 = st.columns(2)
    with col1:
        if pos_win:
            a, b, sumv = pos_win
            st.success(f"**Best Positive Window [{a}-{b}]**\n\nSum: {sumv}")
            for i in range(a, b + 1):
                st.write(f"• {' '.join(sents[i])}")
        else:
            st.info("No positive window")
    
    with col2:
        if neg_win:
            a, b, sumv = neg_win
            st.error(f"**Best Negative Window [{a}-{b}]**\n\nSum: {sumv}")
            for i in range(a, b + 1):
                st.write(f"• {' '.join(sents[i])}")
        else:
            st.info("No negative window")
    
    st.subheader("Variable-Length Segment Analysis")
    pos_seg, neg_seg = best_varlen_segments_from_scores(scores)
    
    col1, col2 = st.columns(2)
    with col1:
        if pos_seg:
            a, b, sumv = pos_seg
            st.success(f"**Best Positive Segment [{a}-{b}]**\n\nSum: {sumv}")
            for i in range(a, b + 1):
                st.write(f"• {' '.join(sents[i])}")
        else:
            st.info("No positive segment")
    
    with col2:
        if neg_seg:
            a, b, sumv = neg_seg
            st.error(f"**Best Negative Segment [{a}-{b}]**\n\nSum: {sumv}")
            for i in range(a, b + 1):
                st.write(f"• {' '.join(sents[i])}")
        else:
            st.info("No negative segment")
