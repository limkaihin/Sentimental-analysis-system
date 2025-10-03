# streamlit_app.py (COMPLETE FINAL)
from __future__ import annotations
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt

from src.lib.lexicon import load_tab_lexicon
from src.lib.preprocessing import normalize_text, split_sentences, set_word_segmentation
from src.lib.sentiment_scoring import sentence_scores
from src.lib.sliding_window import sliding_window_sentiment_over_sentences, extrema_segments
from src.lib.varlen_segments import best_varlen_segments_from_scores

st.set_page_config(page_title="AFINN Sentiment Analysis", layout="centered")
st.title("AFINN Sentiment Analysis")

LEX_ROOT = Path("data") / "lexicon"
AFINN_PATH = LEX_ROOT / "AFINN-en-165.txt"
EMOT_PATH  = LEX_ROOT / "AFINN-emoticon-8.txt"

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
    
    # Visualizations
    if scores:
        st.subheader("Score Visualizations")
        
        # 1. Sorted bar chart
        bar_df = pd.DataFrame({
            "Sentence": list(range(len(scores))),
            "Score": scores,
            "Text": [" ".join(t)[:50] + "..." if len(" ".join(t)) > 50 else " ".join(t) for t in sents],
        }).sort_values("Score", ascending=False)
        
        bar_df["Sentiment"] = np.where(bar_df["Score"] >= 0, "Positive/Neutral", "Negative")
        
        bar = alt.Chart(bar_df).mark_bar().encode(
            x=alt.X("Score:Q", title="Sentence Score"),
            y=alt.Y("Sentence:O", sort="-x", title="Sentence Index (sorted by score)"),
            color=alt.Color("Sentiment:N",
                          scale=alt.Scale(domain=["Negative", "Positive/Neutral"],
                                        range=["#e74c3c", "#3498db"]),
                          legend=alt.Legend(title="Sentiment")),
            tooltip=["Sentence:O", "Score:Q", "Text:N"]
        ).properties(height=max(250, 30 * len(scores)), width=700)
        
        st.altair_chart(bar, use_container_width=True)
        
        # 2. Timeline/sequence chart (replaces confusing histogram)
        timeline_df = pd.DataFrame({
            "Sentence Index": list(range(len(scores))),
            "Score": scores,
            "Sentiment": ["Positive" if s > 0 else ("Negative" if s < 0 else "Neutral") for s in scores],
            "Text": [" ".join(t)[:40] + "..." if len(" ".join(t)) > 40 else " ".join(t) for t in sents]
        })
        
        timeline = alt.Chart(timeline_df).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Sentence Index:Q", title="Sentence Position in Text"),
            y=alt.Y("Score:Q", title="Sentiment Score", scale=alt.Scale(zero=False)),
            color=alt.Color("Sentiment:N",
                          scale=alt.Scale(domain=["Negative", "Neutral", "Positive"],
                                        range=["#e74c3c", "#95a5a6", "#2ecc71"]),
                          legend=alt.Legend(title="Sentiment")),
            tooltip=["Sentence Index:Q", "Score:Q", "Text:N"]
        ).properties(height=300, width=700, title="Sentiment Flow Across Text")
        
        st.altair_chart(timeline, use_container_width=True)
    
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
