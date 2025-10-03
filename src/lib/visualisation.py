import altair as alt
import pandas as pd
import numpy as np

def plot_sentence_scores(scores: list[float], sents: list[list[str]]) -> alt.Chart:
    """
    Create a sorted bar chart for sentiment scores per sentence.

    Args:
        scores: List of sentiment scores per sentence.
        sents: List of tokenized sentences.

    Returns:
        Altair Chart object.
    """
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

    return bar


def plot_sentiment_timeline(scores: list[float], sents: list[list[str]]) -> alt.Chart:
    """
    Create a timeline line chart showing sentiment flow across sentences.

    Args:
        scores: List of sentiment scores per sentence.
        sents: List of tokenized sentences.

    Returns:
        Altair Chart object.
    """
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

    return timeline

def save_chart_as_png(chart: alt.Chart, filename: str):
    """
    Save Altair chart as PNG file.

    Requires 'selenium' and 'altair_saver' packages and a compatible driver installed.

    Args:
        chart: Altair Chart object
        filename: Path to save PNG file
    """
    import altair_saver
    altair_saver.save(chart, filename, method="selenium")


def save_chart_as_html(chart: alt.Chart, filename: str):
    """
    Save Altair chart as standalone HTML file.

    Args:
        chart: Altair Chart object
        filename: Path to save HTML file
    """
    chart.save(filename)
