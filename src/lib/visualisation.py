from __future__ import annotations
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

Segment = Optional[Tuple[int, int, int]]

def plot_review_windows(
    windows: List[Tuple[int, int]],
    k: int,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot the sequence of window sentiment scores versus window start index.
    Caller decides whether to save/show the figure to keep this module I/O-free.
    """
    if not windows:
        fig, ax = plt.subplots()
        ax.set_title(title or "No windows to plot")
        ax.set_xlabel(f"Window start (k={k})")
        ax.set_ylabel("Sentiment")
        return ax

    starts = [s for s, _ in windows]
    scores = [v for _, v in windows]
    fig, ax = plt.subplots()
    ax.plot(starts, scores, marker="o", linestyle="-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Window start (k={k})")
    ax.set_ylabel("Sentiment")
    ax.set_title(title or "Sliding-window sentiment")
    return ax

def annotate_extrema(
    ax: plt.Axes,
    pos_seg: Segment,
    neg_seg: Segment,
    color_pos: str = "green",
    color_neg: str = "red",
) -> None:
    """
    Shade the most positive and negative segments on the plot, if provided.
    """
    for seg, color, label in [(pos_seg, color_pos, "Most positive"),
                              (neg_seg, color_neg, "Most negative")]:
        if seg is None:
            continue
        start, end, s = seg
        ax.axvspan(start, end, color=color, alpha=0.15, label=f"{label} (sum={s})")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False)
