# src/lib/visualisation.py
from __future__ import annotations
from typing import List, Tuple, Optional, Sequence, Union, Dict

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

Window = Union[Tuple[int, int], Tuple[int, int, int]]
Segment = Optional[Tuple[int, int, int]]

def _to_xy(windows: Sequence[Window]) -> Tuple[List[int], List[int]]:
    xs: List[int] = []
    ys: List[int] = []
    for w in windows:
        if len(w) == 2:
            start, score = w  # (start, score)
        else:
            start, _end, score = w  # (start, end, score)
        xs.append(int(start))
        ys.append(int(score))
    return xs, ys

def plot_review_windows(windows: List[Window], k: int, title: str | None = None) -> plt.Axes:
    """Line plot of sliding-window sentiment with labels, grid, and zero baseline."""
    xs, ys = _to_xy(windows)
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    if not windows:
        ax.set_title(title or "No windows to plot")
        ax.set_xlabel(f"Window start index (k={k})")
        ax.set_ylabel("Sentiment score")
        ax.grid(True, alpha=0.3)
        return ax

    ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color="#1f77b4")
    ax.axhline(0, linestyle="--", linewidth=1, color="0.5")  # zero baseline
    ax.set_xticks(xs)  # integer starts only
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    # nice y-lims with padding
    ymin, ymax = min(ys), max(ys)
    pad = max(1, int(0.1 * (ymax - ymin or 10)))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(title or "Sliding-window sentiment")
    ax.set_xlabel(f"Window start index (k={k})")
    ax.set_ylabel("Sentiment score")
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax

def annotate_extrema(
    ax: plt.Axes,
    pos_seg: Segment,
    neg_seg: Segment,
    color_pos: str = "green",
    color_neg: str = "red",
) -> None:
    """Shade the most positive and most negative segments on the window plot."""
    if pos_seg is not None:
        ps, pe, _ = pos_seg
        ax.axvspan(ps - 0.5, pe + 0.5, color=color_pos, alpha=0.15, label="most positive")
    if neg_seg is not None:
        ns, ne, _ = neg_seg
        ax.axvspan(ns - 0.5, ne + 0.5, color=color_neg, alpha=0.15, label="most negative")
    # Add legend only if something was drawn
    if (pos_seg is not None) or (neg_seg is not None):
        handles, labels = ax.get_legend_handles_labels()
        # deduplicate labels
        seen = set(); kept_h, kept_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                kept_h.append(h); kept_l.append(l); seen.add(l)
        ax.legend(kept_h, kept_l, frameon=False, loc="best")

def plot_bar_counts(
    counts: Dict[str, int],
    title: str = "Sentence sentiment distribution",
    xlabel: str = "Sentiment class",
    ylabel: str = "Count",
) -> plt.Axes:
    """Neat labeled bar chart for sentiment class counts."""
    labels = ["negative", "neutral", "positive"]
    values = [int(counts.get(k, 0)) for k in labels]
    colors = ["#d62728", "#7f7f7f", "#2ca02c"]  # red, gray, green

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # integer y-ticks
    ymax = max(values + [1])
    ax.set_ylim(0, ymax + 0.5)
    ax.set_yticks(range(0, ymax + 1))

    # value labels on bars
    for b in bars:
        v = int(b.get_height())
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)
    return ax
