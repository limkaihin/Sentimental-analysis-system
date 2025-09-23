from __future__ import annotations
from typing import List, Tuple, Optional, Sequence, Union
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

def plot_review_windows(
    windows: List[Window],
    k: int,
    title: str | None = None,
) -> plt.Axes:
    if not windows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title or "No windows to plot")
        ax.set_xlabel(f"Window start (k={k})")
        ax.set_ylabel("Sentiment")
        ax.grid(True, alpha=0.3)
        return ax

    xs, ys = _to_xy(windows)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xs, ys, marker="o", linestyle="-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Window start (k={k})")
    ax.set_ylabel("Sentiment")
    ax.set_title(title or "Sliding-window sentiment")
    ax.grid(True, alpha=0.3)
    return ax

def annotate_extrema(
    ax: plt.Axes,
    pos_seg: Segment,
    neg_seg: Segment,
    color_pos: str = "green",
    color_neg: str = "red",
) -> None:
    """
    Shade extrema spans on the x-axis with sign-aware filtering:
    - Shade positive only if score > 0
    - Shade negative only if score < 0
    """
    handles = []
    labels = []

    if pos_seg and pos_seg[2] > 0:
        ps, pe, pv = pos_seg
        h = ax.axvspan(ps, pe, color=color_pos, alpha=0.15, label=f"Most positive (sum={pv})")
        handles.append(h); labels.append("Most positive (sum={pv})")

    if neg_seg and neg_seg[2] < 0:
        ns, ne, nv = neg_seg
        h = ax.axvspan(ns, ne, color=color_neg, alpha=0.15, label=f"Most negative (sum={nv})")
        handles.append(h); labels.append("Most negative (sum={nv})")

    if handles:
        # De-duplicate legend entries
        by_label = {}
        for h, l in zip(handles, labels):
            by_label[l] = h
        ax.legend(by_label.values(), by_label.keys(), frameon=False)

def plot_bar_counts(counts: Dict[str, int], title: str | None = None) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(5, 3))
    order = ["negative", "neutral", "positive"]
    values = [counts.get(k, 0) for k in order]
    colors = ["red", "gray", "green"]
    ax.bar(order, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title(title or "Sentiment class distribution")
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return ax