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
    for seg, color, label in [
        (pos_seg, color_pos, "Most positive"),
        (neg_seg, color_neg, "Most negative"),
    ]:
        if seg is None:
            continue
        start, end, s = seg
        ax.axvspan(start, end, color=color, alpha=0.15, label=f"{label} (sum={s})")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False)
