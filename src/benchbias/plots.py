"""Figures. Sequential single hue for shares; position (small multiples) instead of 11 hues for categories."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HIGHLIGHT, CONTEXT = "#2a6fdb", "#d0d0d0"


def category_cluster_heatmap(df: pd.DataFrame, title: str, path: str | Path | None = None):
    d = df[df["cluster"] != -1]
    ct = pd.crosstab(d["category"], d["cluster"], normalize="index")
    ct = ct[ct.idxmax(0).sort_values().index]
    fig, ax = plt.subplots(figsize=(min(18, 2 + 0.18 * ct.shape[1]), 4.5))
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues", vmin=0, vmax=ct.values.max())
    ax.set_yticks(range(len(ct.index)), ct.index, fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel("clusters (ordered by dominant category)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, label="share of category's items", shrink=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def small_multiples(xy: np.ndarray, categories, title: str, path: str | Path | None = None, ncols: int = 4):
    categories = np.asarray(categories)
    cats = sorted(set(categories))
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.3 * nrows), sharex=True, sharey=True)
    for ax, cat in zip(axes.flat, cats):
        m = categories == cat
        ax.scatter(xy[~m, 0], xy[~m, 1], s=2, c=CONTEXT, linewidths=0)
        ax.scatter(xy[m, 0], xy[m, 1], s=4, c=HIGHLIGHT, linewidths=0)
        ax.set_title(cat, fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.flat[len(cats):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig
