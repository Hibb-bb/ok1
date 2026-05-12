"""
Plotting helpers aligned with ok1/new_style.py::plot_combined_3x3_plus_right
(left 3×3 panels: errorbar style, fonts, no grid on recall axes).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stable series order so default color cycle matches across runs
DEFAULT_MODEL_ORDER = ("Karcher-Flow", "DAM", "MHN", "geo_distance", "square_distance", "identity")


def apply_combined_figure_rcparams(font_size: int = 10) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )


def plot_recall_df_errorbars(
    ax,
    df: pd.DataFrame,
    *,
    model_colors: dict[str, str] | None = None,
    font_size: int = 10,
    show_legend: bool = False,
) -> None:
    """Recall vs M with mean ± SD error bars (matches new_style 3×3 left panels)."""
    models = [m for m in DEFAULT_MODEL_ORDER if m in set(df["model"].unique())]
    models += [m for m in sorted(df["model"].unique()) if m not in models]

    for model in models:
        g = df[df["model"] == model].sort_values("M")
        if g.empty:
            continue
        x = g["M"].to_numpy()
        xu = np.unique(x)
        y_mean = g.groupby("M")["recall rate"].mean().reindex(xu).to_numpy()
        y_sd = g.groupby("M")["recall rate"].std(ddof=1).reindex(xu).to_numpy()

        color = None
        if model_colors is not None and model in model_colors:
            color = model_colors[model]

        ax.errorbar(
            x=xu,
            y=y_mean,
            yerr=y_sd,
            fmt="-o",
            markersize=1,
            alpha=1,
            capsize=1,
            label=str(model),
            linewidth=1,
            elinewidth=0.5,
            color=color,
        )

    ax.set_xscale("log")
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    ax.tick_params(labelsize=font_size)
    ax.grid(False)


def decorate_combined_grid_cell(ax, *, row: int, col: int, font_size: int = 10) -> None:
    """Axis decoration for one cell of the capacity 3×3 grid."""
    ax.set_title("")
    if col == 0 and row == 1:
        ax.set_ylabel("Recall rate", fontsize=font_size)
    else:
        ax.set_ylabel("")

    if col == 0:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.0", "", "0.5", "", "1.0"], fontsize=font_size)
    else:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])

    if row == 2:
        ax.set_xlabel("M", fontsize=font_size)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    ax.grid(False)


def save_single_panel_recall_plot(
    df: pd.DataFrame,
    plot_path: str | Path,
    *,
    title: str,
    model_colors: dict[str, str] | None = None,
    font_size: int = 10,
    figsize: tuple[float, float] | None = None,
) -> None:
    """One recall-rate panel: same errorbar/visual style as plot_combined_3x3_plus_right left cells."""
    apply_combined_figure_rcparams(font_size=font_size)
    w, h = figsize if figsize is not None else (6.0, 5.0)
    fig, ax = plt.subplots(figsize=(w, h))

    plot_recall_df_errorbars(
        ax, df, model_colors=model_colors, font_size=font_size, show_legend=True
    )
    ax.legend(title="Model", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("M", fontsize=font_size)
    ax.set_ylabel("Recall rate", fontsize=font_size)
    ax.grid(False)

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
