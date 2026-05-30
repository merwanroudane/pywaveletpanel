"""
Publication-grade visualisations for wavelet panel econometrics.

All plots use a custom dark/modern journal theme with:
- Vibrant colour palette (teal, coral, gold, violet)
- Smooth gradients and subtle grid
- Professional typography (uses system sans-serif)
- Automatic significance highlighting

Author: Dr. Merwan Roudane (merwanroudane920@gmail.com)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pywaveletpanel.panel_regression import ScaleRegressionResult
    from pywaveletpanel.structural_breaks import BreakDetectionResult
    from pywaveletpanel.unit_root import UnitRootResult

# ══════════════════════════════════════════════════════════════════════════════
# Journal Style Theme
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette — light "journal / paper" theme.
# Line colours follow the colourblind-safe Okabe–Ito palette, which reads
# cleanly on a white background and reproduces well in print/greyscale.
COLORS = {
    "bg":          "#ffffff",  # figure background (white)
    "bg_light":    "#ffffff",  # axes background (white)
    "fg":          "#1a1a1a",  # text / axes (near-black)
    "grid":        "#cfcfcf",  # subtle grey grid
    "teal":        "#0072B2",  # blue
    "coral":       "#D55E00",  # vermillion
    "gold":        "#E69F00",  # orange
    "violet":      "#CC79A7",  # reddish purple
    "blue":        "#56B4E9",  # sky blue
    "green":       "#009E73",  # bluish green
    "orange":      "#E69F00",  # orange
    "pink":        "#CC79A7",  # reddish purple
    "cyan":        "#56B4E9",  # sky blue
    "red":         "#C00000",  # red (break / reference lines)
}

PALETTE = [
    COLORS["teal"], COLORS["coral"], COLORS["green"], COLORS["violet"],
    COLORS["gold"], COLORS["blue"], COLORS["fg"], COLORS["pink"],
]


def set_journal_style():
    """
    Apply the PyWaveletPanel publication theme globally.

    Call this at the start of your script to set a consistent, clean
    journal/paper theme (white background, serif type, black axes) for all
    matplotlib plots.
    """
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     COLORS["fg"],
        "axes.linewidth":     0.8,
        "axes.labelcolor":    COLORS["fg"],
        "axes.grid":          True,
        "grid.color":         COLORS["grid"],
        "grid.alpha":         0.6,
        "grid.linewidth":     0.6,
        "grid.linestyle":     "-",
        "text.color":         COLORS["fg"],
        "xtick.color":        COLORS["fg"],
        "ytick.color":        COLORS["fg"],
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.facecolor":   "white",
        "legend.edgecolor":   COLORS["grid"],
        "legend.framealpha":  0.9,
        "legend.labelcolor":  COLORS["fg"],
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })


def _apply_style(fig, axes):
    """Apply the theme to a specific figure."""
    fig.patch.set_facecolor("white")
    if not isinstance(axes, np.ndarray):
        axes = [axes] if not isinstance(axes, list) else axes
    else:
        axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors=COLORS["fg"])
        ax.spines["bottom"].set_color(COLORS["fg"])
        ax.spines["left"].set_color(COLORS["fg"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.5, linestyle="-", linewidth=0.6, color=COLORS["grid"])
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(COLORS["fg"])
        ax.yaxis.label.set_color(COLORS["fg"])
        ax.title.set_color(COLORS["fg"])


# ══════════════════════════════════════════════════════════════════════════════
# 1. Wavelet Decomposition Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_wavelet_decomposition(
    x: np.ndarray,
    components: dict,
    title: str = "MODWT Multiresolution Decomposition",
    time_index: Optional[np.ndarray] = None,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot original signal and its MRA components (D1..DJ, SJ).

    Parameters
    ----------
    x : ndarray
        Original time series.
    components : dict
        Output of ``modwt_mra()``. Keys: ``'D1', 'D2', ..., 'SJ'``.
    title : str
    time_index : ndarray, optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    labels_map = components.get("labels", {})
    keys = [k for k in components if k != "labels"]
    # Sort: S first, then D in descending order
    smooth_keys = sorted([k for k in keys if k.startswith("S")])
    detail_keys = sorted([k for k in keys if k.startswith("D")],
                         key=lambda k: int(k[1:]), reverse=True)
    ordered = ["Original"] + smooth_keys + detail_keys

    n_panels = len(ordered)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    _apply_style(fig, axes)

    t = time_index if time_index is not None else np.arange(len(x))

    for i, key in enumerate(ordered):
        ax = axes[i]
        color = COLORS["teal"] if i == 0 else PALETTE[(i - 1) % len(PALETTE)]
        if key == "Original":
            ax.plot(t, x, color=color, linewidth=1.2, alpha=0.9)
            ax.set_ylabel("Original", fontsize=10, fontweight="bold")
        else:
            data = components[key]
            ax.plot(t[:len(data)], data, color=color, linewidth=1.0, alpha=0.85)
            label = labels_map.get(key, key)
            ax.set_ylabel(f"{key}\n({label})", fontsize=9, fontweight="bold")
            ax.axhline(y=0, color=COLORS["fg"], alpha=0.2, linewidth=0.5)

    axes[0].set_title(title, fontsize=15, fontweight="bold", pad=12)
    axes[-1].set_xlabel("Time", fontsize=12)
    fig.tight_layout(h_pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scale-Dependent Coefficient Plot (Papers 2, 4)
# ══════════════════════════════════════════════════════════════════════════════

def plot_scale_coefficients(
    result: "ScaleRegressionResult",
    figsize: tuple = (10, 7),
    ci_level: float = 0.05,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Forest plot of regression coefficients at each wavelet scale.

    Shows coefficient estimates with confidence intervals, with
    significant coefficients highlighted in green.

    Parameters
    ----------
    result : ScaleRegressionResult
    figsize : tuple
    ci_level : float
        Significance level for confidence intervals (default 5%).
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from scipy import stats as sp_stats

    scales = []
    coefs = []
    ses = []
    pvals = []
    freq_labels = []

    # Order: D1, D2, ..., DJ, SJ (bottom to top on plot)
    for j in range(1, result.level + 1):
        key = f"D{j}"
        if key in result.scale_results:
            res = result.scale_results[key]
            scales.append(key)
            coefs.append(res["coef"][0])
            ses.append(res["se"][0])
            pvals.append(res["pvalue"][0])
            freq_labels.append(result.scale_labels.get(key, key))

    s_key = f"S{result.level}"
    if s_key in result.scale_results:
        res = result.scale_results[s_key]
        scales.append(s_key)
        coefs.append(res["coef"][0])
        ses.append(res["se"][0])
        pvals.append(res["pvalue"][0])
        freq_labels.append(result.scale_labels.get(s_key, s_key))

    if result.aggregate_result:
        scales.append("Aggregate")
        coefs.append(result.aggregate_result["coef"][0])
        ses.append(result.aggregate_result["se"][0])
        pvals.append(result.aggregate_result["pvalue"][0])
        freq_labels.append("Raw data")

    coefs = np.array(coefs)
    ses = np.array(ses)
    pvals = np.array(pvals)
    n_scales = len(scales)
    z = sp_stats.norm.ppf(1 - ci_level / 2)
    ci_lo = coefs - z * ses
    ci_hi = coefs + z * ses

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    y_pos = np.arange(n_scales)
    for i in range(n_scales):
        is_sig = pvals[i] < ci_level
        color = COLORS["green"] if is_sig else COLORS["coral"]
        marker_size = 12 if is_sig else 8
        alpha = 1.0 if is_sig else 0.6

        # Confidence interval line
        ax.plot([ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]],
                color=color, linewidth=2.5, alpha=alpha, solid_capstyle="round")
        # Coefficient point
        ax.plot(coefs[i], y_pos[i], "o", color=color, markersize=marker_size,
                alpha=alpha, markeredgecolor="white", markeredgewidth=1.5,
                zorder=5)
        # Value annotation
        stars = "***" if pvals[i] <= 0.01 else ("**" if pvals[i] <= 0.05 else
                ("*" if pvals[i] <= 0.10 else ""))
        ax.annotate(
            f"{coefs[i]:.3f}{stars}", (coefs[i], y_pos[i]),
            textcoords="offset points", xytext=(15, 5),
            fontsize=10, color=color, fontweight="bold" if is_sig else "normal",
        )

    # Zero reference line
    ax.axvline(x=0, color=COLORS["fg"], alpha=0.4, linewidth=1, linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s}\n{fl}" for s, fl in zip(scales, freq_labels)],
                       fontsize=10)
    ax.set_xlabel("Coefficient estimate", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Scale-by-Scale Regression Coefficients\n"
        f"Wavelet: {result.wavelet.upper()} | N={result.n_entities}, T={result.n_periods}",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=COLORS["green"], markersize=10,
               label=f"Significant (p < {ci_level})", markerfacecolor=COLORS["green"],
               linewidth=2),
        Line2D([0], [0], marker="o", color=COLORS["coral"], markersize=7,
               label="Not significant", markerfacecolor=COLORS["coral"],
               linewidth=2, alpha=0.6),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Structural Break Plot (Paper 1)
# ══════════════════════════════════════════════════════════════════════════════

def plot_structural_breaks(
    result: "BreakDetectionResult",
    figsize: tuple = (14, 5),
    time_index: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot time-varying coefficients with detected structural breaks.

    Shows the step-function coefficient path with vertical lines
    at detected break points (like Figure 6 in Bada et al. 2021).

    Parameters
    ----------
    result : BreakDetectionResult
    figsize : tuple
    time_index : ndarray, optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    k = len(result.n_breaks)
    fig, axes = plt.subplots(k, 1, figsize=(figsize[0], figsize[1] * k),
                             sharex=True, squeeze=False)
    _apply_style(fig, axes)

    T = result.n_periods
    t = time_index if time_index is not None else np.arange(T - 1)
    if len(t) > T - 1:
        t = t[:T - 1]

    for p in range(k):
        ax = axes[p, 0]
        name = result.regressor_names[p] if p < len(result.regressor_names) else f"x{p+1}"
        color = PALETTE[p % len(PALETTE)]

        # Plot step-function coefficient path
        intervals = result.stability_intervals[p]
        coefs = result.coefficients[p]

        for seg_idx, ((start, end), coef) in enumerate(zip(intervals, coefs)):
            t_start = t[start] if start < len(t) else t[-1]
            t_end = t[min(end - 1, len(t) - 1)] if end > 0 else t[0]
            ax.hlines(coef, t_start, t_end, color=color,
                      linewidth=2.5, alpha=0.9, zorder=3)
            # Fill region
            ax.fill_between([t_start, t_end], coef - 0.02, coef + 0.02,
                            alpha=0.15, color=color)

        # Break lines
        for bp in result.break_locations[p]:
            if bp < len(t):
                ax.axvline(x=t[bp], color=COLORS["red"], linewidth=1.5,
                           linestyle="--", alpha=0.8, zorder=4)
                ax.annotate(
                    f"Break", (t[bp], ax.get_ylim()[1] * 0.9),
                    fontsize=8, color=COLORS["red"], ha="center",
                    fontweight="bold",
                )

        ax.axhline(y=0, color=COLORS["fg"], alpha=0.2, linewidth=0.5)
        ax.set_ylabel(f"γ({name})", fontsize=12, fontweight="bold")

        # Info box
        n_brk = result.n_breaks[p]
        ax.text(
            0.02, 0.92,
            f"Detected breaks: {n_brk}",
            transform=ax.transAxes, fontsize=10,
            color=COLORS["gold"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg"],
                      edgecolor=COLORS["gold"], alpha=0.8),
        )

    axes[0, 0].set_title(
        f"SAW Structural Break Detection\n"
        f"N={result.n_entities}, T={result.n_periods} | "
        f"Threshold: {result.threshold:.4f}",
        fontsize=14, fontweight="bold", pad=12,
    )
    axes[-1, 0].set_xlabel("Time", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Unit Root Comparison Plot (Papers 3, 5)
# ══════════════════════════════════════════════════════════════════════════════

def plot_unit_root_comparison(
    results: List["UnitRootResult"],
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visual comparison of multiple panel unit root tests.

    Shows test statistics, critical values, and rejection decisions
    as a grouped bar chart.

    Parameters
    ----------
    results : list of UnitRootResult
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _apply_style(fig, [ax1, ax2])

    names = []
    pvals = []
    reject_5 = []
    for r in results:
        short_name = r.test_name.split("(")[0].strip()
        if len(short_name) > 20:
            short_name = short_name[:18] + "…"
        names.append(short_name)
        pvals.append(r.pvalue)
        reject_5.append(r.reject_null.get(0.05, False))

    x = np.arange(len(names))
    bar_colors = [COLORS["green"] if r else COLORS["coral"] for r in reject_5]

    # Left panel: p-values
    bars = ax1.bar(x, pvals, color=bar_colors, alpha=0.85, edgecolor="white",
                   linewidth=0.8, width=0.6)
    ax1.axhline(y=0.05, color=COLORS["gold"], linewidth=2, linestyle="--",
                label="5% significance", alpha=0.8)
    ax1.axhline(y=0.01, color=COLORS["orange"], linewidth=1.5, linestyle=":",
                label="1% significance", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("p-value", fontsize=12, fontweight="bold")
    ax1.set_title("P-values by Test", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    # Annotate bars
    for bar, pv in zip(bars, pvals):
        stars = "***" if pv <= 0.01 else ("**" if pv <= 0.05 else
                ("*" if pv <= 0.10 else "n.s."))
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{pv:.3f}\n{stars}", ha="center", fontsize=9,
                 color=COLORS["fg"], fontweight="bold")

    # Right panel: Reject / Accept
    bar_labels = ["Reject H0\n(Stationary)" if r else "Fail to reject\n(Unit root)"
                  for r in reject_5]
    ax2.bar(x, [1] * len(x), color=bar_colors, alpha=0.85,
            edgecolor="white", linewidth=0.8, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_yticks([])
    ax2.set_title("Decision at 5% level", fontsize=13, fontweight="bold")
    for i, (lbl, rej) in enumerate(zip(bar_labels, reject_5)):
        symbol = "✓" if rej else "✗"
        color = COLORS["green"] if rej else COLORS["coral"]
        ax2.text(i, 0.5, f"{symbol}\n{lbl}", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")

    fig.suptitle(
        "Panel Unit Root Test Comparison",
        fontsize=15, fontweight="bold", color=COLORS["fg"], y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Loess Scatter by Country (Paper 4)
# ══════════════════════════════════════════════════════════════════════════════

def plot_loess_by_country(
    x_dict: Dict[str, np.ndarray],
    y_dict: Dict[str, np.ndarray],
    xlabel: str = "Productivity growth",
    ylabel: str = "Unemployment rate",
    title: str = "Nonparametric Loess Fit by Country",
    span: float = 0.5,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plots with LOESS smooth for each country at a given scale.

    Reproduces Figures 2–5 in Gallegati et al. (2015).

    Parameters
    ----------
    x_dict : dict
        ``{country_name: x_array}``
    y_dict : dict
        ``{country_name: y_array}``
    xlabel, ylabel, title : str
    span : float
        LOESS smoothing parameter (0–1).
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    countries = list(x_dict.keys())
    n = len(countries)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    _apply_style(fig, axes)

    for idx, country in enumerate(countries):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        x_data = np.asarray(x_dict[country])
        y_data = np.asarray(y_dict[country])
        color = PALETTE[idx % len(PALETTE)]

        # Scatter
        ax.scatter(x_data, y_data, color=color, alpha=0.5, s=20,
                   edgecolors="white", linewidths=0.3)

        # LOESS fit (using numpy polyfit as simple approximation)
        if len(x_data) > 5:
            sort_idx = np.argsort(x_data)
            x_s = x_data[sort_idx]
            y_s = y_data[sort_idx]
            # Simple moving average smoother
            window = max(int(len(x_s) * span), 3)
            if window % 2 == 0:
                window += 1
            from scipy.ndimage import uniform_filter1d
            y_smooth = uniform_filter1d(y_s, size=window, mode="reflect")
            ax.plot(x_s, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        ax.set_title(country, fontsize=11, fontweight="bold", color=color)
        ax.axhline(y=0, color=COLORS["fg"], alpha=0.15, linewidth=0.5)
        ax.axvline(x=0, color=COLORS["fg"], alpha=0.15, linewidth=0.5)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.supxlabel(xlabel, fontsize=13, fontweight="bold", color=COLORS["fg"])
    fig.supylabel(ylabel, fontsize=13, fontweight="bold", color=COLORS["fg"])
    fig.suptitle(title, fontsize=15, fontweight="bold", color=COLORS["fg"], y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
