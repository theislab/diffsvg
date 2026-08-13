"""Plots for the output of `diffsvg.tl.fit_model`.

`fit_model` writes:

- ``adata.varm[stats_key]`` (default ``"dsvg_stats"``): per-gene statistics,
  including ``integral`` (area between the two predicted condition curves)
  and ``gamma_pval``/``gamma_pval_adj``/``empirical_pval``/``empirical_pval_adj``.
- ``adata.varm[preds_key]`` (default ``"dsvg_preds"``): the per-gene,
  per-bin ``condition==1`` minus ``condition==0`` difference curve.
- Six ``adata.varm[f"{curves_key}_..."]`` entries (default prefix
  ``"dsvg_curves"``) and ``adata.uns[f"{curves_key}_condition_categories"]``:
  the per-condition predicted curves with a credible band.

The three plots below consume those directly.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_volcano(
    stats_df,
    x="integral",
    pvalue="gamma_pval",
    pvalue_adj="gamma_pval_adj",
    alpha=0.05,
    n_label=20,
    highlight_gene=None,
    highlight_offset=(0.025, 1.25),
    ax=None,
    figsize=(4, 6),
    save_path=None,
):
    """Volcano plot of per-gene curve difference vs. significance.

    Parameters
    ----------
    stats_df
        A statistics table indexed by gene, e.g. ``adata.varm["dsvg_stats"]``.
    x
        Column used for the x-axis (area between the two condition curves).
    pvalue
        Column used for the y-axis, plotted as ``-log10(pvalue)``.
    pvalue_adj
        Column used to decide significance (colored blue/red) and to rank
        genes for labeling.
    alpha
        Significance threshold on `pvalue_adj`; also drawn as a horizontal
        reference line at ``-log10(alpha)``.
    n_label
        Label at most this many significant genes (the ones with the
        smallest `pvalue_adj`). Use `None` to label every significant gene,
        or `0` to label none.
    highlight_gene
        If given, draw this one gene as a larger, annotated point on top of
        the rest.
    highlight_offset
        ``(dx, dy)`` offset (in data units) of the `highlight_gene` label
        from its point.
    ax
        Existing axes to draw into; a new figure is created if omitted.
    figsize
        Passed to `plt.subplots` when `ax` is omitted.
    save_path
        If given, save the figure here (``dpi=300, bbox_inches="tight"``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x_vals = stats_df[x]
    minus_log10_p = -np.log10(stats_df[pvalue])
    sig = stats_df[pvalue_adj] < alpha
    pos_sign = x_vals > 0
    neg_sign = x_vals < 0

    ax.scatter(x_vals[~sig], minus_log10_p[~sig], color="black", s=20, alpha=0.7, linewidth=0)
    ax.scatter(x_vals[sig & pos_sign], minus_log10_p[sig & pos_sign], color="blue", s=20, alpha=0.7, linewidth=0)
    ax.scatter(x_vals[sig & neg_sign], minus_log10_p[sig & neg_sign], color="red", s=20, alpha=0.7, linewidth=0)

    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.axhline(-np.log10(alpha), color="gray", linestyle="--", lw=1)

    label_genes = stats_df.index[sig]
    if n_label is not None and len(label_genes) > n_label:
        label_genes = stats_df.loc[label_genes, pvalue_adj].sort_values().index[:n_label]

    for gene in label_genes:
        ax.text(x_vals[gene], minus_log10_p[gene], gene, fontsize=7, ha="left", va="bottom")

    if highlight_gene is not None:
        hx, hy = x_vals[highlight_gene], minus_log10_p[highlight_gene]
        dx, dy = highlight_offset
        ax.scatter(hx, hy, s=80, color="red", edgecolor="black", linewidth=1.5, zorder=10)
        ax.annotate(
            highlight_gene,
            xy=(hx, hy),
            xytext=(hx + dx, hy + dy),
            fontsize=14,
            fontweight="bold",
            color="black",
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "black"},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black", "alpha": 0.9},
            zorder=11,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\int f_0 - f_1$", fontsize=12)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=12)
    ax.tick_params(labelsize=10)

    fig = ax.get_figure()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_prediction_curve(
    adata,
    gene,
    curves_key="dsvg_curves",
    condition_labels=None,
    colors=None,
    ax=None,
    figsize=(6, 5),
    save_path=None,
):
    """Plot the predicted expression curve (posterior mean +/- credible band) for `gene`, for both conditions.

    Reads the six ``adata.varm[f"{curves_key}_cond{0,1}_{mean,ci_low,ci_high}"]``
    tables written by `diffsvg.tl.fit_model`.

    Parameters
    ----------
    adata
        Annotated data matrix that `diffsvg.tl.fit_model` wrote into.
    gene
        Gene to plot (must be in ``adata.var_names``).
    curves_key
        Prefix used by `diffsvg.tl.fit_model` for the per-condition curve
        `adata.varm` entries.
    condition_labels
        ``{"0": label, "1": label}`` for the legend. Defaults to
        ``adata.uns[f"{curves_key}_condition_categories"]`` if present,
        otherwise generic ``"condition 0"``/``"condition 1"`` labels.
    colors
        ``{0: color, 1: color}`` for the two curves.
    ax
        Existing axes to draw into; a new figure is created if omitted.
    figsize
        Passed to `plt.subplots` when `ax` is omitted.
    save_path
        If given, save the figure here (``dpi=300, bbox_inches="tight"``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if condition_labels is None:
        condition_labels = adata.uns.get(
            f"{curves_key}_condition_categories", {"0": "condition 0", "1": "condition 1"}
        )
    if colors is None:
        colors = {0: "steelblue", 1: "darkorange"}

    for cat in (0, 1):
        mean = adata.varm[f"{curves_key}_cond{cat}_mean"].loc[gene]
        lower = adata.varm[f"{curves_key}_cond{cat}_ci_low"].loc[gene]
        upper = adata.varm[f"{curves_key}_cond{cat}_ci_high"].loc[gene]
        x = mean.index.to_numpy(dtype=float)

        ax.plot(x, mean.to_numpy(), color=colors[cat], lw=2, label=condition_labels[str(cat)])
        ax.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=colors[cat], alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Normalized distance", fontsize=14)
    ax.set_ylabel(f"smoothed {gene} expression", fontsize=14)
    ax.legend(frameon=False, fontsize=12)

    fig = ax.get_figure()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_pvalue_histogram(
    stats_df,
    pvalue="empirical_pval",
    bins=20,
    ax=None,
    figsize=(4, 4),
    save_path=None,
):
    """QC histogram of a per-gene p-value column; should look roughly uniform under the null.

    Parameters
    ----------
    stats_df
        A statistics table indexed by gene, e.g. ``adata.varm["dsvg_stats"]``.
    pvalue
        Column to histogram.
    bins
        Passed to `matplotlib.axes.Axes.hist`.
    ax
        Existing axes to draw into; a new figure is created if omitted.
    figsize
        Passed to `plt.subplots` when `ax` is omitted.
    save_path
        If given, save the figure here (``dpi=300, bbox_inches="tight"``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.hist(stats_df[pvalue].dropna(), bins=bins, edgecolor="black", linewidth=1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(pvalue, fontsize=12)
    ax.set_ylabel("count", fontsize=12)

    fig = ax.get_figure()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
