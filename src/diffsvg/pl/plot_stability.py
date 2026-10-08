"""Plots for the output of `diffsvg.tl.check_stability`."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def _clustered_order(similarity):
    """Order runs by hierarchical clustering on ``1 - similarity``, returning ``(order, linkage)``.

    The linkage is what the dendrogram is drawn from; clustering the
    configurations is the point of the plot, since it shows which parameter
    changes land in the same place and which one sits on its own branch.
    """
    distance = 1.0 - similarity.to_numpy()
    distance = (distance + distance.T) / 2.0  # squareform insists on exact symmetry
    np.fill_diagonal(distance, 0.0)
    if distance.shape[0] < 3:
        return list(range(distance.shape[0])), None
    link = linkage(squareform(distance, checks=False), method="average")
    return dendrogram(link, no_plot=True)["leaves"], link


def _heatmap(result, matrix, label, floor, ax, cmap, annotate, title):
    order, link = _clustered_order(matrix)
    ordered = matrix.iloc[order, order]
    values = ordered.to_numpy()
    n = len(ordered)

    fig = ax.get_figure()
    tree_ax, colorbar_ax = None, None
    if ax.get_subplotspec() is not None:
        # Give the dendrogram and the colorbar their own cells rather than
        # letting the colorbar steal width from the heatmap: the dendrogram
        # only lines up with the columns if both share one grid column.
        rows = [1, 4] if link is not None else [0.001, 4]
        grid = ax.get_subplotspec().subgridspec(2, 2, height_ratios=rows, width_ratios=[20, 1], hspace=0.04, wspace=0.04)
        ax.remove()
        if link is not None:
            tree_ax = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[1, 0])
        colorbar_ax = fig.add_subplot(grid[1, 1])

    if tree_ax is not None:
        dendrogram(link, ax=tree_ax, color_threshold=0, above_threshold_color="black", no_labels=True)
        # scipy places leaf i at 10*i + 5; pinning the limits to (0, 10n)
        # makes that the same fractional position as column i under imshow.
        tree_ax.set_xlim(0, 10 * n)
        tree_ax.set_axis_off()

    # Scaled to the data rather than a fixed 0-1: agreement is usually
    # bunched near the top, and a fixed floor would flatten the differences
    # that matter. Negative correlations stay visible.
    image = ax.imshow(values, cmap=cmap, vmin=float(values.min()), vmax=1.0, aspect="auto")
    ax.set_xticks(range(n), ordered.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n), ordered.index, fontsize=7)

    if annotate:
        midpoint = (float(values.min()) + 1.0) / 2
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if values[i, j] < midpoint else "black",
                )

    colorbar = fig.colorbar(image, cax=colorbar_ax, ax=None if colorbar_ax else ax)
    colorbar.set_label(label, fontsize=9)
    if np.isfinite(floor) and float(values.min()) <= floor <= 1.0:
        colorbar.ax.axhline(floor, color="red", lw=1.5)

    # On the dendrogram axes when there is one, so it clears the tree.
    (tree_ax or ax).set_title(title, fontsize=11)
    return ax


def _sensitivity(result, metric, ax, color):
    data = result.sensitivity
    if data.empty:
        raise ValueError(
            "`sensitivity` is empty - it needs configurations that differ from the baseline in exactly "
            "one parameter, which requires at least one grid axis with more than one value."
        )
    labels = [f"{row.parameter}={row.value}" for row in data.itertuples()]
    positions = np.arange(len(data))

    ax.barh(positions, data[metric].to_numpy(), color=color, alpha=0.85)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{metric} vs. baseline", fontsize=10)

    floor = result.noise_floor.get(metric, float("nan"))
    if np.isfinite(floor):
        ax.axvline(floor, color="red", linestyle="--", lw=1.5, label="noise floor")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_stability(
    result,
    kind="heatmap",
    metric="spearman",
    ax=None,
    figsize=None,
    cmap="viridis",
    annotate=True,
    save_path=None,
):
    """Visualize a `diffsvg.tl.check_stability` result.

    Parameters
    ----------
    result
        A `diffsvg.tl.StabilityResult`.
    kind
        ``"heatmap"`` (default) draws the run-by-run agreement matrix with a
        dendrogram above it, so configurations that rank genes the same way
        cluster together. ``"sensitivity"`` draws the one-factor-at-a-time
        view instead: how far each single parameter change moves the ranking
        away from the baseline.
    metric
        ``"spearman"`` (rank correlation over all genes) or ``"top_overlap"``
        (fraction of the top `result.top_k` genes shared). The first is the
        global view, the second is the decision-relevant one.
    ax
        Existing axes to draw into; a new figure is created if omitted. For
        ``kind="heatmap"`` the axes is split to make room for the
        dendrogram, so pass one that can be subdivided (a normal subplot).
    figsize
        Defaults to something that scales with the number of runs.
    cmap, annotate
        Heatmap colormap, and whether to write each cell's value into it
        (readable up to roughly a dozen runs).
    save_path
        If given, save the figure here (``dpi=300, bbox_inches="tight"``).

    Returns
    -------
    matplotlib.axes.Axes
        The axes the data was drawn into. Note that for ``kind="heatmap"``
        this is *not* the `ax` that was passed in, since that one is split.
    """
    if metric not in ("spearman", "top_overlap"):
        raise ValueError(f"Unknown metric {metric!r}; expected 'spearman' or 'top_overlap'.")

    if kind == "heatmap":
        matrix = result.similarity if metric == "spearman" else result.top_overlap
        label = "Spearman rank correlation" if metric == "spearman" else f"Top-{result.top_k} overlap"
        floor = result.noise_floor.get(metric, float("nan"))
        title = f"Ranking agreement across runs (by {result.rank_by})"
        if np.isfinite(floor):
            title += f"\nnoise floor {floor:.3f} (red line on the colorbar)"
        if figsize is None:
            side = max(5.0, 0.55 * len(matrix) + 3.0)
            figsize = (side, side)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        drawn = _heatmap(result, matrix, label, floor, ax, cmap, annotate, title)
    elif kind == "sensitivity":
        if figsize is None:
            figsize = (6.0, max(2.5, 0.4 * len(result.sensitivity) + 1.5))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        drawn = _sensitivity(result, metric, ax, "steelblue")
        drawn.set_title(f"Effect of one parameter change (baseline: {result.baseline})", fontsize=10)
    else:
        raise ValueError(f"Unknown kind {kind!r}; expected 'heatmap' or 'sensitivity'.")

    fig = drawn.get_figure()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return drawn
