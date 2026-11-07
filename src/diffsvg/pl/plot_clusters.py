"""Plots for the output of `diffsvg.tl.cluster_curves`."""

import colorcet
import matplotlib.pyplot as plt
import numpy as np

from diffsvg.tl.model import _gene_profiles


def _display_tables(adata, curves, by, curves_key, profiles_key):
    """``{panel title suffix: genes x bins table}``, one entry per panel of a cluster."""
    # `_gene_profiles` prefers a `fit_profiles` result, which is not the curve `by="average"` compared.
    if curves == "combined" and by != "average":
        return {"": _gene_profiles(adata, curves_key, profiles_key)[0]}

    key0, key1 = f"{curves_key}_cond0_mean", f"{curves_key}_cond1_mean"
    if key0 not in adata.varm:
        raise KeyError(f"adata.varm[{key0!r}] not found - run `diffsvg.tl.fit_model` first, with `inplace=True`.")
    cond0, cond1 = adata.varm[key0], adata.varm[key1]
    labels = adata.uns.get(f"{curves_key}_condition_categories", {"0": "condition 0", "1": "condition 1"})
    if curves == "combined":
        return {"": (cond0 + cond1) / 2.0}
    if curves == "conditions":
        return {labels["0"]: cond0, labels["1"]: cond1}
    if curves == "difference":
        return {f"{labels['1']} - {labels['0']}": cond1 - cond0}
    if curves == labels["0"]:
        return {curves: cond0}
    if curves == labels["1"]:
        return {curves: cond1}
    available = ["combined", "conditions", "difference", labels["0"], labels["1"]]
    raise ValueError(f"Unknown curves {curves!r}. Available: {available}")


def _standardise(values, standardize):
    """Per gene over all of `values` (genes x points, panels side by side), so a gap between panels survives."""
    if standardize == "none":
        return values
    values = values - values.mean(axis=1, keepdims=True)
    if standardize == "zscore":
        sd = values.std(axis=1, keepdims=True)
        values = values / np.where(sd > 0, sd, 1.0)
    return values


def _legend_cols(n_genes):
    """Legend columns for a cluster of `n_genes`, about 24 names to a column (a panel's height at the default size)."""
    return (n_genes + 23) // 24


#: Inches around a panel: the axis labels on its left, the title above and the x-axis label below it.
_PANEL_MARGINS = {"left": 0.65, "right": 0.15, "top": 0.4, "bottom": 0.55}
#: Inches a legend column takes next to a panel.
_LEGEND_COLUMN_WIDTH = 1.0

_Y_LABELS = {"none": "smoothed expression", "center": "centred expression", "zscore": "z-scored expression"}
_DEFAULT_CURVES = {"interaction": "difference", "average": "combined", "profile": "combined"}
_GENE_COLOURS = colorcet.b_glasbey_category10


def plot_cluster_curves(
    adata,
    cluster_key="dsvg_cluster_average",
    curves=None,
    clusters=None,
    standardize=None,
    curves_key="dsvg_curves",
    profiles_key="dsvg_profiles",
    ncols=3,
    figsize=(4, 3.2),
    show_legend=True,
    dpi=300,
    save_path=None,
):
    """One curve plot per cluster, with the fitted curve of each gene overlaid in its own colour.

    Parameters
    ----------
    adata
        Annotated data matrix that `diffsvg.tl.cluster_curves` wrote into.
    cluster_key
        The clustering, i.e. ``key_added`` of `diffsvg.tl.cluster_curves`.
    curves
        Which curves to draw:

        - ``"combined"``: one curve per gene, the mean of the two condition
          curves. For ``by="profile"``, and for labels that did not come from
          `diffsvg.tl.cluster_curves`, the `diffsvg.tl.fit_profiles` fit if
          there is one.
        - ``"conditions"``: two panels per cluster, one per condition, in a
          shared y-range and with the same gene colours.
        - ``"difference"``: the difference curve ``f1 - f0``.
        - A condition label, e.g. ``"SHH"``: that condition's curve only, one
          panel per cluster.

        Defaults to the curves the clustering compared: ``"difference"`` for
        ``by="interaction"``, otherwise ``"combined"``.
    clusters
        Clusters to draw, by default all of them.
    standardize
        Per gene: ``"none"`` (expression as fitted), ``"center"`` (mean
        removed, so levels do not separate the curves; the clustering does not
        see levels either) or ``"zscore"`` (also unit variance). With two
        panels the mean and variance are taken over both, so the gap between
        the conditions stays visible. Defaults to ``"zscore"`` if the
        clustering scaled the genes, otherwise ``"center"``.
    curves_key, profiles_key
        Prefixes passed to `diffsvg.tl.fit_model` / `diffsvg.tl.fit_profiles`.
    ncols
        Panels per row. With ``curves="conditions"`` a row always holds one
        cluster, so this is ignored.
    figsize
        ``(width, height)`` of a single panel in inches.
    show_legend
        Name the genes in a legend, per cluster. Colours repeat only between
        clusters, not within one.
    dpi
        Resolution used when saving to `save_path`.
    save_path
        If given, save the figure here (``bbox_inches="tight"``).

    Returns
    -------
    An array of `matplotlib.axes.Axes`, row by row: one panel per cluster, or
    two with ``curves="conditions"``. Each gene is a line labelled with its name.

    Examples
    --------
    >>> dvg.tl.cluster_curves(adata, by="interaction", n_clusters=4)
    >>> dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", curves="conditions")
    """
    if cluster_key not in adata.var:
        raise KeyError(f"adata.var[{cluster_key!r}] not found - run `diffsvg.tl.cluster_curves` first.")
    params = adata.uns.get(cluster_key, {})
    if curves is None:
        curves = _DEFAULT_CURVES.get(params.get("by"), "combined")
    if standardize is None:
        standardize = "zscore" if params.get("scale", False) else "center"
    if standardize not in _Y_LABELS:
        raise ValueError(f"Unknown standardize {standardize!r}. Available: {sorted(_Y_LABELS)}")

    tables = _display_tables(adata, curves, params.get("by"), curves_key, profiles_key)
    assignment = adata.var[cluster_key].dropna()
    if clusters is None:
        clusters = list(assignment.cat.categories)
    else:
        clusters = [str(c) for c in ([clusters] if isinstance(clusters, str | int) else clusters)]
        unknown = [c for c in clusters if c not in assignment.cat.categories]
        if unknown:
            raise KeyError(f"clusters not found in adata.var[{cluster_key!r}]: {unknown}")

    width, height = figsize
    n_panels = len(clusters) * len(tables)
    ncols = len(tables) if len(tables) > 1 else max(1, min(ncols, n_panels))
    nrows = (n_panels + ncols - 1) // ncols
    # The legends sit next to their panels: the gap holds the widest one and the figure grows by it, so
    # panels keep the size asked for. With two panels per cluster the legend is right of the row, not in its gap.
    largest = max((assignment == c).sum() for c in clusters)
    legend_width = _LEGEND_COLUMN_WIDTH * _legend_cols(largest) if show_legend else 0.0
    gap_x = _PANEL_MARGINS["left"] + _PANEL_MARGINS["right"] + (legend_width if len(tables) == 1 else 0.0)
    gap_y = _PANEL_MARGINS["top"] + _PANEL_MARGINS["bottom"]
    fig_width = ncols * width + (ncols - 1) * gap_x + _PANEL_MARGINS["left"] + _PANEL_MARGINS["right"] + legend_width
    fig_height = nrows * height + (nrows - 1) * gap_y + gap_y
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={
            "left": _PANEL_MARGINS["left"] / fig_width,
            "right": 1.0 - (_PANEL_MARGINS["right"] + legend_width) / fig_width,
            "bottom": _PANEL_MARGINS["bottom"] / fig_height,
            "top": 1.0 - _PANEL_MARGINS["top"] / fig_height,
            "wspace": gap_x / width,
            "hspace": gap_y / height,
        },
    )
    axes = axes.ravel()
    for spare in axes[n_panels:]:
        spare.set_visible(False)

    x = next(iter(tables.values())).columns.to_numpy(dtype=float)
    panels = iter(axes)
    for cluster in clusters:
        genes = list(assignment.index[assignment == cluster])
        joined = _standardise(np.hstack([t.loc[genes].to_numpy(dtype=float) for t in tables.values()]), standardize)
        lines = np.split(joined, len(tables), axis=1)
        # A cluster's panels share a y-range, so the conditions can be compared; clusters keep their own.
        margin = 0.05 * (np.nanmax(joined) - np.nanmin(joined)) or 1.0
        ylim = (float(np.nanmin(joined) - margin), float(np.nanmax(joined) + margin))
        for i, (suffix, values) in enumerate(zip(tables, lines, strict=True)):
            ax = next(panels)
            for g, gene in enumerate(genes):
                ax.plot(x, values[g], color=_GENE_COLOURS[g % len(_GENE_COLOURS)], label=gene, lw=1.2)
            ax.set_ylim(ylim)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Normalized distance")
            ax.set_ylabel(_Y_LABELS[standardize])
            ax.set_title(f"Cluster {cluster} (n={len(genes)})" + (f", {suffix}" if suffix else ""), fontsize=11)
            # With two panels per cluster one legend is enough; it goes next to the second.
            if show_legend and i == len(tables) - 1:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    ncols=_legend_cols(len(genes)),
                    frameon=False,
                    fontsize=7,
                    borderaxespad=0.0,
                    handlelength=1.2,
                    columnspacing=0.8,
                )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return axes[:n_panels]
