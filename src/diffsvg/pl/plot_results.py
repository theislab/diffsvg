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

# ============================================================
# Label placement helpers (used by plot_volcano)
# ============================================================
# Lightweight, dependency-free stand-in for a package like `adjustText`:
# for each label, try a ring of candidate offsets around its point and use
# the first one whose (rendered) bounding box doesn't overlap anything
# already placed; fall back to the least-overlapping candidate. Best-effort,
# not a guarantee - true only "if possible", as requested.


def _get_renderer(fig):
    try:
        fig.canvas.draw()
        return fig.canvas.get_renderer()
    except Exception:  # noqa: BLE001 - collision avoidance is a nice-to-have, never worth crashing the plot
        return None


def _bbox_tuple(bbox):
    return (bbox.x0, bbox.y0, bbox.x1, bbox.y1)


def _boxes_overlap(a, b, pad=2.0):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def _dot_boxes(ax, xs, ys, marker_size, pad_px=2.0):
    """Small display-coordinate boxes around each `(x, y)` scatter point, for label-collision checks."""
    if len(xs) == 0:
        return []
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = np.sqrt(marker_size / np.pi) * ax.get_figure().dpi / 72.0 + pad_px
    return [(px - r, py - r, px + r, py + r) for px, py in pts]


def _place_label(
    ax,
    renderer,
    x,
    y,
    text,
    occupied,
    fontsize=7,
    color="black",
    fontweight="normal",
    preferred_offset=None,
    base_radius=10.0,
    n_angles=10,
    n_rings=4,
    arrow=False,
    bbox_kwargs=None,
    zorder=12,
):
    """Place `text` near data point `(x, y)`, avoiding the display-coordinate boxes in `occupied`.

    Tries `preferred_offset` (data units) first if given, then a ring of
    candidate pixel offsets around the point; picks the first candidate
    whose label box doesn't overlap anything in `occupied`, or the
    least-overlapping one if none is fully clear. The chosen box is
    appended to `occupied` in place, so later calls avoid it too.

    Returns the created `matplotlib.text.Annotation`. If `renderer` is
    `None` (couldn't be obtained for this backend), skips collision
    avoidance and just places the label at `preferred_offset` (or the
    point itself).
    """
    point_disp = ax.transData.transform((x, y))
    arrowprops = {"arrowstyle": "->", "lw": 1.0, "color": color} if arrow else None

    def _measure(dx, dy):
        # No `arrowprops` here: `Annotation.get_window_extent` includes the
        # arrow's own bounding box (which reaches all the way back to the
        # data point), not just the text glyph - measuring that would treat
        # the arrow's sweep as occupied space and reject candidates for the
        # wrong reason. The text-only box is what we actually want to keep
        # labels clear of.
        cand = ax.transData.inverted().transform((point_disp[0] + dx, point_disp[1] + dy))
        ann = ax.annotate(
            text,
            xy=(x, y),
            xytext=cand,
            textcoords="data",
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            ha="center",
            va="center",
            zorder=zorder,
            bbox=bbox_kwargs,
        )
        return cand, ann

    def _finalize(cand, ann):
        if arrowprops is None:
            return ann
        ann.remove()
        return ax.annotate(
            text,
            xy=(x, y),
            xytext=cand,
            textcoords="data",
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            ha="center",
            va="center",
            zorder=zorder,
            bbox=bbox_kwargs,
            arrowprops=arrowprops,
        )

    if renderer is None:
        dx, dy = (
            (0.0, 0.0)
            if preferred_offset is None
            else tuple(ax.transData.transform((x + preferred_offset[0], y + preferred_offset[1])) - point_disp)
        )
        cand, ann = _measure(dx, dy)
        return _finalize(cand, ann)

    candidates = []
    if preferred_offset is not None:
        pdx, pdy = preferred_offset
        candidates.append(tuple(ax.transData.transform((x + pdx, y + pdy)) - point_disp))
    for ring in range(n_rings):
        radius = base_radius * (1 + ring)
        for i in range(n_angles):
            angle = 2 * np.pi * i / n_angles + ring * 0.35
            candidates.append((np.cos(angle) * radius, np.sin(angle) * radius))

    best, best_score, best_box, best_cand = None, None, None, None
    for dx, dy in candidates:
        cand, ann = _measure(dx, dy)
        box = _bbox_tuple(ann.get_window_extent(renderer=renderer))
        score = sum(1 for other in occupied if _boxes_overlap(box, other))
        if score == 0:
            if best is not None:
                best.remove()
            occupied.append(box)
            return _finalize(cand, ann)
        # On ties, prefer the later (farther-explored) candidate over the
        # first one seen: when several labels share nearly the same
        # preferred position, the first-tried candidate is exactly the one
        # most likely to collide with a sibling label placed just before it,
        # so always settling on it would leave them stacked.
        if best_score is None or score <= best_score:
            if best is not None:
                best.remove()
            best, best_score, best_box, best_cand = ann, score, box, cand
        else:
            ann.remove()

    occupied.append(best_box)
    return _finalize(best_cand, best)


def plot_volcano(
    stats_df,
    x="integral",
    pvalue="gamma_pval",
    pvalue_adj="gamma_pval_adj",
    alpha=0.05,
    show_dots=True,
    n_label=20,
    highlight_genes=None,
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
        Column used for the y-axis, plotted as ``-log10(pvalue)``. Every
        point above the ``-log10(alpha)`` reference line is colored by the
        sign of `x` (blue = positive, red = negative); points at or below it
        are black.
    pvalue_adj
        Column used to rank genes for labeling (the `n_label` genes with the
        smallest `pvalue_adj` among those above the line get a text label).
    alpha
        Significance threshold on `pvalue`, drawn as a horizontal reference
        line at ``-log10(alpha)`` and used for the color cutoff described
        under `pvalue`.
    show_dots
        If `True` (default), every gene is drawn as a colored dot. If
        `False`, every gene's name is drawn instead, in the same color, in
        place of a dot - there is no separate background scatter in that
        mode. Best suited to modest gene panels (a few hundred genes);
        subset `stats_df` first for larger ones.
    n_label
        Label at most this many of the genes above the line (the ones with
        the smallest `pvalue_adj`) with their gene name. Use `None` to label
        every gene above the line, or `0` to label none. Labels use the
        same collision-avoidance as `highlight_genes` (see below).
    highlight_genes
        A gene name, or list of gene names, to draw with a larger red point
        and a bold, boxed, arrow-annotated label on top of the rest. Raises
        `KeyError` if any given gene is not in `stats_df.index`.
    highlight_offset
        ``(dx, dy)`` offset (in data units) tried first for each
        `highlight_genes` label; if that spot is already occupied (by a
        dot, another gene's name, or another label), a nearby spot is used
        instead - so with several highlighted genes this is a preference,
        not a guarantee.
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
    fig = ax.get_figure()

    if highlight_genes is None:
        highlight_genes = []
    elif isinstance(highlight_genes, str):
        highlight_genes = [highlight_genes]
    else:
        highlight_genes = list(highlight_genes)
    missing = [g for g in highlight_genes if g not in stats_df.index]
    if missing:
        raise KeyError(f"highlight_genes not found in stats_df.index: {missing}")

    x_vals = stats_df[x]
    minus_log10_p = -np.log10(stats_df[pvalue])
    threshold = -np.log10(alpha)
    sig = minus_log10_p > threshold
    pos_sign = x_vals > 0
    neg_sign = x_vals < 0

    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.axhline(threshold, color="gray", linestyle="--", lw=1)

    renderer = _get_renderer(fig) if (n_label != 0 or highlight_genes or not show_dots) else None
    occupied = []

    if show_dots:
        ax.scatter(x_vals[~sig], minus_log10_p[~sig], color="black", s=20, alpha=0.7, linewidth=0)
        ax.scatter(x_vals[sig & pos_sign], minus_log10_p[sig & pos_sign], color="blue", s=20, alpha=0.7, linewidth=0)
        ax.scatter(x_vals[sig & neg_sign], minus_log10_p[sig & neg_sign], color="red", s=20, alpha=0.7, linewidth=0)
        if renderer is not None:
            occupied.extend(_dot_boxes(ax, x_vals.to_numpy(), minus_log10_p.to_numpy(), marker_size=20))
    else:
        for gene in stats_df.index:
            color = "black" if not sig[gene] else ("blue" if pos_sign[gene] else "red")
            txt = ax.text(
                x_vals[gene], minus_log10_p[gene], gene, fontsize=6, color=color, alpha=0.85, ha="center", va="center"
            )
            if renderer is not None:
                occupied.append(_bbox_tuple(txt.get_window_extent(renderer=renderer)))

    label_genes = stats_df.index[sig]
    if n_label is not None and len(label_genes) > n_label:
        label_genes = stats_df.loc[label_genes, pvalue_adj].sort_values().index[:n_label]

    for gene in label_genes:
        if gene in highlight_genes:
            continue  # gets the larger highlight treatment below instead
        _place_label(ax, renderer, x_vals[gene], minus_log10_p[gene], gene, occupied, fontsize=7, base_radius=8.0)

    for gene in highlight_genes:
        hx, hy = x_vals[gene], minus_log10_p[gene]
        ax.scatter(hx, hy, s=80, color="red", edgecolor="black", linewidth=1.5, zorder=10)
        _place_label(
            ax,
            renderer,
            hx,
            hy,
            gene,
            occupied,
            fontsize=13,
            fontweight="bold",
            preferred_offset=highlight_offset,
            base_radius=45.0,
            n_rings=6,
            arrow=True,
            bbox_kwargs={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black", "alpha": 0.9},
            zorder=13,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\int f_0 - f_1$", fontsize=12)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def _draw_prediction_curve(adata, gene, ax, curves_key, condition_labels, colors):
    """Draw one gene's two condition curves (mean + credible band) into `ax`."""
    for cat in (0, 1):
        curves = adata.varm[f"{curves_key}_cond{cat}_mean"]
        mean = curves.loc[gene]
        lower = adata.varm[f"{curves_key}_cond{cat}_ci_low"].loc[gene]
        upper = adata.varm[f"{curves_key}_cond{cat}_ci_high"].loc[gene]
        # Taken from the columns, not from `mean.index`, so the bin grid is
        # read off the table itself rather than depending on what `.loc`
        # happened to return.
        x = curves.columns.to_numpy(dtype=float)

        ax.plot(x, mean.to_numpy(), color=colors[cat], lw=2, label=condition_labels[str(cat)])
        ax.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=colors[cat], alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_prediction_curve(
    adata,
    gene,
    curves_key="dsvg_curves",
    condition_labels=None,
    colors=None,
    ax=None,
    ncols=4,
    figsize=(6, 5),
    save_path=None,
):
    """Plot the predicted expression curve (posterior mean +/- credible band) per gene, for both conditions.

    Reads the six ``adata.varm[f"{curves_key}_cond{0,1}_{mean,ci_low,ci_high}"]``
    tables written by `diffsvg.tl.fit_model`. Several genes are drawn as a
    grid of panels, one gene each.

    Parameters
    ----------
    adata
        Annotated data matrix that `diffsvg.tl.fit_model` wrote into.
    gene
        A gene name, or a list of them (all must be in ``adata.var_names``).
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
        Existing axes to draw into; for several genes, a sequence of at
        least that many axes. A new figure is created if omitted.
    ncols
        Panels per row when several genes are given. The grid is narrowed
        when there are fewer genes than this, so three genes give one row of
        three rather than a row of four with a gap.
    figsize
        Size of a *single* panel. The figure is scaled by the grid shape, so
        one gene still gets exactly `figsize`.
    save_path
        If given, save the figure here (``dpi=300, bbox_inches="tight"``).

    Returns
    -------
    A single `matplotlib.axes.Axes` when one gene is given, otherwise a list
    of them, one per gene in the order passed.
    """
    single = isinstance(gene, str)
    genes = [gene] if single else list(gene)
    if not genes:
        raise ValueError("`gene` is empty - pass at least one gene name.")
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(f"genes not found in adata.var_names: {missing}")
    # One gene is one panel however it was passed, so a one-element list
    # behaves (and returns) exactly like the bare string.
    single = len(genes) == 1

    if condition_labels is None:
        condition_labels = adata.uns.get(f"{curves_key}_condition_categories", {"0": "condition 0", "1": "condition 1"})
    if colors is None:
        colors = {0: "steelblue", 1: "darkorange"}

    if ax is None:
        ncols = max(1, min(ncols, len(genes)))
        nrows = (len(genes) + ncols - 1) // ncols
        _, grid = plt.subplots(nrows, ncols, figsize=(ncols * figsize[0], nrows * figsize[1]), squeeze=False)
        grid = grid.ravel()
        for spare in grid[len(genes) :]:  # trailing cells of a part-filled last row
            spare.set_visible(False)
        axes = list(grid[: len(genes)])
    elif isinstance(ax, plt.Axes):
        if not single:
            raise ValueError(
                f"`ax` is a single Axes but {len(genes)} genes were given - "
                "pass a sequence of at least that many axes, or omit `ax` to get a grid."
            )
        axes = [ax]
    else:
        axes = list(ax)
        if len(axes) < len(genes):
            raise ValueError(f"`ax` holds {len(axes)} axes but {len(genes)} genes were given.")
        axes = axes[: len(genes)]

    for g, a in zip(genes, axes, strict=True):
        _draw_prediction_curve(adata, g, a, curves_key, condition_labels, colors)
        a.set_xlabel("Normalized distance", fontsize=14)
        if single:
            a.set_ylabel(f"smoothed {g} expression", fontsize=14)
        else:
            a.set_title(g, fontsize=14)
            a.set_ylabel("smoothed expression", fontsize=12)
    # One shared legend: repeating it in every panel is just noise.
    axes[0].legend(frameon=False, fontsize=12)

    fig = axes[0].get_figure()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return axes[0] if single else axes


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
